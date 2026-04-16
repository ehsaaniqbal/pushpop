import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch

from pushpop.pp0_causal import (
    build_mean_ablation_intervention,
    rollout_supervised_examples,
    select_matched_suffix_pairs,
    stack_ids_from_rollout_prediction,
)
from pushpop.pp0_model import ResidualIntervention, TinyTransformer, TinyTransformerConfig
from pushpop.pp0_training import build_supervised_example
from pushpop.pp0_vocab import TOKEN_TO_ID, VOCAB_TOKENS


class ToyRolloutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(context_length=16)
        vocab = {token: index for index, token in enumerate(VOCAB_TOKENS)}
        self.vocab_size = len(VOCAB_TOKENS)
        self.out_id = vocab["OUT"]
        self.stop_id = vocab["STOP"]
        self.first_id = vocab["5"]
        self.second_id = vocab["6"]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -1e9,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for batch_index in range(batch_size):
            for position in range(sequence_length):
                prefix = input_ids[batch_index, : position + 1].tolist()
                if self.out_id not in prefix:
                    next_token_id = self.stop_id
                else:
                    out_index = prefix.index(self.out_id)
                    generated_suffix = prefix[out_index + 1 :]
                    if not generated_suffix:
                        next_token_id = self.first_id
                    elif generated_suffix == [self.first_id]:
                        next_token_id = self.second_id
                    elif generated_suffix == [self.first_id, self.second_id]:
                        next_token_id = self.stop_id
                    else:
                        next_token_id = self.stop_id
                logits[batch_index, position, next_token_id] = 0.0
        return logits


def _row(
    *,
    example_id: str,
    split: str,
    program_text: str,
    tokens: list[str],
    trace: list[dict[str, object]],
    final_stack: list[int],
    ops_used: list[str],
    literal_values_used: list[int],
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "split": split,
        "program_text": program_text,
        "tokens": tokens,
        "trace": trace,
        "final_stack": final_stack,
        "final_top": final_stack[-1],
        "metadata": {
            "program_length": len(tokens) - 1,
            "trace_length": len(trace),
            "final_depth": len(final_stack),
            "max_depth_reached": max(step["depth_after"] for step in trace),
            "ops_used": ops_used,
            "literal_values_used": literal_values_used,
        },
    }


class PP0CausalTests(unittest.TestCase):
    def test_model_forward_replaces_targeted_residual_state(self) -> None:
        model = TinyTransformer(
            TinyTransformerConfig(
                vocab_size=len(VOCAB_TOKENS),
                context_length=8,
                d_model=16,
                d_mlp=32,
                n_layers=2,
                n_heads=4,
            )
        )
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        replacement = torch.full((16,), 7.0)

        _, hidden_states = model(
            input_ids,
            return_hidden_states=True,
            interventions=[
                ResidualIntervention(
                    layer_name="block_0",
                    position_index=1,
                    replacement_vector=replacement,
                )
            ],
        )

        self.assertTrue(torch.equal(hidden_states[1][0, 1], replacement))

    def test_build_mean_ablation_intervention_uses_feature_mean(self) -> None:
        intervention = build_mean_ablation_intervention(
            {"block_0": torch.tensor([[1.0, 3.0], [5.0, 7.0]])},
            layer_name="block_0",
            position_index=4,
        )

        self.assertEqual(intervention.layer_name, "block_0")
        self.assertEqual(intervention.position_index, 4)
        self.assertTrue(torch.equal(intervention.replacement_vector, torch.tensor([3.0, 5.0])))

    def test_model_forward_adds_targeted_residual_state(self) -> None:
        model = TinyTransformer(
            TinyTransformerConfig(
                vocab_size=len(VOCAB_TOKENS),
                context_length=8,
                d_model=16,
                d_mlp=32,
                n_layers=2,
                n_heads=4,
            )
        )
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        _, baseline_hidden_states = model(input_ids, return_hidden_states=True)
        delta = torch.full((16,), 3.0)

        _, hidden_states = model(
            input_ids,
            return_hidden_states=True,
            interventions=[
                ResidualIntervention(
                    layer_name="block_0",
                    position_index=1,
                    replacement_vector=delta,
                    mode="add",
                )
            ],
        )

        self.assertTrue(torch.allclose(hidden_states[1][0, 1], baseline_hidden_states[1][0, 1] + delta))

    def test_rollout_supervised_examples_generates_expected_tokens(self) -> None:
        example = build_supervised_example(
            _row(
                example_id="test-000000",
                split="test",
                program_text="2 3 END",
                tokens=["2", "3", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [2]},
                    {"depth_after": 2, "stack_after": [2, 3]},
                    {"depth_after": 2, "stack_after": [2, 3]},
                ],
                final_stack=[5, 6],
                ops_used=[],
                literal_values_used=[2, 3],
            )
        )

        prediction = rollout_supervised_examples(
            ToyRolloutModel(),
            [example],
            torch.device("cpu"),
            batch_size=1,
        )[0]

        self.assertEqual(
            stack_ids_from_rollout_prediction(prediction),
            [TOKEN_TO_ID["5"], TOKEN_TO_ID["6"]],
        )

    def test_select_matched_suffix_pairs_requires_same_token_and_suffix(self) -> None:
        rows = [
            _row(
                example_id="test-000000",
                split="test",
                program_text="1 2 ADD END",
                tokens=["1", "2", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [1]},
                    {"depth_after": 2, "stack_after": [1, 2]},
                    {"depth_after": 1, "stack_after": [3]},
                    {"depth_after": 1, "stack_after": [3]},
                ],
                final_stack=[3],
                ops_used=["ADD"],
                literal_values_used=[1, 2],
            ),
            _row(
                example_id="test-000001",
                split="test",
                program_text="4 5 ADD END",
                tokens=["4", "5", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [4]},
                    {"depth_after": 2, "stack_after": [4, 5]},
                    {"depth_after": 1, "stack_after": [9]},
                    {"depth_after": 1, "stack_after": [9]},
                ],
                final_stack=[9],
                ops_used=["ADD"],
                literal_values_used=[4, 5],
            ),
            _row(
                example_id="test-000002",
                split="test",
                program_text="6 7 SUB END",
                tokens=["6", "7", "SUB", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [6]},
                    {"depth_after": 2, "stack_after": [6, 7]},
                    {"depth_after": 1, "stack_after": [9]},
                    {"depth_after": 1, "stack_after": [9]},
                ],
                final_stack=[9],
                ops_used=["SUB"],
                literal_values_used=[6, 7],
            ),
            _row(
                example_id="test-000003",
                split="test",
                program_text="8 1 ADD DUP END",
                tokens=["8", "1", "ADD", "DUP", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [8]},
                    {"depth_after": 2, "stack_after": [8, 1]},
                    {"depth_after": 1, "stack_after": [9]},
                    {"depth_after": 2, "stack_after": [9, 9]},
                    {"depth_after": 2, "stack_after": [9, 9]},
                ],
                final_stack=[9, 9],
                ops_used=["ADD", "DUP"],
                literal_values_used=[1, 8],
            ),
        ]

        pairs = select_matched_suffix_pairs(
            rows,
            position_index=2,
            eligible_local_indices=[0, 1, 2, 3],
            pair_target="top",
            max_pairs=None,
            seed=0,
        )

        self.assertTrue(pairs)
        self.assertTrue(
            all({pair.source_local_index, pair.target_local_index} == {0, 1} for pair in pairs)
        )

    def test_causal_ablation_script_runs_end_to_end(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "causal_ablate_pp0.py"
        rows = [
            _row(
                example_id="test-000000",
                split="test",
                program_text="9 6 ADD DUP END",
                tokens=["9", "6", "ADD", "DUP", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [9]},
                    {"depth_after": 2, "stack_after": [9, 6]},
                    {"depth_after": 1, "stack_after": [5]},
                    {"depth_after": 2, "stack_after": [5, 5]},
                    {"depth_after": 2, "stack_after": [5, 5]},
                ],
                final_stack=[5, 5],
                ops_used=["ADD", "DUP"],
                literal_values_used=[6, 9],
            ),
            _row(
                example_id="test-000001",
                split="test",
                program_text="3 DUP SWAP ADD END",
                tokens=["3", "DUP", "SWAP", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [3]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                    {"depth_after": 1, "stack_after": [6]},
                    {"depth_after": 1, "stack_after": [6]},
                ],
                final_stack=[6],
                ops_used=["ADD", "DUP", "SWAP"],
                literal_values_used=[3],
            ),
            _row(
                example_id="test-000002",
                split="test",
                program_text="2 3 ADD END",
                tokens=["2", "3", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [2]},
                    {"depth_after": 2, "stack_after": [2, 3]},
                    {"depth_after": 1, "stack_after": [5]},
                    {"depth_after": 1, "stack_after": [5]},
                ],
                final_stack=[5],
                ops_used=["ADD"],
                literal_values_used=[2, 3],
            ),
            _row(
                example_id="test-000003",
                split="test",
                program_text="4 DUP ADD END",
                tokens=["4", "DUP", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [4]},
                    {"depth_after": 2, "stack_after": [4, 4]},
                    {"depth_after": 1, "stack_after": [8]},
                    {"depth_after": 1, "stack_after": [8]},
                ],
                final_stack=[8],
                ops_used=["ADD", "DUP"],
                literal_values_used=[4],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "causal.jsonl"
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "causal_results.json"

            with data_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            model = TinyTransformer(
                TinyTransformerConfig(
                    vocab_size=len(VOCAB_TOKENS),
                    context_length=16,
                    d_model=16,
                    d_mlp=32,
                    n_layers=1,
                    n_heads=4,
                )
            )
            torch.save(
                {
                    "model_config": model.config.to_dict(),
                    "model_state": model.state_dict(),
                    "epoch": 0,
                    "vocab_tokens": list(VOCAB_TOKENS),
                },
                checkpoint_path,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--data-path",
                    str(data_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output-json",
                    str(output_path),
                    "--positions",
                    "1",
                    "2",
                    "--layers",
                    "block_0",
                    "final_ln",
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["analysis"], "pp0_residual_mean_ablation_v1")
        self.assertEqual(report["layers"], ["block_0", "final_ln"])
        self.assertIn("pc_1", report["positions"])
        self.assertIn("baseline_metrics", report["positions"]["pc_1"])
        self.assertIn("block_0", report["positions"]["pc_1"]["interventions"])
        self.assertIn(
            "exact_match_delta",
            report["positions"]["pc_1"]["interventions"]["block_0"]["deltas"],
        )

    def test_causal_steering_script_runs_end_to_end(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "causal_steer_pp0.py"
        rows = [
            _row(
                example_id="test-000000",
                split="test",
                program_text="1 2 ADD END",
                tokens=["1", "2", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [1]},
                    {"depth_after": 2, "stack_after": [1, 2]},
                    {"depth_after": 1, "stack_after": [3]},
                    {"depth_after": 1, "stack_after": [3]},
                ],
                final_stack=[3],
                ops_used=["ADD"],
                literal_values_used=[1, 2],
            ),
            _row(
                example_id="test-000001",
                split="test",
                program_text="4 5 ADD END",
                tokens=["4", "5", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [4]},
                    {"depth_after": 2, "stack_after": [4, 5]},
                    {"depth_after": 1, "stack_after": [9]},
                    {"depth_after": 1, "stack_after": [9]},
                ],
                final_stack=[9],
                ops_used=["ADD"],
                literal_values_used=[4, 5],
            ),
            _row(
                example_id="test-000002",
                split="test",
                program_text="2 1 ADD END",
                tokens=["2", "1", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [2]},
                    {"depth_after": 2, "stack_after": [2, 1]},
                    {"depth_after": 1, "stack_after": [3]},
                    {"depth_after": 1, "stack_after": [3]},
                ],
                final_stack=[3],
                ops_used=["ADD"],
                literal_values_used=[1, 2],
            ),
            _row(
                example_id="test-000003",
                split="test",
                program_text="5 4 ADD END",
                tokens=["5", "4", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [5]},
                    {"depth_after": 2, "stack_after": [5, 4]},
                    {"depth_after": 1, "stack_after": [9]},
                    {"depth_after": 1, "stack_after": [9]},
                ],
                final_stack=[9],
                ops_used=["ADD"],
                literal_values_used=[4, 5],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "steer.jsonl"
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "steer_results.json"

            with data_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            model = TinyTransformer(
                TinyTransformerConfig(
                    vocab_size=len(VOCAB_TOKENS),
                    context_length=16,
                    d_model=16,
                    d_mlp=32,
                    n_layers=1,
                    n_heads=4,
                )
            )
            torch.save(
                {
                    "model_config": model.config.to_dict(),
                    "model_state": model.state_dict(),
                    "epoch": 0,
                    "vocab_tokens": list(VOCAB_TOKENS),
                },
                checkpoint_path,
            )

            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--data-path",
                    str(data_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output-json",
                    str(output_path),
                    "--positions",
                    "2",
                    "--layers",
                    "block_0",
                    "--alphas",
                    "1.0",
                    "--max-pairs",
                    "2",
                    "--batch-size",
                    "2",
                    "--device",
                    "cpu",
                    "--probe-train-size",
                    "2",
                    "--probe-val-size",
                    "1",
                    "--probe-test-size",
                    "1",
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["analysis"], "pp0_top_direction_steering_v1")
        self.assertEqual(report["layers"], ["block_0"])
        self.assertIn("pc_2", report["positions"])
        self.assertIn("probe", report["positions"]["pc_2"]["layers"]["block_0"])


if __name__ == "__main__":
    unittest.main()
