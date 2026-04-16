import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from pushpop.pp0_causal import build_mean_ablation_intervention
from pushpop.pp0_model import ResidualIntervention, TinyTransformer, TinyTransformerConfig
from pushpop.pp0_vocab import VOCAB_TOKENS


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


if __name__ == "__main__":
    unittest.main()
