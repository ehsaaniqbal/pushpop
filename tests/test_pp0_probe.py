import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    capture_program_position_hidden_states,
    capture_end_hidden_states,
    collect_program_position_surface_control_keys,
    collect_program_position_target_labels,
    collect_end_state_target_labels,
    fit_ridge_probe,
    lookup_majority_control_accuracy,
    program_position_slot_present_indices,
    restrict_split_indices,
    select_feature_rows,
)
from pushpop.pp0_training import (
    PP0SupervisedDataset,
    build_supervised_example,
    collate_supervised_examples,
)
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


class PP0ProbeTests(unittest.TestCase):
    def test_collect_program_position_target_labels_uses_stack_after_at_requested_pc(self) -> None:
        rows = [
            _row(
                example_id="train-000000",
                split="train",
                program_text="3 1 4 SWAP SUB END",
                tokens=["3", "1", "4", "SWAP", "SUB", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [3]},
                    {"depth_after": 2, "stack_after": [3, 1]},
                    {"depth_after": 3, "stack_after": [3, 1, 4]},
                    {"depth_after": 3, "stack_after": [3, 4, 1]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                ],
                final_stack=[3, 3],
                ops_used=["SUB", "SWAP"],
                literal_values_used=[1, 3, 4],
            ),
            _row(
                example_id="train-000001",
                split="train",
                program_text="2 8 SWAP POP END",
                tokens=["2", "8", "SWAP", "POP", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [2]},
                    {"depth_after": 2, "stack_after": [2, 8]},
                    {"depth_after": 2, "stack_after": [8, 2]},
                    {"depth_after": 1, "stack_after": [8]},
                    {"depth_after": 1, "stack_after": [8]},
                ],
                final_stack=[8],
                ops_used=["POP", "SWAP"],
                literal_values_used=[2, 8],
            ),
        ]

        labels = collect_program_position_target_labels(rows, position_index=3, slot_count=2)

        self.assertEqual(labels["depth"], ["3", "1"])
        self.assertEqual(labels["top"], ["1", "8"])
        self.assertEqual(labels["slot_1"], ["4", "EMPTY"])
        self.assertEqual(labels["slot_2"], ["3", "EMPTY"])

    def test_collect_end_state_target_labels_uses_top_relative_slots(self) -> None:
        examples = [
            build_supervised_example(
                _row(
                    example_id="train-000000",
                    split="train",
                    program_text="5 4 3 2 END",
                    tokens=["5", "4", "3", "2", "END"],
                    trace=[
                        {"depth_after": 1, "stack_after": [5]},
                        {"depth_after": 2, "stack_after": [5, 4]},
                        {"depth_after": 3, "stack_after": [5, 4, 3]},
                        {"depth_after": 4, "stack_after": [5, 4, 3, 2]},
                        {"depth_after": 4, "stack_after": [5, 4, 3, 2]},
                    ],
                    final_stack=[5, 4, 3, 2],
                    ops_used=[],
                    literal_values_used=[2, 3, 4, 5],
                )
            ),
            build_supervised_example(
                _row(
                    example_id="train-000001",
                    split="train",
                    program_text="7 END",
                    tokens=["7", "END"],
                    trace=[
                        {"depth_after": 1, "stack_after": [7]},
                        {"depth_after": 1, "stack_after": [7]},
                    ],
                    final_stack=[7],
                    ops_used=[],
                    literal_values_used=[7],
                )
            ),
        ]

        labels = collect_end_state_target_labels(examples, slot_count=3)

        self.assertEqual(labels["depth"], ["4", "1"])
        self.assertEqual(labels["top"], ["2", "7"])
        self.assertEqual(labels["slot_1"], ["3", "EMPTY"])
        self.assertEqual(labels["slot_2"], ["4", "EMPTY"])
        self.assertEqual(labels["slot_3"], ["5", "EMPTY"])

    def test_capture_program_position_hidden_states_filters_to_examples_with_that_pc(self) -> None:
        rows = [
            _row(
                example_id="train-000000",
                split="train",
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
                example_id="train-000001",
                split="train",
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
        ]

        examples = [build_supervised_example(row) for row in rows]
        dataloader = DataLoader(
            examples,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_supervised_examples,
        )
        model = TinyTransformer(
            TinyTransformerConfig(
                vocab_size=len(VOCAB_TOKENS),
                context_length=16,
                d_model=32,
                d_mlp=64,
                n_layers=2,
                n_heads=4,
            )
        )

        features = capture_program_position_hidden_states(
            model,
            dataloader,
            position_index=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(list(features), ["embed", "block_0", "block_1", "final_ln"])
        for tensor in features.values():
            self.assertEqual(tensor.shape, (1, 32))

    def test_capture_end_hidden_states_returns_one_vector_per_layer(self) -> None:
        rows = [
            _row(
                example_id="train-000000",
                split="train",
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
                example_id="train-000001",
                split="train",
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
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            dataset = PP0SupervisedDataset(path)
            dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                collate_fn=collate_supervised_examples,
            )
            model = TinyTransformer(
                TinyTransformerConfig(
                    vocab_size=len(VOCAB_TOKENS),
                    context_length=16,
                    d_model=32,
                    d_mlp=64,
                    n_layers=2,
                    n_heads=4,
                )
            )

            features = capture_end_hidden_states(model, dataloader, device=torch.device("cpu"))

        self.assertEqual(list(features), ["embed", "block_0", "block_1", "final_ln"])
        for tensor in features.values():
            self.assertEqual(tensor.shape, (2, 32))

    def test_fit_ridge_probe_solves_toy_linearly_separable_problem(self) -> None:
        train_features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        train_labels = torch.tensor([0, 0, 1, 1])
        val_features = torch.tensor([[-1.5], [1.5]])
        val_labels = torch.tensor([0, 1])
        test_features = torch.tensor([[-0.5], [0.5]])
        test_labels = torch.tensor([0, 1])

        metrics = fit_ridge_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
            test_features,
            test_labels,
            num_classes=2,
            ridge_lambdas=[1e-6, 1e-4, 1e-2],
        )

        self.assertEqual(metrics.val_accuracy, 1.0)
        self.assertEqual(metrics.test_accuracy, 1.0)

    def test_restrict_split_indices_preserves_global_probe_split_membership(self) -> None:
        split_indices = {"train": [0, 1], "val": [2], "test": [3]}
        restricted = restrict_split_indices([1, 2, 3], split_indices)

        self.assertEqual(restricted, {"train": [0], "val": [1], "test": [2]})

    def test_collect_program_position_surface_control_keys_tracks_token_and_remaining_length(self) -> None:
        rows = [
            _row(
                example_id="train-000000",
                split="train",
                program_text="2 8 SWAP POP END",
                tokens=["2", "8", "SWAP", "POP", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [2]},
                    {"depth_after": 2, "stack_after": [2, 8]},
                    {"depth_after": 2, "stack_after": [8, 2]},
                    {"depth_after": 1, "stack_after": [8]},
                    {"depth_after": 1, "stack_after": [8]},
                ],
                final_stack=[8],
                ops_used=["POP", "SWAP"],
                literal_values_used=[2, 8],
            ),
            _row(
                example_id="train-000001",
                split="train",
                program_text="5 DUP ADD END",
                tokens=["5", "DUP", "ADD", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [5]},
                    {"depth_after": 2, "stack_after": [5, 5]},
                    {"depth_after": 1, "stack_after": [0]},
                    {"depth_after": 1, "stack_after": [0]},
                ],
                final_stack=[0],
                ops_used=["ADD", "DUP"],
                literal_values_used=[5],
            ),
        ]

        keys = collect_program_position_surface_control_keys(rows, position_index=1)

        self.assertEqual(keys["current_token"], ["8", "DUP"])
        self.assertEqual(
            keys["current_token_and_remaining_length"],
            ["8|remaining_length=3", "DUP|remaining_length=2"],
        )

    def test_program_position_slot_present_indices_filters_by_required_depth(self) -> None:
        rows = [
            _row(
                example_id="train-000000",
                split="train",
                program_text="3 1 4 SWAP SUB END",
                tokens=["3", "1", "4", "SWAP", "SUB", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [3]},
                    {"depth_after": 2, "stack_after": [3, 1]},
                    {"depth_after": 3, "stack_after": [3, 1, 4]},
                    {"depth_after": 3, "stack_after": [3, 4, 1]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                ],
                final_stack=[3, 3],
                ops_used=["SUB", "SWAP"],
                literal_values_used=[1, 3, 4],
            ),
            _row(
                example_id="train-000001",
                split="train",
                program_text="2 8 SWAP POP END",
                tokens=["2", "8", "SWAP", "POP", "END"],
                trace=[
                    {"depth_after": 1, "stack_after": [2]},
                    {"depth_after": 2, "stack_after": [2, 8]},
                    {"depth_after": 2, "stack_after": [8, 2]},
                    {"depth_after": 1, "stack_after": [8]},
                    {"depth_after": 1, "stack_after": [8]},
                ],
                final_stack=[8],
                ops_used=["POP", "SWAP"],
                literal_values_used=[2, 8],
            ),
        ]

        present_indices = program_position_slot_present_indices(
            rows,
            position_index=3,
            slot_index=1,
        )

        self.assertEqual(present_indices, [0])

    def test_lookup_majority_control_accuracy_uses_seen_keys_and_global_fallback(self) -> None:
        metrics = lookup_majority_control_accuracy(
            train_keys=["A", "A", "B"],
            train_labels=torch.tensor([0, 0, 1]),
            test_keys=["A", "B", "C"],
            test_labels=torch.tensor([0, 1, 0]),
            num_classes=2,
        )

        self.assertEqual(metrics.test_accuracy, 1.0)
        self.assertEqual(metrics.num_train_groups, 2)
        self.assertEqual(metrics.unseen_test_key_count, 1)

    def test_select_feature_rows_keeps_requested_row_order(self) -> None:
        features = {
            "embed": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            "block_0": torch.tensor([[10.0], [20.0], [30.0]]),
        }

        selected = select_feature_rows(features, [2, 0])

        self.assertTrue(torch.equal(selected["embed"], torch.tensor([[3.0, 3.0], [1.0, 1.0]])))
        self.assertTrue(torch.equal(selected["block_0"], torch.tensor([[30.0], [10.0]])))

    def test_probe_script_runs_end_to_end(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "probe_pp0.py"
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
            data_path = root / "probe.jsonl"
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "probe_results.json"

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
                    "--slot-count",
                    "1",
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["analysis"], "pp0_end_state_linear_probe_v1")
        self.assertEqual(report["layer_names"], ["embed", "block_0", "final_ln"])
        self.assertIn("depth", report["targets"])
        self.assertIn("top", report["targets"])
        self.assertIn("slot_1", report["targets"])

    def test_position_probe_script_runs_end_to_end(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "probe_pp0_positions.py"
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
            data_path = root / "probe.jsonl"
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "position_probe_results.json"

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
                    "--slot-count",
                    "1",
                    "--max-program-position",
                    "2",
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["analysis"], "pp0_program_position_linear_probe_v1")
        self.assertEqual(report["max_program_position"], 2)
        self.assertIn("pc_0", report["positions"])
        self.assertIn("pc_2", report["positions"])
        self.assertIn("depth", report["positions"]["pc_0"]["targets"])

    def test_position_control_probe_script_runs_end_to_end(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "probe_pp0_position_controls.py"
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
            data_path = root / "probe.jsonl"
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "position_probe_controls_results.json"

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
                    "--slot-count",
                    "1",
                    "--max-program-position",
                    "2",
                ],
                check=True,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(report["analysis"], "pp0_program_position_controls_v1")
        self.assertEqual(report["max_program_position"], 2)
        self.assertIn("pc_0", report["positions"])
        self.assertIn("surface_control_definition", report)
        self.assertIn("current_token", report["positions"]["pc_0"]["targets"]["top"]["surface_controls"])
        self.assertIn("slot_1_present_only", report["positions"]["pc_1"]["present_only_targets"])


if __name__ == "__main__":
    unittest.main()
