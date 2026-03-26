import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch

from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_training import (
    PP0SupervisedDataset,
    build_supervised_example,
    collate_supervised_examples,
    evaluate_model,
    operation_composition_key,
    validate_dataset_row,
)
from pushpop.pp0_vocab import STOP_TOKEN, TOKEN_TO_ID


class RolloutGapModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(context_length=16)
        self.vocab_size = len(TOKEN_TO_ID)
        self.out_id = TOKEN_TO_ID["OUT"]
        self.stop_id = TOKEN_TO_ID[STOP_TOKEN]
        self.first_id = TOKEN_TO_ID["4"]
        self.second_id = TOKEN_TO_ID["6"]
        self.gold_first_id = TOKEN_TO_ID["5"]

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
                    elif generated_suffix == [self.gold_first_id]:
                        next_token_id = self.second_id
                    elif generated_suffix == [self.gold_first_id, self.second_id]:
                        next_token_id = self.stop_id
                    else:
                        next_token_id = self.stop_id
                logits[batch_index, position, next_token_id] = 0.0
        return logits


class PP0TrainingTests(unittest.TestCase):
    def test_build_supervised_example_creates_expected_targets_and_mask(self) -> None:
        row = {
            "example_id": "train-000000",
            "split": "train",
            "program_text": "2 DUP SWAP SUB END",
            "tokens": ["2", "DUP", "SWAP", "SUB", "END"],
            "trace": [
                {"depth_after": 1, "stack_after": [2]},
                {"depth_after": 2, "stack_after": [2, 2]},
                {"depth_after": 2, "stack_after": [2, 2]},
                {"depth_after": 1, "stack_after": [0]},
                {"depth_after": 1, "stack_after": [0]},
            ],
            "final_stack": [0],
            "final_top": 0,
            "metadata": {
                "program_length": 4,
                "trace_length": 5,
                "final_depth": 1,
                "max_depth_reached": 2,
                "ops_used": ["DUP", "SUB", "SWAP"],
                "literal_values_used": [2],
            },
        }

        example = build_supervised_example(row)

        self.assertEqual(example.sequence_tokens[-1], STOP_TOKEN)
        self.assertEqual(example.top_target_index, 5)
        self.assertEqual(example.loss_mask, (False, False, False, False, False, True, True))
        self.assertEqual(example.final_stack, (0,))

    def test_validate_dataset_row_catches_program_text_mismatch(self) -> None:
        row = {
            "example_id": "bad-000000",
            "split": "train",
            "program_text": "BROKEN",
            "tokens": ["2", "END"],
            "trace": [
                {"depth_after": 1, "stack_after": [2]},
                {"depth_after": 1, "stack_after": [2]},
            ],
            "final_stack": [2],
            "final_top": 2,
            "metadata": {
                "program_length": 1,
                "trace_length": 2,
                "final_depth": 1,
                "max_depth_reached": 1,
                "ops_used": [],
                "literal_values_used": [2],
            },
        }

        with self.assertRaisesRegex(ValueError, "program_text does not match tokens"):
            validate_dataset_row(row, path=Path("fake.jsonl"), line_number=1)

    def test_dataset_collation_and_model_forward_match_shapes(self) -> None:
        rows = [
            {
                "example_id": "train-000000",
                "split": "train",
                "program_text": "9 6 ADD DUP END",
                "tokens": ["9", "6", "ADD", "DUP", "END"],
                "trace": [
                    {"depth_after": 1, "stack_after": [9]},
                    {"depth_after": 2, "stack_after": [9, 6]},
                    {"depth_after": 1, "stack_after": [5]},
                    {"depth_after": 2, "stack_after": [5, 5]},
                    {"depth_after": 2, "stack_after": [5, 5]},
                ],
                "final_stack": [5, 5],
                "final_top": 5,
                "metadata": {
                    "program_length": 4,
                    "trace_length": 5,
                    "final_depth": 2,
                    "max_depth_reached": 2,
                    "ops_used": ["ADD", "DUP"],
                    "literal_values_used": [6, 9],
                },
            },
            {
                "example_id": "train-000001",
                "split": "train",
                "program_text": "3 DUP SWAP ADD END",
                "tokens": ["3", "DUP", "SWAP", "ADD", "END"],
                "trace": [
                    {"depth_after": 1, "stack_after": [3]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                    {"depth_after": 2, "stack_after": [3, 3]},
                    {"depth_after": 1, "stack_after": [6]},
                    {"depth_after": 1, "stack_after": [6]},
                ],
                "final_stack": [6],
                "final_top": 6,
                "metadata": {
                    "program_length": 4,
                    "trace_length": 5,
                    "final_depth": 1,
                    "max_depth_reached": 2,
                    "ops_used": ["ADD", "DUP", "SWAP"],
                    "literal_values_used": [3],
                },
            },
        ]
        examples = [build_supervised_example(row) for row in rows]
        batch = collate_supervised_examples(examples)

        model = TinyTransformer(
            TinyTransformerConfig(vocab_size=19, context_length=16, d_model=64, d_mlp=128, n_layers=2, n_heads=4)
        )
        logits = model(batch["input_ids"])

        self.assertEqual(logits.shape[:2], batch["input_ids"].shape)
        self.assertEqual(logits.shape[-1], 19)

    def test_dataset_loader_reads_written_rows(self) -> None:
        row = {
            "example_id": "train-000000",
            "split": "train",
            "program_text": "9 6 ADD DUP END",
            "tokens": ["9", "6", "ADD", "DUP", "END"],
            "trace": [
                {"depth_after": 1, "stack_after": [9]},
                {"depth_after": 2, "stack_after": [9, 6]},
                {"depth_after": 1, "stack_after": [5]},
                {"depth_after": 2, "stack_after": [5, 5]},
                {"depth_after": 2, "stack_after": [5, 5]},
            ],
            "final_stack": [5, 5],
            "final_top": 5,
            "metadata": {
                "program_length": 4,
                "trace_length": 5,
                "final_depth": 2,
                "max_depth_reached": 2,
                "ops_used": ["ADD", "DUP"],
                "literal_values_used": [6, 9],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = PP0SupervisedDataset(path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.max_sequence_length, len(dataset[0].input_ids))

    def test_operation_composition_key_is_stable(self) -> None:
        self.assertEqual(operation_composition_key(["SWAP", "ADD", "DUP"]), "ADD+DUP+SWAP")

    def test_evaluate_model_uses_rollout_for_sequence_metrics(self) -> None:
        row = {
            "example_id": "train-000002",
            "split": "train",
            "program_text": "5 3 DUP ADD END",
            "tokens": ["5", "3", "DUP", "ADD", "END"],
            "trace": [
                {"depth_after": 1, "stack_after": [5]},
                {"depth_after": 2, "stack_after": [5, 3]},
                {"depth_after": 3, "stack_after": [5, 3, 3]},
                {"depth_after": 2, "stack_after": [5, 6]},
                {"depth_after": 2, "stack_after": [5, 6]},
            ],
            "final_stack": [5, 6],
            "final_top": 6,
            "metadata": {
                "program_length": 4,
                "trace_length": 5,
                "final_depth": 2,
                "max_depth_reached": 3,
                "ops_used": ["ADD", "DUP"],
                "literal_values_used": [3, 5],
            },
        }

        example = build_supervised_example(row)
        batch = collate_supervised_examples([example])
        metrics = evaluate_model(RolloutGapModel(), [batch], torch.device("cpu"))

        self.assertEqual(metrics["overall"]["count"], 1)
        self.assertEqual(metrics["overall"]["exact_match"], 0.0)
        self.assertEqual(metrics["overall"]["top_accuracy"], 0.0)
        self.assertEqual(metrics["overall"]["stop_accuracy"], 0.0)
        self.assertAlmostEqual(metrics["overall"]["token_accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
