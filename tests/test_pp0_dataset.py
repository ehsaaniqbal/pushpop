import json
import tempfile
import unittest
from pathlib import Path
import random

from pushpop.pp0 import ARITHMETIC_TOKENS, HALT_TOKEN, STRUCTURAL_TOKENS, execute
from pushpop.pp0_dataset import (
    DatasetConfig,
    anti_leakage_checks,
    generate_dataset,
    generate_program,
    write_dataset,
)


class PP0DatasetTests(unittest.TestCase):
    def test_generate_program_produces_valid_program_within_limits(self) -> None:
        config = DatasetConfig(
            train_size=1,
            val_size=0,
            test_size=0,
            seed=7,
            min_program_length=6,
            max_program_length=8,
            max_stack_depth=4,
        )

        program = generate_program(random.Random(config.seed), config)
        result = execute(program)
        ops_used = [token for token in program if token in ARITHMETIC_TOKENS | STRUCTURAL_TOKENS]
        max_depth_reached = max(step["depth_after"] for step in result.trace_rows())

        self.assertEqual(program[-1], HALT_TOKEN)
        self.assertGreaterEqual(len(program) - 1, config.min_program_length)
        self.assertLessEqual(len(program) - 1, config.max_program_length)
        self.assertGreaterEqual(len(result.final_stack), 1)
        self.assertLessEqual(max_depth_reached, config.max_stack_depth)
        self.assertTrue(any(token in ARITHMETIC_TOKENS for token in ops_used))
        self.assertTrue(any(token in STRUCTURAL_TOKENS for token in ops_used))

    def test_generate_dataset_is_deterministic_for_fixed_seed(self) -> None:
        config = DatasetConfig(
            train_size=4,
            val_size=2,
            test_size=2,
            seed=11,
            min_program_length=4,
            max_program_length=6,
            max_stack_depth=4,
        )

        dataset_a = generate_dataset(config)
        dataset_b = generate_dataset(config)

        rows_a = [example.program_text for split in dataset_a.split_map().values() for example in split]
        rows_b = [example.program_text for split in dataset_b.split_map().values() for example in split]
        self.assertEqual(rows_a, rows_b)

    def test_instruction_set_is_respected(self) -> None:
        config = DatasetConfig(
            train_size=2,
            val_size=1,
            test_size=1,
            seed=5,
            min_program_length=3,
            max_program_length=3,
            max_stack_depth=3,
            instruction_set=("0", "1", "ADD"),
            require_structural=False,
        )

        dataset = generate_dataset(config)
        allowed_tokens = set(config.instruction_set) | {HALT_TOKEN}

        for split_examples in dataset.split_map().values():
            for example in split_examples:
                self.assertTrue(set(example.tokens).issubset(allowed_tokens))

    def test_examples_include_trace_and_metadata(self) -> None:
        config = DatasetConfig(
            train_size=2,
            val_size=1,
            test_size=1,
            seed=13,
            min_program_length=4,
            max_program_length=5,
            max_stack_depth=4,
        )

        dataset = generate_dataset(config)
        example = dataset.train[0]

        self.assertEqual(example.program_text, " ".join(example.tokens))
        self.assertEqual(example.final_top, example.final_stack[-1])
        self.assertIn("max_depth_reached", example.metadata)
        self.assertIn("ops_used", example.metadata)
        self.assertEqual(example.metadata["trace_length"], len(example.trace))
        self.assertEqual(example.metadata["final_depth"], len(example.final_stack))

    def test_anti_leakage_checks_report_zero_overlap(self) -> None:
        config = DatasetConfig(
            train_size=5,
            val_size=2,
            test_size=2,
            seed=17,
            min_program_length=4,
            max_program_length=6,
            max_stack_depth=4,
        )

        dataset = generate_dataset(config)
        checks = anti_leakage_checks(dataset)

        self.assertTrue(checks["passed"])
        self.assertEqual(checks["cross_split_program_overlap"]["train_val"], 0)
        self.assertEqual(checks["cross_split_program_overlap"]["train_test"], 0)
        self.assertEqual(checks["cross_split_program_overlap"]["val_test"], 0)

    def test_write_dataset_persists_jsonl_and_metadata(self) -> None:
        config = DatasetConfig(
            train_size=2,
            val_size=1,
            test_size=1,
            seed=19,
            min_program_length=4,
            max_program_length=5,
            max_stack_depth=4,
        )
        dataset = generate_dataset(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_dataset(dataset, output_dir)

            train_path = output_dir / "train.jsonl"
            metadata_path = output_dir / "metadata.json"

            self.assertTrue(train_path.exists())
            self.assertTrue(metadata_path.exists())

            train_rows = train_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(train_rows), config.train_size)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["anti_leakage_checks"]["passed"])
            self.assertEqual(metadata["config"]["seed"], config.seed)


if __name__ == "__main__":
    unittest.main()
