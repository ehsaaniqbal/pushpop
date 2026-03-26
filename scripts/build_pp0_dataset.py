from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pushpop.pp0_dataset import (
    DEFAULT_INSTRUCTION_SET,
    DatasetConfig,
    generate_dataset,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic PP0 dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--val-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-program-length", type=int, default=6)
    parser.add_argument("--max-program-length", type=int, default=18)
    parser.add_argument("--max-stack-depth", type=int, default=4)
    parser.add_argument(
        "--instruction-set",
        nargs="+",
        default=list(DEFAULT_INSTRUCTION_SET),
        help="Space-separated PP0 tokens. END is appended automatically.",
    )
    parser.add_argument("--max-consecutive-literals", type=int, default=3)
    parser.add_argument(
        "--allow-no-arithmetic",
        action="store_true",
        help="Do not require ADD or SUB in every program.",
    )
    parser.add_argument(
        "--allow-no-structural",
        action="store_true",
        help="Do not require DUP, POP, or SWAP in every program.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        min_program_length=args.min_program_length,
        max_program_length=args.max_program_length,
        max_stack_depth=args.max_stack_depth,
        instruction_set=tuple(args.instruction_set),
        require_arithmetic=not args.allow_no_arithmetic,
        require_structural=not args.allow_no_structural,
        max_consecutive_literals=args.max_consecutive_literals,
    )
    bundle = generate_dataset(config)
    write_dataset(bundle, args.output_dir)
    print(json.dumps(bundle.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
