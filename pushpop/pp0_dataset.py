"""Synthetic dataset generation for Pushpop-0 (PP0)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Final

from pushpop.pp0 import (
    ARITHMETIC_TOKENS,
    HALT_TOKEN,
    LITERAL_VALUES,
    STRUCTURAL_TOKENS,
    VALID_TOKENS,
    execute,
)

DEFAULT_INSTRUCTION_SET: Final[tuple[str, ...]] = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "DUP",
    "POP",
    "SWAP",
    "ADD",
    "SUB",
)
SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")
_CATEGORY_WEIGHTS: Final[dict[str, float]] = {
    "literal": 0.40,
    "structural": 0.30,
    "arithmetic": 0.30,
}


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    train_size: int
    val_size: int
    test_size: int
    seed: int = 0
    min_program_length: int = 6
    max_program_length: int = 18
    max_stack_depth: int = 4
    instruction_set: tuple[str, ...] = DEFAULT_INSTRUCTION_SET
    require_arithmetic: bool = True
    require_structural: bool = True
    max_consecutive_literals: int = 3

    def __post_init__(self) -> None:
        if self.train_size < 0 or self.val_size < 0 or self.test_size < 0:
            raise ValueError("split sizes must be non-negative")
        if self.train_size + self.val_size + self.test_size <= 0:
            raise ValueError("at least one split must request examples")
        if self.min_program_length < 1:
            raise ValueError("min_program_length must be >= 1")
        if self.max_program_length < self.min_program_length:
            raise ValueError("max_program_length must be >= min_program_length")
        if self.max_stack_depth < 1:
            raise ValueError("max_stack_depth must be >= 1")
        if self.max_consecutive_literals < 1:
            raise ValueError("max_consecutive_literals must be >= 1")

        deduped_instruction_set = _dedupe_tokens(self.instruction_set)
        if not deduped_instruction_set:
            raise ValueError("instruction_set must not be empty")
        if HALT_TOKEN in deduped_instruction_set:
            raise ValueError("instruction_set should not include END; it is appended automatically")

        invalid_tokens = [token for token in deduped_instruction_set if token not in VALID_TOKENS]
        if invalid_tokens:
            raise ValueError(f"unsupported instruction tokens: {invalid_tokens}")

        literal_tokens = [token for token in deduped_instruction_set if token in LITERAL_VALUES]
        if not literal_tokens:
            raise ValueError("instruction_set must include at least one literal token")

        if self.require_arithmetic and not any(
            token in ARITHMETIC_TOKENS for token in deduped_instruction_set
        ):
            raise ValueError("require_arithmetic=True but no arithmetic op is available")

        if self.require_structural and not any(
            token in STRUCTURAL_TOKENS for token in deduped_instruction_set
        ):
            raise ValueError("require_structural=True but no structural op is available")

        object.__setattr__(self, "instruction_set", deduped_instruction_set)

    def split_sizes(self) -> dict[str, int]:
        return {
            "train": self.train_size,
            "val": self.val_size,
            "test": self.test_size,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "seed": self.seed,
            "min_program_length": self.min_program_length,
            "max_program_length": self.max_program_length,
            "max_stack_depth": self.max_stack_depth,
            "instruction_set": list(self.instruction_set),
            "require_arithmetic": self.require_arithmetic,
            "require_structural": self.require_structural,
            "max_consecutive_literals": self.max_consecutive_literals,
        }


@dataclass(frozen=True, slots=True)
class ProgramExample:
    example_id: str
    split: str
    program_text: str
    tokens: tuple[str, ...]
    trace: tuple[dict[str, object], ...]
    final_stack: tuple[int, ...]
    final_top: int | None
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "program_text": self.program_text,
            "tokens": list(self.tokens),
            "trace": _json_ready(self.trace),
            "final_stack": list(self.final_stack),
            "final_top": self.final_top,
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    config: DatasetConfig
    train: tuple[ProgramExample, ...]
    val: tuple[ProgramExample, ...]
    test: tuple[ProgramExample, ...]
    metadata: dict[str, object]

    def split_map(self) -> dict[str, tuple[ProgramExample, ...]]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


def generate_program(rng: random.Random, config: DatasetConfig) -> tuple[str, ...]:
    """Sample one valid PP0 program ending in END."""

    candidate_pairs = [
        (program_length, final_depth)
        for program_length in range(config.min_program_length, config.max_program_length + 1)
        for final_depth in range(1, config.max_stack_depth + 1)
    ]
    rng.shuffle(candidate_pairs)

    for program_length, final_depth in candidate_pairs:
        program = _build_program_with_target(
            rng=rng,
            config=config,
            program_length=program_length,
            target_final_depth=final_depth,
        )
        if program is not None:
            return program

    raise RuntimeError("unable to generate a valid program with the current config")


def generate_dataset(config: DatasetConfig) -> DatasetBundle:
    """Generate deterministic train/val/test splits with exact-program deduplication."""

    rng = random.Random(config.seed)
    split_sizes = config.split_sizes()
    seen_programs: set[str] = set()
    total_requested = sum(split_sizes.values())
    attempt_limit = max(10_000, total_requested * 500)
    attempts = 0
    program_pool: list[tuple[str, ...]] = []

    while len(program_pool) < total_requested:
        attempts += 1
        if attempts > attempt_limit:
            raise RuntimeError(
                "dataset generation exceeded the attempt limit; the config may be too restrictive"
            )

        tokens = generate_program(rng, config)
        program_text = " ".join(tokens)
        if program_text in seen_programs:
            continue

        seen_programs.add(program_text)
        program_pool.append(tokens)

    program_pool.sort(key=_program_sort_key)
    train_cutoff = split_sizes["train"]
    val_cutoff = train_cutoff + split_sizes["val"]

    examples_by_split = {
        "train": tuple(
            _build_example(tokens=tokens, split="train", example_index=index)
            for index, tokens in enumerate(program_pool[:train_cutoff])
        ),
        "val": tuple(
            _build_example(tokens=tokens, split="val", example_index=index)
            for index, tokens in enumerate(program_pool[train_cutoff:val_cutoff])
        ),
        "test": tuple(
            _build_example(tokens=tokens, split="test", example_index=index)
            for index, tokens in enumerate(program_pool[val_cutoff:])
        ),
    }

    bundle = DatasetBundle(
        config=config,
        train=examples_by_split["train"],
        val=examples_by_split["val"],
        test=examples_by_split["test"],
        metadata={},
    )
    metadata = _build_dataset_metadata(bundle)
    return DatasetBundle(
        config=config,
        train=bundle.train,
        val=bundle.val,
        test=bundle.test,
        metadata=metadata,
    )


def write_dataset(bundle: DatasetBundle, output_dir: str | Path) -> None:
    """Write train/val/test JSONL files plus dataset metadata."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split, examples in bundle.split_map().items():
        split_path = output_path / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.to_dict(), sort_keys=True))
                handle.write("\n")

    metadata_path = output_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(bundle.metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")


def anti_leakage_checks(bundle: DatasetBundle) -> dict[str, object]:
    """Check for exact duplicate program leakage within and across splits."""

    split_sets = {
        split: {example.program_text for example in examples}
        for split, examples in bundle.split_map().items()
    }
    within_split_duplicates = {
        split: len(examples) - len(split_sets[split])
        for split, examples in bundle.split_map().items()
    }
    cross_split_overlap = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    passed = all(count == 0 for count in within_split_duplicates.values()) and all(
        count == 0 for count in cross_split_overlap.values()
    )
    return {
        "within_split_duplicates": within_split_duplicates,
        "cross_split_program_overlap": cross_split_overlap,
        "passed": passed,
    }


def _build_program_with_target(
    rng: random.Random,
    config: DatasetConfig,
    program_length: int,
    target_final_depth: int,
) -> tuple[str, ...] | None:
    instruction_set = config.instruction_set

    @lru_cache(maxsize=None)
    def can_finish(
        steps_remaining: int,
        depth: int,
        used_arithmetic: bool,
        used_structural: bool,
        consecutive_literals: int,
    ) -> bool:
        if steps_remaining == 0:
            return (
                depth == target_final_depth
                and (used_arithmetic or not config.require_arithmetic)
                and (used_structural or not config.require_structural)
            )

        for token in _candidate_tokens(
            instruction_set=instruction_set,
            depth=depth,
            consecutive_literals=consecutive_literals,
            max_stack_depth=config.max_stack_depth,
            max_consecutive_literals=config.max_consecutive_literals,
        ):
            next_depth = _next_depth(depth, token)
            if can_finish(
                steps_remaining - 1,
                next_depth,
                used_arithmetic or token in ARITHMETIC_TOKENS,
                used_structural or token in STRUCTURAL_TOKENS,
                consecutive_literals + 1 if token in LITERAL_VALUES else 0,
            ):
                return True
        return False

    if not can_finish(program_length, 0, False, False, 0):
        return None

    tokens: list[str] = []
    steps_remaining = program_length
    depth = 0
    used_arithmetic = False
    used_structural = False
    consecutive_literals = 0

    while steps_remaining > 0:
        reachable_tokens: list[str] = []
        for token in _candidate_tokens(
            instruction_set=instruction_set,
            depth=depth,
            consecutive_literals=consecutive_literals,
            max_stack_depth=config.max_stack_depth,
            max_consecutive_literals=config.max_consecutive_literals,
        ):
            next_depth = _next_depth(depth, token)
            if can_finish(
                steps_remaining - 1,
                next_depth,
                used_arithmetic or token in ARITHMETIC_TOKENS,
                used_structural or token in STRUCTURAL_TOKENS,
                consecutive_literals + 1 if token in LITERAL_VALUES else 0,
            ):
                reachable_tokens.append(token)

        if not reachable_tokens:
            return None

        token = _sample_token(rng, reachable_tokens)
        tokens.append(token)
        depth = _next_depth(depth, token)
        used_arithmetic = used_arithmetic or token in ARITHMETIC_TOKENS
        used_structural = used_structural or token in STRUCTURAL_TOKENS
        consecutive_literals = consecutive_literals + 1 if token in LITERAL_VALUES else 0
        steps_remaining -= 1

    return tuple(tokens + [HALT_TOKEN])


def _build_example(tokens: tuple[str, ...], split: str, example_index: int) -> ProgramExample:
    result = execute(tokens)
    trace = tuple(result.trace_rows())
    program_text = " ".join(tokens)
    program_ops = sorted(
        {token for token in tokens if token in STRUCTURAL_TOKENS or token in ARITHMETIC_TOKENS}
    )
    literal_values_used = sorted({int(token) for token in tokens if token in LITERAL_VALUES})
    max_depth_reached = max((int(step["depth_after"]) for step in trace), default=0)
    metadata = {
        "program_length": len(tokens) - 1,
        "trace_length": len(trace),
        "final_depth": len(result.final_stack),
        "max_depth_reached": max_depth_reached,
        "ops_used": program_ops,
        "literal_values_used": literal_values_used,
    }
    return ProgramExample(
        example_id=f"{split}-{example_index:06d}",
        split=split,
        program_text=program_text,
        tokens=tokens,
        trace=trace,
        final_stack=result.final_stack,
        final_top=result.final_top,
        metadata=metadata,
    )


def _build_dataset_metadata(bundle: DatasetBundle) -> dict[str, object]:
    split_summaries = {
        split: _summarize_split(examples)
        for split, examples in bundle.split_map().items()
    }
    return {
        "config": bundle.config.to_dict(),
        "split_strategy": (
            "global exact-program deduplication, then stable hash sort of canonical "
            "program text, then contiguous train/val/test slices"
        ),
        "anti_leakage_checks": anti_leakage_checks(bundle),
        "split_summaries": split_summaries,
    }


def _summarize_split(examples: tuple[ProgramExample, ...]) -> dict[str, object]:
    length_histogram = Counter(example.metadata["program_length"] for example in examples)
    max_depth_histogram = Counter(example.metadata["max_depth_reached"] for example in examples)
    final_depth_histogram = Counter(example.metadata["final_depth"] for example in examples)
    op_histogram = Counter(op for example in examples for op in example.metadata["ops_used"])
    return {
        "num_examples": len(examples),
        "program_length_histogram": _sorted_histogram(length_histogram),
        "max_depth_histogram": _sorted_histogram(max_depth_histogram),
        "final_depth_histogram": _sorted_histogram(final_depth_histogram),
        "op_histogram": dict(sorted(op_histogram.items())),
    }


def _candidate_tokens(
    instruction_set: tuple[str, ...],
    depth: int,
    consecutive_literals: int,
    max_stack_depth: int,
    max_consecutive_literals: int,
) -> tuple[str, ...]:
    candidates: list[str] = []
    for token in instruction_set:
        if token in LITERAL_VALUES:
            if depth < max_stack_depth and consecutive_literals < max_consecutive_literals:
                candidates.append(token)
            continue

        if token == "DUP":
            if 1 <= depth < max_stack_depth:
                candidates.append(token)
            continue

        if token == "POP":
            if depth >= 1:
                candidates.append(token)
            continue

        if token in {"SWAP", "ADD", "SUB"} and depth >= 2:
            candidates.append(token)

    return tuple(candidates)


def _sample_token(rng: random.Random, tokens: list[str]) -> str:
    grouped_tokens = {
        "literal": [token for token in tokens if token in LITERAL_VALUES],
        "structural": [token for token in tokens if token in STRUCTURAL_TOKENS],
        "arithmetic": [token for token in tokens if token in ARITHMETIC_TOKENS],
    }
    available_categories = [
        category for category, grouped in grouped_tokens.items() if grouped
    ]
    if len(available_categories) == 1:
        chosen_category = available_categories[0]
    else:
        category_weights = [_CATEGORY_WEIGHTS[category] for category in available_categories]
        chosen_category = rng.choices(available_categories, weights=category_weights, k=1)[0]

    return rng.choice(grouped_tokens[chosen_category])


def _next_depth(depth: int, token: str) -> int:
    if token in LITERAL_VALUES or token == "DUP":
        return depth + 1
    if token == "SWAP":
        return depth
    return depth - 1


def _dedupe_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            deduped.append(token)
    return tuple(deduped)


def _program_sort_key(tokens: tuple[str, ...]) -> tuple[str, str]:
    program_text = " ".join(tokens)
    program_hash = sha256(program_text.encode("utf-8")).hexdigest()
    return (program_hash, program_text)


def _sorted_histogram(counter: Counter[object]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
