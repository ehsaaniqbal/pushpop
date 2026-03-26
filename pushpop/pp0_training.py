"""Shared training and evaluation utilities for PP0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from pushpop.pp0_vocab import (
    OUT_TOKEN,
    PAD_ID,
    STOP_TOKEN,
    TOKEN_TO_ID,
    VOCAB_TOKENS,
    decode_ids,
    encode_tokens,
)

IGNORE_INDEX = -100
STOP_ID = TOKEN_TO_ID[STOP_TOKEN]


@dataclass(frozen=True, slots=True)
class SupervisedExample:
    example_id: str
    split: str
    program_text: str
    sequence_tokens: tuple[str, ...]
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    top_target_index: int
    final_stack: tuple[int, ...]
    final_top: int
    metadata: dict[str, Any]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


class PP0SupervisedDataset(Dataset[SupervisedExample]):
    def __init__(self, path: str | Path, *, sanity_checks: bool = True) -> None:
        self.path = Path(path)
        self.examples = load_supervised_examples(self.path, sanity_checks=sanity_checks)
        if not self.examples:
            raise ValueError(f"dataset split is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SupervisedExample:
        return self.examples[index]

    @property
    def max_sequence_length(self) -> int:
        return max(example.sequence_length for example in self.examples)


@dataclass(slots=True)
class MetricAccumulator:
    num_examples: int = 0
    exact_correct: int = 0
    top_correct: int = 0
    stop_correct: int = 0
    token_correct: int = 0
    token_total: int = 0
    loss_numerator: float = 0.0
    loss_token_total: int = 0

    def add_loss(self, loss_value: float, token_count: int) -> None:
        self.loss_numerator += loss_value * token_count
        self.loss_token_total += token_count

    def add_example(
        self,
        *,
        exact_correct: bool,
        top_correct: bool,
        stop_correct: bool,
        token_correct: int,
        token_total: int,
    ) -> None:
        self.num_examples += 1
        self.exact_correct += int(exact_correct)
        self.top_correct += int(top_correct)
        self.stop_correct += int(stop_correct)
        self.token_correct += token_correct
        self.token_total += token_total

    def to_metrics(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {"count": self.num_examples}
        metrics["exact_match"] = (
            self.exact_correct / self.num_examples if self.num_examples else 0.0
        )
        metrics["top_accuracy"] = self.top_correct / self.num_examples if self.num_examples else 0.0
        metrics["stop_accuracy"] = (
            self.stop_correct / self.num_examples if self.num_examples else 0.0
        )
        metrics["token_accuracy"] = self.token_correct / self.token_total if self.token_total else 0.0
        if self.loss_token_total:
            metrics["loss"] = self.loss_numerator / self.loss_token_total
        return metrics


def load_supervised_examples(
    path: str | Path,
    *,
    sanity_checks: bool = True,
) -> list[SupervisedExample]:
    path = Path(path)
    examples: list[SupervisedExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if sanity_checks:
                validate_dataset_row(row, path=path, line_number=line_number)
            examples.append(build_supervised_example(row))
    return examples


def build_supervised_example(row: dict[str, Any]) -> SupervisedExample:
    tokens = tuple(row["tokens"])
    final_stack = tuple(int(value) for value in row["final_stack"])
    if not final_stack:
        raise ValueError("final_stack must be non-empty for the supervised task")

    target_stack_tokens = tuple(str(value) for value in final_stack)
    sequence_tokens = tokens + (OUT_TOKEN,) + target_stack_tokens + (STOP_TOKEN,)
    input_ids = tuple(encode_tokens(sequence_tokens[:-1]))
    target_ids = tuple(encode_tokens(sequence_tokens[1:]))
    loss_mask = tuple([False] * len(tokens) + [True] * (len(target_stack_tokens) + 1))

    decoded = tuple(decode_ids(input_ids))
    if decoded != sequence_tokens[:-1]:
        raise ValueError("token encoding/decoding round-trip failed")

    return SupervisedExample(
        example_id=str(row["example_id"]),
        split=str(row["split"]),
        program_text=str(row["program_text"]),
        sequence_tokens=sequence_tokens,
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        top_target_index=len(tokens) + len(target_stack_tokens) - 1,
        final_stack=final_stack,
        final_top=int(row["final_top"]),
        metadata=dict(row["metadata"]),
    )


def validate_dataset_row(row: dict[str, Any], *, path: Path, line_number: int) -> None:
    tokens = tuple(row["tokens"])
    final_stack = tuple(int(value) for value in row["final_stack"])
    trace = row["trace"]
    metadata = row["metadata"]

    if row["program_text"] != " ".join(tokens):
        raise ValueError(f"{path}:{line_number}: program_text does not match tokens")
    if not final_stack:
        raise ValueError(f"{path}:{line_number}: final_stack must be non-empty")
    if int(row["final_top"]) != final_stack[-1]:
        raise ValueError(f"{path}:{line_number}: final_top does not match final_stack")
    if tuple(trace[-1]["stack_after"]) != final_stack:
        raise ValueError(f"{path}:{line_number}: trace final stack does not match final_stack")
    if int(metadata["program_length"]) != len(tokens) - 1:
        raise ValueError(f"{path}:{line_number}: program_length metadata mismatch")

    max_depth = max(int(step["depth_after"]) for step in trace)
    if int(metadata["max_depth_reached"]) != max_depth:
        raise ValueError(f"{path}:{line_number}: max_depth_reached metadata mismatch")

    sequence_tokens = tokens + (OUT_TOKEN,) + tuple(str(value) for value in final_stack) + (STOP_TOKEN,)
    if tuple(decode_ids(encode_tokens(sequence_tokens))) != sequence_tokens:
        raise ValueError(f"{path}:{line_number}: tokenization round-trip failed")


def collate_supervised_examples(examples: list[SupervisedExample]) -> dict[str, Any]:
    batch_size = len(examples)
    max_length = max(example.sequence_length for example in examples)

    input_ids = torch.full((batch_size, max_length), PAD_ID, dtype=torch.long)
    target_ids = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    top_target_index = torch.full((batch_size,), -1, dtype=torch.long)
    example_ids: list[str] = []
    metadata: list[dict[str, Any]] = []

    for row_index, example in enumerate(examples):
        sequence_length = example.sequence_length
        input_ids[row_index, :sequence_length] = torch.tensor(example.input_ids, dtype=torch.long)
        target_ids[row_index, :sequence_length] = torch.tensor(example.target_ids, dtype=torch.long)
        example_mask = torch.tensor(example.loss_mask, dtype=torch.bool)
        loss_mask[row_index, :sequence_length] = example_mask
        target_ids[row_index, :sequence_length][~example_mask] = IGNORE_INDEX
        top_target_index[row_index] = example.top_target_index
        example_ids.append(example.example_id)
        metadata.append(example.metadata)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
        "top_target_index": top_target_index,
        "example_ids": example_ids,
        "metadata": metadata,
    }


def masked_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def learning_rate_for_step(
    step_index: int,
    *,
    base_learning_rate: float,
    total_steps: int,
    scheduler_name: str = "none",
    warmup_steps: int = 0,
    min_learning_rate_scale: float = 0.1,
) -> float:
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if base_learning_rate <= 0.0:
        raise ValueError("base_learning_rate must be positive")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    if not 0.0 <= min_learning_rate_scale <= 1.0:
        raise ValueError("min_learning_rate_scale must be between 0.0 and 1.0")
    if scheduler_name == "none":
        return base_learning_rate
    if scheduler_name != "warmup_cosine":
        raise ValueError(f"unsupported scheduler: {scheduler_name!r}")

    if warmup_steps > 0 and step_index < warmup_steps:
        return base_learning_rate * (step_index + 1) / warmup_steps

    min_learning_rate = base_learning_rate * min_learning_rate_scale
    decay_steps = total_steps - warmup_steps
    decay_progress = (step_index - warmup_steps) / max(1, decay_steps - 1)
    cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_learning_rate + (base_learning_rate - min_learning_rate) * cosine_multiplier


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


@torch.no_grad()
def greedy_rollout_batch(
    model: torch.nn.Module,
    prefixes: list[list[int]],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> list[list[int]]:
    if not prefixes:
        return []

    context_length = getattr(getattr(model, "config", None), "context_length", None)
    if context_length is None:
        raise ValueError("model must expose config.context_length for rollout evaluation")

    sequences = [list(prefix) for prefix in prefixes]
    generated = [[] for _ in prefixes]
    finished = [len(prefix) >= context_length for prefix in prefixes]

    for _ in range(max_new_tokens):
        active_indices = [index for index, is_finished in enumerate(finished) if not is_finished]
        if not active_indices:
            break

        max_length = max(len(sequences[index]) for index in active_indices)
        input_ids = torch.full(
            (len(active_indices), max_length),
            PAD_ID,
            dtype=torch.long,
            device=device,
        )
        last_positions: list[int] = []
        for row_index, example_index in enumerate(active_indices):
            sequence = sequences[example_index]
            input_ids[row_index, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=device,
            )
            last_positions.append(len(sequence) - 1)

        logits = model(input_ids)
        for row_index, example_index in enumerate(active_indices):
            next_token_id = int(logits[row_index, last_positions[row_index]].argmax().item())
            sequences[example_index].append(next_token_id)
            generated[example_index].append(next_token_id)
            if next_token_id == STOP_ID or len(sequences[example_index]) >= context_length:
                finished[example_index] = True

    return generated


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    compute_slices: bool = False,
) -> dict[str, Any]:
    model.eval()
    overall = MetricAccumulator()
    slice_accumulators: dict[str, dict[str, MetricAccumulator]] = {
        "program_length": defaultdict(MetricAccumulator),
        "max_stack_depth": defaultdict(MetricAccumulator),
        "operation_composition": defaultdict(MetricAccumulator),
    }

    for batch in dataloader:
        input_ids_cpu = batch["input_ids"]
        target_ids_cpu = batch["target_ids"]
        loss_mask_cpu = batch["loss_mask"]
        top_target_index = batch["top_target_index"].to(device)

        input_ids = input_ids_cpu.to(device)
        target_ids = target_ids_cpu.to(device)
        loss_mask = loss_mask_cpu.to(device)

        logits = model(input_ids)
        loss = masked_cross_entropy(logits, target_ids)
        teacher_forced_predictions = logits.argmax(dim=-1)
        masked_tokens_in_batch = int(loss_mask.sum().item())
        overall.add_loss(float(loss.item()), masked_tokens_in_batch)

        rollout_prefixes: list[list[int]] = []
        rollout_targets: list[list[int]] = []
        teacher_forced_positions: list[torch.Tensor] = []

        for row_index in range(len(batch["metadata"])):
            positions_cpu = loss_mask_cpu[row_index].nonzero(as_tuple=False).flatten()
            teacher_forced_positions.append(loss_mask[row_index].nonzero(as_tuple=False).flatten())
            first_target_position = int(positions_cpu[0].item())
            rollout_prefixes.append(input_ids_cpu[row_index, : first_target_position + 1].tolist())
            rollout_targets.append(target_ids_cpu[row_index, positions_cpu].tolist())

        rollout_predictions = greedy_rollout_batch(
            model,
            rollout_prefixes,
            device=device,
            max_new_tokens=max(len(target_tokens) for target_tokens in rollout_targets) + 2,
        )

        for row_index, metadata in enumerate(batch["metadata"]):
            positions = teacher_forced_positions[row_index]
            row_token_total = int(positions.numel())
            row_token_correct = int(
                (
                    teacher_forced_predictions[row_index, positions]
                    == target_ids[row_index, positions]
                ).sum().item()
            )
            rollout_target = rollout_targets[row_index]
            rollout_prediction = rollout_predictions[row_index]
            exact_correct = rollout_prediction == rollout_target

            predicted_stack = rollout_prediction
            if STOP_ID in predicted_stack:
                predicted_stack = predicted_stack[: predicted_stack.index(STOP_ID)]

            target_top_id = int(target_ids[row_index, int(top_target_index[row_index].item())].item())
            stop_position = len(rollout_target) - 1
            top_correct = bool(predicted_stack and predicted_stack[-1] == target_top_id)
            stop_correct = len(rollout_prediction) > stop_position and (
                rollout_prediction[stop_position] == STOP_ID
            )

            overall.add_example(
                exact_correct=exact_correct,
                top_correct=top_correct,
                stop_correct=stop_correct,
                token_correct=row_token_correct,
                token_total=row_token_total,
            )

            if not compute_slices:
                continue

            slice_keys = {
                "program_length": str(metadata["program_length"]),
                "max_stack_depth": str(metadata["max_depth_reached"]),
                "operation_composition": operation_composition_key(metadata["ops_used"]),
            }
            for slice_name, slice_key in slice_keys.items():
                slice_accumulators[slice_name][slice_key].add_example(
                    exact_correct=exact_correct,
                    top_correct=top_correct,
                    stop_correct=stop_correct,
                    token_correct=row_token_correct,
                    token_total=row_token_total,
                )

    results: dict[str, Any] = {"overall": overall.to_metrics()}
    if compute_slices:
        results["slices"] = {
            slice_name: {
                key: accumulator.to_metrics()
                for key, accumulator in sorted(slice_accumulators[slice_name].items())
            }
            for slice_name in slice_accumulators
        }
    return results


def operation_composition_key(ops_used: list[str] | tuple[str, ...]) -> str:
    if not ops_used:
        return "NONE"
    return "+".join(sorted(ops_used))


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_config": model_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_config": train_config,
        "epoch": epoch,
        "metrics": metrics,
        "vocab_tokens": list(VOCAB_TOKENS),
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if checkpoint.get("vocab_tokens") != list(VOCAB_TOKENS):
        raise ValueError("checkpoint vocabulary does not match the current PP0 vocabulary")
    return checkpoint


def choose_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
