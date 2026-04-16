"""Simple linear probes for Phase 2 PP0 representation analysis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F

from pushpop.pp0_model import TinyTransformer
from pushpop.pp0_training import (
    SupervisedExample,
    build_supervised_example,
    validate_dataset_row,
)

EMPTY_LABEL: Final[str] = "EMPTY"


@dataclass(frozen=True, slots=True)
class LinearProbeMetrics:
    ridge_lambda: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float


@dataclass(frozen=True, slots=True)
class LookupControlMetrics:
    test_accuracy: float
    num_train_groups: int
    unseen_test_key_count: int
    unseen_test_fraction: float


@dataclass(frozen=True, slots=True)
class FittedRidgeProbe:
    label_vocab: tuple[str, ...]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    weights: torch.Tensor
    ridge_lambda: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float

    def class_direction(self, label: str) -> torch.Tensor:
        label_to_index = {name: index for index, name in enumerate(self.label_vocab)}
        if label not in label_to_index:
            raise ValueError(f"unknown probe label: {label!r}")
        weight_vector = self.weights[:-1, label_to_index[label]]
        raw_direction = weight_vector / self.feature_std.squeeze(0).to(dtype=self.weights.dtype)
        return raw_direction.to(dtype=torch.float32)

    def class_difference_direction(self, source_label: str, target_label: str) -> torch.Tensor:
        return self.class_direction(source_label) - self.class_direction(target_label)


def hidden_state_names(model: TinyTransformer) -> tuple[str, ...]:
    return ("embed", *(f"block_{index}" for index in range(model.config.n_layers)), "final_ln")


def build_probe_splits(
    total_examples: int,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
) -> dict[str, list[int]]:
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("probe train/val/test sizes must all be positive")

    used_example_count = train_size + val_size + test_size
    if used_example_count > total_examples:
        raise ValueError(
            f"requested {used_example_count} probe examples, but only {total_examples} are available"
        )

    return {
        "train": list(range(0, train_size)),
        "val": list(range(train_size, train_size + val_size)),
        "test": list(range(train_size + val_size, used_example_count)),
    }


def load_probe_rows(
    path: str | Path,
    *,
    sanity_checks: bool = True,
) -> list[dict[str, Any]]:
    probe_path = Path(path)
    rows: list[dict[str, Any]] = []
    with probe_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if sanity_checks:
                validate_dataset_row(row, path=probe_path, line_number=line_number)
            rows.append(row)
    return rows


def build_supervised_examples_from_rows(rows: Sequence[dict[str, Any]]) -> list[SupervisedExample]:
    return [build_supervised_example(row) for row in rows]


def collect_end_state_target_labels(
    examples: Sequence[SupervisedExample],
    *,
    slot_count: int = 3,
) -> dict[str, list[str]]:
    if slot_count < 0:
        raise ValueError("slot_count must be non-negative")

    labels: dict[str, list[str]] = {"depth": [], "top": []}
    for slot_index in range(1, slot_count + 1):
        labels[f"slot_{slot_index}"] = []

    for example in examples:
        stack = example.final_stack
        if not stack:
            raise ValueError("END-state probes require non-empty final stacks")

        labels["depth"].append(str(len(stack)))
        labels["top"].append(str(stack[-1]))
        for slot_index in range(1, slot_count + 1):
            label = str(stack[-1 - slot_index]) if len(stack) > slot_index else EMPTY_LABEL
            labels[f"slot_{slot_index}"].append(label)

    return labels


@torch.no_grad()
def capture_end_hidden_states(
    model: TinyTransformer,
    dataloader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    layer_names = hidden_state_names(model)
    feature_chunks = {name: [] for name in layer_names}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"]
        _, hidden_states = model(input_ids, return_hidden_states=True)

        end_positions = [
            int(row_loss_mask.nonzero(as_tuple=False).flatten()[0].item()) - 1
            for row_loss_mask in loss_mask
        ]
        row_indices = torch.arange(input_ids.shape[0], device=device)
        end_positions_tensor = torch.tensor(end_positions, dtype=torch.long, device=device)

        for layer_name, hidden_state in zip(layer_names, hidden_states, strict=True):
            feature_chunks[layer_name].append(
                hidden_state[row_indices, end_positions_tensor].cpu()
            )

    return {
        layer_name: torch.cat(chunks, dim=0)
        for layer_name, chunks in feature_chunks.items()
    }


@torch.no_grad()
def capture_program_position_hidden_states(
    model: TinyTransformer,
    dataloader: Iterable[dict[str, Any]],
    *,
    position_index: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if position_index < 0:
        raise ValueError("position_index must be non-negative")

    model.eval()
    layer_names = hidden_state_names(model)
    feature_chunks = {name: [] for name in layer_names}

    for batch in dataloader:
        available_rows = [
            row_index
            for row_index, metadata in enumerate(batch["metadata"])
            if position_index <= int(metadata["program_length"])
        ]
        if not available_rows:
            continue

        input_ids = batch["input_ids"].to(device)
        _, hidden_states = model(input_ids, return_hidden_states=True)
        row_indices = torch.tensor(available_rows, dtype=torch.long, device=device)
        position_tensor = torch.full(
            (len(available_rows),),
            position_index,
            dtype=torch.long,
            device=device,
        )

        for layer_name, hidden_state in zip(layer_names, hidden_states, strict=True):
            feature_chunks[layer_name].append(hidden_state[row_indices, position_tensor].cpu())

    return {
        layer_name: torch.cat(chunks, dim=0)
        for layer_name, chunks in feature_chunks.items()
        if chunks
    }


def fit_ridge_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int,
    ridge_lambdas: Sequence[float],
) -> LinearProbeMetrics:
    if num_classes <= 1:
        raise ValueError("num_classes must be at least 2 for a probe")
    if not ridge_lambdas:
        raise ValueError("ridge_lambdas must not be empty")

    prepared_train, prepared_val, prepared_test = _prepare_probe_features(
        train_features,
        val_features,
        test_features,
    )

    best_metrics: LinearProbeMetrics | None = None
    for ridge_lambda in ridge_lambdas:
        if ridge_lambda <= 0.0:
            raise ValueError("ridge_lambdas must be strictly positive")

        weights = _fit_ridge_weights(
            prepared_train,
            train_labels,
            num_classes=num_classes,
            ridge_lambda=float(ridge_lambda),
        )
        metrics = LinearProbeMetrics(
            ridge_lambda=float(ridge_lambda),
            train_accuracy=_accuracy(prepared_train, weights, train_labels),
            val_accuracy=_accuracy(prepared_val, weights, val_labels),
            test_accuracy=_accuracy(prepared_test, weights, test_labels),
        )
        if best_metrics is None or metrics.val_accuracy > best_metrics.val_accuracy:
            best_metrics = metrics
            continue
        if math.isclose(metrics.val_accuracy, best_metrics.val_accuracy) and (
            metrics.ridge_lambda < best_metrics.ridge_lambda
        ):
            best_metrics = metrics

    if best_metrics is None:
        raise AssertionError("ridge probe selection produced no result")
    return best_metrics


def fit_ridge_probe_model(
    features: torch.Tensor,
    labels: Sequence[str],
    split_indices: dict[str, list[int]],
    *,
    ridge_lambdas: Sequence[float],
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 0,
) -> FittedRidgeProbe:
    encoded_labels, label_vocab = _encode_labels(labels)
    if len(label_vocab) <= 1:
        raise ValueError("ridge probe model requires at least two labels")

    train_indices = _index_tensor(split_indices["train"])
    val_indices = _index_tensor(split_indices["val"])
    test_indices = _index_tensor(split_indices["test"])

    train_features = features.index_select(0, train_indices).to(dtype=torch.float64)
    val_features = features.index_select(0, val_indices).to(dtype=torch.float64)
    test_features = features.index_select(0, test_indices).to(dtype=torch.float64)

    train_labels = encoded_labels.index_select(0, train_indices)
    val_labels = encoded_labels.index_select(0, val_indices)
    test_labels = encoded_labels.index_select(0, test_indices)
    if shuffle_train_labels:
        generator = torch.Generator().manual_seed(shuffle_seed)
        train_labels = train_labels[torch.randperm(train_labels.shape[0], generator=generator)]

    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    prepared_train = _append_bias_column((train_features - feature_mean) / feature_std)
    prepared_val = _append_bias_column((val_features - feature_mean) / feature_std)
    prepared_test = _append_bias_column((test_features - feature_mean) / feature_std)

    best_model: FittedRidgeProbe | None = None
    for ridge_lambda in ridge_lambdas:
        if ridge_lambda <= 0.0:
            raise ValueError("ridge_lambdas must be strictly positive")
        weights = _fit_ridge_weights(
            prepared_train,
            train_labels,
            num_classes=len(label_vocab),
            ridge_lambda=float(ridge_lambda),
        )
        candidate = FittedRidgeProbe(
            label_vocab=label_vocab,
            feature_mean=feature_mean.cpu(),
            feature_std=feature_std.cpu(),
            weights=weights.cpu(),
            ridge_lambda=float(ridge_lambda),
            train_accuracy=_accuracy(prepared_train, weights, train_labels),
            val_accuracy=_accuracy(prepared_val, weights, val_labels),
            test_accuracy=_accuracy(prepared_test, weights, test_labels),
        )
        if best_model is None or candidate.val_accuracy > best_model.val_accuracy:
            best_model = candidate
            continue
        if math.isclose(candidate.val_accuracy, best_model.val_accuracy) and (
            candidate.ridge_lambda < best_model.ridge_lambda
        ):
            best_model = candidate

    if best_model is None:
        raise AssertionError("ridge probe model selection produced no result")
    return best_model


def majority_baseline_accuracy(
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int,
) -> float:
    class_counts = torch.bincount(train_labels, minlength=num_classes)
    majority_class = int(class_counts.argmax().item())
    return float((test_labels == majority_class).float().mean().item())


def run_probe_suite(
    layer_features: dict[str, torch.Tensor],
    target_labels: dict[str, list[str]],
    split_indices: dict[str, list[int]],
    *,
    ridge_lambdas: Sequence[float],
    shuffle_seed: int = 0,
    control_keys: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    train_indices = _index_tensor(split_indices["train"])
    val_indices = _index_tensor(split_indices["val"])
    test_indices = _index_tensor(split_indices["test"])

    results: dict[str, Any] = {}
    for target_name, label_strings in target_labels.items():
        encoded_labels, label_vocab = _encode_labels(label_strings)
        train_labels = encoded_labels.index_select(0, train_indices)
        val_labels = encoded_labels.index_select(0, val_indices)
        test_labels = encoded_labels.index_select(0, test_indices)
        num_classes = len(label_vocab)

        target_result: dict[str, Any] = {
            "label_vocab": list(label_vocab),
            "train_class_counts": {
                label: int((train_labels == label_index).sum().item())
                for label_index, label in enumerate(label_vocab)
            },
            "chance_test_accuracy": 1.0 / num_classes,
            "majority_test_accuracy": majority_baseline_accuracy(
                train_labels,
                test_labels,
                num_classes=num_classes,
            ),
            "is_constant_label": num_classes == 1,
            "surface_controls": {},
            "layers": {},
        }
        if control_keys is not None:
            for control_name, keys in control_keys.items():
                if len(keys) != len(label_strings):
                    raise ValueError(
                        f"control key count for {control_name} does not match target {target_name}"
                    )
                train_keys = [keys[index] for index in split_indices["train"]]
                test_keys = [keys[index] for index in split_indices["test"]]
                metrics = lookup_majority_control_accuracy(
                    train_keys,
                    train_labels,
                    test_keys,
                    test_labels,
                    num_classes=num_classes,
                )
                target_result["surface_controls"][control_name] = {
                    "test_accuracy": metrics.test_accuracy,
                    "num_train_groups": metrics.num_train_groups,
                    "unseen_test_key_count": metrics.unseen_test_key_count,
                    "unseen_test_fraction": metrics.unseen_test_fraction,
                }
        if num_classes == 1:
            for layer_name in layer_features:
                target_result["layers"][layer_name] = {
                    "probe_train_accuracy": 1.0,
                    "probe_val_accuracy": 1.0,
                    "probe_test_accuracy": 1.0,
                    "best_ridge_lambda": None,
                    "shuffled_train_accuracy": 1.0,
                    "shuffled_val_accuracy": 1.0,
                    "shuffled_test_accuracy": 1.0,
                    "shuffled_best_ridge_lambda": None,
                }
            results[target_name] = target_result
            continue

        generator = torch.Generator().manual_seed(shuffle_seed)
        shuffled_train_labels = train_labels[torch.randperm(train_labels.shape[0], generator=generator)]

        for layer_name, features in layer_features.items():
            train_features = features.index_select(0, train_indices)
            val_features = features.index_select(0, val_indices)
            test_features = features.index_select(0, test_indices)

            probe_metrics = fit_ridge_probe(
                train_features,
                train_labels,
                val_features,
                val_labels,
                test_features,
                test_labels,
                num_classes=num_classes,
                ridge_lambdas=ridge_lambdas,
            )
            shuffled_metrics = fit_ridge_probe(
                train_features,
                shuffled_train_labels,
                val_features,
                val_labels,
                test_features,
                test_labels,
                num_classes=num_classes,
                ridge_lambdas=ridge_lambdas,
            )
            target_result["layers"][layer_name] = {
                "probe_train_accuracy": probe_metrics.train_accuracy,
                "probe_val_accuracy": probe_metrics.val_accuracy,
                "probe_test_accuracy": probe_metrics.test_accuracy,
                "best_ridge_lambda": probe_metrics.ridge_lambda,
                "shuffled_train_accuracy": shuffled_metrics.train_accuracy,
                "shuffled_val_accuracy": shuffled_metrics.val_accuracy,
                "shuffled_test_accuracy": shuffled_metrics.test_accuracy,
                "shuffled_best_ridge_lambda": shuffled_metrics.ridge_lambda,
            }

        results[target_name] = target_result

    return results


def run_end_state_probe_suite(
    layer_features: dict[str, torch.Tensor],
    target_labels: dict[str, list[str]],
    split_indices: dict[str, list[int]],
    *,
    ridge_lambdas: Sequence[float],
    shuffle_seed: int = 0,
) -> dict[str, Any]:
    return run_probe_suite(
        layer_features,
        target_labels,
        split_indices,
        ridge_lambdas=ridge_lambdas,
        shuffle_seed=shuffle_seed,
    )


def collect_program_position_target_labels(
    rows: Sequence[dict[str, Any]],
    *,
    position_index: int,
    slot_count: int = 3,
) -> dict[str, list[str]]:
    if position_index < 0:
        raise ValueError("position_index must be non-negative")
    if slot_count < 0:
        raise ValueError("slot_count must be non-negative")

    labels: dict[str, list[str]] = {"depth": [], "top": []}
    for slot_index in range(1, slot_count + 1):
        labels[f"slot_{slot_index}"] = []

    for row in rows:
        if position_index >= len(row["trace"]):
            continue

        trace_row = row["trace"][position_index]
        stack_after = [int(value) for value in trace_row["stack_after"]]
        labels["depth"].append(str(len(stack_after)))
        labels["top"].append(str(stack_after[-1]) if stack_after else EMPTY_LABEL)
        for slot_index in range(1, slot_count + 1):
            label = (
                str(stack_after[-1 - slot_index])
                if len(stack_after) > slot_index
                else EMPTY_LABEL
            )
            labels[f"slot_{slot_index}"].append(label)

    return labels


def collect_program_position_surface_control_keys(
    rows: Sequence[dict[str, Any]],
    *,
    position_index: int,
) -> dict[str, list[str]]:
    if position_index < 0:
        raise ValueError("position_index must be non-negative")

    keys = {
        "current_token": [],
        "current_token_and_remaining_length": [],
    }
    for row in rows:
        if position_index >= len(row["tokens"]):
            continue

        current_token = str(row["tokens"][position_index])
        remaining_length = int(row["metadata"]["program_length"]) - position_index
        keys["current_token"].append(current_token)
        keys["current_token_and_remaining_length"].append(
            f"{current_token}|remaining_length={remaining_length}"
        )
    return keys


def program_position_slot_present_indices(
    rows: Sequence[dict[str, Any]],
    *,
    position_index: int,
    slot_index: int,
) -> list[int]:
    if position_index < 0:
        raise ValueError("position_index must be non-negative")
    if slot_index <= 0:
        raise ValueError("slot_index must be positive")

    indices: list[int] = []
    for row_index, row in enumerate(rows):
        if position_index >= len(row["trace"]):
            continue
        stack_after = row["trace"][position_index]["stack_after"]
        if len(stack_after) > slot_index:
            indices.append(row_index)
    return indices


def program_position_available_indices(
    rows: Sequence[dict[str, Any]],
    *,
    position_index: int,
) -> list[int]:
    if position_index < 0:
        raise ValueError("position_index must be non-negative")
    return [row_index for row_index, row in enumerate(rows) if position_index < len(row["tokens"])]


def collect_program_position_metadata(
    rows: Sequence[dict[str, Any]],
    *,
    position_index: int,
) -> dict[str, Any]:
    available_rows = [
        row
        for row in rows
        if position_index < len(row["tokens"])
    ]
    token_histogram = Counter(str(row["tokens"][position_index]) for row in available_rows)
    depth_histogram = Counter(len(row["trace"][position_index]["stack_after"]) for row in available_rows)
    return {
        "example_count": len(available_rows),
        "token_histogram": dict(sorted(token_histogram.items())),
        "depth_histogram": {str(key): depth_histogram[key] for key in sorted(depth_histogram)},
    }


def restrict_split_indices(
    available_indices: Sequence[int],
    split_indices: dict[str, list[int]],
) -> dict[str, list[int]]:
    split_index_sets = {
        split_name: set(indices)
        for split_name, indices in split_indices.items()
    }
    restricted: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for local_index, global_index in enumerate(available_indices):
        for split_name in ("train", "val", "test"):
            if global_index in split_index_sets[split_name]:
                restricted[split_name].append(local_index)
    return restricted


def summarize_best_probe_results(
    results_by_position: dict[str, dict[str, Any]],
    *,
    target_field: str = "targets",
) -> dict[str, Any]:
    best_by_target: dict[str, dict[str, Any]] = {}
    for position_name, position_result in results_by_position.items():
        for target_name, target_result in position_result[target_field].items():
            if target_result.get("skipped"):
                continue
            if "layers" not in target_result:
                continue
            if target_result.get("is_constant_label"):
                continue
            for layer_name, layer_metrics in target_result["layers"].items():
                candidate = {
                    "position": position_name,
                    "layer": layer_name,
                    "probe_test_accuracy": layer_metrics["probe_test_accuracy"],
                    "majority_test_accuracy": target_result["majority_test_accuracy"],
                    "chance_test_accuracy": target_result["chance_test_accuracy"],
                    "example_count": position_result["example_count"],
                }
                current = best_by_target.get(target_name)
                if current is None or candidate["probe_test_accuracy"] > current["probe_test_accuracy"]:
                    best_by_target[target_name] = candidate
    return best_by_target


def _prepare_probe_features(*feature_sets: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not feature_sets:
        raise ValueError("feature_sets must not be empty")

    train_features = feature_sets[0].to(dtype=torch.float64)
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    standardized = tuple((features.to(dtype=torch.float64) - mean) / std for features in feature_sets)
    return tuple(_append_bias_column(features) for features in standardized)


def _fit_ridge_weights(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ridge_lambda: float,
) -> torch.Tensor:
    targets = F.one_hot(labels, num_classes=num_classes).to(dtype=features.dtype)
    gram = features.T @ features
    regularizer = torch.eye(features.shape[1], dtype=features.dtype)
    regularizer[-1, -1] = 0.0
    return torch.linalg.solve(gram + ridge_lambda * regularizer, features.T @ targets)


def _accuracy(features: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (features @ weights).argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def _append_bias_column(features: torch.Tensor) -> torch.Tensor:
    bias = torch.ones((features.shape[0], 1), dtype=features.dtype)
    return torch.cat((features, bias), dim=1)


def select_feature_rows(
    layer_features: dict[str, torch.Tensor],
    row_indices: Sequence[int],
) -> dict[str, torch.Tensor]:
    index_tensor = _index_tensor(row_indices)
    return {
        layer_name: features.index_select(0, index_tensor)
        for layer_name, features in layer_features.items()
    }


def _encode_labels(labels: Sequence[str]) -> tuple[torch.Tensor, tuple[str, ...]]:
    label_vocab = tuple(sorted(set(labels), key=_label_sort_key))
    label_to_index = {label: index for index, label in enumerate(label_vocab)}
    return torch.tensor([label_to_index[label] for label in labels], dtype=torch.long), label_vocab


def _label_sort_key(label: str) -> tuple[int, int | str]:
    if label == EMPTY_LABEL:
        return (1, label)
    return (0, int(label))


def _index_tensor(indices: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(indices), dtype=torch.long)


def lookup_majority_control_accuracy(
    train_keys: Sequence[str],
    train_labels: torch.Tensor,
    test_keys: Sequence[str],
    test_labels: torch.Tensor,
    *,
    num_classes: int,
) -> LookupControlMetrics:
    if len(train_keys) != int(train_labels.shape[0]):
        raise ValueError("train key count does not match train label count")
    if len(test_keys) != int(test_labels.shape[0]):
        raise ValueError("test key count does not match test label count")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    default_prediction = int(torch.bincount(train_labels, minlength=num_classes).argmax().item())
    key_to_counts: dict[str, torch.Tensor] = {}
    for key, label in zip(train_keys, train_labels.tolist(), strict=True):
        if key not in key_to_counts:
            key_to_counts[key] = torch.zeros(num_classes, dtype=torch.long)
        key_to_counts[key][label] += 1

    predictions: list[int] = []
    unseen_test_key_count = 0
    for key in test_keys:
        if key not in key_to_counts:
            unseen_test_key_count += 1
            predictions.append(default_prediction)
            continue
        predictions.append(int(key_to_counts[key].argmax().item()))

    prediction_tensor = torch.tensor(predictions, dtype=torch.long)
    test_accuracy = float((prediction_tensor == test_labels).float().mean().item())
    return LookupControlMetrics(
        test_accuracy=test_accuracy,
        num_train_groups=len(key_to_counts),
        unseen_test_key_count=unseen_test_key_count,
        unseen_test_fraction=(
            float(unseen_test_key_count / len(test_keys))
            if test_keys
            else 0.0
        ),
    )
