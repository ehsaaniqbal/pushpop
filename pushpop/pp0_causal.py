"""Targeted causal interventions for PP0 Phase 2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import random
from typing import Any

import torch

from pushpop.pp0_model import ResidualIntervention, TinyTransformer
from pushpop.pp0_training import STOP_ID, SupervisedExample, evaluate_model, greedy_rollout_batch


def build_mean_ablation_intervention(
    layer_features: dict[str, torch.Tensor],
    *,
    layer_name: str,
    position_index: int,
) -> ResidualIntervention:
    if layer_name not in layer_features:
        raise ValueError(f"unknown layer_name for mean ablation: {layer_name!r}")
    features = layer_features[layer_name]
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"layer_features[{layer_name!r}] must have shape (n, d_model) with n > 0")
    return ResidualIntervention(
        layer_name=layer_name,
        position_index=position_index,
        replacement_vector=features.mean(dim=0).cpu(),
    )


def build_zero_ablation_intervention(
    layer_features: dict[str, torch.Tensor],
    *,
    layer_name: str,
    position_index: int,
) -> ResidualIntervention:
    if layer_name not in layer_features:
        raise ValueError(f"unknown layer_name for zero ablation: {layer_name!r}")
    features = layer_features[layer_name]
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"layer_features[{layer_name!r}] must have shape (n, d_model) with n > 0")
    return ResidualIntervention(
        layer_name=layer_name,
        position_index=position_index,
        replacement_vector=torch.zeros_like(features[0]).cpu(),
    )


def make_intervention_forward_fn(
    model: TinyTransformer,
    interventions: Sequence[ResidualIntervention],
) -> Callable[[torch.Tensor], torch.Tensor]:
    intervention_tuple = tuple(interventions)

    def forward_fn(input_ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids, interventions=intervention_tuple)

    return forward_fn


@torch.no_grad()
def evaluate_model_with_intervention(
    model: TinyTransformer,
    dataloader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    interventions: Sequence[ResidualIntervention],
) -> dict[str, Any]:
    return evaluate_model(
        model,
        dataloader,
        device,
        compute_slices=False,
        forward_fn=make_intervention_forward_fn(model, interventions),
    )


def summarize_metric_deltas(
    baseline_metrics: dict[str, Any],
    intervention_metrics: dict[str, Any],
    *,
    metric_names: Sequence[str] = ("exact_match", "top_accuracy", "stop_accuracy", "token_accuracy", "loss"),
) -> dict[str, float]:
    baseline_overall = baseline_metrics["overall"]
    intervention_overall = intervention_metrics["overall"]
    deltas: dict[str, float] = {}
    for metric_name in metric_names:
        if metric_name not in baseline_overall or metric_name not in intervention_overall:
            continue
        deltas[f"{metric_name}_delta"] = float(
            intervention_overall[metric_name] - baseline_overall[metric_name]
        )
    return deltas


def rollout_supervised_example(
    model: TinyTransformer,
    example: SupervisedExample,
    device: torch.device,
    *,
    forward_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[int]:
    return rollout_supervised_examples(
        model,
        [example],
        device,
        batch_size=1,
        forward_fn=forward_fn,
    )[0]


def rollout_supervised_examples(
    model: TinyTransformer,
    examples: Sequence[SupervisedExample],
    device: torch.device,
    *,
    batch_size: int = 64,
    forward_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> list[list[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not examples:
        return []

    predictions: list[list[int]] = []
    for start_index in range(0, len(examples), batch_size):
        batch_examples = examples[start_index : start_index + batch_size]
        rollout_prefixes = [rollout_prefix_for_example(example) for example in batch_examples]
        rollout_targets = [rollout_target_for_example(example) for example in batch_examples]
        batch_predictions = greedy_rollout_batch(
            model,
            rollout_prefixes,
            device=device,
            max_new_tokens=max(len(target_tokens) for target_tokens in rollout_targets) + 2,
            forward_fn=forward_fn,
        )
        predictions.extend(batch_predictions)
    return predictions


def rollout_prefix_for_example(example: SupervisedExample) -> list[int]:
    first_target_position = next(
        position_index
        for position_index, is_supervised in enumerate(example.loss_mask)
        if is_supervised
    )
    return list(example.input_ids[: first_target_position + 1])


def rollout_target_for_example(example: SupervisedExample) -> list[int]:
    return [
        target_id
        for target_id, is_supervised in zip(example.target_ids, example.loss_mask, strict=True)
        if is_supervised
    ]


def stack_ids_from_rollout_prediction(prediction: Sequence[int]) -> list[int]:
    prediction_list = list(prediction)
    if STOP_ID in prediction_list:
        prediction_list = prediction_list[: prediction_list.index(STOP_ID)]
    return prediction_list


@dataclass(frozen=True, slots=True)
class MatchedSuffixPair:
    source_local_index: int
    target_local_index: int
    position_index: int
    current_token: str
    suffix_tokens: tuple[str, ...]
    source_top: str
    target_top: str
    source_depth: int
    target_depth: int


def select_matched_suffix_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    position_index: int,
    eligible_local_indices: Sequence[int],
    pair_target: str = "top",
    max_pairs: int | None = 64,
    seed: int = 0,
) -> list[MatchedSuffixPair]:
    if position_index < 0:
        raise ValueError("position_index must be non-negative")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if pair_target not in {"top", "depth", "stack"}:
        raise ValueError("pair_target must be one of: top, depth, stack")

    eligible_set = set(eligible_local_indices)
    grouped_indices: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for local_index, row in enumerate(rows):
        if local_index not in eligible_set:
            continue
        if position_index >= len(row["tokens"]):
            continue
        group_key = (
            str(row["tokens"][position_index]),
            *(str(token) for token in row["tokens"][position_index + 1 :]),
        )
        grouped_indices[group_key].append(local_index)

    candidate_pairs: list[MatchedSuffixPair] = []
    for group_key, group_local_indices in grouped_indices.items():
        for source_index in group_local_indices:
            source_state = _stack_state_after_position(rows[source_index], position_index)
            for target_index in group_local_indices:
                if source_index == target_index:
                    continue
                target_state = _stack_state_after_position(rows[target_index], position_index)
                if pair_target == "top" and source_state["top"] == target_state["top"]:
                    continue
                if pair_target == "depth" and source_state["depth"] == target_state["depth"]:
                    continue
                if pair_target == "stack" and source_state["stack"] == target_state["stack"]:
                    continue
                candidate_pairs.append(
                    MatchedSuffixPair(
                        source_local_index=source_index,
                        target_local_index=target_index,
                        position_index=position_index,
                        current_token=group_key[0],
                        suffix_tokens=group_key[1:],
                        source_top=str(source_state["top"]),
                        target_top=str(target_state["top"]),
                        source_depth=int(source_state["depth"]),
                        target_depth=int(target_state["depth"]),
                    )
                )

    generator = random.Random(seed)
    generator.shuffle(candidate_pairs)
    if max_pairs is None:
        return candidate_pairs
    return candidate_pairs[:max_pairs]


def _stack_state_after_position(row: dict[str, Any], position_index: int) -> dict[str, Any]:
    stack_after = [int(value) for value in row["trace"][position_index]["stack_after"]]
    return {
        "stack": tuple(stack_after),
        "top": stack_after[-1] if stack_after else "EMPTY",
        "depth": len(stack_after),
    }


def run_site_patch_panel(
    model: TinyTransformer,
    position_examples: Sequence[SupervisedExample],
    selected_pairs: Sequence[MatchedSuffixPair],
    *,
    source_site_features: torch.Tensor,
    site_name: str,
    position_index: int,
    device: torch.device,
) -> dict[str, object]:
    def intervention_builder(pair: MatchedSuffixPair) -> list[ResidualIntervention]:
        return [
            ResidualIntervention(
                layer_name=site_name,
                position_index=position_index,
                replacement_vector=source_site_features[pair.source_local_index].clone(),
            )
        ]

    return run_pair_intervention_panel(
        model,
        position_examples,
        selected_pairs,
        intervention_builder=intervention_builder,
        device=device,
    )


def run_pair_intervention_panel(
    model: TinyTransformer,
    position_examples: Sequence[SupervisedExample],
    selected_pairs: Sequence[MatchedSuffixPair],
    *,
    intervention_builder: Callable[[MatchedSuffixPair], Sequence[ResidualIntervention]],
    device: torch.device,
) -> dict[str, object]:
    if not selected_pairs:
        return {
            "pair_count": 0,
            "patched_matches_source_full_rate": 0.0,
            "patched_matches_target_full_rate": 0.0,
            "patched_source_top_rate": 0.0,
            "patched_target_top_rate": 0.0,
            "patched_source_depth_rate": 0.0,
            "patched_target_depth_rate": 0.0,
            "sample_results": [],
        }

    patched_matches_source_full = 0
    patched_matches_target_full = 0
    patched_source_top = 0
    patched_target_top = 0
    patched_source_depth = 0
    patched_target_depth = 0
    sample_results: list[dict[str, object]] = []

    for pair in selected_pairs:
        source_example = position_examples[pair.source_local_index]
        target_example = position_examples[pair.target_local_index]
        source_target = rollout_target_for_example(source_example)
        target_target = rollout_target_for_example(target_example)
        source_stack = stack_ids_from_rollout_prediction(source_target)
        target_stack = stack_ids_from_rollout_prediction(target_target)

        interventions = list(intervention_builder(pair))
        patched_prediction = rollout_supervised_example(
            model,
            target_example,
            device,
            forward_fn=make_intervention_forward_fn(model, interventions),
        )
        patched_stack = stack_ids_from_rollout_prediction(patched_prediction)

        source_full_match = patched_prediction == source_target
        target_full_match = patched_prediction == target_target
        source_top_match = bool(patched_stack and source_stack and patched_stack[-1] == source_stack[-1])
        target_top_match = bool(patched_stack and target_stack and patched_stack[-1] == target_stack[-1])
        source_depth_match = len(patched_stack) == len(source_stack)
        target_depth_match = len(patched_stack) == len(target_stack)

        patched_matches_source_full += int(source_full_match)
        patched_matches_target_full += int(target_full_match)
        patched_source_top += int(source_top_match)
        patched_target_top += int(target_top_match)
        patched_source_depth += int(source_depth_match)
        patched_target_depth += int(target_depth_match)

        if len(sample_results) < 5:
            sample_results.append(
                {
                    "source_example_id": source_example.example_id,
                    "target_example_id": target_example.example_id,
                    "source_top": pair.source_top,
                    "target_top": pair.target_top,
                    "source_depth": pair.source_depth,
                    "target_depth": pair.target_depth,
                    "patched_top": patched_stack[-1] if patched_stack else None,
                    "patched_depth": len(patched_stack),
                    "patched_matches_source_full": source_full_match,
                    "patched_matches_target_full": target_full_match,
                    "patched_source_top_match": source_top_match,
                    "patched_target_top_match": target_top_match,
                }
            )

    pair_count = len(selected_pairs)
    return {
        "pair_count": pair_count,
        "patched_matches_source_full_rate": patched_matches_source_full / pair_count,
        "patched_matches_target_full_rate": patched_matches_target_full / pair_count,
        "patched_source_top_rate": patched_source_top / pair_count,
        "patched_target_top_rate": patched_target_top / pair_count,
        "patched_source_depth_rate": patched_source_depth / pair_count,
        "patched_target_depth_rate": patched_target_depth / pair_count,
        "sample_results": sample_results,
    }
