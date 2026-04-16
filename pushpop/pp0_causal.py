"""Targeted causal interventions for PP0 Phase 2."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch

from pushpop.pp0_model import ResidualIntervention, TinyTransformer
from pushpop.pp0_training import evaluate_model


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
