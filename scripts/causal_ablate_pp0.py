from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pushpop.pp0_causal import (
    build_mean_ablation_intervention,
    evaluate_model_with_intervention,
    summarize_metric_deltas,
)
from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_supervised_examples_from_rows,
    capture_program_position_hidden_states,
    hidden_state_names,
    load_probe_rows,
    program_position_available_indices,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, evaluate_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted residual mean-ablation on PP0 program positions."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=str, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-sanity-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model_config = TinyTransformerConfig.from_dict(checkpoint["model_config"])
    model = TinyTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    rows = load_probe_rows(
        args.data_path,
        sanity_checks=not args.no_sanity_checks,
    )
    available_example_count = len(rows)
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided")
        rows = rows[: min(args.max_examples, len(rows))]
    examples = build_supervised_examples_from_rows(rows)

    max_sequence_length = max(example.sequence_length for example in examples)
    if max_sequence_length > model_config.context_length:
        raise ValueError(
            f"dataset sequence length {max_sequence_length} exceeds model context length "
            f"{model_config.context_length}"
        )

    layer_names = hidden_state_names(model)
    default_layers = [name for name in layer_names if name != "embed"]
    selected_layers = args.layers if args.layers is not None else default_layers
    for layer_name in selected_layers:
        if layer_name not in layer_names:
            raise ValueError(f"unknown layer name: {layer_name!r}")

    full_dataloader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_supervised_examples,
    )
    full_baseline = evaluate_model(model, full_dataloader, device, compute_slices=False)

    results_by_position: dict[str, dict[str, object]] = {}
    for position_index in args.positions:
        if position_index < 0:
            raise ValueError("positions must be non-negative")
        available_indices = program_position_available_indices(rows, position_index=position_index)
        if not available_indices:
            raise ValueError(f"no examples reach program position {position_index}")

        position_rows = [rows[index] for index in available_indices]
        position_examples = [examples[index] for index in available_indices]
        position_dataloader = DataLoader(
            position_examples,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_supervised_examples,
        )
        baseline_metrics = evaluate_model(model, position_dataloader, device, compute_slices=False)
        layer_features = capture_program_position_hidden_states(
            model,
            position_dataloader,
            position_index=position_index,
            device=device,
        )

        interventions_by_layer: dict[str, dict[str, object]] = {}
        for layer_name in selected_layers:
            intervention = build_mean_ablation_intervention(
                layer_features,
                layer_name=layer_name,
                position_index=position_index,
            )
            intervention_metrics = evaluate_model_with_intervention(
                model,
                position_dataloader,
                device,
                interventions=[intervention],
            )
            interventions_by_layer[layer_name] = {
                "mean_replacement_norm": float(intervention.replacement_vector.norm().item()),
                "metrics": intervention_metrics,
                "deltas": summarize_metric_deltas(baseline_metrics, intervention_metrics),
            }

        results_by_position[f"pc_{position_index}"] = {
            "position_index": position_index,
            "example_count": len(position_rows),
            "affected_fraction": float(len(position_rows) / len(rows)),
            "baseline_metrics": baseline_metrics,
            "interventions": interventions_by_layer,
            "summary": summarize_position_effects(interventions_by_layer),
        }

    output = {
        "analysis": "pp0_residual_mean_ablation_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "positions": results_by_position,
        "layers": selected_layers,
        "full_baseline_metrics": full_baseline,
        "intervention_definition": (
            "Replace one residual vector at (layer, pc_k) with the mean vector from the same "
            "site over the evaluated subset, then rerun rollout evaluation."
        ),
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def summarize_position_effects(
    interventions_by_layer: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    by_exact = min(
        interventions_by_layer.items(),
        key=lambda item: item[1]["deltas"].get("exact_match_delta", 0.0),
    )
    by_top = min(
        interventions_by_layer.items(),
        key=lambda item: item[1]["deltas"].get("top_accuracy_delta", 0.0),
    )
    return {
        "largest_exact_drop": {
            "layer": by_exact[0],
            **by_exact[1]["deltas"],
        },
        "largest_top_drop": {
            "layer": by_top[0],
            **by_top[1]["deltas"],
        },
    }


if __name__ == "__main__":
    main()
