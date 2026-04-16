from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pushpop.pp0_causal import (
    make_intervention_forward_fn,
    rollout_supervised_example,
    rollout_supervised_examples,
    rollout_target_for_example,
    select_matched_suffix_pairs,
    stack_ids_from_rollout_prediction,
)
from pushpop.pp0_model import ResidualIntervention, TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_probe_splits,
    build_supervised_examples_from_rows,
    capture_program_position_hidden_states,
    collect_program_position_target_labels,
    fit_ridge_probe_model,
    hidden_state_names,
    load_probe_rows,
    program_position_available_indices,
    restrict_split_indices,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-direction steering experiments on PP0."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=str, nargs="+", default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--probe-train-size", type=int, default=8000)
    parser.add_argument("--probe-val-size", type=int, default=1000)
    parser.add_argument("--probe-test-size", type=int, default=1000)
    parser.add_argument(
        "--ridge-lambdas",
        type=float,
        nargs="+",
        default=[1e-6, 1e-4, 1e-2, 1.0, 100.0],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
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

    split_indices = build_probe_splits(
        len(rows),
        train_size=args.probe_train_size,
        val_size=args.probe_val_size,
        test_size=args.probe_test_size,
    )
    used_example_count = args.probe_train_size + args.probe_val_size + args.probe_test_size
    rows = rows[:used_example_count]
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

    results_by_position: dict[str, dict[str, object]] = {}
    for position_index in args.positions:
        available_indices = program_position_available_indices(rows, position_index=position_index)
        if not available_indices:
            raise ValueError(f"no examples reach program position {position_index}")
        position_rows = [rows[index] for index in available_indices]
        position_examples = [examples[index] for index in available_indices]
        position_split_indices = restrict_split_indices(available_indices, split_indices)
        if not all(position_split_indices[split_name] for split_name in ("train", "val", "test")):
            raise ValueError(
                f"position {position_index} does not have examples in every probe split"
            )

        position_dataloader = DataLoader(
            position_examples,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_supervised_examples,
        )
        layer_features = capture_program_position_hidden_states(
            model,
            position_dataloader,
            position_index=position_index,
            device=device,
        )
        top_labels = collect_program_position_target_labels(
            position_rows,
            position_index=position_index,
            slot_count=0,
        )["top"]

        probe_models: dict[str, dict[str, object]] = {}
        for layer_name in selected_layers:
            probe_model = fit_ridge_probe_model(
                layer_features[layer_name],
                top_labels,
                position_split_indices,
                ridge_lambdas=args.ridge_lambdas,
            )
            shuffled_probe_model = fit_ridge_probe_model(
                layer_features[layer_name],
                top_labels,
                position_split_indices,
                ridge_lambdas=args.ridge_lambdas,
                shuffle_train_labels=True,
                shuffle_seed=args.seed,
            )
            probe_models[layer_name] = {
                "probe": probe_model,
                "shuffled_probe": shuffled_probe_model,
            }

        baseline_predictions = rollout_supervised_examples(
            model,
            position_examples,
            device,
            batch_size=args.batch_size,
        )
        baseline_targets = [rollout_target_for_example(example) for example in position_examples]
        exact_correct_test_local_indices = [
            local_index
            for local_index in position_split_indices["test"]
            if baseline_predictions[local_index] == baseline_targets[local_index]
        ]
        candidate_pairs = select_matched_suffix_pairs(
            position_rows,
            position_index=position_index,
            eligible_local_indices=exact_correct_test_local_indices,
            pair_target="top",
            max_pairs=None,
            seed=args.seed,
        )
        selected_pairs = select_matched_suffix_pairs(
            position_rows,
            position_index=position_index,
            eligible_local_indices=exact_correct_test_local_indices,
            pair_target="top",
            max_pairs=args.max_pairs,
            seed=args.seed,
        )

        layer_results: dict[str, dict[str, object]] = {}
        for layer_name in selected_layers:
            train_indices = position_split_indices["train"]
            train_features = layer_features[layer_name].index_select(
                0,
                torch.tensor(train_indices, dtype=torch.long),
            )
            layer_results[layer_name] = run_layer_steering_panel(
                model,
                position_examples,
                selected_pairs,
                source_layer_features=layer_features[layer_name],
                train_features=train_features,
                layer_name=layer_name,
                position_index=position_index,
                probe_model=probe_models[layer_name]["probe"],
                shuffled_probe_model=probe_models[layer_name]["shuffled_probe"],
                alphas=args.alphas,
                device=device,
            )

        results_by_position[f"pc_{position_index}"] = {
            "position_index": position_index,
            "example_count": len(position_examples),
            "split_example_counts": {
                split_name: len(position_split_indices[split_name])
                for split_name in ("train", "val", "test")
            },
            "baseline_exact_correct_test_count": len(exact_correct_test_local_indices),
            "candidate_pair_count": len(candidate_pairs),
            "selected_pair_count": len(selected_pairs),
            "layers": layer_results,
            "summary": summarize_steering_position(layer_results),
        }

    output = {
        "analysis": "pp0_top_direction_steering_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": used_example_count,
        "positions": results_by_position,
        "layers": selected_layers,
        "alphas": args.alphas,
        "intervention_definition": (
            "Add alpha times a probe-defined source-minus-target top direction at one site, "
            "scaled by the train-set projection standard deviation at that site."
        ),
        "control_definition": "Repeat the same procedure with a shuffled-label top probe.",
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def run_layer_steering_panel(
    model: TinyTransformer,
    position_examples: list[object],
    selected_pairs: list[object],
    *,
    source_layer_features: torch.Tensor,
    train_features: torch.Tensor,
    layer_name: str,
    position_index: int,
    probe_model,
    shuffled_probe_model,
    alphas: list[float],
    device,
) -> dict[str, object]:
    return {
        "probe_model": {
            "ridge_lambda": probe_model.ridge_lambda,
            "train_accuracy": probe_model.train_accuracy,
            "val_accuracy": probe_model.val_accuracy,
            "test_accuracy": probe_model.test_accuracy,
        },
        "shuffled_probe_model": {
            "ridge_lambda": shuffled_probe_model.ridge_lambda,
            "train_accuracy": shuffled_probe_model.train_accuracy,
            "val_accuracy": shuffled_probe_model.val_accuracy,
            "test_accuracy": shuffled_probe_model.test_accuracy,
        },
        "probe": run_direction_panel(
            model,
            position_examples,
            selected_pairs,
            source_layer_features=source_layer_features,
            train_features=train_features,
            layer_name=layer_name,
            position_index=position_index,
            probe_model=probe_model,
            alphas=alphas,
            device=device,
        ),
        "shuffled_probe": run_direction_panel(
            model,
            position_examples,
            selected_pairs,
            source_layer_features=source_layer_features,
            train_features=train_features,
            layer_name=layer_name,
            position_index=position_index,
            probe_model=shuffled_probe_model,
            alphas=alphas,
            device=device,
        ),
    }


def run_direction_panel(
    model: TinyTransformer,
    position_examples: list[object],
    selected_pairs: list[object],
    *,
    source_layer_features: torch.Tensor,
    train_features: torch.Tensor,
    layer_name: str,
    position_index: int,
    probe_model,
    alphas: list[float],
    device,
) -> dict[str, object]:
    results_by_alpha: dict[str, dict[str, object]] = {}
    for alpha in alphas:
        results_by_alpha[str(alpha)] = run_alpha_steering(
            model,
            position_examples,
            selected_pairs,
            source_layer_features=source_layer_features,
            train_features=train_features,
            layer_name=layer_name,
            position_index=position_index,
            probe_model=probe_model,
            alpha=float(alpha),
            device=device,
        )
    best_alpha, best_result = max(
        results_by_alpha.items(),
        key=lambda item: item[1]["source_top_rate"],
    )
    return {
        "results_by_alpha": results_by_alpha,
        "best_alpha_by_source_top": {
            "alpha": float(best_alpha),
            "source_top_rate": best_result["source_top_rate"],
            "target_top_rate": best_result["target_top_rate"],
            "source_full_rate": best_result["source_full_rate"],
            "target_full_rate": best_result["target_full_rate"],
        },
    }


def run_alpha_steering(
    model: TinyTransformer,
    position_examples: list[object],
    selected_pairs: list[object],
    *,
    source_layer_features: torch.Tensor,
    train_features: torch.Tensor,
    layer_name: str,
    position_index: int,
    probe_model,
    alpha: float,
    device,
) -> dict[str, object]:
    if not selected_pairs:
        return {
            "pair_count": 0,
            "source_top_rate": 0.0,
            "target_top_rate": 0.0,
            "source_full_rate": 0.0,
            "target_full_rate": 0.0,
            "sample_results": [],
        }

    source_top_matches = 0
    target_top_matches = 0
    source_full_matches = 0
    target_full_matches = 0
    sample_results: list[dict[str, object]] = []

    for pair in selected_pairs:
        source_example = position_examples[pair.source_local_index]
        target_example = position_examples[pair.target_local_index]
        source_target = rollout_target_for_example(source_example)
        target_target = rollout_target_for_example(target_example)
        source_stack = stack_ids_from_rollout_prediction(source_target)
        target_stack = stack_ids_from_rollout_prediction(target_target)

        direction = probe_model.class_difference_direction(pair.source_top, pair.target_top)
        direction_norm = float(direction.norm().item())
        if direction_norm == 0.0:
            delta_vector = torch.zeros_like(direction)
            projection_std = 0.0
        else:
            unit_direction = direction / direction_norm
            projection_std = float(
                (train_features.to(dtype=torch.float32) @ unit_direction).std(unbiased=False).item()
            )
            delta_vector = alpha * projection_std * unit_direction

        intervention = ResidualIntervention(
            layer_name=layer_name,
            position_index=position_index,
            replacement_vector=delta_vector,
            mode="add",
        )
        steered_prediction = rollout_supervised_example(
            model,
            target_example,
            device,
            forward_fn=make_intervention_forward_fn(model, [intervention]),
        )
        steered_stack = stack_ids_from_rollout_prediction(steered_prediction)

        source_top_match = bool(steered_stack and source_stack and steered_stack[-1] == source_stack[-1])
        target_top_match = bool(steered_stack and target_stack and steered_stack[-1] == target_stack[-1])
        source_full_match = steered_prediction == source_target
        target_full_match = steered_prediction == target_target

        source_top_matches += int(source_top_match)
        target_top_matches += int(target_top_match)
        source_full_matches += int(source_full_match)
        target_full_matches += int(target_full_match)

        if len(sample_results) < 5:
            sample_results.append(
                {
                    "source_example_id": source_example.example_id,
                    "target_example_id": target_example.example_id,
                    "source_top": pair.source_top,
                    "target_top": pair.target_top,
                    "steered_top": steered_stack[-1] if steered_stack else None,
                    "direction_norm": direction_norm,
                    "projection_std": projection_std,
                    "source_top_match": source_top_match,
                    "target_top_match": target_top_match,
                    "source_full_match": source_full_match,
                    "target_full_match": target_full_match,
                }
            )

    pair_count = len(selected_pairs)
    return {
        "pair_count": pair_count,
        "source_top_rate": source_top_matches / pair_count,
        "target_top_rate": target_top_matches / pair_count,
        "source_full_rate": source_full_matches / pair_count,
        "target_full_rate": target_full_matches / pair_count,
        "sample_results": sample_results,
    }


def summarize_steering_position(
    layer_results: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    best_probe = max(
        (
            (
                layer_name,
                layer_result["probe"]["best_alpha_by_source_top"],
            )
            for layer_name, layer_result in layer_results.items()
        ),
        key=lambda item: item[1]["source_top_rate"],
    )
    best_shuffled = max(
        (
            (
                layer_name,
                layer_result["shuffled_probe"]["best_alpha_by_source_top"],
            )
            for layer_name, layer_result in layer_results.items()
        ),
        key=lambda item: item[1]["source_top_rate"],
    )
    return {
        "best_probe_source_top_transfer": {
            "layer": best_probe[0],
            **best_probe[1],
        },
        "best_shuffled_source_top_transfer": {
            "layer": best_shuffled[0],
            **best_shuffled[1],
        },
    }


if __name__ == "__main__":
    main()
