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
    make_intervention_forward_fn,
    rollout_supervised_example,
    rollout_supervised_examples,
    rollout_target_for_example,
    select_matched_suffix_pairs,
    stack_ids_from_rollout_prediction,
)
from pushpop.pp0_model import ResidualIntervention, TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_supervised_examples_from_rows,
    capture_program_position_hidden_states,
    hidden_state_names,
    load_probe_rows,
    program_position_available_indices,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched-suffix activation patching on PP0 program positions."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=str, nargs="+", default=None)
    parser.add_argument("--pair-target", choices=["top", "depth", "stack"], default="top")
    parser.add_argument("--max-pairs", type=int, default=64)
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
        layer_features = capture_program_position_hidden_states(
            model,
            position_dataloader,
            position_index=position_index,
            device=device,
        )
        baseline_predictions = rollout_supervised_examples(
            model,
            position_examples,
            device,
            batch_size=args.batch_size,
        )
        baseline_targets = [rollout_target_for_example(example) for example in position_examples]
        exact_correct_local_indices = [
            local_index
            for local_index, (prediction, target) in enumerate(
                zip(baseline_predictions, baseline_targets, strict=True)
            )
            if prediction == target
        ]
        all_pairs = select_matched_suffix_pairs(
            position_rows,
            position_index=position_index,
            eligible_local_indices=exact_correct_local_indices,
            pair_target=args.pair_target,
            max_pairs=None,
            seed=args.seed,
        )
        selected_pairs = select_matched_suffix_pairs(
            position_rows,
            position_index=position_index,
            eligible_local_indices=exact_correct_local_indices,
            pair_target=args.pair_target,
            max_pairs=args.max_pairs,
            seed=args.seed,
        )

        layer_results: dict[str, dict[str, object]] = {}
        for layer_name in selected_layers:
            layer_results[layer_name] = run_layer_patch_panel(
                model,
                position_rows,
                position_examples,
                selected_pairs,
                source_layer_features=layer_features[layer_name],
                layer_name=layer_name,
                position_index=position_index,
                device=device,
            )

        results_by_position[f"pc_{position_index}"] = {
            "position_index": position_index,
            "example_count": len(position_rows),
            "baseline_exact_correct_count": len(exact_correct_local_indices),
            "baseline_exact_correct_fraction": (
                float(len(exact_correct_local_indices) / len(position_rows))
                if position_rows
                else 0.0
            ),
            "candidate_pair_count": len(all_pairs),
            "selected_pair_count": len(selected_pairs),
            "pair_target": args.pair_target,
            "layers": layer_results,
            "sample_pairs": [
                {
                    "source_example_id": position_examples[pair.source_local_index].example_id,
                    "target_example_id": position_examples[pair.target_local_index].example_id,
                    "current_token": pair.current_token,
                    "suffix_length": len(pair.suffix_tokens),
                    "source_top": pair.source_top,
                    "target_top": pair.target_top,
                    "source_depth": pair.source_depth,
                    "target_depth": pair.target_depth,
                }
                for pair in selected_pairs[:5]
            ],
            "summary": summarize_patch_position(layer_results),
        }

    output = {
        "analysis": "pp0_matched_suffix_activation_patching_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "positions": results_by_position,
        "layers": selected_layers,
        "pair_target": args.pair_target,
        "matching_definition": (
            "Pairs have the same current token at pc_k and the same suffix after pc_k, but differ "
            "on the selected pair_target state."
        ),
        "intervention_definition": (
            "Patch the source hidden state at (layer, pc_k) into the target example at the same site, "
            "then rerun target rollout."
        ),
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def run_layer_patch_panel(
    model: TinyTransformer,
    position_rows: list[dict[str, object]],
    position_examples: list[object],
    selected_pairs: list[object],
    *,
    source_layer_features,
    layer_name: str,
    position_index: int,
    device,
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

        intervention = ResidualIntervention(
            layer_name=layer_name,
            position_index=position_index,
            replacement_vector=source_layer_features[pair.source_local_index].clone(),
        )
        patched_prediction = rollout_supervised_example(
            model,
            target_example,
            device,
            forward_fn=make_intervention_forward_fn(model, [intervention]),
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


def summarize_patch_position(
    layer_results: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    best_source_top = max(
        layer_results.items(),
        key=lambda item: item[1]["patched_source_top_rate"],
    )
    best_source_full = max(
        layer_results.items(),
        key=lambda item: item[1]["patched_matches_source_full_rate"],
    )
    return {
        "best_source_top_transfer": {
            "layer": best_source_top[0],
            "patched_source_top_rate": best_source_top[1]["patched_source_top_rate"],
            "patched_target_top_rate": best_source_top[1]["patched_target_top_rate"],
        },
        "best_source_full_transfer": {
            "layer": best_source_full[0],
            "patched_matches_source_full_rate": best_source_full[1]["patched_matches_source_full_rate"],
            "patched_matches_target_full_rate": best_source_full[1]["patched_matches_target_full_rate"],
        },
    }


if __name__ == "__main__":
    main()
