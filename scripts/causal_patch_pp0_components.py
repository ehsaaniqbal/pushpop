from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pushpop.pp0_causal import (
    rollout_supervised_examples,
    rollout_target_for_example,
    run_site_patch_panel,
    select_matched_suffix_pairs,
)
from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_supervised_examples_from_rows,
    capture_program_position_component_outputs,
    load_probe_rows,
    program_position_available_indices,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched-suffix component-family patching on PP0 program positions."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=str, nargs="+", default=None)
    parser.add_argument("--components", choices=["attn", "mlp"], nargs="+", default=["attn", "mlp"])
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

    available_blocks = [f"block_{index}" for index in range(model.config.n_layers)]
    selected_blocks = args.blocks if args.blocks is not None else available_blocks
    for block_name in selected_blocks:
        if block_name not in available_blocks:
            raise ValueError(f"unknown block name: {block_name!r}")

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
        component_features = capture_program_position_component_outputs(
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

        block_results: dict[str, dict[str, dict[str, object]]] = {}
        for block_name in selected_blocks:
            component_results: dict[str, dict[str, object]] = {}
            for component_name in args.components:
                site_name = f"{block_name}.{component_name}"
                component_results[component_name] = run_site_patch_panel(
                    model,
                    position_examples,
                    selected_pairs,
                    source_site_features=component_features[site_name],
                    site_name=site_name,
                    position_index=position_index,
                    device=device,
                )
            block_results[block_name] = component_results

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
            "components": list(args.components),
            "blocks": block_results,
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
            "summary": summarize_component_position(block_results),
        }

    output = {
        "analysis": "pp0_component_family_activation_patching_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "positions": results_by_position,
        "blocks": selected_blocks,
        "components": list(args.components),
        "pair_target": args.pair_target,
        "matching_definition": (
            "Pairs have the same current token at pc_k and the same suffix after pc_k, but differ "
            "on the selected pair_target state."
        ),
        "intervention_definition": (
            "Patch the source component output at (block, component, pc_k) into the target example "
            "at the same site, then rerun target rollout."
        ),
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def summarize_component_position(
    block_results: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    best_site_name: str | None = None
    best_site_metrics: dict[str, object] | None = None
    per_block: dict[str, dict[str, object]] = {}

    for block_name, component_results in block_results.items():
        attn_result = component_results.get("attn")
        mlp_result = component_results.get("mlp")
        better_component = "tie"
        if attn_result is not None and mlp_result is not None:
            attn_score = float(attn_result["patched_source_top_rate"])
            mlp_score = float(mlp_result["patched_source_top_rate"])
            if not math.isclose(attn_score, mlp_score):
                better_component = "attn" if attn_score > mlp_score else "mlp"
        elif attn_result is not None:
            better_component = "attn"
        elif mlp_result is not None:
            better_component = "mlp"

        per_block[block_name] = {
            "better_component_for_top": better_component,
            "attn_patched_source_top_rate": (
                float(attn_result["patched_source_top_rate"]) if attn_result is not None else None
            ),
            "mlp_patched_source_top_rate": (
                float(mlp_result["patched_source_top_rate"]) if mlp_result is not None else None
            ),
        }

        for component_name, metrics in component_results.items():
            if best_site_metrics is None or (
                float(metrics["patched_source_top_rate"])
                > float(best_site_metrics["patched_source_top_rate"])
            ):
                best_site_name = f"{block_name}.{component_name}"
                best_site_metrics = metrics

    best_source_top_transfer = {
        "site": best_site_name,
        "patched_source_top_rate": (
            float(best_site_metrics["patched_source_top_rate"]) if best_site_metrics is not None else 0.0
        ),
        "patched_target_top_rate": (
            float(best_site_metrics["patched_target_top_rate"]) if best_site_metrics is not None else 0.0
        ),
    }
    return {
        "best_source_top_transfer": best_source_top_transfer,
        "by_block": per_block,
    }


if __name__ == "__main__":
    main()
