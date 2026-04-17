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
    rollout_supervised_examples,
    rollout_target_for_example,
    run_site_patch_panel,
    select_matched_suffix_pairs,
)
from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_supervised_examples_from_rows,
    capture_program_position_attention_head_outputs,
    load_probe_rows,
    program_position_available_indices,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched-suffix attention-head patching on PP0 program positions."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=str, nargs="+", default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
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

    available_heads = list(range(model.config.n_heads))
    selected_heads = args.heads if args.heads is not None else available_heads
    for head_index in selected_heads:
        if head_index not in available_heads:
            raise ValueError(f"unknown head index: {head_index}")

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
        head_features = capture_program_position_attention_head_outputs(
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

        block_results: dict[str, dict[str, object]] = {}
        for block_name in selected_blocks:
            head_results: dict[str, object] = {}
            for head_index in selected_heads:
                site_name = f"{block_name}.attn_head_{head_index}"
                head_results[f"head_{head_index}"] = run_site_patch_panel(
                    model,
                    position_examples,
                    selected_pairs,
                    source_site_features=head_features[site_name],
                    site_name=site_name,
                    position_index=position_index,
                    device=device,
                )
            block_results[block_name] = {
                "heads": head_results,
                "summary": summarize_head_block(head_results),
            }

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
            "head_indices": list(selected_heads),
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
            "summary": summarize_head_position(block_results),
        }

    output = {
        "analysis": "pp0_attention_head_activation_patching_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "positions": results_by_position,
        "blocks": selected_blocks,
        "head_indices": list(selected_heads),
        "pair_target": args.pair_target,
        "matching_definition": (
            "Pairs have the same current token at pc_k and the same suffix after pc_k, but differ "
            "on the selected pair_target state."
        ),
        "intervention_definition": (
            "Patch the source attention-head residual contribution at (block, head, pc_k) into the "
            "target example at the same site, then rerun target rollout."
        ),
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def summarize_head_block(head_results: dict[str, object]) -> dict[str, object]:
    ranked_heads = sorted(
        head_results.items(),
        key=lambda item: (
            float(item[1]["patched_source_top_rate"]),
            -float(item[1]["patched_target_top_rate"]),
        ),
        reverse=True,
    )
    best_head_name, best_head_metrics = ranked_heads[0]
    return {
        "best_source_top_transfer": {
            "head": best_head_name,
            "patched_source_top_rate": float(best_head_metrics["patched_source_top_rate"]),
            "patched_target_top_rate": float(best_head_metrics["patched_target_top_rate"]),
        },
        "top_heads_by_source_top_transfer": [
            {
                "head": head_name,
                "patched_source_top_rate": float(metrics["patched_source_top_rate"]),
                "patched_target_top_rate": float(metrics["patched_target_top_rate"]),
            }
            for head_name, metrics in ranked_heads[:3]
        ],
    }


def summarize_head_position(
    block_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    overall_best_block_name: str | None = None
    overall_best_head_name: str | None = None
    overall_best_metrics: dict[str, object] | None = None

    for block_name, block_result in block_results.items():
        head_results = block_result["heads"]
        for head_name, metrics in head_results.items():
            if overall_best_metrics is None or (
                float(metrics["patched_source_top_rate"])
                > float(overall_best_metrics["patched_source_top_rate"])
            ):
                overall_best_block_name = block_name
                overall_best_head_name = head_name
                overall_best_metrics = metrics

    return {
        "best_source_top_transfer": {
            "block": overall_best_block_name,
            "head": overall_best_head_name,
            "patched_source_top_rate": (
                float(overall_best_metrics["patched_source_top_rate"])
                if overall_best_metrics is not None
                else 0.0
            ),
            "patched_target_top_rate": (
                float(overall_best_metrics["patched_target_top_rate"])
                if overall_best_metrics is not None
                else 0.0
            ),
        }
    }


if __name__ == "__main__":
    main()
