from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import torch
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
        description="Analyze whether PP0 head-7 patching survives previous-token identity controls."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--block", type=str, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--pair-target", choices=["top", "depth", "stack"], default="top")
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-sanity-checks", action="store_true")
    parser.add_argument("--use-all-examples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.position <= 0:
        raise ValueError("position must be at least 1 so there is a previous position to inspect")

    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    model_config = TinyTransformerConfig.from_dict(checkpoint["model_config"])
    model = TinyTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    available_blocks = [f"block_{index}" for index in range(model.config.n_layers)]
    if args.block not in available_blocks:
        raise ValueError(f"unknown block name: {args.block!r}")
    if not 0 <= args.head < model.config.n_heads:
        raise ValueError(f"unknown head index: {args.head}")

    rows = load_probe_rows(args.data_path, sanity_checks=not args.no_sanity_checks)
    available_example_count = len(rows)
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided")
        rows = rows[: min(args.max_examples, len(rows))]
    examples = build_supervised_examples_from_rows(rows)

    available_indices = program_position_available_indices(rows, position_index=args.position)
    if not available_indices:
        raise ValueError(f"no examples reach program position {args.position}")

    position_rows = [rows[index] for index in available_indices]
    position_examples = [examples[index] for index in available_indices]
    position_dataloader = DataLoader(
        position_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_supervised_examples,
    )
    site_name = f"{args.block}.attn_head_{args.head}"
    head_features = capture_program_position_attention_head_outputs(
        model,
        position_dataloader,
        position_index=args.position,
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
    eligible_local_indices = (
        list(range(len(position_rows))) if args.use_all_examples else exact_correct_local_indices
    )
    all_pairs = select_matched_suffix_pairs(
        position_rows,
        position_index=args.position,
        eligible_local_indices=eligible_local_indices,
        pair_target=args.pair_target,
        max_pairs=None,
        seed=args.seed,
    )
    previous_position = args.position - 1
    bucket_candidates = build_pair_buckets(
        all_pairs,
        position_rows=position_rows,
        previous_position=previous_position,
    )

    bucket_results: dict[str, dict[str, object]] = {}
    for bucket_name, candidate_pairs in bucket_candidates.items():
        selected_pairs = candidate_pairs[: args.max_pairs]
        bucket_results[bucket_name] = {
            "candidate_pair_count": len(candidate_pairs),
            "selected_pair_count": len(selected_pairs),
            "previous_position_stats": summarize_previous_position_stats(
                candidate_pairs,
                position_rows=position_rows,
                previous_position=previous_position,
            ),
            "patch_metrics": run_site_patch_panel(
                model,
                position_examples,
                selected_pairs,
                source_site_features=head_features[site_name],
                site_name=site_name,
                position_index=args.position,
                device=device,
            ),
            "sample_pairs": summarize_sample_pairs(
                selected_pairs,
                position_rows=position_rows,
                previous_position=previous_position,
            ),
        }

    output = {
        "analysis": "pp0_head_prev_position_control_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "position_index": args.position,
        "previous_position": previous_position,
        "block": args.block,
        "head_index": args.head,
        "site_name": site_name,
        "pair_target": args.pair_target,
        "eligible_subset_definition": (
            "all examples reaching the query position"
            if args.use_all_examples
            else "examples reaching the query position with exact-correct baseline rollout"
        ),
        "position_example_count": len(position_rows),
        "eligible_example_count": len(eligible_local_indices),
        "baseline_exact_correct_count": len(exact_correct_local_indices),
        "baseline_exact_correct_fraction": (
            float(len(exact_correct_local_indices) / len(position_rows))
            if position_rows
            else 0.0
        ),
        "candidate_pair_count": len(all_pairs),
        "bucket_results": bucket_results,
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def build_pair_buckets(
    candidate_pairs: list[object],
    *,
    position_rows: list[dict[str, object]],
    previous_position: int,
) -> dict[str, list[object]]:
    buckets: dict[str, list[object]] = defaultdict(list)
    for pair in candidate_pairs:
        source_prev_token = str(position_rows[pair.source_local_index]["tokens"][previous_position])
        target_prev_token = str(position_rows[pair.target_local_index]["tokens"][previous_position])
        source_prev_state = stack_state_after_position(
            position_rows[pair.source_local_index],
            previous_position,
        )
        target_prev_state = stack_state_after_position(
            position_rows[pair.target_local_index],
            previous_position,
        )

        buckets["all"].append(pair)
        if source_prev_token == target_prev_token:
            buckets["same_prev_token"].append(pair)
            if source_prev_state["top"] == target_prev_state["top"]:
                buckets["same_prev_token_same_prev_top"].append(pair)
        else:
            buckets["different_prev_token"].append(pair)
    return dict(buckets)


def stack_state_after_position(row: dict[str, object], position_index: int) -> dict[str, object]:
    stack_after = [int(value) for value in row["trace"][position_index]["stack_after"]]
    return {
        "stack": tuple(stack_after),
        "top": stack_after[-1] if stack_after else "EMPTY",
        "depth": len(stack_after),
    }


def summarize_previous_position_stats(
    candidate_pairs: list[object],
    *,
    position_rows: list[dict[str, object]],
    previous_position: int,
) -> dict[str, object]:
    if not candidate_pairs:
        return {
            "same_prev_token_fraction": 0.0,
            "same_prev_top_fraction": 0.0,
            "same_prev_depth_fraction": 0.0,
            "same_prev_stack_fraction": 0.0,
            "source_prev_token_counts": {},
        }

    same_prev_token = 0
    same_prev_top = 0
    same_prev_depth = 0
    same_prev_stack = 0
    source_prev_token_counts: Counter[str] = Counter()

    for pair in candidate_pairs:
        source_prev_token = str(position_rows[pair.source_local_index]["tokens"][previous_position])
        target_prev_token = str(position_rows[pair.target_local_index]["tokens"][previous_position])
        source_prev_state = stack_state_after_position(
            position_rows[pair.source_local_index],
            previous_position,
        )
        target_prev_state = stack_state_after_position(
            position_rows[pair.target_local_index],
            previous_position,
        )

        source_prev_token_counts[source_prev_token] += 1
        same_prev_token += int(source_prev_token == target_prev_token)
        same_prev_top += int(source_prev_state["top"] == target_prev_state["top"])
        same_prev_depth += int(source_prev_state["depth"] == target_prev_state["depth"])
        same_prev_stack += int(source_prev_state["stack"] == target_prev_state["stack"])

    pair_count = len(candidate_pairs)
    return {
        "same_prev_token_fraction": same_prev_token / pair_count,
        "same_prev_top_fraction": same_prev_top / pair_count,
        "same_prev_depth_fraction": same_prev_depth / pair_count,
        "same_prev_stack_fraction": same_prev_stack / pair_count,
        "source_prev_token_counts": dict(sorted(source_prev_token_counts.items())),
    }


def summarize_sample_pairs(
    selected_pairs: list[object],
    *,
    position_rows: list[dict[str, object]],
    previous_position: int,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for pair in selected_pairs[:5]:
        source_row = position_rows[pair.source_local_index]
        target_row = position_rows[pair.target_local_index]
        source_prev_state = stack_state_after_position(source_row, previous_position)
        target_prev_state = stack_state_after_position(target_row, previous_position)
        samples.append(
            {
                "source_example_id": source_row["example_id"],
                "target_example_id": target_row["example_id"],
                "source_prev_token": source_row["tokens"][previous_position],
                "target_prev_token": target_row["tokens"][previous_position],
                "source_prev_top": source_prev_state["top"],
                "target_prev_top": target_prev_state["top"],
                "source_prev_depth": source_prev_state["depth"],
                "target_prev_depth": target_prev_state["depth"],
                "source_current_top": pair.source_top,
                "target_current_top": pair.target_top,
            }
        )
    return samples


if __name__ == "__main__":
    main()
