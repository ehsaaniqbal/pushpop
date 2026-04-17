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
    evaluate_model_with_intervention,
    rollout_supervised_examples,
    rollout_target_for_example,
    run_site_patch_panel,
    select_matched_suffix_pairs,
    summarize_metric_deltas,
)
from pushpop.pp0_model import ResidualIntervention, TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_supervised_examples_from_rows,
    capture_program_position_attention_head_outputs,
    load_probe_rows,
    program_position_available_indices,
)
from pushpop.pp0_training import (
    choose_device,
    collate_supervised_examples,
    evaluate_model,
    load_checkpoint,
)

OPS = {"ADD", "SUB", "SWAP", "POP", "DUP"}
CATEGORY_ORDER = ("ADD", "SUB", "SWAP", "POP", "DUP", "DIGIT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze winner head-7 effects by previous-token category."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--block", type=str, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--pair-target", choices=["top", "depth", "stack"], default="top")
    parser.add_argument("--max-pairs-per-bucket", type=int, default=256)
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
        raise ValueError("position must be at least 1 so a previous token exists")

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

    ablation_results = run_ablation_by_prev_token_category(
        model,
        position_rows=position_rows,
        position_examples=position_examples,
        eligible_local_indices=eligible_local_indices,
        site_name=site_name,
        position_index=args.position,
        batch_size=args.batch_size,
        device=device,
    )

    all_pairs = select_matched_suffix_pairs(
        position_rows,
        position_index=args.position,
        eligible_local_indices=eligible_local_indices,
        pair_target=args.pair_target,
        max_pairs=None,
        seed=args.seed,
    )
    patch_results = run_patch_by_source_prev_token_category(
        model,
        position_rows=position_rows,
        position_examples=position_examples,
        all_pairs=all_pairs,
        site_name=site_name,
        source_site_features=head_features[site_name],
        position_index=args.position,
        device=device,
        max_pairs_per_bucket=args.max_pairs_per_bucket,
    )

    output = {
        "analysis": "pp0_head_prev_token_type_analysis_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "position_index": args.position,
        "previous_position": args.position - 1,
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
        "ablation_by_previous_token_category": ablation_results,
        "patch_by_source_previous_token_category": patch_results,
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def previous_token_category(token: str) -> str:
    if token in OPS:
        return token
    if token.isdigit():
        return "DIGIT"
    return token


def category_sort_key(category: str) -> tuple[int, str]:
    if category in CATEGORY_ORDER:
        return (CATEGORY_ORDER.index(category), category)
    return (len(CATEGORY_ORDER), category)


def exact_prev_token(row: dict[str, object], position_index: int) -> str:
    return str(row["tokens"][position_index - 1])


def run_ablation_by_prev_token_category(
    model: TinyTransformer,
    *,
    position_rows: list[dict[str, object]],
    position_examples: list[object],
    eligible_local_indices: list[int],
    site_name: str,
    position_index: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    zero_intervention = ResidualIntervention(
        layer_name=site_name,
        position_index=position_index,
        replacement_vector=torch.zeros(model.config.d_model),
    )
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for local_index in eligible_local_indices:
        prev_token = exact_prev_token(position_rows[local_index], position_index)
        grouped_indices[previous_token_category(prev_token)].append(local_index)

    category_results: dict[str, object] = {}
    for category in sorted(grouped_indices, key=category_sort_key):
        local_indices = grouped_indices[category]
        subset_rows = [position_rows[index] for index in local_indices]
        subset_examples = [position_examples[index] for index in local_indices]
        subset_dataloader = DataLoader(
            subset_examples,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_supervised_examples,
        )
        baseline_metrics = evaluate_model(model, subset_dataloader, device, compute_slices=False)
        intervention_metrics = evaluate_model_with_intervention(
            model,
            subset_dataloader,
            device,
            interventions=[zero_intervention],
        )
        category_results[category] = {
            "example_count": len(local_indices),
            "exact_previous_token_counts": dict(
                sorted(
                    Counter(exact_prev_token(row, position_index) for row in subset_rows).items()
                )
            ),
            "baseline_metrics": baseline_metrics,
            "intervention_metrics": intervention_metrics,
            "deltas": summarize_metric_deltas(baseline_metrics, intervention_metrics),
        }

    ranked = sorted(
        (
            (category, result)
            for category, result in category_results.items()
        ),
        key=lambda item: (
            -float(-item[1]["deltas"].get("top_accuracy_delta", 0.0)),
            -float(-item[1]["deltas"].get("exact_match_delta", 0.0)),
        ),
    )
    summary = {
        "largest_top_drop": (
            {
                "category": ranked[0][0],
                **ranked[0][1]["deltas"],
                "example_count": int(ranked[0][1]["example_count"]),
            }
            if ranked
            else None
        )
    }
    return {
        "categories": category_results,
        "summary": summary,
    }


def run_patch_by_source_prev_token_category(
    model: TinyTransformer,
    *,
    position_rows: list[dict[str, object]],
    position_examples: list[object],
    all_pairs: list[object],
    site_name: str,
    source_site_features: torch.Tensor,
    position_index: int,
    device: torch.device,
    max_pairs_per_bucket: int,
) -> dict[str, object]:
    bucketed_pairs: dict[str, list[object]] = defaultdict(list)
    for pair in all_pairs:
        source_prev_token = exact_prev_token(position_rows[pair.source_local_index], position_index)
        target_prev_token = exact_prev_token(position_rows[pair.target_local_index], position_index)
        if source_prev_token == target_prev_token:
            continue
        bucketed_pairs[previous_token_category(source_prev_token)].append(pair)

    category_results: dict[str, object] = {}
    for category in sorted(bucketed_pairs, key=category_sort_key):
        candidate_pairs = bucketed_pairs[category]
        selected_pairs = candidate_pairs[:max_pairs_per_bucket]
        category_results[category] = {
            "candidate_pair_count": len(candidate_pairs),
            "selected_pair_count": len(selected_pairs),
            "source_exact_previous_token_counts": dict(
                sorted(
                    Counter(
                        exact_prev_token(position_rows[pair.source_local_index], position_index)
                        for pair in candidate_pairs
                    ).items()
                )
            ),
            "target_previous_token_category_counts": dict(
                sorted(
                    Counter(
                        previous_token_category(
                            exact_prev_token(position_rows[pair.target_local_index], position_index)
                        )
                        for pair in candidate_pairs
                    ).items(),
                    key=lambda item: category_sort_key(item[0]),
                )
            ),
            "patch_metrics": run_site_patch_panel(
                model,
                position_examples,
                selected_pairs,
                source_site_features=source_site_features,
                site_name=site_name,
                position_index=position_index,
                device=device,
            ),
        }

    ranked = sorted(
        (
            (category, result)
            for category, result in category_results.items()
        ),
        key=lambda item: (
            float(item[1]["patch_metrics"]["patched_source_top_rate"]),
            -float(item[1]["patch_metrics"]["patched_target_top_rate"]),
        ),
        reverse=True,
    )
    summary = {
        "best_source_top_transfer": (
            {
                "category": ranked[0][0],
                **ranked[0][1]["patch_metrics"],
            }
            if ranked
            else None
        )
    }
    return {
        "categories": category_results,
        "summary": summary,
    }


if __name__ == "__main__":
    main()
