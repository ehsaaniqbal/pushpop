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

from pushpop.pp0_causal import rollout_supervised_examples, rollout_target_for_example
from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_supervised_examples_from_rows,
    capture_program_position_attention_patterns,
    load_probe_rows,
    program_position_available_indices,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect winner attention patterns for a single PP0 attention head."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--block", type=str, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-sanity-checks", action="store_true")
    parser.add_argument("--use-all-examples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.position < 0:
        raise ValueError("position must be non-negative")

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
    attention_patterns = capture_program_position_attention_patterns(
        model,
        position_dataloader,
        position_index=args.position,
        device=device,
    )
    site_name = f"{args.block}.attn_pattern"
    site_patterns = attention_patterns[site_name]

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
    selected_local_indices = (
        list(range(len(position_rows))) if args.use_all_examples else exact_correct_local_indices
    )
    if not selected_local_indices:
        raise ValueError("selected example subset is empty")

    selection_index = torch.tensor(selected_local_indices, dtype=torch.long)
    selected_head_patterns = site_patterns.index_select(0, selection_index)[:, args.head, :]
    selected_rows = [position_rows[index] for index in selected_local_indices]

    output = {
        "analysis": "pp0_attention_head_inspection_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "position_index": args.position,
        "block": args.block,
        "head_index": args.head,
        "subset_definition": (
            "all examples reaching the query position"
            if args.use_all_examples
            else "examples reaching the query position with exact-correct baseline rollout"
        ),
        "position_example_count": len(position_rows),
        "subset_example_count": len(selected_rows),
        "baseline_exact_correct_count": len(exact_correct_local_indices),
        "baseline_exact_correct_fraction": (
            float(len(exact_correct_local_indices) / len(position_rows))
            if position_rows
            else 0.0
        ),
        "query_token_counts": dict(sorted(Counter(row["tokens"][args.position] for row in selected_rows).items())),
        "attention_row_sum_stats": summarize_attention_row_sums(selected_head_patterns),
        "average_attention_by_source_position": summarize_average_attention_by_position(
            selected_head_patterns,
            selected_rows,
            position_index=args.position,
        ),
        "source_token_attention_mass": summarize_source_token_attention_mass(
            selected_head_patterns,
            selected_rows,
            position_index=args.position,
        ),
        "top_position_token_pairs": summarize_position_token_pairs(
            selected_head_patterns,
            selected_rows,
            position_index=args.position,
        ),
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def summarize_attention_row_sums(head_patterns: torch.Tensor) -> dict[str, float]:
    row_sums = head_patterns.sum(dim=1)
    return {
        "mean": float(row_sums.mean().item()),
        "min": float(row_sums.min().item()),
        "max": float(row_sums.max().item()),
    }


def summarize_average_attention_by_position(
    head_patterns: torch.Tensor,
    selected_rows: list[dict[str, object]],
    *,
    position_index: int,
) -> list[dict[str, object]]:
    average_by_position = head_patterns.mean(dim=0)
    token_counter_by_position: dict[int, Counter[str]] = defaultdict(Counter)
    for row in selected_rows:
        for source_position in range(position_index + 1):
            token_counter_by_position[source_position][str(row["tokens"][source_position])] += 1

    return [
        {
            "source_position": source_position,
            "mean_attention": float(average_by_position[source_position].item()),
            "most_common_token": token_counter_by_position[source_position].most_common(1)[0][0],
        }
        for source_position in range(position_index + 1)
    ]


def summarize_source_token_attention_mass(
    head_patterns: torch.Tensor,
    selected_rows: list[dict[str, object]],
    *,
    position_index: int,
) -> list[dict[str, object]]:
    total_mass_by_token: dict[str, float] = defaultdict(float)
    occurrence_count_by_token: Counter[str] = Counter()

    for row_index, row in enumerate(selected_rows):
        for source_position in range(position_index + 1):
            token = str(row["tokens"][source_position])
            mass = float(head_patterns[row_index, source_position].item())
            total_mass_by_token[token] += mass
            occurrence_count_by_token[token] += 1

    result = [
        {
            "token": token,
            "total_attention_mass": total_mass_by_token[token],
            "mean_attention_per_example": total_mass_by_token[token] / len(selected_rows),
            "mean_attention_per_occurrence": total_mass_by_token[token] / occurrence_count_by_token[token],
            "occurrence_count": int(occurrence_count_by_token[token]),
        }
        for token in total_mass_by_token
    ]
    result.sort(key=lambda item: item["total_attention_mass"], reverse=True)
    return result


def summarize_position_token_pairs(
    head_patterns: torch.Tensor,
    selected_rows: list[dict[str, object]],
    *,
    position_index: int,
) -> list[dict[str, object]]:
    total_mass_by_pair: dict[tuple[int, str], float] = defaultdict(float)
    occurrence_count_by_pair: Counter[tuple[int, str]] = Counter()

    for row_index, row in enumerate(selected_rows):
        for source_position in range(position_index + 1):
            token = str(row["tokens"][source_position])
            key = (source_position, token)
            mass = float(head_patterns[row_index, source_position].item())
            total_mass_by_pair[key] += mass
            occurrence_count_by_pair[key] += 1

    top_pairs = sorted(total_mass_by_pair.items(), key=lambda item: item[1], reverse=True)[:10]
    return [
        {
            "source_position": source_position,
            "token": token,
            "total_attention_mass": total_mass,
            "mean_attention_per_example": total_mass / len(selected_rows),
            "mean_attention_per_occurrence": total_mass / occurrence_count_by_pair[(source_position, token)],
            "occurrence_count": int(occurrence_count_by_pair[(source_position, token)]),
        }
        for (source_position, token), total_mass in top_pairs
    ]


if __name__ == "__main__":
    main()
