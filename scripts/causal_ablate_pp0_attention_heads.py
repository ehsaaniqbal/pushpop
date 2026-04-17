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
    build_zero_ablation_intervention,
    evaluate_model_with_intervention,
    summarize_metric_deltas,
)
from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_supervised_examples_from_rows,
    capture_program_position_attention_head_outputs,
    load_probe_rows,
    program_position_available_indices,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, evaluate_model, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted attention-head ablations on PP0 program positions."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=str, nargs="+", default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--ablation-mode", choices=["zero", "mean"], default="zero")
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
        head_features = capture_program_position_attention_head_outputs(
            model,
            position_dataloader,
            position_index=position_index,
            device=device,
        )

        block_results: dict[str, dict[str, object]] = {}
        for block_name in selected_blocks:
            head_results: dict[str, object] = {}
            for head_index in selected_heads:
                site_name = f"{block_name}.attn_head_{head_index}"
                if args.ablation_mode == "zero":
                    intervention = build_zero_ablation_intervention(
                        head_features,
                        layer_name=site_name,
                        position_index=position_index,
                    )
                else:
                    intervention = build_mean_ablation_intervention(
                        head_features,
                        layer_name=site_name,
                        position_index=position_index,
                    )

                intervention_metrics = evaluate_model_with_intervention(
                    model,
                    position_dataloader,
                    device,
                    interventions=[intervention],
                )
                deltas = summarize_metric_deltas(baseline_metrics, intervention_metrics)
                head_results[f"head_{head_index}"] = {
                    "replacement_norm": float(intervention.replacement_vector.norm().item()),
                    "metrics": intervention_metrics,
                    "deltas": deltas,
                    "specificity": summarize_top_specificity(deltas),
                }
            block_results[block_name] = {
                "heads": head_results,
                "summary": summarize_head_ablation_block(head_results),
            }

        results_by_position[f"pc_{position_index}"] = {
            "position_index": position_index,
            "example_count": len(position_rows),
            "affected_fraction": float(len(position_rows) / len(rows)),
            "baseline_metrics": baseline_metrics,
            "ablation_mode": args.ablation_mode,
            "head_indices": list(selected_heads),
            "blocks": block_results,
            "summary": summarize_head_ablation_position(block_results),
        }

    output = {
        "analysis": "pp0_attention_head_ablation_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "positions": results_by_position,
        "blocks": selected_blocks,
        "head_indices": list(selected_heads),
        "ablation_mode": args.ablation_mode,
        "full_baseline_metrics": full_baseline,
        "intervention_definition": (
            "Replace one attention-head residual contribution at (block, head, pc_k) with either "
            "zero or the mean vector from the evaluated subset, then rerun rollout evaluation."
        ),
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def summarize_top_specificity(deltas: dict[str, float]) -> dict[str, float]:
    top_drop = max(0.0, -deltas.get("top_accuracy_delta", 0.0))
    stop_drop = max(0.0, -deltas.get("stop_accuracy_delta", 0.0))
    token_drop = max(0.0, -deltas.get("token_accuracy_delta", 0.0))
    exact_drop = max(0.0, -deltas.get("exact_match_delta", 0.0))
    return {
        "top_drop": top_drop,
        "stop_drop": stop_drop,
        "token_drop": token_drop,
        "exact_drop": exact_drop,
        "top_minus_stop_drop": top_drop - stop_drop,
        "top_minus_token_drop": top_drop - token_drop,
        "top_minus_exact_drop": top_drop - exact_drop,
    }


def summarize_head_ablation_block(head_results: dict[str, object]) -> dict[str, object]:
    ranked_heads = sorted(
        head_results.items(),
        key=lambda item: (
            item[1]["specificity"]["top_drop"],
            item[1]["specificity"]["top_minus_token_drop"],
        ),
        reverse=True,
    )
    best_head_name, best_head_result = ranked_heads[0]
    return {
        "largest_top_drop": {
            "head": best_head_name,
            **best_head_result["deltas"],
            **best_head_result["specificity"],
        },
        "top_heads_by_top_drop": [
            {
                "head": head_name,
                **head_result["deltas"],
                **head_result["specificity"],
            }
            for head_name, head_result in ranked_heads[:3]
        ],
    }


def summarize_head_ablation_position(
    block_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    overall_best_block_name: str | None = None
    overall_best_head_name: str | None = None
    overall_best_result: dict[str, object] | None = None

    for block_name, block_result in block_results.items():
        for head_name, head_result in block_result["heads"].items():
            if overall_best_result is None or (
                head_result["specificity"]["top_drop"] > overall_best_result["specificity"]["top_drop"]
            ):
                overall_best_block_name = block_name
                overall_best_head_name = head_name
                overall_best_result = head_result

    return {
        "largest_top_drop": {
            "block": overall_best_block_name,
            "head": overall_best_head_name,
            **(overall_best_result["deltas"] if overall_best_result is not None else {}),
            **(overall_best_result["specificity"] if overall_best_result is not None else {}),
        }
    }


if __name__ == "__main__":
    main()
