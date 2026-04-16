from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_probe_splits,
    build_supervised_examples_from_rows,
    capture_program_position_hidden_states,
    collect_program_position_metadata,
    collect_program_position_target_labels,
    load_probe_rows,
    program_position_available_indices,
    restrict_split_indices,
    run_probe_suite,
    summarize_best_probe_results,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run linear probes across PP0 program positions."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--probe-train-size", type=int, default=8000)
    parser.add_argument("--probe-val-size", type=int, default=1000)
    parser.add_argument("--probe-test-size", type=int, default=1000)
    parser.add_argument("--slot-count", type=int, default=3)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-program-position", type=int, default=None)
    parser.add_argument(
        "--ridge-lambdas",
        type=float,
        nargs="+",
        default=[1e-6, 1e-4, 1e-2, 1.0, 100.0],
    )
    parser.add_argument("--seed", type=int, default=0)
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
    selected_rows = rows[:used_example_count]
    selected_examples = build_supervised_examples_from_rows(selected_rows)

    max_sequence_length = max(example.sequence_length for example in selected_examples)
    if max_sequence_length > model_config.context_length:
        raise ValueError(
            f"dataset sequence length {max_sequence_length} exceeds model context length "
            f"{model_config.context_length}"
        )

    dataloader = DataLoader(
        selected_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_supervised_examples,
    )

    max_position_index = max(len(row["tokens"]) for row in selected_rows) - 1
    if args.max_program_position is not None:
        if args.max_program_position < 0:
            raise ValueError("max_program_position must be non-negative")
        max_position_index = min(max_position_index, args.max_program_position)

    results_by_position: dict[str, dict[str, object]] = {}
    for position_index in range(max_position_index + 1):
        available_indices = program_position_available_indices(
            selected_rows,
            position_index=position_index,
        )
        position_split_indices = restrict_split_indices(available_indices, split_indices)
        if not all(position_split_indices[split_name] for split_name in ("train", "val", "test")):
            raise ValueError(
                f"position {position_index} does not have examples in every probe split"
            )

        position_rows = [selected_rows[index] for index in available_indices]
        layer_features = capture_program_position_hidden_states(
            model,
            dataloader,
            position_index=position_index,
            device=device,
        )
        targets = run_probe_suite(
            layer_features,
            collect_program_position_target_labels(
                position_rows,
                position_index=position_index,
                slot_count=args.slot_count,
            ),
            position_split_indices,
            ridge_lambdas=args.ridge_lambdas,
            shuffle_seed=args.seed,
        )
        results_by_position[f"pc_{position_index}"] = {
            "position_index": position_index,
            "example_count": len(available_indices),
            "split_example_counts": {
                split_name: len(position_split_indices[split_name])
                for split_name in ("train", "val", "test")
            },
            "metadata": collect_program_position_metadata(
                selected_rows,
                position_index=position_index,
            ),
            "targets": targets,
            "best_layers": summarize_best_layers(targets),
        }

    output = {
        "analysis": "pp0_program_position_linear_probe_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": used_example_count,
        "split_sizes": {
            "train": args.probe_train_size,
            "val": args.probe_val_size,
            "test": args.probe_test_size,
        },
        "position_definition": (
            "pc_k means the hidden state at program token index k (0-based), aligned with "
            "the true stack_after state after executing token k"
        ),
        "slot_definition": "slot_k is the value k below the top at that program position; EMPTY if absent",
        "model_config": checkpoint["model_config"],
        "max_program_position": max_position_index,
        "positions": results_by_position,
        "summary": {
            "best_by_target": summarize_best_probe_results(results_by_position),
        },
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def summarize_best_layers(targets: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    best_layers: dict[str, dict[str, object]] = {}
    for target_name, target_result in targets.items():
        best_layer_name, best_layer_metrics = max(
            target_result["layers"].items(),
            key=lambda item: item[1]["probe_test_accuracy"],
        )
        best_layers[target_name] = {
            "layer": best_layer_name,
            "probe_test_accuracy": best_layer_metrics["probe_test_accuracy"],
            "majority_test_accuracy": target_result["majority_test_accuracy"],
            "chance_test_accuracy": target_result["chance_test_accuracy"],
            "is_constant_label": bool(target_result["is_constant_label"]),
        }
    return best_layers


if __name__ == "__main__":
    main()
