from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pushpop.pp0_model import TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_probe_splits,
    capture_end_hidden_states,
    collect_end_state_target_labels,
    hidden_state_names,
    run_end_state_probe_suite,
)
from pushpop.pp0_training import (
    PP0SupervisedDataset,
    choose_device,
    collate_supervised_examples,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple END-state linear probes on a fixed PP0 corpus."
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

    dataset = PP0SupervisedDataset(
        args.data_path,
        sanity_checks=not args.no_sanity_checks,
    )
    if dataset.max_sequence_length > model_config.context_length:
        raise ValueError(
            f"dataset sequence length {dataset.max_sequence_length} exceeds model context length "
            f"{model_config.context_length}"
        )

    available_example_count = len(dataset)
    if args.max_examples is None:
        used_example_count = available_example_count
    else:
        if args.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided")
        used_example_count = min(args.max_examples, available_example_count)

    split_indices = build_probe_splits(
        used_example_count,
        train_size=args.probe_train_size,
        val_size=args.probe_val_size,
        test_size=args.probe_test_size,
    )

    selected_indices = list(range(args.probe_train_size + args.probe_val_size + args.probe_test_size))
    selected_examples = [dataset[index] for index in selected_indices]
    dataloader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_supervised_examples,
    )

    layer_features = capture_end_hidden_states(model, dataloader, device=device)
    target_labels = collect_end_state_target_labels(
        selected_examples,
        slot_count=args.slot_count,
    )
    targets = run_end_state_probe_suite(
        layer_features,
        target_labels,
        split_indices,
        ridge_lambdas=args.ridge_lambdas,
        shuffle_seed=args.seed,
    )

    output = {
        "analysis": "pp0_end_state_linear_probe_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(selected_indices),
        "split_sizes": {
            "train": args.probe_train_size,
            "val": args.probe_val_size,
            "test": args.probe_test_size,
        },
        "slot_definition": "slot_k is the value k below the top of the stack at END; EMPTY if absent",
        "model_config": checkpoint["model_config"],
        "layer_names": list(hidden_state_names(model)),
        "targets": targets,
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
