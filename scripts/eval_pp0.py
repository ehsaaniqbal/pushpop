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
from pushpop.pp0_training import (
    PP0SupervisedDataset,
    choose_device,
    collate_supervised_examples,
    evaluate_model,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a tiny PP0 transformer checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
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
        args.data_dir / f"{args.split}.jsonl",
        sanity_checks=not args.no_sanity_checks,
    )
    if dataset.max_sequence_length > model_config.context_length:
        raise ValueError(
            f"dataset sequence length {dataset.max_sequence_length} exceeds model context length "
            f"{model_config.context_length}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_supervised_examples,
    )
    metrics = evaluate_model(model, dataloader, device, compute_slices=True)
    metrics["split"] = args.split
    metrics["checkpoint"] = str(args.checkpoint)

    output = json.dumps(metrics, indent=2, sort_keys=True)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
