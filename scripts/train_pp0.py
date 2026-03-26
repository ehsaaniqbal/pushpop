from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
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
    masked_cross_entropy,
    save_checkpoint,
    set_random_seed,
)
from pushpop.pp0_vocab import VOCAB_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny PP0 transformer.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-mlp", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-sanity-checks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    device = choose_device(args.device)

    train_dataset = PP0SupervisedDataset(
        args.data_dir / "train.jsonl",
        sanity_checks=not args.no_sanity_checks,
    )
    val_dataset = PP0SupervisedDataset(
        args.data_dir / "val.jsonl",
        sanity_checks=not args.no_sanity_checks,
    )

    max_sequence_length = max(train_dataset.max_sequence_length, val_dataset.max_sequence_length)
    if max_sequence_length > args.context_length:
        raise ValueError(
            f"dataset sequence length {max_sequence_length} exceeds context length {args.context_length}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_supervised_examples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_supervised_examples,
    )

    model_config = TinyTransformerConfig(
        vocab_size=len(VOCAB_TOKENS),
        context_length=args.context_length,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    )
    model = TinyTransformer(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "device": str(device),
        "context_length": args.context_length,
        "d_model": args.d_model,
        "d_mlp": args.d_mlp,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_sequence_length": max_sequence_length,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best_exact_match = float("-inf")
    metrics_path = args.output_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss_numerator = 0.0
        running_token_total = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = masked_cross_entropy(logits, target_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            token_count = int(loss_mask.sum().item())
            running_loss_numerator += float(loss.item()) * token_count
            running_token_total += token_count

            if step % args.log_every == 0 or step == len(train_loader):
                running_loss = running_loss_numerator / running_token_total
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} train_loss={running_loss:.4f}"
                )

        train_loss = running_loss_numerator / running_token_total
        val_metrics = evaluate_model(model, val_loader, device, compute_slices=False)
        record = {
            "epoch": epoch,
            "train": {"loss": train_loss},
            "val": val_metrics["overall"],
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['overall']['loss']:.4f} "
            f"val_exact={val_metrics['overall']['exact_match']:.4f} "
            f"val_top={val_metrics['overall']['top_accuracy']:.4f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            model_config=model_config.to_dict(),
            train_config=run_config,
            epoch=epoch,
            metrics=record,
        )

        if float(val_metrics["overall"]["exact_match"]) >= best_exact_match:
            best_exact_match = float(val_metrics["overall"]["exact_match"])
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                model_config=model_config.to_dict(),
                train_config=run_config,
                epoch=epoch,
                metrics=record,
            )


if __name__ == "__main__":
    main()
