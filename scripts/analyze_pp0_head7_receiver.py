from __future__ import annotations

import argparse
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
    run_pair_intervention_panel,
    select_matched_suffix_pairs,
)
from pushpop.pp0_model import ResidualIntervention, TinyTransformer, TinyTransformerConfig
from pushpop.pp0_probe import (
    build_probe_splits,
    build_supervised_examples_from_rows,
    capture_program_position_attention_head_outputs,
    capture_program_position_hidden_states,
    collect_program_position_target_labels,
    fit_ridge_probe_model,
    hidden_state_names,
    load_probe_rows,
)
from pushpop.pp0_training import choose_device, collate_supervised_examples, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a downstream receiver candidate for winner head-7 and run a small mediation test."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-position", type=int, default=15)
    parser.add_argument("--positions", type=int, nargs="+", default=[15, 16, 17])
    parser.add_argument("--block", type=str, default="block_0")
    parser.add_argument("--head", type=int, default=7)
    parser.add_argument("--pair-target", choices=["top", "depth", "stack"], default="top")
    parser.add_argument("--max-pairs", type=int, default=256)
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
    if not args.positions:
        raise ValueError("positions must not be empty")
    if args.query_position not in args.positions:
        raise ValueError("query-position must be included in positions")

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

    common_rows = [
        row
        for row in rows
        if all(position_index < len(row["tokens"]) for position_index in args.positions)
    ]
    if not common_rows:
        raise ValueError("no examples reach all requested positions")
    common_examples = build_supervised_examples_from_rows(common_rows)
    common_dataloader = DataLoader(
        common_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_supervised_examples,
    )

    source_site_name = f"{args.block}.attn_head_{args.head}"
    source_site_features = capture_program_position_attention_head_outputs(
        model,
        common_dataloader,
        position_index=args.query_position,
        device=device,
    )[source_site_name]
    head7_zero_ablation = ResidualIntervention(
        layer_name=source_site_name,
        position_index=args.query_position,
        replacement_vector=torch.zeros(model.config.d_model),
    )

    candidate_sites = build_candidate_sites(model, query_position=args.query_position, positions=args.positions)
    receiver_search_results, candidate_receiver = run_receiver_search(
        model,
        common_rows=common_rows,
        common_dataloader=common_dataloader,
        candidate_sites=candidate_sites,
        positions=args.positions,
        ablation=head7_zero_ablation,
        device=device,
    )

    mediation_results = run_mediation_test(
        model,
        common_rows=common_rows,
        common_examples=common_examples,
        common_dataloader=common_dataloader,
        query_position=args.query_position,
        pair_target=args.pair_target,
        source_site_name=source_site_name,
        source_site_features=source_site_features,
        receiver_site=candidate_receiver,
        device=device,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        seed=args.seed,
        use_all_examples=args.use_all_examples,
    )

    output = {
        "analysis": "pp0_head_receiver_analysis_v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_path": str(args.data_path),
        "available_example_count": available_example_count,
        "used_example_count": len(rows),
        "common_subset_example_count": len(common_rows),
        "positions": list(args.positions),
        "query_position": args.query_position,
        "source_site_name": source_site_name,
        "eligible_subset_definition": (
            "all examples reaching every searched position"
            if args.use_all_examples
            else "examples reaching every searched position with exact-correct baseline rollout"
        ),
        "receiver_search": receiver_search_results,
        "candidate_receiver": candidate_receiver,
        "mediation": mediation_results,
    }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


def build_candidate_sites(
    model: TinyTransformer,
    *,
    query_position: int,
    positions: list[int],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    layer_names = hidden_state_names(model)
    for position_index in positions:
        for layer_name in layer_names:
            if layer_name == "embed":
                continue
            if position_index == query_position and layer_name == "block_0":
                continue
            candidates.append(
                {
                    "position_index": position_index,
                    "layer_name": layer_name,
                }
            )
    return candidates


def run_receiver_search(
    model: TinyTransformer,
    *,
    common_rows: list[dict[str, object]],
    common_dataloader: DataLoader,
    candidate_sites: list[dict[str, object]],
    positions: list[int],
    ablation: ResidualIntervention,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    if len(common_rows) < 5:
        raise ValueError("receiver search requires at least 5 common examples")

    split_indices = build_ratio_split_indices(len(common_rows))
    ridge_lambdas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

    baseline_features_by_position = {
        position_index: capture_program_position_hidden_states(
            model,
            common_dataloader,
            position_index=position_index,
            device=device,
        )
        for position_index in positions
    }
    ablated_features_by_position = {
        position_index: capture_program_position_hidden_states(
            model,
            common_dataloader,
            position_index=position_index,
            device=device,
            interventions=[ablation],
        )
        for position_index in positions
    }

    site_results: list[dict[str, object]] = []
    for candidate_site in candidate_sites:
        position_index = int(candidate_site["position_index"])
        layer_name = str(candidate_site["layer_name"])
        labels = collect_program_position_target_labels(
            common_rows,
            position_index=position_index,
        )["top"]
        label_vocab = sorted(set(labels))
        if len(label_vocab) <= 1:
            site_results.append(
                {
                    "position_index": position_index,
                    "layer_name": layer_name,
                    "baseline_probe_train_accuracy": 1.0,
                    "baseline_probe_val_accuracy": 1.0,
                    "baseline_probe_test_accuracy": 1.0,
                    "ablated_probe_test_accuracy": 1.0,
                    "readout_drop": 0.0,
                    "best_ridge_lambda": None,
                    "is_constant_label": True,
                    "label_vocab": label_vocab,
                }
            )
            continue
        fitted_probe = fit_ridge_probe_model(
            baseline_features_by_position[position_index][layer_name],
            labels,
            split_indices,
            ridge_lambdas=ridge_lambdas,
        )
        ablated_test_accuracy = probe_test_accuracy(
            fitted_probe,
            ablated_features_by_position[position_index][layer_name],
            labels,
            split_indices,
        )
        readout_drop = fitted_probe.test_accuracy - ablated_test_accuracy
        site_results.append(
            {
                "position_index": position_index,
                "layer_name": layer_name,
                "baseline_probe_train_accuracy": fitted_probe.train_accuracy,
                "baseline_probe_val_accuracy": fitted_probe.val_accuracy,
                "baseline_probe_test_accuracy": fitted_probe.test_accuracy,
                "ablated_probe_test_accuracy": ablated_test_accuracy,
                "readout_drop": readout_drop,
                "best_ridge_lambda": fitted_probe.ridge_lambda,
                "is_constant_label": False,
                "label_vocab": list(fitted_probe.label_vocab),
            }
        )

    ranked_sites = sorted(
        site_results,
        key=lambda result: (
            not bool(result["is_constant_label"]),
            float(result["readout_drop"]),
            float(result["baseline_probe_test_accuracy"]),
        ),
        reverse=True,
    )
    max_readout_candidate = ranked_sites[0]
    candidate_receiver = select_mediation_receiver(ranked_sites)
    return (
        {
            "candidate_sites": ranked_sites,
            "split_sizes": {split_name: len(indices) for split_name, indices in split_indices.items()},
            "summary": {
                "largest_readout_drop": max_readout_candidate,
                "mediation_receiver_candidate": candidate_receiver,
            },
        },
        candidate_receiver,
    )


def probe_test_accuracy(
    probe: object,
    features: torch.Tensor,
    labels: list[str],
    split_indices: dict[str, list[int]],
) -> float:
    label_to_index = {label: index for index, label in enumerate(probe.label_vocab)}
    encoded_labels = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
    test_indices = torch.tensor(split_indices["test"], dtype=torch.long)
    test_features = features.index_select(0, test_indices).to(dtype=torch.float64)
    standardized = (test_features - probe.feature_mean.to(dtype=torch.float64)) / probe.feature_std.to(
        dtype=torch.float64
    )
    prepared = torch.cat(
        [standardized, torch.ones((standardized.shape[0], 1), dtype=standardized.dtype)],
        dim=1,
    )
    logits = prepared @ probe.weights.to(dtype=torch.float64)
    predictions = logits.argmax(dim=1).cpu()
    test_labels = encoded_labels.index_select(0, test_indices)
    return float((predictions == test_labels).float().mean().item())


def select_mediation_receiver(ranked_sites: list[dict[str, object]]) -> dict[str, object]:
    non_constant_sites = [site for site in ranked_sites if not bool(site["is_constant_label"])]
    if not non_constant_sites:
        raise ValueError("receiver search produced no non-constant candidate sites")

    max_drop = float(non_constant_sites[0]["readout_drop"])
    near_best_sites = [
        site
        for site in non_constant_sites
        if max_drop - float(site["readout_drop"]) <= 0.01
    ]
    return min(near_best_sites, key=mediation_receiver_sort_key)


def mediation_receiver_sort_key(site: dict[str, object]) -> tuple[int, int]:
    layer_order = {
        "block_0": 0,
        "block_1": 1,
        "block_2": 2,
        "block_3": 3,
        "final_ln": 4,
    }
    return (
        int(site["position_index"]),
        layer_order.get(str(site["layer_name"]), 99),
    )


def build_ratio_split_indices(total_examples: int) -> dict[str, list[int]]:
    train_size = int(total_examples * 0.6)
    val_size = int(total_examples * 0.2)
    test_size = total_examples - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("receiver search split sizes must all be positive")
    return build_probe_splits(
        total_examples,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )


def run_mediation_test(
    model: TinyTransformer,
    *,
    common_rows: list[dict[str, object]],
    common_examples: list[object],
    common_dataloader: DataLoader,
    query_position: int,
    pair_target: str,
    source_site_name: str,
    source_site_features: torch.Tensor,
    receiver_site: dict[str, object],
    device: torch.device,
    batch_size: int,
    max_pairs: int,
    seed: int,
    use_all_examples: bool,
) -> dict[str, object]:
    baseline_predictions = rollout_supervised_examples(
        model,
        common_examples,
        device,
        batch_size=batch_size,
    )
    baseline_targets = [rollout_target_for_example(example) for example in common_examples]
    exact_correct_local_indices = [
        local_index
        for local_index, (prediction, target) in enumerate(
            zip(baseline_predictions, baseline_targets, strict=True)
        )
        if prediction == target
    ]
    eligible_local_indices = (
        list(range(len(common_rows))) if use_all_examples else exact_correct_local_indices
    )
    all_pairs = select_matched_suffix_pairs(
        common_rows,
        position_index=query_position,
        eligible_local_indices=eligible_local_indices,
        pair_target=pair_target,
        max_pairs=None,
        seed=seed,
    )
    selected_pairs = all_pairs[:max_pairs]
    if not selected_pairs:
        raise ValueError("no matched-suffix pairs available for mediation test")

    receiver_position = int(receiver_site["position_index"])
    receiver_layer_name = str(receiver_site["layer_name"])
    receiver_features = capture_program_position_hidden_states(
        model,
        common_dataloader,
        position_index=receiver_position,
        device=device,
    )[receiver_layer_name]
    receiver_mean = receiver_features.mean(dim=0).cpu()

    patch_only = run_pair_intervention_panel(
        model,
        common_examples,
        selected_pairs,
        intervention_builder=lambda pair: [
            ResidualIntervention(
                layer_name=source_site_name,
                position_index=query_position,
                replacement_vector=source_site_features[pair.source_local_index].clone(),
            )
        ],
        device=device,
    )
    receiver_mean_ablation_only = run_pair_intervention_panel(
        model,
        common_examples,
        selected_pairs,
        intervention_builder=lambda pair: [
            ResidualIntervention(
                layer_name=receiver_layer_name,
                position_index=receiver_position,
                replacement_vector=receiver_mean,
            )
        ],
        device=device,
    )
    patch_plus_receiver_mean_ablation = run_pair_intervention_panel(
        model,
        common_examples,
        selected_pairs,
        intervention_builder=lambda pair: [
            ResidualIntervention(
                layer_name=source_site_name,
                position_index=query_position,
                replacement_vector=source_site_features[pair.source_local_index].clone(),
            ),
            ResidualIntervention(
                layer_name=receiver_layer_name,
                position_index=receiver_position,
                replacement_vector=receiver_mean,
            ),
        ],
        device=device,
    )

    return {
        "eligible_example_count": len(eligible_local_indices),
        "baseline_exact_correct_count": len(exact_correct_local_indices),
        "candidate_pair_count": len(all_pairs),
        "selected_pair_count": len(selected_pairs),
        "receiver_site_name": receiver_layer_name,
        "receiver_position_index": receiver_position,
        "patch_only": patch_only,
        "receiver_mean_ablation_only": receiver_mean_ablation_only,
        "patch_plus_receiver_mean_ablation": patch_plus_receiver_mean_ablation,
    }


if __name__ == "__main__":
    main()
