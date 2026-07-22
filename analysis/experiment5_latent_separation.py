"""Layer-wise Success-Failure latent separation at first-error positions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.common import discover_layers, load_sae, resolve_device
from analysis.experiment5_retrieval_success_failure import (
    build_activation_metadata,
    load_json_records,
    record_query_id,
)


GROUPS = ("S1_vs_F1", "S2_vs_F2", "S3_vs_F3", "S4_vs_F4")
RED_COLORS = ("#fcae91", "#fb6a4a", "#cb181d")
BLUE_COLORS = ("#9ecae1", "#4292c6", "#08519c")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/activation_cache"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("out/sae_train_8x"))
    parser.add_argument("--dev-data", type=Path, default=Path("dataset/nq320k/dev.json"))
    parser.add_argument(
        "--semantic-id-path", type=Path,
        default=Path("dataset/nq320k_id/id.semantic.bert.json"),
    )
    parser.add_argument(
        "--outcome-root", type=Path,
        default=Path("dataset/nq320k/generation_outcomes"),
    )
    parser.add_argument(
        "--matched-root", type=Path,
        default=Path("dataset/nq320k/generation_outcomes/matched_success_controls"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/layerwise_success_failure_separation"),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--active-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--sae-format", choices=("auto", "inference", "training"),
        default="inference",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pair_groups(
    outcome_root: Path, matched_root: Path
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], int]]:
    result = {}
    for position in range(1, 5):
        success = load_json_records(matched_root / f"S{position}.json")
        failure = load_json_records(outcome_root / f"failure_{position}.json")
        if len(success) != len(failure):
            raise ValueError(f"S{position} and F{position} sizes differ")
        result[f"S{position}_vs_F{position}"] = (success, failure, position - 1)
    return result


def make_row_weights(
    records: Sequence[dict[str, Any]], query_ids: np.ndarray,
    positions: np.ndarray, position: int,
) -> np.ndarray:
    multiplicity = Counter(record_query_id(row) for row in records)
    weights = np.fromiter(
        (multiplicity.get(int(query_id), 0) for query_id in query_ids),
        dtype=np.float32, count=len(query_ids),
    )
    weights[positions != position] = 0
    return weights


def choose_symmetric_thresholds(scores: np.ndarray) -> tuple[float, float, float, str]:
    absolute = np.abs(scores[np.isfinite(scores)])
    nonzero = absolute[absolute > 0]
    if len(nonzero) == 0:
        return 0.01, 0.02, 0.05, "fallback_no_nonzero_scores"
    if float(np.quantile(nonzero, 0.95)) >= 0.05:
        return 0.05, 0.10, 0.20, "nominal"
    values = np.quantile(nonzero, [0.50, 0.80, 0.95]).astype(float)
    if len(np.unique(values)) < 3 or values[0] <= 0:
        maximum = float(nonzero.max())
        values = np.asarray([maximum * 0.25, maximum * 0.50, maximum * 0.75])
    epsilon = max(float(nonzero.min()) * 0.1, 1e-8)
    values[0] = max(values[0], epsilon)
    values[1] = max(values[1], values[0] + epsilon)
    values[2] = max(values[2], values[1] + epsilon)
    return float(values[0]), float(values[1]), float(values[2]), "data_quantiles_50_80_95"


def bin_definitions(thresholds: tuple[float, float, float]) -> list[dict[str, Any]]:
    a, b, c = thresholds
    return [
        {"name": "success_weak", "side": "success", "strength": 1,
         "label": f"Success ({a:.4g}, {b:.4g}]", "low": a, "high": b},
        {"name": "success_moderate", "side": "success", "strength": 2,
         "label": f"Success ({b:.4g}, {c:.4g}]", "low": b, "high": c},
        {"name": "success_strong", "side": "success", "strength": 3,
         "label": f"Success ({c:.4g}, 1]", "low": c, "high": 1.0},
        {"name": "failure_weak", "side": "failure", "strength": 1,
         "label": f"Failure [-{b:.4g}, -{a:.4g})", "low": a, "high": b},
        {"name": "failure_moderate", "side": "failure", "strength": 2,
         "label": f"Failure [-{c:.4g}, -{b:.4g})", "low": b, "high": c},
        {"name": "failure_strong", "side": "failure", "strength": 3,
         "label": f"Failure [-1, -{c:.4g})", "low": c, "high": 1.0},
    ]


def bin_mask(scores: np.ndarray, definition: dict[str, Any]) -> np.ndarray:
    magnitude = np.abs(scores)
    sign = scores > 0 if definition["side"] == "success" else scores < 0
    if definition["strength"] < 3:
        return sign & (magnitude > definition["low"]) & (magnitude <= definition["high"])
    return sign & (magnitude > definition["low"]) & (magnitude <= 1.0)


def plot_stacked(
    bin_rows: Sequence[dict[str, Any]], layers: Sequence[int], definitions: list[dict[str, Any]],
    output_dir: Path, value_column: str, ylabel: str, stem: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), sharex=True, sharey=True)
    handles = []
    labels = []
    maximum = 0.0
    for axis, group in zip(axes.flat, GROUPS):
        group_rows = [row for row in bin_rows if row["group"] == group]
        x = np.arange(len(layers))
        success_bottom = np.zeros(len(layers))
        failure_bottom = np.zeros(len(layers))
        bar_width = 0.39
        for definition, color in zip(definitions[:3], RED_COLORS):
            values = np.asarray([
                next(row[value_column] for row in group_rows
                     if row["layer"] == layer and row["bin"] == definition["name"])
                for layer in layers
            ], dtype=float)
            bars = axis.bar(
                x - bar_width / 2, values, bottom=success_bottom,
                color=color, width=bar_width,
            )
            success_bottom += values
            if group == GROUPS[0]:
                handles.append(bars[0]); labels.append(definition["label"])
        for definition, color in zip(definitions[3:], BLUE_COLORS):
            values = np.asarray([
                next(row[value_column] for row in group_rows
                     if row["layer"] == layer and row["bin"] == definition["name"])
                for layer in layers
            ], dtype=float)
            bars = axis.bar(
                x + bar_width / 2, values, bottom=failure_bottom,
                color=color, width=bar_width,
            )
            failure_bottom += values
            if group == GROUPS[0]:
                handles.append(bars[0]); labels.append(definition["label"])
        maximum = max(maximum, float(success_bottom.max()), float(failure_bottom.max()))
        position = GROUPS.index(group) + 1
        axis.set_title(f"S{position} vs F{position} — position {position}")
        axis.set_xticks(x, layers, rotation=90)
        axis.tick_params(axis="x", labelbottom=True)
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    limit = max(maximum * 1.08, 1e-9)
    for axis in axes.flat:
        axis.set_ylim(0, limit)
        axis.set_xlabel("Decoder Layer")
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    fig.suptitle("Layer-wise Success–Failure Latent Separation at First Error Position", fontsize=16)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_distributions(
    results: dict[str, dict[int, dict[str, np.ndarray]]], layers: Sequence[int],
    output_dir: Path,
) -> None:
    valid_scores = [
        values["separation"][values["valid"]]
        for group in GROUPS for layer, values in results[group].items()
    ]
    limit = max(float(np.max(np.abs(np.concatenate(valid_scores)))), 1e-6)
    bins = np.linspace(-limit, limit, 101)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, GROUPS):
        scores = np.concatenate([
            results[group][layer]["separation"][results[group][layer]["valid"]]
            for layer in layers
        ])
        axis.hist(scores[np.isfinite(scores)], bins=bins, color="#777777", alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.8)
        position = GROUPS.index(group) + 1
        axis.set_title(f"S{position} vs F{position} — position {position}")
        axis.set_yscale("log")
        axis.set_xlabel("Separation score")
        axis.set_ylabel("Latent-layer count (log)")
    fig.tight_layout()
    fig.savefig(output_dir / "separation_score_distributions.png", dpi=320)
    fig.savefig(output_dir / "separation_score_distributions.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.active_epsilon < 0:
        raise ValueError("batch-size must be positive and active-epsilon non-negative")
    available = discover_layers(args.cache_dir, args.checkpoint_root)
    layers = available if args.layers is None else sorted(set(args.layers))
    if set(layers) - set(available):
        raise FileNotFoundError("Some requested layers have no cache/SAE checkpoint")
    query_ids, positions, _ = build_activation_metadata(args.dev_data, args.semantic_id_path)
    pair_groups = load_pair_groups(args.outcome_root, args.matched_root)
    weights: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for group, (success, failure, position) in pair_groups.items():
        success_weights = make_row_weights(success, query_ids, positions, position)
        failure_weights = make_row_weights(failure, query_ids, positions, position)
        n_pairs = len(failure)
        if int(success_weights.sum()) != n_pairs or int(failure_weights.sum()) != n_pairs:
            raise RuntimeError(f"{group}: activation-row weights do not equal N={n_pairs}")
        weights[group] = (success_weights, failure_weights, n_pairs)

    device = resolve_device(args.device)
    results: dict[str, dict[int, dict[str, np.ndarray]]] = {group: {} for group in GROUPS}
    nonfinite_by_layer: dict[int, int] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer in tqdm(layers, desc="Latent separation", unit="layer"):
        activations = load_file(str(args.cache_dir / f"layer_{layer}.safetensors"))["activations"]
        if len(activations) != len(query_ids):
            raise RuntimeError(f"Layer {layer}: activation metadata length mismatch")
        sae, config, loaded_format = load_sae(
            args.checkpoint_root / f"layer_{layer}", device, args.sae_format
        )
        d_sae = int(config["d_sae"])
        counts = {
            group: [torch.zeros(d_sae, device=device), torch.zeros(d_sae, device=device)]
            for group in GROUPS
        }
        nonfinite = 0
        with torch.inference_mode():
            for start in range(0, len(activations), args.batch_size):
                end = min(start + args.batch_size, len(activations))
                latent = sae.encode(activations[start:end].to(device).float())
                finite = torch.isfinite(latent)
                nonfinite += int((~finite).sum().item())
                active = (finite & (latent > args.active_epsilon)).float()
                for group in GROUPS:
                    for side in (0, 1):
                        row_weights = torch.from_numpy(weights[group][side][start:end]).to(device)
                        if torch.count_nonzero(row_weights):
                            counts[group][side] += row_weights @ active
        nonfinite_by_layer[layer] = nonfinite
        for group in GROUPS:
            n_pairs = weights[group][2]
            success_count = counts[group][0].cpu().numpy()
            failure_count = counts[group][1].cpu().numpy()
            success_rate = np.nan_to_num(success_count / n_pairs)
            failure_rate = np.nan_to_num(failure_count / n_pairs)
            separation = np.nan_to_num(success_rate - failure_rate)
            minimum_support = max(20, math.ceil(0.01 * n_pairs))
            valid = np.maximum(success_count, failure_count) >= minimum_support
            results[group][layer] = {
                "success_count": success_count, "failure_count": failure_count,
                "success_rate": success_rate, "failure_rate": failure_rate,
                "separation": separation, "valid": valid,
                "minimum_support": np.asarray(minimum_support),
                "d_sae": np.asarray(d_sae), "sae_format": np.asarray(loaded_format),
            }
        del sae, activations
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_valid_scores = np.concatenate([
        values["separation"][values["valid"]]
        for group in GROUPS for values in results[group].values()
    ])
    a, b, c, bin_method = choose_symmetric_thresholds(all_valid_scores)
    definitions = bin_definitions((a, b, c))
    bin_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for group in GROUPS:
        group_scores = []
        candidates = []
        position = GROUPS.index(group) + 1
        for layer in layers:
            values = results[group][layer]
            valid = values["valid"]
            scores = values["separation"]
            valid_count = int(valid.sum())
            group_scores.append(scores[valid])
            for definition in definitions:
                count = int((valid & bin_mask(scores, definition)).sum())
                bin_rows.append({
                    "group": group, "position": position, "layer": layer,
                    "bin": definition["name"], "side": definition["side"],
                    "strength": definition["strength"], "interval": definition["label"],
                    "latent_count": count, "valid_latent_count": valid_count,
                    "latent_fraction_percent": 100.0 * count / valid_count if valid_count else float("nan"),
                    "minimum_support_count": int(values["minimum_support"]),
                })
            for latent_id in np.flatnonzero(valid):
                candidates.append((float(scores[latent_id]), layer, int(latent_id)))
            key = f"{group}_layer_{layer:02d}"
            for field in ("success_rate", "failure_rate", "separation", "valid"):
                arrays[f"{key}_{field}"] = values[field]
        concatenated = np.concatenate(group_scores)
        finite = concatenated[np.isfinite(concatenated)]
        distribution_rows.append({
            "group": group, "position": position, "n_valid_latent_layers": len(finite),
            "mean": float(np.mean(finite)), "std": float(np.std(finite)),
            "minimum": float(np.min(finite)), "q01": float(np.quantile(finite, .01)),
            "q05": float(np.quantile(finite, .05)), "median": float(np.median(finite)),
            "q95": float(np.quantile(finite, .95)), "q99": float(np.quantile(finite, .99)),
            "maximum": float(np.max(finite)),
            "mean_absolute": float(np.mean(np.abs(finite))),
            "q95_absolute": float(np.quantile(np.abs(finite), .95)),
        })
        for selection, ordered in (
            ("success-selective", sorted(candidates, reverse=True)[:20]),
            ("failure-selective", sorted(candidates)[:20]),
        ):
            for rank, (score, layer, latent_id) in enumerate(ordered, 1):
                values = results[group][layer]
                top_rows.append({
                    "group": group, "position": position, "selection": selection,
                    "rank": rank, "layer": layer, "latent_id": latent_id,
                    "success_activation_rate": float(values["success_rate"][latent_id]),
                    "failure_activation_rate": float(values["failure_rate"][latent_id]),
                    "separation_score": score,
                    "support": int(max(values["success_count"][latent_id], values["failure_count"][latent_id])),
                })

    write_csv(args.output_dir / "separation_bins_by_layer.csv", bin_rows)
    write_csv(args.output_dir / "separation_distribution_summary.csv", distribution_rows)
    write_csv(args.output_dir / "top_selective_latents.csv", top_rows)
    np.savez_compressed(args.output_dir / "separation_scores.npz", **arrays)
    plot_stacked(
        bin_rows, layers, definitions, args.output_dir,
        "latent_fraction_percent", "Effective latent fraction (%)",
        "layerwise_latent_separation_fraction",
    )
    plot_stacked(
        bin_rows, layers, definitions, args.output_dir,
        "latent_count", "Latent count", "layerwise_latent_separation_count",
    )
    plot_distributions(results, layers, args.output_dir)
    config = {
        "title": "Layer-wise Success–Failure Latent Separation at First Error Position",
        "groups": {group: {"position": GROUPS.index(group) + 1, "n_pairs": weights[group][2]}
                   for group in GROUPS},
        "support_rule": "max(success_active_count, failure_active_count) >= max(20, ceil(0.01*N))",
        "activation_rule": "JumpReLU SAE.encode applies tau_i; active iff encoded z_i > active_epsilon",
        "active_epsilon": args.active_epsilon,
        "separation": "success activation rate minus failure activation rate",
        "bin_method": bin_method, "symmetric_thresholds": [a, b, c],
        "bin_definitions": definitions, "layers": layers,
        "nonfinite_encoded_values_by_layer": nonfinite_by_layer,
        "matched_success_multiplicity_retained": True,
    }
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f"Thresholds: {a:.6g}, {b:.6g}, {c:.6g} ({bin_method})")
    print(f"Saved latent separation analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
