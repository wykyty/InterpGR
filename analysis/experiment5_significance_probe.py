"""Paired latent significance tests and layer-wise sparse failure probes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from safetensors.torch import load_file
from scipy.stats import binom, norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.common import discover_layers, load_sae, resolve_device
from analysis.experiment5_retrieval_success_failure import (
    build_activation_metadata,
    load_json_records,
)


GROUPS = ("S1_vs_F1", "S2_vs_F2", "S3_vs_F3", "S4_vs_F4")


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
        default=Path("results/success_failure_significance_probe"),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--active-epsilon", type=float, default=0.0)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--probe-folds", type=int, default=3)
    parser.add_argument("--probe-permutations", type=int, default=1)
    parser.add_argument("--probe-alpha", type=float, default=1e-3)
    parser.add_argument("--probe-max-iter", type=int, default=80)
    parser.add_argument("--probe-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
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
        writer.writeheader(); writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_pairs(matched_root: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for position in range(1, 5):
        rows = read_jsonl(matched_root / f"S{position}_pairs.jsonl")
        result[f"S{position}_vs_F{position}"] = {
            "success_ids": np.asarray([row["success_query_id"] for row in rows], dtype=np.int64),
            "failure_ids": np.asarray([row["matched_failure_query_id"] for row in rows], dtype=np.int64),
            "position": np.asarray(position - 1),
        }
    return result


def row_lookup(query_ids: np.ndarray, positions: np.ndarray) -> dict[tuple[int, int], int]:
    lookup = {}
    for row, (query_id, position) in enumerate(zip(query_ids, positions)):
        key = (int(query_id), int(position))
        if key in lookup:
            raise ValueError(f"Duplicate activation metadata key: {key}")
        lookup[key] = row
    return lookup


def pair_row_indices(pair: dict[str, np.ndarray], lookup: dict[tuple[int, int], int]) -> tuple[np.ndarray, np.ndarray]:
    position = int(pair["position"])
    try:
        success = np.asarray([lookup[(int(q), position)] for q in pair["success_ids"]], dtype=np.int64)
        failure = np.asarray([lookup[(int(q), position)] for q in pair["failure_ids"]], dtype=np.int64)
    except KeyError as error:
        raise KeyError(f"Missing paired activation row: {error}") from error
    return success, failure


def encode_rows(
    activations: torch.Tensor, indices: np.ndarray, sae: torch.nn.Module,
    batch_size: int, device: torch.device,
) -> tuple[sp.csr_matrix, int]:
    blocks = []
    nonfinite = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = torch.from_numpy(indices[start:start + batch_size]).long()
            hidden = activations.index_select(0, batch_indices).to(device).float()
            latent = sae.encode(hidden)
            finite = torch.isfinite(latent)
            nonfinite += int((~finite).sum().item())
            latent = torch.where(finite, latent, torch.zeros_like(latent))
            blocks.append(sp.csr_matrix(latent.float().cpu().numpy()))
    return sp.vstack(blocks, format="csr"), nonfinite


def paired_statistics(
    success_latent: sp.csr_matrix, failure_latent: sp.csr_matrix,
    success_cluster_ids: np.ndarray, epsilon: float,
) -> dict[str, np.ndarray]:
    success_active = success_latent.copy(); failure_active = failure_latent.copy()
    success_active.data = (success_active.data > epsilon).astype(np.float32)
    failure_active.data = (failure_active.data > epsilon).astype(np.float32)
    success_active.eliminate_zeros(); failure_active.eliminate_zeros()
    n_pairs, d_sae = success_active.shape
    success_count = np.asarray(success_active.sum(axis=0)).ravel()
    failure_count = np.asarray(failure_active.sum(axis=0)).ravel()
    separation = (success_count - failure_count) / float(n_pairs)
    support_threshold = max(20, int(np.ceil(0.01 * n_pairs)))
    valid = np.maximum(success_count, failure_count) >= support_threshold

    unique_clusters, cluster_inverse = np.unique(success_cluster_ids, return_inverse=True)
    n_clusters = len(unique_clusters)
    aggregation = sp.csr_matrix(
        (np.ones(n_pairs), (cluster_inverse, np.arange(n_pairs))),
        shape=(n_clusters, n_pairs),
    )
    differences = success_active - failure_active
    cluster_sums = aggregation @ differences
    cluster_sizes = np.bincount(cluster_inverse).astype(np.float64)
    cluster_sq = np.asarray(cluster_sums.power(2).sum(axis=0)).ravel()
    weighted_sum = np.asarray(cluster_sizes @ cluster_sums).ravel()
    residual_ss = (
        cluster_sq - 2.0 * separation * weighted_sum
        + np.square(separation) * np.square(cluster_sizes).sum()
    )
    variance = np.maximum(residual_ss, 0.0) / float(n_pairs * n_pairs)
    if n_clusters > 1:
        variance *= n_clusters / (n_clusters - 1.0)
    standard_error = np.sqrt(variance)
    z_score = np.zeros_like(separation)
    positive_error = standard_error > 0
    z_score[positive_error] = separation[positive_error] / standard_error[positive_error]
    zero_error_effect = (~positive_error) & (separation != 0)
    z_score[zero_error_effect] = np.sign(separation[zero_error_effect]) * np.inf
    p_cluster = np.clip(2.0 * norm.sf(np.abs(z_score)), 0.0, 1.0)

    overlap = np.asarray(success_active.multiply(failure_active).sum(axis=0)).ravel()
    success_only = success_count - overlap
    failure_only = failure_count - overlap
    discordant = success_only + failure_only
    smaller = np.minimum(success_only, failure_only)
    p_mcnemar = np.ones(d_sae)
    has_discordance = discordant > 0
    p_mcnemar[has_discordance] = np.minimum(
        1.0,
        2.0 * binom.cdf(smaller[has_discordance], discordant[has_discordance], 0.5),
    )
    return {
        "success_count": success_count, "failure_count": failure_count,
        "success_rate": success_count / n_pairs, "failure_rate": failure_count / n_pairs,
        "separation": separation, "standard_error": standard_error,
        "ci_low": separation - 1.96 * standard_error,
        "ci_high": separation + 1.96 * standard_error,
        "p_cluster": p_cluster, "p_mcnemar": p_mcnemar,
        "valid": valid, "support_threshold": np.asarray(support_threshold),
        "n_clusters": np.asarray(n_clusters),
    }


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    q_values = np.full_like(p_values, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if len(finite_indices) == 0:
        return q_values
    values = p_values[finite_indices]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted); restored[order] = np.clip(adjusted, 0, 1)
    q_values[finite_indices] = restored
    return q_values


def evaluate_probe(
    success_latent: sp.csr_matrix, failure_latent: sp.csr_matrix,
    success_ids: np.ndarray, folds: int, permutations: int,
    alpha: float, max_iter: int, learning_rate: float, seed: int,
    device: torch.device,
) -> dict[str, float]:
    # Binary latents are the same thresholded features used by the significance
    # test. A dense GPU matrix is substantially faster than CPU sparse solvers
    # for later T5 layers, whose JumpReLU codes are less sparse.
    x_sparse = sp.vstack([success_latent, failure_latent], format="csr")
    x_sparse.data = np.ones_like(x_sparse.data, dtype=np.float32)
    x_sparse.eliminate_zeros()
    x = torch.from_numpy(x_sparse.toarray().astype(np.float32, copy=False)).to(device)
    n_pairs = success_latent.shape[0]
    y = np.concatenate([np.ones(n_pairs, dtype=np.int8), np.zeros(n_pairs, dtype=np.int8)])
    # Each matched pair, and all pairs sharing a reused Success, stay in one fold.
    groups = np.concatenate([success_ids, success_ids])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))

    def oof(labels: np.ndarray, run_seed: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        probabilities = np.zeros(len(labels), dtype=np.float64)
        predictions = np.zeros(len(labels), dtype=bool)
        nonzero = []
        for fold, (train, test) in enumerate(splits):
            torch.manual_seed(run_seed + fold)
            model = torch.nn.Linear(x.shape[1], 1, device=device)
            torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            train_index = torch.from_numpy(train).long().to(device)
            test_index = torch.from_numpy(test).long().to(device)
            train_labels = torch.from_numpy(labels[train].astype(np.float32)).to(device)
            for _ in range(max_iter):
                logits = model(x.index_select(0, train_index)).squeeze(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                # Proximal L1 step gives genuinely sparse coefficients.
                with torch.no_grad():
                    weight = model.weight
                    weight.copy_(weight.sign() * torch.relu(weight.abs() - learning_rate * alpha))
            with torch.no_grad():
                train_probabilities = torch.sigmoid(
                    model(x.index_select(0, train_index)).squeeze(1)
                ).cpu().numpy()
                test_probabilities = torch.sigmoid(
                    model(x.index_select(0, test_index)).squeeze(1)
                ).cpu().numpy()
                probabilities[test] = test_probabilities
                nonzero.append(int(torch.count_nonzero(model.weight).item()))
            # Calibrate the threshold strictly inside the training fold. AUROC
            # and AUPRC remain threshold-free; this only affects hard metrics.
            fpr, tpr, thresholds = roc_curve(labels[train], train_probabilities)
            best_index = int(np.nanargmax(tpr - fpr))
            decision_threshold = float(thresholds[best_index])
            if not np.isfinite(decision_threshold):
                decision_threshold = 0.5
            predictions[test] = test_probabilities >= decision_threshold
        return probabilities, predictions, nonzero

    probability, prediction, nonzero = oof(y, seed)
    rng = np.random.default_rng(seed)
    null_aucs = []
    unique_groups, inverse = np.unique(success_ids, return_inverse=True)
    for permutation in range(permutations):
        flips = rng.integers(0, 2, size=len(unique_groups), dtype=np.int8)
        pair_flips = flips[inverse]
        permuted = y.copy()
        permuted[:n_pairs] = 1 - pair_flips
        permuted[n_pairs:] = pair_flips
        null_probability, _, _ = oof(permuted, seed + 10_000 + permutation * 100)
        null_aucs.append(float(roc_auc_score(permuted, null_probability)))
    result = {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
        "mean_nonzero_coefficients": float(np.mean(nonzero)),
        "std_nonzero_coefficients": float(np.std(nonzero)),
        "permutation_auc_mean": float(np.mean(null_aucs)) if null_aucs else float("nan"),
        "permutation_auc_std": float(np.std(null_aucs)) if null_aucs else float("nan"),
    }
    del x
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def plot_significance(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, GROUPS):
        selected = [row for row in rows if row["group"] == group]
        layers = [row["layer"] for row in selected]
        axis.plot(layers, [row["success_significant_fraction_percent"] for row in selected],
                  color="#cb181d", marker="o", markersize=3, label="Success-selective")
        axis.plot(layers, [row["failure_significant_fraction_percent"] for row in selected],
                  color="#08519c", marker="o", markersize=3, label="Failure-selective")
        position = GROUPS.index(group) + 1
        axis.set_title(f"S{position} vs F{position} — position {position}")
        axis.set_xticks(range(24)); axis.tick_params(axis="x", rotation=90, labelbottom=True)
        axis.grid(alpha=.22); axis.set_xlabel("Decoder Layer")
    axes[0, 0].set_ylabel("Significant effective latents (%)")
    axes[1, 0].set_ylabel("Significant effective latents (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("FDR-significant Success- and Failure-selective SAE Latents")
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output_dir / "layerwise_significant_latent_fraction.png", dpi=320)
    fig.savefig(output_dir / "layerwise_significant_latent_fraction.pdf")
    plt.close(fig)


def plot_probe(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    for axis, group in zip(axes.flat, GROUPS):
        selected = [row for row in rows if row["group"] == group]
        layers = [row["layer"] for row in selected]
        axis.plot(layers, [row["roc_auc"] for row in selected], color="#7b3294", label="AUROC")
        axis.plot(layers, [row["balanced_accuracy"] for row in selected], color="#008837", label="Balanced accuracy")
        axis.plot(layers, [row["average_precision"] for row in selected], color="#c51b7d", label="AUPRC")
        axis.plot(layers, [row["permutation_auc_mean"] for row in selected], color="#666666",
                  linestyle="--", label="Permutation AUROC")
        position = GROUPS.index(group) + 1
        axis.set_title(f"S{position} vs F{position} — position {position}")
        axis.set_xticks(range(24)); axis.tick_params(axis="x", rotation=90, labelbottom=True)
        axis.set_ylim(.35, 1.01); axis.grid(alpha=.22); axis.set_xlabel("Decoder Layer")
    axes[0, 0].set_ylabel("Cross-validated score")
    axes[1, 0].set_ylabel("Cross-validated score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Layer-wise Sparse SAE Probe: Retrieval Success vs Failure")
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(output_dir / "layerwise_sparse_probe.png", dpi=320)
    fig.savefig(output_dir / "layerwise_sparse_probe.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    available = discover_layers(args.cache_dir, args.checkpoint_root)
    layers = available if args.layers is None else sorted(set(args.layers))
    if not layers or set(layers) - set(available):
        raise ValueError("Invalid or unavailable layers")
    query_ids, positions, _ = build_activation_metadata(args.dev_data, args.semantic_id_path)
    lookup = row_lookup(query_ids, positions)
    pairs = load_pairs(args.matched_root)
    pair_indices = {group: pair_row_indices(pair, lookup) for group, pair in pairs.items()}
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    statistics: dict[str, dict[int, dict[str, np.ndarray]]] = {group: {} for group in GROUPS}
    probe_rows = []
    nonfinite = {}
    for layer in tqdm(layers, desc="Significance + sparse probe", unit="layer"):
        activations = load_file(str(args.cache_dir / f"layer_{layer}.safetensors"))["activations"]
        sae, config, _ = load_sae(args.checkpoint_root / f"layer_{layer}", device, args.sae_format)
        for group_index, group in enumerate(GROUPS):
            success_indices, failure_indices = pair_indices[group]
            all_indices = np.concatenate([success_indices, failure_indices])
            latent, invalid_count = encode_rows(
                activations, all_indices, sae, args.batch_size, device
            )
            nonfinite[f"{group}_layer_{layer}"] = invalid_count
            n_pairs = len(success_indices)
            success_latent, failure_latent = latent[:n_pairs], latent[n_pairs:]
            stats = paired_statistics(
                success_latent, failure_latent, pairs[group]["success_ids"], args.active_epsilon
            )
            statistics[group][layer] = stats
            metrics = evaluate_probe(
                success_latent, failure_latent, pairs[group]["success_ids"],
                args.probe_folds, args.probe_permutations, args.probe_alpha,
                args.probe_max_iter, args.probe_learning_rate,
                args.seed + layer * 100 + group_index, device,
            )
            probe_rows.append({
                "group": group, "position": group_index + 1, "layer": layer,
                "n_pairs": n_pairs, "n_unique_success_clusters": int(stats["n_clusters"]),
                "d_sae": int(config["d_sae"]), "probe_input": "thresholded_binary_sae_latents",
                **metrics,
            })
            del latent, success_latent, failure_latent
        del sae, activations
        if device.type == "cuda": torch.cuda.empty_cache()

    latent_rows = []
    layer_rows = []
    for group_index, group in enumerate(GROUPS):
        valid_p = np.concatenate([
            statistics[group][layer]["p_cluster"][statistics[group][layer]["valid"]]
            for layer in layers
        ])
        valid_q = bh_fdr(valid_p)
        offset = 0
        for layer in layers:
            stats = statistics[group][layer]; valid_ids = np.flatnonzero(stats["valid"])
            q_values = valid_q[offset:offset + len(valid_ids)]; offset += len(valid_ids)
            significant = q_values < args.fdr_alpha
            separations = stats["separation"][valid_ids]
            success_significant = significant & (separations > 0)
            failure_significant = significant & (separations < 0)
            for local, latent_id in enumerate(valid_ids):
                latent_rows.append({
                    "group": group, "position": group_index + 1, "layer": layer,
                    "latent_id": int(latent_id),
                    "success_activation_rate": float(stats["success_rate"][latent_id]),
                    "failure_activation_rate": float(stats["failure_rate"][latent_id]),
                    "separation_score": float(stats["separation"][latent_id]),
                    "cluster_robust_se": float(stats["standard_error"][latent_id]),
                    "ci_low": float(stats["ci_low"][latent_id]),
                    "ci_high": float(stats["ci_high"][latent_id]),
                    "cluster_robust_p": float(stats["p_cluster"][latent_id]),
                    "mcnemar_exact_p_naive": float(stats["p_mcnemar"][latent_id]),
                    "fdr_q": float(q_values[local]), "fdr_significant": bool(significant[local]),
                    "selectivity": "success" if separations[local] > 0 else "failure" if separations[local] < 0 else "none",
                    "support": int(max(stats["success_count"][latent_id], stats["failure_count"][latent_id])),
                    "support_threshold": int(stats["support_threshold"]),
                })
            denominator = len(valid_ids)
            layer_rows.append({
                "group": group, "position": group_index + 1, "layer": layer,
                "valid_latent_count": denominator,
                "success_significant_count": int(success_significant.sum()),
                "failure_significant_count": int(failure_significant.sum()),
                "success_significant_fraction_percent": 100 * success_significant.sum() / denominator if denominator else np.nan,
                "failure_significant_fraction_percent": 100 * failure_significant.sum() / denominator if denominator else np.nan,
                "total_significant_fraction_percent": 100 * significant.sum() / denominator if denominator else np.nan,
                "median_separation": float(np.median(separations)) if denominator else np.nan,
            })

    with gzip.open(args.output_dir / "latent_significance.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latent_rows[0])); writer.writeheader(); writer.writerows(latent_rows)
    write_csv(args.output_dir / "layer_significance_summary.csv", layer_rows)
    write_csv(args.output_dir / "sparse_probe_summary.csv", probe_rows)
    plot_significance(layer_rows, args.output_dir); plot_probe(probe_rows, args.output_dir)
    config_out = {
        "primary_test": "paired cluster-robust Wald test clustered by reused success_query_id",
        "secondary_test": "exact McNemar/binomial test without reuse correction",
        "fdr": f"Benjamini-Hochberg within each group over all valid layer-latent tests; alpha={args.fdr_alpha}",
        "support": "max active count >= max(20, ceil(0.01*N))",
        "probe": "GPU proximal-L1 logistic regression on thresholded binary SAE latents",
        "probe_cv": "StratifiedGroupKFold; matched pair and all reused-Success pairs stay in one fold",
        "probe_decision_threshold": "Youden-J threshold selected on each training fold and applied to its held-out fold",
        "probe_folds": args.probe_folds, "probe_permutations": args.probe_permutations,
        "probe_alpha": args.probe_alpha, "probe_learning_rate": args.probe_learning_rate,
        "probe_epochs": args.probe_max_iter, "layers": layers, "nonfinite_counts": nonfinite,
    }
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_out, handle, indent=2)
    print(f"Saved significance and probe results to {args.output_dir}")


if __name__ == "__main__":
    main()
