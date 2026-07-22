"""Experiment 5: retrieval Success-vs-Failure active-fraction analysis.

For each cached teacher-forced decoder activation, the layer SAE produces a
latent vector z.  JumpReLU ``encode`` has already applied each feature's learned
threshold tau_i, so a feature is active exactly when its encoded value is
positive.  The per-row metric is::

    ActiveFraction = count(z_i > 0) / d_sae

The experiment compares five pairs over 24 decoder layers x the first four
DocID generation positions and plots::

    Delta_AF(layer, position) = E[AF | Success group] - E[AF | Failure group]

Matched Success datasets may contain a Success query more than once.  Those
occurrences are deliberately retained as integer weights so the matched target-
token distribution is not changed by accidental deduplication.
"""

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


N_POSITIONS = 4
COMPARISONS = (
    ("Success_vs_All_Failure", "Success", "All Failure"),
    ("S1_vs_F1", "S1", "F1"),
    ("S2_vs_F2", "S2", "F2"),
    ("S3_vs_F3", "S3", "F3"),
    ("S4_vs_F4", "S4", "F4"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAE Active Fraction for retrieval success and failure."
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/activation_cache")
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("out/sae_train_8x")
    )
    parser.add_argument(
        "--dev-data", type=Path, default=Path("dataset/nq320k/dev.json")
    )
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("dataset/nq320k_id/id.semantic.bert.json"),
    )
    parser.add_argument(
        "--outcome-root",
        type=Path,
        default=Path("dataset/nq320k/generation_outcomes"),
    )
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path(
            "dataset/nq320k/generation_outcomes/matched_success_controls"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/retrieval_success_failure"),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--active-epsilon",
        type=float,
        default=0.0,
        help="Numerical epsilon after SAE thresholding; active means z_i > epsilon.",
    )
    parser.add_argument(
        "--sae-format",
        choices=("auto", "inference", "training"),
        default="inference",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def unwrap_scalar(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def parse_dev_record(record: Any) -> tuple[str, int]:
    if isinstance(record, dict):
        query, doc_id = record.get("query", ""), record.get("doc_id")
    elif isinstance(record, (list, tuple)) and len(record) >= 2:
        query, doc_id = record[0], record[1]
    else:
        raise TypeError(f"Unsupported dev record type: {type(record)!r}")
    query, doc_id = unwrap_scalar(query), unwrap_scalar(doc_id)
    if doc_id is None:
        raise KeyError("Dev record is missing doc_id")
    return str(query).strip(), int(doc_id)


def build_activation_metadata(
    dev_path: Path, semantic_path: Path
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """Reproduce cache row order: query-major, then gold DocID position."""
    with dev_path.open("r", encoding="utf-8") as handle:
        dev = json.load(handle)
    with semantic_path.open("r", encoding="utf-8") as handle:
        semantic_ids = json.load(handle)

    query_ids: list[int] = []
    positions: list[int] = []
    valid_queries: set[int] = set()
    for raw_index, record in enumerate(dev):
        query, document_id = parse_dev_record(record)
        if not query:
            continue
        if not 0 <= document_id < len(semantic_ids):
            raise IndexError(f"doc_id {document_id} at dev row {raw_index} is invalid")
        valid_queries.add(raw_index)
        for position in range(len(semantic_ids[document_id])):
            query_ids.append(raw_index)
            positions.append(position)
    return (
        np.asarray(query_ids, dtype=np.int64),
        np.asarray(positions, dtype=np.int8),
        valid_queries,
    )


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise TypeError(f"{path} must contain a JSON array of objects")
    return records


def record_query_id(record: dict[str, Any]) -> int:
    value = record.get("sample_index", record.get("query_id"))
    if value is None:
        raise KeyError("Outcome record is missing sample_index/query_id")
    return int(value)


def load_groups(outcome_root: Path, matched_root: Path) -> dict[str, Counter[int]]:
    paths = {
        "Success": outcome_root / "success.json",
        "All Failure": outcome_root / "failure.json",
        "F1": outcome_root / "failure_1.json",
        "F2": outcome_root / "failure_2.json",
        "F3": outcome_root / "failure_3.json",
        "F4": outcome_root / "failure_4.json",
        "S1": matched_root / "S1.json",
        "S2": matched_root / "S2.json",
        "S3": matched_root / "S3.json",
        "S4": matched_root / "S4.json",
    }
    groups: dict[str, Counter[int]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        groups[name] = Counter(record_query_id(row) for row in load_json_records(path))
    return groups


def build_weighted_selectors(
    query_ids: np.ndarray,
    positions: np.ndarray,
    groups: dict[str, Counter[int]],
    valid_query_ids: set[int],
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    selectors: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for group_name, query_weights in groups.items():
        missing = set(query_weights) - valid_query_ids
        if missing:
            raise ValueError(
                f"{group_name} has {len(missing)} query IDs absent from activation metadata"
            )
        row_weights = np.fromiter(
            (query_weights.get(int(query_id), 0) for query_id in query_ids),
            dtype=np.int32,
            count=len(query_ids),
        )
        selectors[group_name] = {}
        for position in range(N_POSITIONS):
            indices = np.flatnonzero((positions == position) & (row_weights > 0))
            weights = row_weights[indices].astype(np.float64)
            selectors[group_name][position] = (indices, weights)
    return selectors


def weighted_summary(
    values: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float, int, int]:
    if len(indices) == 0 or weights.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    selected = values[indices].astype(np.float64)
    total_weight = float(weights.sum())
    value_mean = float(np.average(selected, weights=weights))
    variance = float(np.average((selected - value_mean) ** 2, weights=weights))
    standard_deviation = math.sqrt(max(variance, 0.0))
    standard_error = standard_deviation / math.sqrt(total_weight)
    return value_mean, standard_deviation, standard_error, int(total_weight), len(indices)


def encode_active_fraction(
    activations: torch.Tensor,
    sae: torch.nn.Module,
    d_sae: int,
    batch_size: int,
    epsilon: float,
    device: torch.device,
) -> np.ndarray:
    fractions = torch.empty(len(activations), dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(activations), batch_size):
            end = min(start + batch_size, len(activations))
            batch = activations[start:end].to(device).float()
            latent_acts = sae.encode(batch)
            fractions[start:end] = (
                (latent_acts > epsilon).sum(dim=1).float().div(float(d_sae)).cpu()
            )
    return fractions.numpy()


def safe_stem(name: str) -> str:
    return name.lower().replace(" ", "_")


def plot_individual_heatmaps(
    matrices: dict[str, np.ndarray], output_dir: Path, color_limit: float
) -> None:
    for comparison, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(5.8, 9.0), constrained_layout=True)
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="nearest",
        )
        ax.set_title(comparison.replace("_", " "))
        ax.set_xlabel("DocID generation position")
        ax.set_ylabel("Decoder layer")
        ax.set_xticks(range(N_POSITIONS), range(1, N_POSITIONS + 1))
        ax.set_yticks(range(matrix.shape[0]), range(matrix.shape[0]))
        for layer in range(matrix.shape[0]):
            for position in range(N_POSITIONS):
                value = matrix[layer, position]
                if np.isfinite(value):
                    ax.text(
                        position,
                        layer,
                        f"{value:+.3f}",
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="white" if abs(value) > 0.55 * color_limit else "black",
                    )
        colorbar = fig.colorbar(image, ax=ax, shrink=0.78)
        colorbar.set_label("Delta Active Fraction (Success - Failure)")
        stem = safe_stem(comparison)
        fig.savefig(output_dir / f"heatmap_{stem}.png", dpi=220)
        fig.savefig(output_dir / f"heatmap_{stem}.pdf")
        plt.close(fig)


def plot_combined_heatmap(
    matrices: dict[str, np.ndarray], output_dir: Path, color_limit: float
) -> None:
    fig, axes = plt.subplots(
        1, len(COMPARISONS), figsize=(17.5, 8.5), sharey=True, constrained_layout=True
    )
    image = None
    for axis, (comparison, _, _) in zip(axes, COMPARISONS):
        matrix = matrices[comparison]
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="nearest",
        )
        axis.set_title(comparison.replace("_", "\n"), fontsize=10)
        axis.set_xlabel("Position")
        axis.set_xticks(range(N_POSITIONS), range(1, N_POSITIONS + 1))
    axes[0].set_ylabel("Decoder layer")
    axes[0].set_yticks(range(len(next(iter(matrices.values())))))
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.72, location="right")
        colorbar.set_label("Delta Active Fraction (Success - Failure)")
    fig.savefig(output_dir / "heatmap_all_comparisons.png", dpi=220)
    fig.savefig(output_dir / "heatmap_all_comparisons.pdf")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.active_epsilon < 0:
        raise ValueError("--active-epsilon must be non-negative")
    for path in (args.dev_data, args.semantic_id_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    available_layers = discover_layers(args.cache_dir, args.checkpoint_root)
    layers = available_layers if args.layers is None else sorted(set(args.layers))
    missing_layers = set(layers) - set(available_layers)
    if missing_layers:
        raise FileNotFoundError(f"Missing cache/SAE for layers: {sorted(missing_layers)}")
    if not layers:
        raise ValueError("No layers selected")

    query_ids, positions, valid_query_ids = build_activation_metadata(
        args.dev_data, args.semantic_id_path
    )
    groups = load_groups(args.outcome_root, args.matched_root)
    selectors = build_weighted_selectors(
        query_ids, positions, groups, valid_query_ids
    )
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    matrices = {
        comparison: np.full((len(layers), N_POSITIONS), np.nan, dtype=np.float64)
        for comparison, _, _ in COMPARISONS
    }
    for layer_index, layer in enumerate(tqdm(layers, desc="Experiment 5", unit="layer")):
        activation_path = args.cache_dir / f"layer_{layer}.safetensors"
        activations = load_file(str(activation_path))["activations"]
        if len(activations) != len(query_ids):
            raise RuntimeError(
                f"Layer {layer}: cache has {len(activations)} rows, metadata has "
                f"{len(query_ids)}"
            )
        sae, config, loaded_format = load_sae(
            args.checkpoint_root / f"layer_{layer}", device, args.sae_format
        )
        d_sae = int(config["d_sae"])
        active_fraction = encode_active_fraction(
            activations,
            sae,
            d_sae,
            args.batch_size,
            args.active_epsilon,
            device,
        )
        for comparison, success_group, failure_group in COMPARISONS:
            for position in range(N_POSITIONS):
                success_summary = weighted_summary(
                    active_fraction, *selectors[success_group][position]
                )
                failure_summary = weighted_summary(
                    active_fraction, *selectors[failure_group][position]
                )
                success_mean, success_std, success_sem, success_n, success_unique = (
                    success_summary
                )
                failure_mean, failure_std, failure_sem, failure_n, failure_unique = (
                    failure_summary
                )
                delta = success_mean - failure_mean
                matrices[comparison][layer_index, position] = delta
                rows.append(
                    {
                        "comparison": comparison,
                        "success_group": success_group,
                        "failure_group": failure_group,
                        "layer": layer,
                        "position": position + 1,
                        "d_sae": d_sae,
                        "sae_format": loaded_format,
                        "success_active_fraction_mean": success_mean,
                        "failure_active_fraction_mean": failure_mean,
                        "delta_active_fraction": delta,
                        "success_std": success_std,
                        "failure_std": failure_std,
                        "success_sem": success_sem,
                        "failure_sem": failure_sem,
                        "success_weighted_n": success_n,
                        "failure_weighted_n": failure_n,
                        "success_unique_query_positions": success_unique,
                        "failure_unique_query_positions": failure_unique,
                    }
                )
        del sae, activations, active_fraction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    finite_values = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in matrices.values()]
    )
    color_limit = float(np.max(np.abs(finite_values))) if len(finite_values) else 1.0
    color_limit = max(color_limit, 1e-9)
    plot_individual_heatmaps(matrices, args.output_dir, color_limit)
    plot_combined_heatmap(matrices, args.output_dir, color_limit)
    write_csv(args.output_dir / "active_fraction_summary.csv", rows)
    with (args.output_dir / "active_fraction_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    np.savez_compressed(
        args.output_dir / "delta_active_fraction_matrices.npz",
        layers=np.asarray(layers),
        positions=np.arange(1, N_POSITIONS + 1),
        **{safe_stem(name): matrix for name, matrix in matrices.items()},
    )
    config = {
        "metric": "count(SAE.encode(x)_i > active_epsilon) / d_sae",
        "threshold_interpretation": (
            "JumpReLU SAE.encode applies learned per-latent tau_i; active_epsilon "
            "is only a post-encoding numerical cutoff"
        ),
        "delta": "mean Active Fraction of Success group minus Failure group",
        "comparisons": [comparison for comparison, _, _ in COMPARISONS],
        "layers": layers,
        "positions": list(range(1, N_POSITIONS + 1)),
        "cache_dir": str(args.cache_dir),
        "checkpoint_root": str(args.checkpoint_root),
        "outcome_root": str(args.outcome_root),
        "matched_root": str(args.matched_root),
        "sae_format": args.sae_format,
        "active_epsilon": args.active_epsilon,
        "matched_success_weighting": (
            "Repeated Success controls retain their multiplicity from S1-S4"
        ),
        "shared_heatmap_color_limit": color_limit,
    }
    with (args.output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    print(f"Saved Experiment 5 to {args.output_dir}")


if __name__ == "__main__":
    main()
