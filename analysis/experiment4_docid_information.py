"""Experiment 4: layer-wise and position-wise DocID information analysis.

The experiment binarizes JumpReLU SAE latents (an encoded value greater than
zero means the learned per-feature threshold was crossed) and measures their
mutual information with five targets: current DocID token, DocID prefix,
top-level document cluster, full DocID, and an independent query-topic control.

For every layer/position/target cell it saves raw MI, label-entropy-normalized
MI, permutation-corrected MI, an empirical-null significance mask, and summary
statistics. A grouped sparse linear probe predicts the current DocID token from
continuous SAE activations while keeping every query's positions in one split.

Example
-------
uv run python analysis/experiment4_docid_information.py \
    --cache-dir data/activation_cache \
    --checkpoint-root out/sae_train_8x \
    --output-dir results/layer_docid_information
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import (  # noqa: E402
    discover_layers,
    linear_trend,
    load_sae,
    resolve_device,
)


TARGETS = ("current_token", "prefix", "document_cluster", "full_docid", "query_topic")


@dataclass
class ActivationMetadata:
    queries: list[str]
    query_doc_ids: np.ndarray
    query_indices: np.ndarray
    positions: np.ndarray
    current_tokens: np.ndarray
    prefix_codes: np.ndarray
    document_clusters: np.ndarray
    full_doc_ids: np.ndarray
    full_docid_eligible: np.ndarray
    max_position: int


def unwrap_doc_id(doc_id) -> int:
    while isinstance(doc_id, list):
        doc_id = doc_id[0]
    return int(doc_id)


def load_activation_metadata(
    dev_data_path: Path,
    semantic_id_path: Path,
    min_full_doc_samples: int,
) -> ActivationMetadata:
    """Reconstruct all labels in exactly the activation-cache row order."""
    with open(dev_data_path, encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(semantic_id_path, encoding="utf-8") as f:
        semantic_ids = json.load(f)

    queries: list[str] = []
    query_doc_ids: list[int] = []
    for raw_query, raw_doc_id in dev_data:
        query = raw_query[0] if isinstance(raw_query, list) else raw_query
        query = str(query).strip()
        if not query:
            continue
        queries.append(query)
        query_doc_ids.append(unwrap_doc_id(raw_doc_id))

    doc_frequency = Counter(query_doc_ids)
    eligible_docs = {
        doc_id for doc_id, count in doc_frequency.items() if count >= min_full_doc_samples
    }
    prefix_maps: dict[int, dict[tuple[int, ...], int]] = defaultdict(dict)
    query_indices: list[int] = []
    positions: list[int] = []
    current_tokens: list[int] = []
    prefix_codes: list[int] = []
    clusters: list[int] = []
    full_doc_ids: list[int] = []
    full_eligible: list[bool] = []

    for query_index, doc_id in enumerate(query_doc_ids):
        semantic_id = [int(token) for token in semantic_ids[doc_id]]
        for position, token in enumerate(semantic_id):
            prefix = tuple(semantic_id[: position + 1])
            prefix_map = prefix_maps[position]
            if prefix not in prefix_map:
                prefix_map[prefix] = len(prefix_map)
            query_indices.append(query_index)
            positions.append(position)
            current_tokens.append(token)
            prefix_codes.append(prefix_map[prefix])
            clusters.append(semantic_id[0])
            full_doc_ids.append(doc_id)
            full_eligible.append(doc_id in eligible_docs)

    return ActivationMetadata(
        queries=queries,
        query_doc_ids=np.asarray(query_doc_ids, dtype=np.int64),
        query_indices=np.asarray(query_indices, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.int64),
        current_tokens=np.asarray(current_tokens, dtype=np.int64),
        prefix_codes=np.asarray(prefix_codes, dtype=np.int64),
        document_clusters=np.asarray(clusters, dtype=np.int64),
        full_doc_ids=np.asarray(full_doc_ids, dtype=np.int64),
        full_docid_eligible=np.asarray(full_eligible, dtype=bool),
        max_position=max(positions),
    )


def build_query_topic_labels(
    queries: list[str],
    n_topics: int,
    max_features: int,
    seed: int,
    output_dir: Path,
) -> np.ndarray:
    """Fit lexical query topics used only as a semantic-confound control."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(queries)
    actual_topics = min(n_topics, len(queries))
    n_components = min(128, matrix.shape[0] - 1, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    semantic_matrix = svd.fit_transform(matrix)
    semantic_matrix = Normalizer(copy=False).fit_transform(semantic_matrix)
    model = KMeans(
        n_clusters=actual_topics,
        random_state=seed,
        n_init=10,
        max_iter=300,
    )
    labels = model.fit_predict(semantic_matrix).astype(np.int64)
    terms = np.asarray(vectorizer.get_feature_names_out())
    lexical_centers = svd.inverse_transform(model.cluster_centers_)
    top_terms = []
    for center in lexical_centers:
        indices = np.argsort(center)[-10:][::-1]
        top_terms.append(terms[indices].tolist())
    with open(output_dir / "query_topic_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "TF-IDF + 128-dimensional LSA + L2 normalization + KMeans",
                "role": "semantic confound control",
                "n_queries": len(queries),
                "n_topics": actual_topics,
                "lsa_components": n_components,
                "max_features": max_features,
                "top_terms": top_terms,
                "topic_sizes": dict(sorted(Counter(map(int, labels)).items())),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return labels


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    _, encoded = np.unique(labels, return_inverse=True)
    return encoded.astype(np.int64), int(encoded.max() + 1) if len(encoded) else 0


def mutual_information_from_counts(
    counts: torch.Tensor,
    class_counts: torch.Tensor,
    n_samples: int,
    feature_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute binary-feature MI and MI/H(Y) from categorical contingency counts."""
    device = counts.device
    n_features = counts.shape[1]
    n = torch.tensor(float(n_samples), device=device)
    class_counts = class_counts.float()
    class_probabilities = class_counts / n
    positive_classes = class_probabilities > 0
    label_entropy = float(
        -(
            class_probabilities[positive_classes]
            * torch.log(class_probabilities[positive_classes])
        ).sum().item()
    )
    active_counts = counts.sum(dim=0)
    mi = torch.zeros(n_features, dtype=torch.float32, device=device)

    for start in range(0, n_features, feature_chunk_size):
        end = min(start + feature_chunk_size, n_features)
        n11 = counts[:, start:end]
        n01 = class_counts[:, None] - n11
        n1 = active_counts[start:end][None, :]
        n0 = n - n1
        class_marginal = class_counts[:, None]
        value = torch.zeros(end - start, dtype=torch.float32, device=device)
        valid11 = n11 > 0
        ratio11 = (n11 * n) / (class_marginal * n1).clamp_min(1e-30)
        value += torch.where(
            valid11, (n11 / n) * torch.log(ratio11.clamp_min(1e-30)), 0.0
        ).sum(dim=0)
        valid01 = n01 > 0
        ratio01 = (n01 * n) / (class_marginal * n0).clamp_min(1e-30)
        value += torch.where(
            valid01, (n01 / n) * torch.log(ratio01.clamp_min(1e-30)), 0.0
        ).sum(dim=0)
        mi[start:end] = value

    mi_np = mi.cpu().numpy()
    nmi = mi_np / label_entropy if label_entropy > 0 else np.zeros_like(mi_np)
    return mi_np, nmi, label_entropy


def compute_binary_latent_mi(
    active_features: torch.Tensor,
    raw_labels: np.ndarray,
    n_permutations: int,
    null_quantile: float,
    activation_rate_bins: int,
    feature_chunk_size: int,
    seed: int,
) -> dict[str, np.ndarray | float | int]:
    """Compute observed and stratified empirical-null corrected MI vectors."""
    labels_np, n_classes = encode_labels(raw_labels)
    if len(labels_np) != len(active_features):
        raise ValueError("Feature rows and labels do not match")
    device = active_features.device
    labels = torch.from_numpy(labels_np).to(device)
    class_counts = torch.bincount(labels, minlength=n_classes).float()
    n_samples, n_features = active_features.shape

    def counts_for(label_order: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(
            (n_classes, n_features), dtype=torch.float32, device=device
        )
        counts.index_add_(0, label_order, active_features)
        return counts

    observed_counts = counts_for(labels)
    observed_mi, observed_nmi, label_entropy = mutual_information_from_counts(
        observed_counts, class_counts, n_samples, feature_chunk_size
    )
    active_rates = (observed_counts.sum(dim=0) / n_samples).cpu().numpy()
    del observed_counts

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    null_mi = np.empty((n_permutations, n_features), dtype=np.float32)
    null_nmi = np.empty_like(null_mi)
    for permutation in range(n_permutations):
        order = torch.randperm(n_samples, generator=generator, device=device)
        permuted_counts = counts_for(labels[order])
        perm_mi, perm_nmi, _ = mutual_information_from_counts(
            permuted_counts, class_counts, n_samples, feature_chunk_size
        )
        null_mi[permutation] = perm_mi
        null_nmi[permutation] = perm_nmi
        del permuted_counts

    corrected_mi = observed_mi - null_mi.mean(axis=0)
    corrected_nmi = observed_nmi - null_nmi.mean(axis=0)
    quantile_edges = np.quantile(
        active_rates, np.linspace(0.0, 1.0, activation_rate_bins + 1)
    )
    rate_bin = np.digitize(active_rates, quantile_edges[1:-1], right=True)
    significant = np.zeros(n_features, dtype=bool)
    null_threshold = np.zeros(n_features, dtype=np.float32)
    for bin_index in range(activation_rate_bins):
        feature_mask = rate_bin == bin_index
        if not feature_mask.any():
            continue
        threshold = float(np.quantile(null_nmi[:, feature_mask], null_quantile))
        null_threshold[feature_mask] = threshold
        significant[feature_mask] = (
            (observed_nmi[feature_mask] > threshold)
            & (corrected_nmi[feature_mask] > 0)
        )

    return {
        "mi": observed_mi,
        "nmi": observed_nmi,
        "permutation_corrected_mi": corrected_mi,
        "permutation_corrected_nmi": corrected_nmi,
        "null_nmi_mean": null_nmi.mean(axis=0),
        "null_nmi_threshold": null_threshold,
        "significant": significant,
        "active_rate": active_rates,
        "label_entropy": label_entropy,
        "n_samples": n_samples,
        "n_classes": n_classes,
    }


def summarize_mi_cell(
    layer: int,
    position: int,
    target: str,
    metrics: dict,
    top_fraction: float,
) -> dict:
    corrected_mi = np.asarray(metrics["permutation_corrected_mi"])
    corrected_nmi = np.asarray(metrics["permutation_corrected_nmi"])
    n_features = len(corrected_mi)
    top_k = max(1, math.ceil(n_features * top_fraction))
    top_indices = np.argpartition(corrected_nmi, -top_k)[-top_k:]
    return {
        "layer": layer,
        "generation_position": position + 1,
        "target": target,
        "n_samples": int(metrics["n_samples"]),
        "n_classes": int(metrics["n_classes"]),
        "label_entropy": float(metrics["label_entropy"]),
        "median_mi": float(np.median(metrics["mi"])),
        "median_nmi": float(np.median(metrics["nmi"])),
        "median_permutation_corrected_mi": float(np.median(corrected_mi)),
        "median_permutation_corrected_nmi": float(np.median(corrected_nmi)),
        "top_1pct_mi_mean": float(np.mean(np.asarray(metrics["mi"])[top_indices])),
        "top_1pct_nmi_mean": float(np.mean(np.asarray(metrics["nmi"])[top_indices])),
        "top_1pct_permutation_corrected_mi_mean": float(
            np.mean(corrected_mi[top_indices])
        ),
        "top_1pct_permutation_corrected_nmi_mean": float(
            np.mean(corrected_nmi[top_indices])
        ),
        "significant_latent_fraction": float(np.mean(metrics["significant"])),
        "active_rate_median": float(np.median(metrics["active_rate"])),
    }


def fit_sparse_current_token_probe(
    latent_activations: torch.Tensor,
    labels: np.ndarray,
    query_groups: np.ndarray,
    test_size: float,
    alpha: float,
    max_iter: int,
    seed: int,
    csr_block_rows: int,
) -> dict:
    """Fit an L1 multinomial SGD probe with a query-grouped split."""
    from scipy.sparse import csr_matrix, vstack

    blocks = []
    for start in range(0, len(latent_activations), csr_block_rows):
        dense_block = latent_activations[start : start + csr_block_rows].float().numpy()
        blocks.append(csr_matrix(dense_block))
    features = vstack(blocks, format="csr", dtype=np.float32)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_indices, test_indices = next(
        splitter.split(features, labels, groups=query_groups)
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l1",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    pipeline = make_pipeline(StandardScaler(with_mean=False), classifier)
    pipeline.fit(features[train_indices], labels[train_indices])
    predictions = pipeline.predict(features[test_indices])
    coefficients = pipeline.named_steps["sgdclassifier"].coef_
    test_labels = labels[test_indices]
    majority_baseline = float(
        np.bincount(test_labels).max() / len(test_labels)
    )
    result = {
        "probe_accuracy": float(accuracy_score(test_labels, predictions)),
        "probe_balanced_accuracy": float(
            balanced_accuracy_score(test_labels, predictions)
        ),
        "probe_majority_baseline": majority_baseline,
        "probe_nonzero_coefficient_fraction": float(
            np.mean(np.abs(coefficients) > 1e-12)
        ),
        "probe_n_iter": int(pipeline.named_steps["sgdclassifier"].n_iter_),
        "probe_train_rows": len(train_indices),
        "probe_test_rows": len(test_indices),
        "probe_classes": int(len(np.unique(labels))),
    }
    del features, blocks, pipeline, classifier
    gc.collect()
    return result


@torch.inference_mode()
def analyze_layer(
    layer: int,
    cache_dir: Path,
    checkpoint_root: Path,
    metadata: ActivationMetadata,
    row_topic_labels: np.ndarray,
    device: torch.device,
    sae_format: str,
    batch_size: int,
    activation_threshold: float,
    n_permutations: int,
    null_quantile: float,
    activation_rate_bins: int,
    mi_feature_chunk_size: int,
    top_fraction: float,
    min_position_samples: int,
    probe_test_size: float,
    probe_alpha: float,
    probe_max_iter: int,
    probe_csr_block_rows: int,
    seed: int,
    distribution_dir: Path,
) -> tuple[list[dict], dict]:
    cached = load_file(str(cache_dir / f"layer_{layer}.safetensors"))
    base_activations = cached["activations"]
    if base_activations.shape[0] != len(metadata.positions):
        raise ValueError(
            f"Layer {layer}: {base_activations.shape[0]} cache rows != "
            f"{len(metadata.positions)} metadata rows"
        )
    sae, sae_cfg, loaded_format = load_sae(
        checkpoint_root / f"layer_{layer}", device, sae_format
    )
    d_sae = int(sae_cfg["d_sae"])
    latent_cache = torch.empty(
        (len(base_activations), d_sae), dtype=torch.float16, device="cpu"
    )
    for start in tqdm(
        range(0, len(base_activations), batch_size),
        desc=f"Layer {layer:02d} SAE encoding",
    ):
        end = min(start + batch_size, len(base_activations))
        batch = base_activations[start:end].to(device, non_blocking=True)
        latent_cache[start:end] = sae.encode(batch).float().cpu().half()
    del sae, base_activations, cached
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cell_summaries: list[dict] = []
    distribution_arrays: dict[str, np.ndarray] = {}
    current_corrected_mi: list[np.ndarray] = []
    current_corrected_nmi: list[np.ndarray] = []
    current_significant: list[np.ndarray] = []

    target_source = {
        "current_token": metadata.current_tokens,
        "prefix": metadata.prefix_codes,
        "document_cluster": metadata.document_clusters,
        "full_docid": metadata.full_doc_ids,
        "query_topic": row_topic_labels,
    }
    for position in range(metadata.max_position + 1):
        position_mask = metadata.positions == position
        row_indices = np.flatnonzero(position_mask)
        position_latents = latent_cache[row_indices].to(device)
        active_features = position_latents.abs().gt(activation_threshold).float()
        del position_latents
        for target_index, target in enumerate(TARGETS):
            target_mask = np.ones(len(row_indices), dtype=bool)
            if target == "full_docid":
                target_mask = metadata.full_docid_eligible[row_indices]
            if target_mask.sum() < 2:
                continue
            target_active = (
                active_features
                if target_mask.all()
                else active_features[
                    torch.from_numpy(np.flatnonzero(target_mask)).to(device)
                ]
            )
            labels = target_source[target][row_indices][target_mask]
            if len(np.unique(labels)) < 2:
                continue
            metrics = compute_binary_latent_mi(
                active_features=target_active,
                raw_labels=labels,
                n_permutations=n_permutations,
                null_quantile=null_quantile,
                activation_rate_bins=activation_rate_bins,
                feature_chunk_size=mi_feature_chunk_size,
                seed=seed + layer * 10_000 + position * 100 + target_index,
            )
            cell_summaries.append(
                summarize_mi_cell(layer, position, target, metrics, top_fraction)
            )
            prefix = f"p{position + 1}_{target}"
            for metric_name in (
                "mi",
                "nmi",
                "permutation_corrected_mi",
                "permutation_corrected_nmi",
                "null_nmi_mean",
                "null_nmi_threshold",
                "significant",
                "active_rate",
            ):
                distribution_arrays[f"{prefix}_{metric_name}"] = np.asarray(
                    metrics[metric_name]
                )
            if target == "current_token" and metrics["n_samples"] >= min_position_samples:
                current_corrected_mi.append(metrics["permutation_corrected_mi"])
                current_corrected_nmi.append(metrics["permutation_corrected_nmi"])
                current_significant.append(metrics["significant"])
            del target_active, metrics
        del active_features

    distribution_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        distribution_dir / f"layer_{layer:02d}_latent_mi.npz",
        **distribution_arrays,
    )

    probe = fit_sparse_current_token_probe(
        latent_activations=latent_cache,
        labels=metadata.current_tokens,
        query_groups=metadata.query_indices,
        test_size=probe_test_size,
        alpha=probe_alpha,
        max_iter=probe_max_iter,
        seed=seed,
        csr_block_rows=probe_csr_block_rows,
    )
    # Layer-level statistics treat a latent as the unit: average its current-token
    # MI across generation positions and mark it selective if significant anywhere.
    latent_mi = np.mean(np.stack(current_corrected_mi), axis=0)
    latent_nmi = np.mean(np.stack(current_corrected_nmi), axis=0)
    latent_significant = np.any(np.stack(current_significant), axis=0)
    top_k = max(1, math.ceil(len(latent_nmi) * top_fraction))
    top_indices = np.argpartition(latent_nmi, -top_k)[-top_k:]
    layer_summary = {
        "layer": layer,
        "n_tokens": len(metadata.positions),
        "d_sae": d_sae,
        "sae_format": loaded_format,
        "median_current_token_mi": float(np.median(latent_mi)),
        "median_current_token_nmi": float(np.median(latent_nmi)),
        "top_1pct_current_token_mi_mean": float(np.mean(latent_mi[top_indices])),
        "top_1pct_current_token_nmi_mean": float(np.mean(latent_nmi[top_indices])),
        "significant_docid_selective_latent_fraction": float(
            np.mean(latent_significant)
        ),
        "current_token_nmi_q25": float(np.quantile(latent_nmi, 0.25)),
        "current_token_nmi_q75": float(np.quantile(latent_nmi, 0.75)),
        "current_token_nmi_q95": float(np.quantile(latent_nmi, 0.95)),
        "current_token_nmi_q99": float(np.quantile(latent_nmi, 0.99)),
        **probe,
    }
    del latent_cache, distribution_arrays
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cell_summaries, layer_summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_progress(
    cell_summaries: list[dict], layer_summaries: list[dict], output_dir: Path
) -> None:
    write_csv(output_dir / "position_target_mi_summary.csv", cell_summaries)
    write_csv(output_dir / "layer_docid_information_summary.csv", layer_summaries)
    with open(output_dir / "position_target_mi_summary.json", "w", encoding="utf-8") as f:
        json.dump(cell_summaries, f, ensure_ascii=False, indent=2)
    with open(output_dir / "layer_docid_information_summary.json", "w", encoding="utf-8") as f:
        json.dump(layer_summaries, f, ensure_ascii=False, indent=2)


def matrix_from_cells(
    rows: list[dict],
    layers: list[int],
    positions: list[int],
    target: str,
    metric: str,
    min_samples: int,
) -> np.ndarray:
    matrix = np.full((len(layers), len(positions)), np.nan, dtype=np.float64)
    layer_index = {layer: index for index, layer in enumerate(layers)}
    position_index = {position: index for index, position in enumerate(positions)}
    for row in rows:
        if row["target"] == target and row["n_samples"] >= min_samples:
            matrix[layer_index[row["layer"]], position_index[row["generation_position"]]] = row[metric]
    return matrix


def plot_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    positions: list[int],
    title: str,
    xlabel: str,
    output_stem: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 10))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Decoder Layer")
    ax.set_xticks(range(len(positions)), positions)
    ax.set_yticks(range(len(layers)), layers)
    fig.colorbar(image, ax=ax, label="Top 1% permutation-corrected NMI")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_results(
    cells: list[dict],
    layers_summary: list[dict],
    output_dir: Path,
    min_position_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [row["layer"] for row in layers_summary]
    positions = sorted({row["generation_position"] for row in cells})
    metric = "top_1pct_permutation_corrected_nmi_mean"
    current_matrix = matrix_from_cells(
        cells, layers, positions, "current_token", metric, min_position_samples
    )
    prefix_matrix = matrix_from_cells(
        cells, layers, positions, "prefix", metric, min_position_samples
    )
    plot_heatmap(
        current_matrix,
        layers,
        positions,
        "Current DocID Token Information by Layer and Position",
        "Generation Position",
        output_dir / "heatmap_layer_position_current_token_nmi",
    )
    plot_heatmap(
        prefix_matrix,
        layers,
        positions,
        "DocID Prefix Information by Layer and Prefix Depth",
        "DocID Prefix Depth",
        output_dir / "heatmap_layer_prefix_depth_nmi",
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        layers,
        [row["top_1pct_current_token_mi_mean"] for row in layers_summary],
        marker="o",
        linewidth=2,
    )
    ax.set_title("Layer-wise Top 1% Permutation-corrected Current-token MI")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Top 1% corrected MI (nats)")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "curve_layer_top_1pct_mi.png", dpi=200)
    fig.savefig(output_dir / "curve_layer_top_1pct_mi.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        layers,
        [row["probe_accuracy"] for row in layers_summary],
        marker="o",
        linewidth=2,
        label="Sparse probe accuracy",
    )
    ax.plot(
        layers,
        [row["probe_majority_baseline"] for row in layers_summary],
        linestyle="--",
        label="Majority baseline",
    )
    ax.set_title("Layer-wise Sparse Probe: Current DocID Token")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Held-out accuracy")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "curve_layer_probe_accuracy.png", dpi=200)
    fig.savefig(output_dir / "curve_layer_probe_accuracy.pdf")
    plt.close(fig)


def compute_trends(layer_summaries: list[dict]) -> dict:
    layers = np.asarray([row["layer"] for row in layer_summaries], dtype=np.float64)
    metrics = (
        "top_1pct_current_token_mi_mean",
        "top_1pct_current_token_nmi_mean",
        "significant_docid_selective_latent_fraction",
        "probe_accuracy",
    )
    trends = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in layer_summaries], dtype=np.float64)
        slope, correlation = linear_trend(layers, values)
        trends[metric] = {
            "slope_per_layer": slope,
            "layer_correlation": correlation,
            "peak_layer": int(layer_summaries[int(np.argmax(values))]["layer"]),
            "peak_value": float(np.max(values)),
        }
    return trends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Layer-wise DocID Information Analysis"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/activation_cache"))
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("out/sae_train_8x")
    )
    parser.add_argument("--dev-data", type=Path, default=Path("dataset/nq320k/dev.json"))
    parser.add_argument(
        "--semantic-id-path",
        type=Path,
        default=Path("dataset/nq320k_id/id.semantic.bert.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/layer_docid_information")
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--null-quantile", type=float, default=0.99)
    parser.add_argument("--activation-rate-bins", type=int, default=10)
    parser.add_argument("--mi-feature-chunk-size", type=int, default=512)
    parser.add_argument("--top-fraction", type=float, default=0.01)
    parser.add_argument("--min-position-samples", type=int, default=100)
    parser.add_argument("--min-full-doc-samples", type=int, default=2)
    parser.add_argument("--n-query-topics", type=int, default=50)
    parser.add_argument("--topic-max-features", type=int, default=20_000)
    parser.add_argument("--probe-test-size", type=float, default=0.2)
    parser.add_argument("--probe-alpha", type=float, default=1e-4)
    parser.add_argument("--probe-max-iter", type=int, default=200)
    parser.add_argument("--probe-csr-block-rows", type=int, default=1024)
    parser.add_argument(
        "--sae-format", choices=("inference", "training", "auto"), default="inference"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    positive = (
        "batch_size",
        "n_permutations",
        "activation_rate_bins",
        "mi_feature_chunk_size",
        "min_full_doc_samples",
        "min_position_samples",
        "n_query_topics",
        "topic_max_features",
        "probe_max_iter",
        "probe_csr_block_rows",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("null_quantile", "top_fraction", "probe_test_size"):
        if not 0 < getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1)")
    if args.activation_threshold < 0 or args.probe_alpha <= 0:
        parser.error("activation threshold must be non-negative and probe alpha positive")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_dir = args.output_dir / "mi_distributions"
    available_layers = discover_layers(args.cache_dir, args.checkpoint_root)
    if args.layers is None:
        layers = available_layers
    else:
        missing = sorted(set(args.layers) - set(available_layers))
        if missing:
            raise FileNotFoundError(f"Missing cache/checkpoint layers: {missing}")
        layers = sorted(set(args.layers))

    metadata = load_activation_metadata(
        args.dev_data, args.semantic_id_path, args.min_full_doc_samples
    )
    print(
        f"Mapped {len(metadata.positions)} activation rows to "
        f"{len(metadata.queries)} queries; positions 1-{metadata.max_position + 1}"
    )
    query_topics = build_query_topic_labels(
        metadata.queries,
        args.n_query_topics,
        args.topic_max_features,
        args.seed,
        args.output_dir,
    )
    row_topic_labels = query_topics[metadata.query_indices]
    with open(args.output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "resolved_layers": layers,
                "latent_binary_definition": (
                    "abs(SAE encoded activation) > activation_threshold; for JumpReLU "
                    "this indicates crossing the learned feature-specific threshold"
                ),
                "nmi_definition": "MI(binary latent; target) / entropy(target)",
                "permutation_correction": (
                    "observed MI minus per-feature empirical permutation-null mean; "
                    "significance uses a null quantile stratified by activation-rate bins"
                ),
                "probe_target": "current DocID token across all generation positions",
                "probe_split": "grouped by query instance",
                "query_topic_definition": (
                    "TF-IDF + 128-dimensional LSA + L2 normalization + KMeans"
                ),
                "layer_latent_aggregation": (
                    "mean corrected MI across generation positions meeting "
                    "min_position_samples; a latent is selective if significant at "
                    "any retained position"
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    all_cells: list[dict] = []
    layer_summaries: list[dict] = []
    for layer in layers:
        cells, summary = analyze_layer(
            layer=layer,
            cache_dir=args.cache_dir,
            checkpoint_root=args.checkpoint_root,
            metadata=metadata,
            row_topic_labels=row_topic_labels,
            device=device,
            sae_format=args.sae_format,
            batch_size=args.batch_size,
            activation_threshold=args.activation_threshold,
            n_permutations=args.n_permutations,
            null_quantile=args.null_quantile,
            activation_rate_bins=args.activation_rate_bins,
            mi_feature_chunk_size=args.mi_feature_chunk_size,
            top_fraction=args.top_fraction,
            min_position_samples=args.min_position_samples,
            probe_test_size=args.probe_test_size,
            probe_alpha=args.probe_alpha,
            probe_max_iter=args.probe_max_iter,
            probe_csr_block_rows=args.probe_csr_block_rows,
            seed=args.seed,
            distribution_dir=distribution_dir,
        )
        all_cells.extend(cells)
        layer_summaries.append(summary)
        save_progress(all_cells, layer_summaries, args.output_dir)
        print(
            f"Layer {layer:02d}: top1% corrected MI="
            f"{summary['top_1pct_current_token_mi_mean']:.6f}, "
            f"selective={summary['significant_docid_selective_latent_fraction']:.4f}, "
            f"probe={summary['probe_accuracy']:.4f}"
        )

    trends = compute_trends(layer_summaries)
    with open(args.output_dir / "trend_summary.json", "w", encoding="utf-8") as f:
        json.dump(trends, f, ensure_ascii=False, indent=2)
    plot_results(
        all_cells,
        layer_summaries,
        args.output_dir,
        args.min_position_samples,
    )
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
