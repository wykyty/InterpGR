"""Experiment 2: layer-wise semantic purity of SAE latents.

The experiment samples reproducible random latents from a high-activation
candidate pool at every decoder layer, finds their top activated documents,
and measures whether later layers become more document-specific.

Metrics
-------
Document Entropy
    Entropy of a latent's activation mass, normalized by log(number of dev
    documents). Lower is more document-specific and comparable across layers.
Embedding Cluster Purity
    Weighted dominant-cluster share among top activated documents, where the
    cluster is the first BERT semantic-DocID component.
Topic Purity
    Weighted dominant-topic share among top activated documents. Topic labels
    are independently produced with TF-IDF + MiniBatchKMeans on dev documents.

Example
-------
uv run python analysis/experiment2_document_purity.py \
    --cache-dir data/activation_cache \
    --checkpoint-root out/sae_train_8x \
    --output-dir results/layer_semantic_purity
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import (  # noqa: E402
    discover_layers,
    linear_trend,
    load_sae,
    resolve_device,
    sample_top_latents,
    summarize_values,
    weighted_purity,
)


def unwrap_doc_id(doc_id) -> int:
    while isinstance(doc_id, list):
        doc_id = doc_id[0]
    return int(doc_id)


def load_activation_document_mapping(
    dev_data_path: Path,
    semantic_id_path: Path,
) -> tuple[list[list[int]], list[int], list[int], torch.Tensor]:
    """Reconstruct the cache-row -> document mapping used by cache_activations."""
    with open(dev_data_path, encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(semantic_id_path, encoding="utf-8") as f:
        semantic_ids = json.load(f)

    query_doc_ids: list[int] = []
    activation_doc_ids: list[int] = []
    for query, raw_doc_id in dev_data:
        if isinstance(query, list):
            query = query[0]
        if not query or not str(query).strip():
            continue
        doc_id = unwrap_doc_id(raw_doc_id)
        query_doc_ids.append(doc_id)
        activation_doc_ids.extend([doc_id] * len(semantic_ids[doc_id]))

    unique_doc_ids = sorted(set(query_doc_ids))
    doc_to_local = {doc_id: i for i, doc_id in enumerate(unique_doc_ids)}
    activation_local_docs = torch.tensor(
        [doc_to_local[doc_id] for doc_id in activation_doc_ids],
        dtype=torch.long,
    )
    return semantic_ids, query_doc_ids, unique_doc_ids, activation_local_docs


def build_topic_labels(
    corpus_path: Path,
    unique_doc_ids: list[int],
    n_topics: int,
    max_features: int,
    seed: int,
    output_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Create independent lexical topic labels for dev-referenced documents."""
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    if unique_doc_ids and max(unique_doc_ids) >= len(corpus):
        raise IndexError("A dev document ID is outside corpus_lite.json")

    documents = [str(corpus[doc_id]) for doc_id in unique_doc_ids]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(documents)
    actual_topics = min(n_topics, len(documents))
    topic_model = MiniBatchKMeans(
        n_clusters=actual_topics,
        random_state=seed,
        batch_size=min(1024, len(documents)),
        n_init=10,
        max_iter=200,
    )
    topic_labels = topic_model.fit_predict(tfidf)

    terms = np.asarray(vectorizer.get_feature_names_out())
    top_terms: list[list[str]] = []
    for center in topic_model.cluster_centers_:
        top_indices = np.argsort(center)[-10:][::-1]
        top_terms.append(terms[top_indices].tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "topic_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "TF-IDF + MiniBatchKMeans",
                "n_documents": len(documents),
                "n_topics": actual_topics,
                "max_features": max_features,
                "top_terms": top_terms,
                "topic_sizes": dict(sorted(Counter(map(int, topic_labels)).items())),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    del corpus, documents, tfidf
    gc.collect()
    return topic_labels.astype(np.int64), [", ".join(x[:5]) for x in top_terms]


def normalized_document_entropy(scores: np.ndarray) -> tuple[float, float, int]:
    """Compute activation-mass entropy normalized by the document universe.

    Using log(total documents), rather than log(active documents), makes values
    comparable across layers and lets a latent active on fewer documents receive
    lower entropy even when its mass is uniform inside that smaller support.
    """
    positive = scores[scores > 0]
    n_active = len(positive)
    if n_active == 0:
        return 0.0, 0.0, 0
    probabilities = positive / positive.sum()
    raw_entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-30)))
    normalized_entropy = raw_entropy / math.log(len(scores)) if len(scores) > 1 else 0.0
    return raw_entropy, normalized_entropy, n_active


@torch.inference_mode()
def analyze_layer(
    layer: int,
    cache_dir: Path,
    checkpoint_root: Path,
    activation_local_docs: torch.Tensor,
    unique_doc_ids: list[int],
    semantic_ids: list[list[int]],
    topic_labels: np.ndarray,
    topic_names: list[str],
    corpus: list[str],
    device: torch.device,
    sae_format: str,
    batch_size: int,
    n_latents: int,
    candidate_pool_size: int,
    top_documents: int,
    min_active_tokens: int,
    activation_threshold: float,
    document_aggregation: str,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Analyze sampled high-activation latents for one layer."""
    cache_path = cache_dir / f"layer_{layer}.safetensors"
    cached = load_file(str(cache_path))
    activations = cached["activations"]
    if activations.ndim != 2:
        raise ValueError(f"Layer {layer}: invalid activation shape {activations.shape}")
    if activations.shape[0] != len(activation_local_docs):
        raise ValueError(
            f"Layer {layer}: cache has {activations.shape[0]} rows but reconstructed "
            f"document mapping has {len(activation_local_docs)}"
        )

    sae, sae_cfg, loaded_format = load_sae(
        checkpoint_root / f"layer_{layer}", device, sae_format
    )
    d_sae = int(sae_cfg["d_sae"])
    if activations.shape[1] != int(sae_cfg["d_in"]):
        raise ValueError(f"Layer {layer}: cache and SAE d_in do not match")

    # Pass 1: rank latents by total activation mass, excluding nearly dead ones.
    activation_mass = torch.zeros(d_sae, dtype=torch.float64)
    active_token_count = torch.zeros(d_sae, dtype=torch.long)
    for start in tqdm(
        range(0, len(activations), batch_size),
        desc=f"Layer {layer:02d} candidate pass",
    ):
        batch = activations[start : start + batch_size].to(device, non_blocking=True)
        latent_acts = sae.encode(batch).float()
        activation_mass += latent_acts.abs().sum(dim=0).double().cpu()
        active_token_count += (
            latent_acts.abs().gt(activation_threshold).sum(dim=0).long().cpu()
        )

    try:
        sampled_latents, pool_size = sample_top_latents(
            activation_mass=activation_mass,
            active_token_count=active_token_count,
            n_latents=n_latents,
            candidate_pool_size=candidate_pool_size,
            min_active_tokens=min_active_tokens,
            seed=seed + layer,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"Layer {layer}: {exc}") from exc

    # Pass 2: aggregate each selected latent over all occurrences of a document.
    n_docs = len(unique_doc_ids)
    selected_tensor = torch.tensor(sampled_latents, dtype=torch.long, device=device)
    if document_aggregation == "mean":
        document_scores = torch.zeros((n_docs, n_latents), device=device)
        document_counts = torch.bincount(
            activation_local_docs, minlength=n_docs
        ).to(device=device, dtype=torch.float32)
    else:
        document_scores = torch.full((n_docs, n_latents), -torch.inf, device=device)
        document_counts = None

    for start in tqdm(
        range(0, len(activations), batch_size),
        desc=f"Layer {layer:02d} document pass",
    ):
        end = min(start + batch_size, len(activations))
        batch = activations[start:end].to(device, non_blocking=True)
        selected_acts = sae.encode(batch).float().index_select(1, selected_tensor)
        local_docs = activation_local_docs[start:end].to(device, non_blocking=True)
        if document_aggregation == "mean":
            document_scores.index_add_(0, local_docs, selected_acts)
        else:
            scatter_index = local_docs[:, None].expand(-1, n_latents)
            document_scores.scatter_reduce_(
                0, scatter_index, selected_acts, reduce="amax", include_self=True
            )

    if document_aggregation == "mean":
        document_scores /= document_counts.clamp_min(1.0)[:, None]
    else:
        document_scores.masked_fill_(torch.isinf(document_scores), 0.0)
    document_scores_np = document_scores.cpu().numpy()

    embedding_clusters = np.asarray(
        [int(semantic_ids[doc_id][0]) for doc_id in unique_doc_ids], dtype=np.int64
    )
    details: list[dict] = []
    raw_entropy_values: list[float] = []
    entropy_values: list[float] = []
    embedding_purity_values: list[float] = []
    topic_purity_values: list[float] = []
    specificity_values: list[float] = []
    active_document_counts: list[float] = []

    for feature_column, feature_id in enumerate(sampled_latents):
        scores = document_scores_np[:, feature_column]
        raw_entropy, entropy, active_document_count = normalized_document_entropy(
            scores
        )
        positive_indices = np.where(scores > 0)[0]
        k = min(top_documents, len(positive_indices))
        if k:
            positive_scores = scores[positive_indices]
            top_local = positive_indices[np.argsort(positive_scores)[-k:][::-1]]
            top_weights = scores[top_local]
            embedding_purity, dominant_embedding_cluster = weighted_purity(
                embedding_clusters[top_local], top_weights
            )
            topic_purity, dominant_topic = weighted_purity(
                topic_labels[top_local], top_weights
            )
        else:
            top_local = np.asarray([], dtype=np.int64)
            embedding_purity, dominant_embedding_cluster = 0.0, -1
            topic_purity, dominant_topic = 0.0, -1

        specificity = ((1.0 - entropy) + embedding_purity + topic_purity) / 3.0
        raw_entropy_values.append(raw_entropy)
        entropy_values.append(entropy)
        embedding_purity_values.append(embedding_purity)
        topic_purity_values.append(topic_purity)
        specificity_values.append(specificity)
        active_document_counts.append(float(active_document_count))

        top_document_records = []
        for local_doc in top_local:
            doc_id = unique_doc_ids[int(local_doc)]
            text = re.sub(r"\s+", " ", str(corpus[doc_id])).strip()
            topic = int(topic_labels[int(local_doc)])
            top_document_records.append(
                {
                    "doc_id": doc_id,
                    "activation": float(scores[int(local_doc)]),
                    "semantic_id": semantic_ids[doc_id],
                    "embedding_cluster": int(embedding_clusters[int(local_doc)]),
                    "topic_cluster": topic,
                    "topic_terms": topic_names[topic],
                    "text": text[:300],
                }
            )

        details.append(
            {
                "layer": layer,
                "feature_id": int(feature_id),
                "candidate_activation_mass": float(activation_mass[feature_id]),
                "active_token_count": int(active_token_count[feature_id]),
                "active_document_count": active_document_count,
                "document_entropy_raw": raw_entropy,
                "document_entropy": entropy,
                "embedding_cluster_purity": embedding_purity,
                "dominant_embedding_cluster": dominant_embedding_cluster,
                "topic_purity": topic_purity,
                "dominant_topic": dominant_topic,
                "dominant_topic_terms": (
                    topic_names[dominant_topic] if dominant_topic >= 0 else ""
                ),
                "document_specificity": specificity,
                "top_activated_documents": top_document_records,
            }
        )

    raw_entropy_mean, raw_entropy_sem = summarize_values(raw_entropy_values)
    document_entropy_mean, document_entropy_sem = summarize_values(entropy_values)
    embedding_purity_mean, embedding_purity_sem = summarize_values(
        embedding_purity_values
    )
    topic_purity_mean, topic_purity_sem = summarize_values(topic_purity_values)
    specificity_mean, specificity_sem = summarize_values(specificity_values)
    active_docs_mean, active_docs_sem = summarize_values(active_document_counts)

    summary = {
        "layer": layer,
        "n_tokens": len(activations),
        "n_documents": n_docs,
        "d_sae": d_sae,
        "sae_format": loaded_format,
        "sampled_latents": n_latents,
        "candidate_pool_size": pool_size,
        "document_aggregation": document_aggregation,
        "document_entropy_raw_mean": raw_entropy_mean,
        "document_entropy_raw_sem": raw_entropy_sem,
        "document_entropy_mean": document_entropy_mean,
        "document_entropy_sem": document_entropy_sem,
        "embedding_cluster_purity_mean": embedding_purity_mean,
        "embedding_cluster_purity_sem": embedding_purity_sem,
        "topic_purity_mean": topic_purity_mean,
        "topic_purity_sem": topic_purity_sem,
        "document_specificity_mean": specificity_mean,
        "document_specificity_sem": specificity_sem,
        "active_documents_mean": active_docs_mean,
        "active_documents_sem": active_docs_sem,
    }

    del sae, activations, cached, document_scores
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, details


def save_layer_summaries(summaries: list[dict], output_dir: Path) -> None:
    with open(output_dir / "layer_semantic_purity.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with open(
        output_dir / "layer_semantic_purity.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def compute_trends(summaries: list[dict]) -> dict:
    layers = np.asarray([row["layer"] for row in summaries], dtype=np.float64)
    metrics = {
        "document_entropy": np.asarray(
            [row["document_entropy_mean"] for row in summaries], dtype=np.float64
        ),
        "embedding_cluster_purity": np.asarray(
            [row["embedding_cluster_purity_mean"] for row in summaries],
            dtype=np.float64,
        ),
        "topic_purity": np.asarray(
            [row["topic_purity_mean"] for row in summaries], dtype=np.float64
        ),
        "document_specificity": np.asarray(
            [row["document_specificity_mean"] for row in summaries],
            dtype=np.float64,
        ),
    }
    trend = {}
    for name, values in metrics.items():
        slope, correlation = linear_trend(layers, values)
        trend[name] = {"slope_per_layer": slope, "layer_correlation": correlation}

    positive_specificity_signals = sum(
        [
            trend["document_entropy"]["slope_per_layer"] < 0,
            trend["embedding_cluster_purity"]["slope_per_layer"] > 0,
            trend["topic_purity"]["slope_per_layer"] > 0,
        ]
    )
    later_more_specific = (
        trend["document_specificity"]["slope_per_layer"] > 0
        and positive_specificity_signals >= 2
    )
    trend["later_layers_more_document_specific"] = later_more_specific
    trend["interpretation"] = (
        "Later layers tend to be more document-specific."
        if later_more_specific
        else "Later layers do not show a consistent increase in document specificity."
    )
    phase_ranges = {"early_0_7": (0, 7), "middle_8_15": (8, 15), "late_16_23": (16, 23)}
    trend["phase_means"] = {}
    for phase, (first, last) in phase_ranges.items():
        rows = [row for row in summaries if first <= row["layer"] <= last]
        if rows:
            trend["phase_means"][phase] = {
                name: float(np.mean([row[f"{name}_mean"] for row in rows]))
                for name in metrics
            }
    peak_row = max(summaries, key=lambda row: row["document_specificity_mean"])
    trend["peak_document_specificity"] = {
        "layer": peak_row["layer"],
        "value": peak_row["document_specificity_mean"],
    }
    return trend


def plot_results(summaries: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [row["layer"] for row in summaries]
    plots = [
        ("document_entropy", "Document Entropy (lower = specific)"),
        ("embedding_cluster_purity", "Embedding Cluster Purity"),
        ("topic_purity", "Topic Purity"),
        ("document_specificity", "Document Specificity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, (metric, title) in zip(axes.flat, plots):
        means = [row[f"{metric}_mean"] for row in summaries]
        sems = [row[f"{metric}_sem"] for row in summaries]
        ax.errorbar(layers, means, yerr=sems, marker="o", linewidth=2, capsize=3)
        ax.set_title(title)
        ax.set_xlabel("Decoder Layer")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Experiment 2: Layer Semantic Purity", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_semantic_purity.png", dpi=200)
    fig.savefig(output_dir / "layer_semantic_purity.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: Layer Semantic Purity")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/activation_cache"))
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
        "--corpus-path",
        type=Path,
        default=Path("dataset/nq320k/corpus_lite.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/layer_semantic_purity"),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-latents", type=int, default=64)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--top-documents", type=int, default=20)
    parser.add_argument("--min-active-tokens", type=int, default=10)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--n-topics", type=int, default=50)
    parser.add_argument("--topic-max-features", type=int, default=20_000)
    parser.add_argument("--document-aggregation", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--sae-format", choices=("inference", "training", "auto"), default="inference"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    for name in (
        "batch_size",
        "n_latents",
        "candidate_pool_size",
        "top_documents",
        "min_active_tokens",
        "n_topics",
        "topic_max_features",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.n_latents > args.candidate_pool_size:
        parser.error("--n-latents cannot exceed --candidate-pool-size")
    if args.activation_threshold < 0:
        parser.error("--activation-threshold must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    available_layers = discover_layers(args.cache_dir, args.checkpoint_root)
    if args.layers is None:
        layers = available_layers
    else:
        missing = sorted(set(args.layers) - set(available_layers))
        if missing:
            raise FileNotFoundError(f"Missing cache/checkpoint layers: {missing}")
        layers = sorted(set(args.layers))

    with open(args.output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "resolved_layers": layers,
                "latent_sampling": (
                    "fixed-seed random sample from latents ranked in the top "
                    "candidate pool by total activation mass"
                ),
                "document_entropy_normalization": "log(number of unique dev documents)",
                "embedding_cluster_label": "first component of BERT semantic DocID",
                "topic_label": "TF-IDF + MiniBatchKMeans over unique dev documents",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    semantic_ids, _, unique_doc_ids, activation_local_docs = (
        load_activation_document_mapping(args.dev_data, args.semantic_id_path)
    )
    print(
        f"Mapped {len(activation_local_docs)} activation rows to "
        f"{len(unique_doc_ids)} unique documents"
    )
    topic_labels, topic_names = build_topic_labels(
        args.corpus_path,
        unique_doc_ids,
        args.n_topics,
        args.topic_max_features,
        args.seed,
        args.output_dir,
    )
    with open(args.corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    details_path = args.output_dir / "sampled_latent_details.jsonl"
    summaries: list[dict] = []
    with open(details_path, "w", encoding="utf-8") as details_file:
        for layer in layers:
            summary, details = analyze_layer(
                layer=layer,
                cache_dir=args.cache_dir,
                checkpoint_root=args.checkpoint_root,
                activation_local_docs=activation_local_docs,
                unique_doc_ids=unique_doc_ids,
                semantic_ids=semantic_ids,
                topic_labels=topic_labels,
                topic_names=topic_names,
                corpus=corpus,
                device=device,
                sae_format=args.sae_format,
                batch_size=args.batch_size,
                n_latents=args.n_latents,
                candidate_pool_size=args.candidate_pool_size,
                top_documents=args.top_documents,
                min_active_tokens=args.min_active_tokens,
                activation_threshold=args.activation_threshold,
                document_aggregation=args.document_aggregation,
                seed=args.seed,
            )
            summaries.append(summary)
            for detail in details:
                details_file.write(json.dumps(detail, ensure_ascii=False) + "\n")
            details_file.flush()
            save_layer_summaries(summaries, args.output_dir)
            print(
                f"Layer {layer:02d}: entropy={summary['document_entropy_mean']:.4f}, "
                f"embedding_purity={summary['embedding_cluster_purity_mean']:.4f}, "
                f"topic_purity={summary['topic_purity_mean']:.4f}, "
                f"specificity={summary['document_specificity_mean']:.4f}"
            )

    trend = compute_trends(summaries)
    with open(args.output_dir / "trend_summary.json", "w", encoding="utf-8") as f:
        json.dump(trend, f, ensure_ascii=False, indent=2)
    plot_results(summaries, args.output_dir)
    print(f"\n{trend['interpretation']}")
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
