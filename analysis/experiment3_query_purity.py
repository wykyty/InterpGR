"""Experiment 3: layer-wise query purity of SAE latents.

For each decoder layer, this script reproducibly samples latents from a
high-activation candidate pool, aggregates every latent across all DocID
positions belonging to the same query, and reports the top activated queries.

Metrics
-------
Semantic Similarity
    Activation-weighted mean pairwise cosine similarity among the top queries.
    Query embeddings use masked-mean representations from the fine-tuned DSI
    T5-large encoder, directly measuring similarity in the retrieval model's
    query-representation space.
Topic Purity
    Activation-weighted dominant-topic share among the top queries. Query topic
    labels are independently fitted with TF-IDF + MiniBatchKMeans.

Example
-------
uv run python analysis/experiment3_query_purity.py \
    --cache-dir data/activation_cache \
    --checkpoint-root out/sae_train_8x \
    --output-dir results/layer_query_purity
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
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


def load_query_activation_mapping(
    dev_data_path: Path,
    semantic_id_path: Path,
) -> tuple[list[str], list[int], torch.Tensor]:
    """Reconstruct cache row -> query index using every DocID position."""
    with open(dev_data_path, encoding="utf-8") as f:
        dev_data = json.load(f)
    with open(semantic_id_path, encoding="utf-8") as f:
        semantic_ids = json.load(f)

    queries: list[str] = []
    query_doc_ids: list[int] = []
    activation_query_indices: list[int] = []
    for raw_query, raw_doc_id in dev_data:
        query = raw_query[0] if isinstance(raw_query, list) else raw_query
        query = str(query).strip()
        if not query:
            continue
        doc_id = unwrap_doc_id(raw_doc_id)
        query_index = len(queries)
        queries.append(query)
        query_doc_ids.append(doc_id)
        activation_query_indices.extend(
            [query_index] * len(semantic_ids[doc_id])
        )

    return (
        queries,
        query_doc_ids,
        torch.tensor(activation_query_indices, dtype=torch.long),
    )


def query_fingerprint(queries: list[str]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        digest.update(query.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@torch.inference_mode()
def load_or_encode_query_embeddings(
    queries: list[str],
    base_model: str,
    dsi_checkpoint: Path,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Load normalized embeddings or encode with the fine-tuned DSI encoder."""
    fingerprint = query_fingerprint(queries)
    checkpoint_identity = f"{dsi_checkpoint.resolve()}:{dsi_checkpoint.stat().st_size}"
    if cache_path.is_file():
        with safe_open(str(cache_path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            embeddings = f.get_tensor("embeddings")
        if (
            metadata.get("query_fingerprint") == fingerprint
            and metadata.get("base_model") == base_model
            and metadata.get("dsi_checkpoint") == checkpoint_identity
            and embeddings.shape[0] == len(queries)
        ):
            print(f"Loaded query embeddings from {cache_path}")
            return F.normalize(embeddings.float(), dim=1)
        print("Ignoring stale query embedding cache and rebuilding it")

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    print(f"Encoding {len(queries)} queries with the DSI T5 encoder...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_tokens([f"${i}$" for i in range(30)])
    model = T5ForConditionalGeneration.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))
    state_dict = torch.load(dsi_checkpoint, map_location="cpu", weights_only=True)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=False)
    encoder = model.encoder.to(device).eval()
    del model, state_dict
    batches: list[torch.Tensor] = []
    for start in tqdm(
        range(0, len(queries), batch_size), desc="Encoding query semantics"
    ):
        encoded = tokenizer(
            queries[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = encoder(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
        ).last_hidden_state
        mask = encoded.attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        batches.append(F.normalize(pooled.float(), dim=1).cpu())
    embeddings = torch.cat(batches, dim=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"embeddings": embeddings.contiguous()},
        str(cache_path),
        metadata={
            "base_model": base_model,
            "dsi_checkpoint": checkpoint_identity,
            "query_fingerprint": fingerprint,
            "pooling": "attention-mask mean over DSI T5 encoder hidden states",
            "normalized": "true",
        },
    )
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved query embeddings to {cache_path}")
    return embeddings


def build_query_topics(
    queries: list[str],
    n_topics: int,
    max_features: int,
    seed: int,
    output_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Fit independent lexical topic labels for query instances."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(queries)
    actual_topics = min(n_topics, len(queries))
    model = MiniBatchKMeans(
        n_clusters=actual_topics,
        random_state=seed,
        batch_size=min(1024, len(queries)),
        n_init=10,
        max_iter=200,
    )
    labels = model.fit_predict(tfidf).astype(np.int64)
    terms = np.asarray(vectorizer.get_feature_names_out())
    top_terms: list[list[str]] = []
    for center in model.cluster_centers_:
        indices = np.argsort(center)[-10:][::-1]
        top_terms.append(terms[indices].tolist())

    with open(output_dir / "query_topic_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "TF-IDF + MiniBatchKMeans",
                "n_queries": len(queries),
                "n_topics": actual_topics,
                "max_features": max_features,
                "top_terms": top_terms,
                "topic_sizes": dict(sorted(Counter(map(int, labels)).items())),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return labels, [", ".join(items[:5]) for items in top_terms]


def weighted_pairwise_cosine(
    embeddings: np.ndarray, weights: np.ndarray
) -> float:
    """Efficient weighted mean cosine over distinct unordered query pairs."""
    if len(embeddings) <= 1:
        return 1.0 if len(embeddings) == 1 else 0.0
    normalized_weights = weights.astype(np.float64)
    normalized_weights /= normalized_weights.sum()
    vectors = embeddings.astype(np.float64, copy=False)
    weighted_centroid = np.sum(normalized_weights[:, None] * vectors, axis=0)
    diagonal_mass = float(
        np.sum(np.square(normalized_weights) * np.sum(np.square(vectors), axis=1))
    )
    denominator = 1.0 - float(np.sum(np.square(normalized_weights)))
    if denominator <= 0:
        return 1.0
    similarity = (float(np.dot(weighted_centroid, weighted_centroid)) - diagonal_mass) / denominator
    return float(np.clip(similarity, -1.0, 1.0))


@torch.inference_mode()
def analyze_layer(
    layer: int,
    cache_dir: Path,
    checkpoint_root: Path,
    activation_query_indices: torch.Tensor,
    queries: list[str],
    query_doc_ids: list[int],
    query_embeddings: np.ndarray,
    topic_labels: np.ndarray,
    topic_names: list[str],
    device: torch.device,
    sae_format: str,
    batch_size: int,
    n_latents: int,
    candidate_pool_size: int,
    top_queries: int,
    min_active_tokens: int,
    activation_threshold: float,
    query_aggregation: str,
    seed: int,
) -> tuple[dict, list[dict]]:
    cache_path = cache_dir / f"layer_{layer}.safetensors"
    cached = load_file(str(cache_path))
    activations = cached["activations"]
    if activations.ndim != 2:
        raise ValueError(f"Layer {layer}: invalid activation shape {activations.shape}")
    if activations.shape[0] != len(activation_query_indices):
        raise ValueError(
            f"Layer {layer}: cache has {activations.shape[0]} rows but query "
            f"mapping has {len(activation_query_indices)}"
        )

    sae, sae_cfg, loaded_format = load_sae(
        checkpoint_root / f"layer_{layer}", device, sae_format
    )
    d_sae = int(sae_cfg["d_sae"])
    if activations.shape[1] != int(sae_cfg["d_in"]):
        raise ValueError(f"Layer {layer}: cache and SAE d_in do not match")

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

    n_queries = len(queries)
    selected = torch.tensor(sampled_latents, dtype=torch.long, device=device)
    if query_aggregation == "mean":
        query_scores = torch.zeros((n_queries, n_latents), device=device)
        query_counts = torch.bincount(
            activation_query_indices, minlength=n_queries
        ).to(device=device, dtype=torch.float32)
    else:
        query_scores = torch.full(
            (n_queries, n_latents), -torch.inf, device=device
        )
        query_counts = None

    for start in tqdm(
        range(0, len(activations), batch_size),
        desc=f"Layer {layer:02d} query pass",
    ):
        end = min(start + batch_size, len(activations))
        batch = activations[start:end].to(device, non_blocking=True)
        selected_acts = sae.encode(batch).float().index_select(1, selected)
        query_indices = activation_query_indices[start:end].to(
            device, non_blocking=True
        )
        if query_aggregation == "mean":
            query_scores.index_add_(0, query_indices, selected_acts)
        else:
            scatter_index = query_indices[:, None].expand(-1, n_latents)
            query_scores.scatter_reduce_(
                0, scatter_index, selected_acts, reduce="amax", include_self=True
            )
    if query_aggregation == "mean":
        query_scores /= query_counts.clamp_min(1.0)[:, None]
    else:
        query_scores.masked_fill_(torch.isinf(query_scores), 0.0)
    scores_np = query_scores.cpu().numpy()

    semantic_values: list[float] = []
    topic_values: list[float] = []
    combined_values: list[float] = []
    active_query_counts: list[float] = []
    details: list[dict] = []
    for feature_column, feature_id in enumerate(sampled_latents):
        scores = scores_np[:, feature_column]
        positive_indices = np.where(scores > 0)[0]
        k = min(top_queries, len(positive_indices))
        if k:
            positive_scores = scores[positive_indices]
            top_indices = positive_indices[np.argsort(positive_scores)[-k:][::-1]]
            weights = scores[top_indices]
            semantic_similarity = weighted_pairwise_cosine(
                query_embeddings[top_indices], weights
            )
            topic_purity, dominant_topic = weighted_purity(
                topic_labels[top_indices], weights
            )
        else:
            top_indices = np.asarray([], dtype=np.int64)
            semantic_similarity, topic_purity, dominant_topic = 0.0, 0.0, -1
        combined_purity = (semantic_similarity + topic_purity) / 2.0
        semantic_values.append(semantic_similarity)
        topic_values.append(topic_purity)
        combined_values.append(combined_purity)
        active_query_counts.append(float(len(positive_indices)))

        top_query_records = []
        for query_index in top_indices:
            index = int(query_index)
            topic = int(topic_labels[index])
            top_query_records.append(
                {
                    "query_index": index,
                    "query": re.sub(r"\s+", " ", queries[index]).strip(),
                    "doc_id": query_doc_ids[index],
                    "activation": float(scores[index]),
                    "topic_cluster": topic,
                    "topic_terms": topic_names[topic],
                }
            )
        details.append(
            {
                "layer": layer,
                "feature_id": int(feature_id),
                "candidate_activation_mass": float(activation_mass[feature_id]),
                "active_token_count": int(active_token_count[feature_id]),
                "active_query_count": len(positive_indices),
                "semantic_similarity": semantic_similarity,
                "topic_purity": topic_purity,
                "dominant_topic": dominant_topic,
                "dominant_topic_terms": (
                    topic_names[dominant_topic] if dominant_topic >= 0 else ""
                ),
                "query_semantic_purity": combined_purity,
                "top_activated_queries": top_query_records,
            }
        )

    semantic_mean, semantic_sem = summarize_values(semantic_values)
    topic_mean, topic_sem = summarize_values(topic_values)
    combined_mean, combined_sem = summarize_values(combined_values)
    active_mean, active_sem = summarize_values(active_query_counts)
    summary = {
        "layer": layer,
        "n_tokens": len(activations),
        "n_queries": n_queries,
        "d_sae": d_sae,
        "sae_format": loaded_format,
        "sampled_latents": n_latents,
        "candidate_pool_size": pool_size,
        "query_aggregation": query_aggregation,
        "semantic_similarity_mean": semantic_mean,
        "semantic_similarity_sem": semantic_sem,
        "topic_purity_mean": topic_mean,
        "topic_purity_sem": topic_sem,
        "query_semantic_purity_mean": combined_mean,
        "query_semantic_purity_sem": combined_sem,
        "active_queries_mean": active_mean,
        "active_queries_sem": active_sem,
    }

    del sae, activations, cached, query_scores
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, details


def save_summaries(summaries: list[dict], output_dir: Path) -> None:
    with open(output_dir / "layer_query_purity.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    with open(
        output_dir / "layer_query_purity.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def compute_trends(summaries: list[dict]) -> dict:
    layers = np.asarray([row["layer"] for row in summaries], dtype=np.float64)
    metrics = {
        "semantic_similarity": np.asarray(
            [row["semantic_similarity_mean"] for row in summaries],
            dtype=np.float64,
        ),
        "topic_purity": np.asarray(
            [row["topic_purity_mean"] for row in summaries], dtype=np.float64
        ),
        "query_semantic_purity": np.asarray(
            [row["query_semantic_purity_mean"] for row in summaries],
            dtype=np.float64,
        ),
    }
    trends = {}
    for name, values in metrics.items():
        slope, correlation = linear_trend(layers, values)
        trends[name] = {
            "slope_per_layer": slope,
            "layer_correlation": correlation,
        }
    increasingly_clustered = (
        trends["semantic_similarity"]["slope_per_layer"] > 0
        and trends["topic_purity"]["slope_per_layer"] > 0
    )
    trends["query_semantics_increasingly_clustered"] = increasingly_clustered
    trends["interpretation"] = (
        "Query semantics become increasingly clustered in later layers."
        if increasingly_clustered
        else "Query semantics do not show a consistent increase in clustering."
    )
    phases = {"early_0_7": (0, 7), "middle_8_15": (8, 15), "late_16_23": (16, 23)}
    trends["phase_means"] = {}
    for phase, (first, last) in phases.items():
        rows = [row for row in summaries if first <= row["layer"] <= last]
        if rows:
            trends["phase_means"][phase] = {
                name: float(np.mean([row[f"{name}_mean"] for row in rows]))
                for name in metrics
            }
    peak = max(summaries, key=lambda row: row["query_semantic_purity_mean"])
    trends["peak_query_semantic_purity"] = {
        "layer": peak["layer"],
        "value": peak["query_semantic_purity_mean"],
    }
    return trends


def plot_results(summaries: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [row["layer"] for row in summaries]
    plots = [
        ("semantic_similarity", "Top-query Semantic Similarity"),
        ("topic_purity", "Top-query Topic Purity"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, (metric, title) in zip(axes, plots):
        means = [row[f"{metric}_mean"] for row in summaries]
        sems = [row[f"{metric}_sem"] for row in summaries]
        ax.errorbar(layers, means, yerr=sems, marker="o", linewidth=2, capsize=3)
        ax.set_title(title)
        ax.set_xlabel("Decoder Layer")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Experiment 3: Layer Query Purity", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_query_purity.png", dpi=200)
    fig.savefig(output_dir / "layer_query_purity.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 3: Layer Query Purity")
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
        "--output-dir", type=Path, default=Path("results/layer_query_purity")
    )
    parser.add_argument(
        "--query-embedding-cache",
        type=Path,
        default=Path("data/analysis_cache/dev_query_embeddings_dsi_t5.safetensors"),
    )
    parser.add_argument("--base-model", default="google-t5/t5-large")
    parser.add_argument(
        "--dsi-checkpoint",
        type=Path,
        default=Path("out/dsi-semantic-bert/99.pt"),
    )
    parser.add_argument("--query-embedding-batch-size", type=int, default=32)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-latents", type=int, default=64)
    parser.add_argument("--candidate-pool-size", type=int, default=512)
    parser.add_argument("--top-queries", type=int, default=20)
    parser.add_argument("--min-active-tokens", type=int, default=10)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--n-topics", type=int, default=50)
    parser.add_argument("--topic-max-features", type=int, default=20_000)
    parser.add_argument("--query-aggregation", choices=("max", "mean"), default="max")
    parser.add_argument(
        "--sae-format", choices=("inference", "training", "auto"), default="inference"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    positive_names = (
        "batch_size",
        "query_embedding_batch_size",
        "query_max_length",
        "n_latents",
        "candidate_pool_size",
        "top_queries",
        "min_active_tokens",
        "n_topics",
        "topic_max_features",
    )
    for name in positive_names:
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

    queries, query_doc_ids, activation_query_indices = load_query_activation_mapping(
        args.dev_data, args.semantic_id_path
    )
    print(
        f"Mapped {len(activation_query_indices)} activation rows to "
        f"{len(queries)} query instances"
    )
    query_embeddings = load_or_encode_query_embeddings(
        queries=queries,
        base_model=args.base_model,
        dsi_checkpoint=args.dsi_checkpoint,
        cache_path=args.query_embedding_cache,
        device=device,
        batch_size=args.query_embedding_batch_size,
        max_length=args.query_max_length,
    ).numpy()
    topic_labels, topic_names = build_query_topics(
        queries, args.n_topics, args.topic_max_features, args.seed, args.output_dir
    )

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
                "semantic_similarity": (
                    "activation-weighted mean pairwise cosine among top queries "
                    "using masked-mean DSI T5 encoder representations"
                ),
                "topic_label": "TF-IDF + MiniBatchKMeans over query instances",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    details_path = args.output_dir / "sampled_latent_query_details.jsonl"
    summaries: list[dict] = []
    with open(details_path, "w", encoding="utf-8") as details_file:
        for layer in layers:
            summary, details = analyze_layer(
                layer=layer,
                cache_dir=args.cache_dir,
                checkpoint_root=args.checkpoint_root,
                activation_query_indices=activation_query_indices,
                queries=queries,
                query_doc_ids=query_doc_ids,
                query_embeddings=query_embeddings,
                topic_labels=topic_labels,
                topic_names=topic_names,
                device=device,
                sae_format=args.sae_format,
                batch_size=args.batch_size,
                n_latents=args.n_latents,
                candidate_pool_size=args.candidate_pool_size,
                top_queries=args.top_queries,
                min_active_tokens=args.min_active_tokens,
                activation_threshold=args.activation_threshold,
                query_aggregation=args.query_aggregation,
                seed=args.seed,
            )
            summaries.append(summary)
            for detail in details:
                details_file.write(json.dumps(detail, ensure_ascii=False) + "\n")
            details_file.flush()
            save_summaries(summaries, args.output_dir)
            print(
                f"Layer {layer:02d}: semantic_similarity="
                f"{summary['semantic_similarity_mean']:.4f}, "
                f"topic_purity={summary['topic_purity_mean']:.4f}"
            )

    trends = compute_trends(summaries)
    with open(args.output_dir / "trend_summary.json", "w", encoding="utf-8") as f:
        json.dump(trends, f, ensure_ascii=False, indent=2)
    plot_results(summaries, args.output_dir)
    print(f"\n{trends['interpretation']}")
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
