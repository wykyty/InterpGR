"""
Analyze Saved Latent Concepts

Provides utilities for analyzing and visualizing latent concepts extracted from SAE.

Usage:
    uv run python sae/analyze_latent_concepts.py \
        --input_path results/latent_concepts/layer_12.jsonl \
        --statistics_path results/latent_concepts/layer_12_statistics.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_concepts(input_path: str) -> list[dict]:
    """Load latent concepts from JSONL file."""
    concepts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            concepts.append(json.loads(line.strip()))
    return concepts


def load_statistics(stats_path: str) -> dict:
    """Load statistics from JSON file."""
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_feature_frequency(concepts: list[dict]) -> dict:
    """Analyze feature activation frequency across all tokens."""
    feature_counts = Counter()
    total_tokens = len(concepts)

    for concept in concepts:
        for feat_id in concept["ids"]:
            feature_counts[feat_id] += 1

    # Convert to frequency
    feature_freq = {feat_id: count / total_tokens for feat_id, count in feature_counts.items()}

    # Sort by frequency
    sorted_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)

    return {
        "feature_counts": dict(feature_counts),
        "feature_freq": feature_freq,
        "sorted_features": sorted_features,
        "total_tokens": total_tokens,
    }


def analyze_activation_patterns(concepts: list[dict]) -> dict:
    """Analyze activation patterns and co-occurrence."""
    # Feature co-occurrence matrix (sparse representation)
    co_occurrence = defaultdict(int)
    feature_pairs = Counter()

    for concept in concepts:
        features = concept["ids"]
        weights = concept["weight"]

        # Update co-occurrence for feature pairs
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                pair = (min(features[i], features[j]), max(features[i], features[j]))
                feature_pairs[pair] += 1

        # Update co-occurrence counts
        for feat in features:
            co_occurrence[feat] += 1

    # Get top co-occurring pairs
    top_pairs = feature_pairs.most_common(20)

    return {
        "co_occurrence": dict(co_occurrence),
        "top_co_occurring_pairs": [
            {"features": pair, "count": count} for pair, count in top_pairs
        ],
    }


def analyze_activation_strength(concepts: list[dict]) -> dict:
    """Analyze activation strength distributions."""
    all_weights = []
    feature_avg_weights = defaultdict(list)

    for concept in concepts:
        for feat_id, weight in zip(concept["ids"], concept["weight"]):
            all_weights.append(weight)
            feature_avg_weights[feat_id].append(weight)

    # Compute statistics
    all_weights = np.array(all_weights)
    feature_avg = {
        feat_id: np.mean(weights) for feat_id, weights in feature_avg_weights.items()
    }

    # Get features with highest average activation
    top_by_strength = sorted(feature_avg.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "global_stats": {
            "mean": float(np.mean(all_weights)),
            "std": float(np.std(all_weights)),
            "min": float(np.min(all_weights)),
            "max": float(np.max(all_weights)),
            "median": float(np.median(all_weights)),
        },
        "feature_avg_weights": feature_avg,
        "top_by_strength": [
            {"feature": feat, "avg_weight": avg} for feat, avg in top_by_strength
        ],
    }


def analyze_token_patterns(concepts: list[dict]) -> dict:
    """Analyze patterns in token-level activations."""
    num_active_per_token = [len(concept["ids"]) for concept in concepts]
    total_activation_per_token = [sum(concept["weight"]) for concept in concepts]

    # Tokens with most active features
    most_active_tokens = sorted(
        enumerate(num_active_per_token), key=lambda x: x[1], reverse=True
    )[:20]

    # Tokens with highest total activation
    highest_activation_tokens = sorted(
        enumerate(total_activation_per_token), key=lambda x: x[1], reverse=True
    )[:20]

    return {
        "stats": {
            "avg_active_features": np.mean(num_active_per_token),
            "std_active_features": np.std(num_active_per_token),
            "max_active_features": max(num_active_per_token),
            "min_active_features": min(num_active_per_token),
        },
        "most_active_tokens": [
            {"token_idx": idx, "num_active": count} for idx, count in most_active_tokens
        ],
        "highest_activation_tokens": [
            {"token_idx": idx, "total_activation": total}
            for idx, total in highest_activation_tokens
        ],
    }


def generate_report(concepts: list[dict], statistics: dict = None) -> str:
    """Generate a comprehensive analysis report."""
    report = []
    report.append("=" * 60)
    report.append("LATENT CONCEPTS ANALYSIS REPORT")
    report.append("=" * 60)

    # Basic statistics
    report.append(f"\n📊 BASIC STATISTICS")
    report.append(f"  Total tokens analyzed: {len(concepts)}")
    if statistics:
        report.append(f"  Total features: {statistics['total_features']}")
        report.append(f"  Dead features: {statistics['dead_features']} ({statistics['dead_ratio']:.2%})")
        report.append(f"  Average active features per token: {statistics['avg_active_features']:.1f}")

    # Feature frequency analysis
    freq_analysis = analyze_feature_frequency(concepts)
    report.append(f"\n📈 FEATURE FREQUENCY ANALYSIS")
    report.append(f"  Unique features activated: {len(freq_analysis['feature_freq'])}")
    report.append(f"  Top 5 most frequent features:")
    for feat_id, freq in freq_analysis["sorted_features"][:5]:
        report.append(f"    Feature {feat_id}: {freq:.4f} ({freq * 100:.1f}%)")

    # Activation strength analysis
    strength_analysis = analyze_activation_strength(concepts)
    report.append(f"\n💪 ACTIVATION STRENGTH ANALYSIS")
    report.append(f"  Global statistics:")
    report.append(f"    Mean: {strength_analysis['global_stats']['mean']:.4f}")
    report.append(f"    Std: {strength_analysis['global_stats']['std']:.4f}")
    report.append(f"    Min: {strength_analysis['global_stats']['min']:.4f}")
    report.append(f"    Max: {strength_analysis['global_stats']['max']:.4f}")
    report.append(f"  Top 5 features by average strength:")
    for item in strength_analysis["top_by_strength"][:5]:
        report.append(f"    Feature {item['feature']}: {item['avg_weight']:.4f}")

    # Token pattern analysis
    token_analysis = analyze_token_patterns(concepts)
    report.append(f"\n🎯 TOKEN PATTERN ANALYSIS")
    report.append(f"  Average active features per token: {token_analysis['stats']['avg_active_features']:.1f}")
    report.append(f"  Max active features in a single token: {token_analysis['stats']['max_active_features']}")
    report.append(f"  Tokens with most active features:")
    for item in token_analysis["most_active_tokens"][:5]:
        report.append(f"    Token {item['token_idx']}: {item['num_active']} features")

    # Co-occurrence analysis
    co_analysis = analyze_activation_patterns(concepts)
    report.append(f"\n🔗 CO-OCCURRENCE ANALYSIS")
    report.append(f"  Top 5 co-occurring feature pairs:")
    for item in co_analysis["top_co_occurring_pairs"][:5]:
        report.append(f"    Features {item['features'][0]} & {item['features'][1]}: {item['count']} times")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze saved latent concepts")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to latent concepts JSONL file")
    parser.add_argument("--statistics_path", type=str, default=None,
                        help="Path to statistics JSON file (optional)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save analysis results (optional)")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top items to show in reports")
    args = parser.parse_args()

    # Load concepts
    print(f"Loading concepts from {args.input_path}...")
    concepts = load_concepts(args.input_path)
    print(f"Loaded {len(concepts)} token concepts")

    # Load statistics if available
    statistics = None
    if args.statistics_path and Path(args.statistics_path).exists():
        print(f"Loading statistics from {args.statistics_path}...")
        statistics = load_statistics(args.statistics_path)

    # Generate report
    print("\nGenerating analysis report...")
    report = generate_report(concepts, statistics)
    print(report)

    # Save detailed analysis if requested
    if args.output_path:
        print(f"\nSaving detailed analysis to {args.output_path}...")
        analysis = {
            "feature_frequency": analyze_feature_frequency(concepts),
            "activation_strength": analyze_activation_strength(concepts),
            "token_patterns": analyze_token_patterns(concepts),
            "co_occurrence": analyze_activation_patterns(concepts),
        }

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        analysis = convert_numpy(analysis)

        import os
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"✅ Detailed analysis saved to {args.output_path}")


if __name__ == "__main__":
    main()