"""
Example: Feature Analysis with Latent Concepts

This script demonstrates how to use extracted latent concepts
for feature interpretability analysis.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def load_concepts(concepts_path: str) -> list[dict]:
    """Load latent concepts from JSONL file."""
    concepts = []
    with open(concepts_path, "r", encoding="utf-8") as f:
        for line in f:
            concepts.append(json.loads(line.strip()))
    return concepts


def get_feature_contexts(concepts: list[dict], feature_id: int, top_k: int = 10) -> list[dict]:
    """
    Get contexts where a specific feature is most active.

    Args:
        concepts: List of concept dictionaries
        feature_id: ID of the feature to analyze
        top_k: Number of top contexts to return

    Returns:
        List of contexts sorted by activation strength
    """
    contexts = []

    for concept in concepts:
        if feature_id in concept["ids"]:
            idx = concept["ids"].index(feature_id)
            contexts.append({
                "docid": concept["docid"],
                "activation": concept["weight"][idx],
                "all_features": concept["ids"],
                "all_weights": concept["weight"],
            })

    # Sort by activation strength
    contexts.sort(key=lambda x: x["activation"], reverse=True)
    return contexts[:top_k]


def find_coactivated_features(concepts: list[dict], feature_id: int, top_k: int = 10) -> list[dict]:
    """
    Find features that frequently co-activate with a given feature.

    Args:
        concepts: List of concept dictionaries
        feature_id: ID of the feature to analyze
        top_k: Number of top co-activated features to return

    Returns:
        List of co-activated features with their co-occurrence counts
    """
    co_occurrence = defaultdict(int)
    total_occurrences = 0

    for concept in concepts:
        if feature_id in concept["ids"]:
            total_occurrences += 1
            for other_feature in concept["ids"]:
                if other_feature != feature_id:
                    co_occurrence[other_feature] += 1

    # Convert to frequency
    co_freq = {
        feat: count / total_occurrences
        for feat, count in co_occurrence.items()
    }

    # Sort by frequency
    sorted_features = sorted(co_freq.items(), key=lambda x: x[1], reverse=True)

    return [
        {"feature": feat, "frequency": freq, "count": co_occurrence[feat]}
        for feat, freq in sorted_features[:top_k]
    ]


def analyze_feature_specialization(concepts: list[dict], feature_id: int) -> dict:
    """
    Analyze how specialized a feature is.

    A feature is specialized if it activates in specific contexts
    rather than everywhere.
    """
    contexts_with_feature = []
    contexts_without_feature = []

    for concept in concepts:
        if feature_id in concept["ids"]:
            contexts_with_feature.append(concept)
        else:
            contexts_without_feature.append(concept)

    # Calculate statistics
    total_tokens = len(concepts)
    activation_rate = len(contexts_with_feature) / total_tokens

    # Calculate average number of other active features
    avg_other_features_with = np.mean([
        len(c["ids"]) - 1 for c in contexts_with_feature
    ]) if contexts_with_feature else 0

    avg_other_features_without = np.mean([
        len(c["ids"]) for c in contexts_without_feature
    ]) if contexts_without_feature else 0

    return {
        "feature_id": feature_id,
        "activation_rate": activation_rate,
        "total_activations": len(contexts_with_feature),
        "avg_other_features_when_active": avg_other_features_with,
        "avg_other_features_when_inactive": avg_other_features_without,
        "specialization_score": 1.0 - activation_rate,  # Higher = more specialized
    }


def main():
    # Example usage
    concepts_path = "results/latent_concepts/layer_12.jsonl"

    if not Path(concepts_path).exists():
        print(f"❌ Concepts file not found: {concepts_path}")
        print("Please run save_latent_concepts.py first.")
        return

    print("Loading concepts...")
    concepts = load_concepts(concepts_path)
    print(f"Loaded {len(concepts)} token concepts")

    # Example 1: Analyze a specific feature
    feature_id = 1234  # Replace with an actual feature ID
    print(f"\n{'='*60}")
    print(f"ANALYZING FEATURE {feature_id}")
    print(f"{'='*60}")

    # Get contexts where this feature is most active
    top_contexts = get_feature_contexts(concepts, feature_id, top_k=5)
    print(f"\nTop 5 contexts where feature {feature_id} is most active:")
    for i, ctx in enumerate(top_contexts, 1):
        print(f"  {i}. Token {ctx['docid']}: activation={ctx['activation']:.4f}")
        print(f"     Other features: {ctx['all_features'][:5]}...")

    # Find co-activated features
    co_activated = find_coactivated_features(concepts, feature_id, top_k=5)
    print(f"\nTop 5 features that co-activate with feature {feature_id}:")
    for item in co_activated:
        print(f"  Feature {item['feature']}: {item['frequency']:.2%} ({item['count']} times)")

    # Analyze specialization
    specialization = analyze_feature_specialization(concepts, feature_id)
    print(f"\nFeature specialization analysis:")
    print(f"  Activation rate: {specialization['activation_rate']:.2%}")
    print(f"  Specialization score: {specialization['specialization_score']:.2f}")
    print(f"  Avg other features when active: {specialization['avg_other_features_when_active']:.1f}")

    # Example 2: Find most specialized features
    print(f"\n{'='*60}")
    print("FINDING MOST SPECIALIZED FEATURES")
    print(f"{'='*60}")

    # Sample a subset of features for analysis
    all_features = set()
    for concept in concepts[:1000]:  # Use first 1000 tokens for speed
        all_features.update(concept["ids"])

    specialization_scores = []
    for feat_id in list(all_features)[:100]:  # Analyze first 100 features
        spec = analyze_feature_specialization(concepts, feat_id)
        specialization_scores.append((feat_id, spec["specialization_score"]))

    # Sort by specialization
    specialization_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 most specialized features:")
    for feat_id, score in specialization_scores[:10]:
        print(f"  Feature {feat_id}: specialization={score:.2f}")

    # Example 3: Analyze feature co-occurrence patterns
    print(f"\n{'='*60}")
    print("FEATURE CO-OCCURRENCE PATTERNS")
    print(f"{'='*60}")

    # Build co-occurrence matrix for top features
    top_features = [feat_id for feat_id, _ in specialization_scores[:20]]
    co_occurrence_matrix = defaultdict(lambda: defaultdict(int))

    for concept in concepts[:5000]:  # Use first 5000 tokens
        active_features = [f for f in concept["ids"] if f in top_features]
        for i, f1 in enumerate(active_features):
            for f2 in active_features[i+1:]:
                co_occurrence_matrix[f1][f2] += 1
                co_occurrence_matrix[f2][f1] += 1

    print(f"\nCo-occurrence between top specialized features:")
    for f1 in top_features[:5]:
        top_co = sorted(co_occurrence_matrix[f1].items(), key=lambda x: x[1], reverse=True)[:3]
        if top_co:
            print(f"  Feature {f1} co-occurs with:")
            for f2, count in top_co:
                print(f"    Feature {f2}: {count} times")


if __name__ == "__main__":
    main()