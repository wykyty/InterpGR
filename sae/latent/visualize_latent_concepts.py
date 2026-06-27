"""
Visualize Latent Concepts Analysis Results

Creates visualizations for the latent concepts analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_analysis_results(analysis_path: str) -> dict:
    """Load analysis results from JSON file."""
    with open(analysis_path, "r") as f:
        return json.load(f)


def plot_feature_frequency_distribution(feature_freq: dict, output_path: str):
    """Plot distribution of feature activation frequencies."""
    frequencies = list(feature_freq.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(frequencies, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Activation Frequency')
    axes[0].set_ylabel('Number of Features')
    axes[0].set_title('Distribution of Feature Activation Frequencies')
    axes[0].axvline(x=0.5, color='r', linestyle='--', label='50% threshold')
    axes[0].legend()

    # Log scale histogram
    axes[1].hist(frequencies, bins=50, edgecolor='black', alpha=0.7, log=True)
    axes[1].set_xlabel('Activation Frequency')
    axes[1].set_ylabel('Number of Features (log scale)')
    axes[1].set_title('Distribution of Feature Activation Frequencies (Log Scale)')
    axes[1].axvline(x=0.5, color='r', linestyle='--', label='50% threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved feature frequency distribution to {output_path}")


def plot_activation_strength_distribution(global_stats: dict, output_path: str):
    """Plot activation strength statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot of statistics
    stats_names = ['Mean', 'Std', 'Min', 'Max', 'Median']
    stats_values = [
        global_stats['mean'],
        global_stats['std'],
        global_stats['min'],
        global_stats['max'],
        global_stats['median']
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    bars = axes[0].bar(stats_names, stats_values, color=colors, edgecolor='black')
    axes[0].set_ylabel('Activation Value')
    axes[0].set_title('Activation Strength Statistics')
    axes[0].set_yscale('log')

    # Add value labels on bars
    for bar, val in zip(bars, stats_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Violin plot representation
    data = [global_stats['min'], global_stats['median'],
            global_stats['mean'], global_stats['max']]
    axes[1].violinplot(data, showmeans=True, showmedians=True)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(['Activation Values'])
    axes[1].set_ylabel('Activation Value')
    axes[1].set_title('Activation Strength Distribution')
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved activation strength distribution to {output_path}")


def plot_token_patterns(token_stats: dict, output_path: str):
    """Plot token-level activation patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot of token statistics
    stats_names = ['Avg', 'Std', 'Max', 'Min']
    stats_values = [
        token_stats['avg_active_features'],
        token_stats['std_active_features'],
        token_stats['max_active_features'],
        token_stats['min_active_features']
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    bars = axes[0].bar(stats_names, stats_values, color=colors, edgecolor='black')
    axes[0].set_ylabel('Number of Active Features')
    axes[0].set_title('Token-Level Activation Statistics')

    # Add value labels
    for bar, val in zip(bars, stats_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom')

    # Pie chart of feature categories
    categories = ['Always On\n(>99%)', 'Often On\n(50-99%)', 'Sometimes\n(10-50%)',
                  'Rarely On\n(0-10%)', 'Dead\n(0%)']
    sizes = [7, 26, 24, 3642, 4493]
    colors_pie = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#95a5a6']
    explode = (0.1, 0, 0, 0, 0)

    axes[1].pie(sizes, explode=explode, labels=categories, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90)
    axes[1].set_title('Feature Activation Categories')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved token patterns to {output_path}")


def plot_co_occurrence_heatmap(co_occurrence_data: list, output_path: str, top_n: int = 10):
    """Plot co-occurrence heatmap for top features."""
    # Get top features
    top_pairs = co_occurrence_data[:top_n]
    features = set()
    for pair in top_pairs:
        features.add(pair['features'][0])
        features.add(pair['features'][1])
    features = sorted(list(features))[:top_n]

    # Build co-occurrence matrix
    n = len(features)
    matrix = np.zeros((n, n))
    feature_to_idx = {f: i for i, f in enumerate(features)}

    for pair in co_occurrence_data:
        f1, f2 = pair['features']
        if f1 in feature_to_idx and f2 in feature_to_idx:
            i, j = feature_to_idx[f1], feature_to_idx[f2]
            matrix[i][j] = pair['count']
            matrix[j][i] = pair['count']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd')

    # Add labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([f'F{f}' for f in features], rotation=45, ha='right')
    ax.set_yticklabels([f'F{f}' for f in features])

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Co-occurrence Count')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                text = ax.text(j, i, f'{int(matrix[i, j])}',
                              ha='center', va='center', color='black', fontsize=8)

    ax.set_title('Feature Co-occurrence Heatmap (Top Features)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved co-occurrence heatmap to {output_path}")


def plot_top_features_comparison(analysis_data: dict, output_path: str):
    """Plot comparison of top features by different metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top features by frequency
    freq_sorted = sorted(analysis_data['feature_frequency']['feature_freq'].items(),
                        key=lambda x: x[1], reverse=True)[:10]
    features_freq = [f'F{f[0]}' for f in freq_sorted]
    values_freq = [f[1] * 100 for f in freq_sorted]

    axes[0, 0].barh(features_freq, values_freq, color='#3498db')
    axes[0, 0].set_xlabel('Activation Frequency (%)')
    axes[0, 0].set_title('Top 10 Features by Frequency')
    axes[0, 0].invert_yaxis()

    # Top features by strength
    strength_sorted = analysis_data['activation_strength']['top_by_strength'][:10]
    features_str = [f'F{f["feature"]}' for f in strength_sorted]
    values_str = [f['avg_weight'] for f in strength_sorted]

    axes[0, 1].barh(features_str, values_str, color='#e74c3c')
    axes[0, 1].set_xlabel('Average Activation Strength')
    axes[0, 1].set_title('Top 10 Features by Strength')
    axes[0, 1].invert_yaxis()

    # Feature activation distribution
    freq_values = list(analysis_data['feature_frequency']['feature_freq'].values())
    axes[1, 0].hist(freq_values, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Activation Frequency')
    axes[1, 0].set_ylabel('Number of Features')
    axes[1, 0].set_title('Feature Frequency Distribution')
    axes[1, 0].axvline(x=0.5, color='r', linestyle='--', label='50%')
    axes[1, 0].legend()

    # Token activation distribution
    token_stats = analysis_data['token_patterns']['stats']
    categories = ['Avg', 'Std', 'Max', 'Min']
    values = [token_stats['avg_active_features'], token_stats['std_active_features'],
              token_stats['max_active_features'], token_stats['min_active_features']]

    axes[1, 1].bar(categories, values, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    axes[1, 1].set_ylabel('Number of Active Features')
    axes[1, 1].set_title('Token Activation Statistics')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved top features comparison to {output_path}")


def main():
    # Paths
    analysis_path = "/home/zyq/wyk/InterpGR/results/latent_concepts/layer_12_analysis.json"
    output_dir = "/home/zyq/wyk/InterpGR/results/latent_concepts/visualizations"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load analysis results
    print("Loading analysis results...")
    analysis = load_analysis_results(analysis_path)

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_feature_frequency_distribution(
        analysis['feature_frequency']['feature_freq'],
        f"{output_dir}/feature_frequency_distribution.png"
    )

    plot_activation_strength_distribution(
        analysis['activation_strength']['global_stats'],
        f"{output_dir}/activation_strength_distribution.png"
    )

    plot_token_patterns(
        analysis['token_patterns']['stats'],
        f"{output_dir}/token_patterns.png"
    )

    plot_co_occurrence_heatmap(
        analysis['co_occurrence']['top_co_occurring_pairs'],
        f"{output_dir}/co_occurrence_heatmap.png"
    )

    plot_top_features_comparison(
        analysis,
        f"{output_dir}/top_features_comparison.png"
    )

    print(f"\n✅ All visualizations saved to {output_dir}")
    print("\nGenerated files:")
    print("  1. feature_frequency_distribution.png")
    print("  2. activation_strength_distribution.png")
    print("  3. token_patterns.png")
    print("  4. co_occurrence_heatmap.png")
    print("  5. top_features_comparison.png")


if __name__ == "__main__":
    main()