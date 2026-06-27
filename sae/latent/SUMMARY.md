# Latent Concepts Extraction and Analysis Summary

This document summarizes the tools created for extracting and analyzing latent concepts from trained Sparse Autoencoders (SAEs).

## 🎯 Overview

The goal is to extract interpretable feature representations from trained SAEs for downstream interpretability analysis. This follows the approach from `reconstruct_embedding.py` but adapted for the current SAE codebase.

## 📁 Created Files

### Core Scripts

1. **`save_latent_concepts.py`** - Main extraction script
   - Loads trained SAE and activation cache
   - Encodes activations through SAE
   - Saves active features and weights as JSONL
   - Supports metadata and statistics saving

2. **`analyze_latent_concepts.py`** - Comprehensive analysis
   - Feature frequency analysis
   - Activation strength statistics
   - Co-occurrence patterns
   - Token-level analysis
   - Generates detailed reports

3. **`feature_analysis_example.py`** - Example analysis code
   - Demonstrates feature investigation
   - Shows co-activation analysis
   - Includes specialization metrics

### Documentation

4. **`LATENT_CONCEPTS_GUIDE.md`** - Detailed usage guide
5. **`README_latent_concepts.md`** - Project overview
6. **`QUICK_START.md`** - Quick start guide
7. **`SUMMARY.md`** - This file

### Utility Scripts

8. **`run_latent_concepts_analysis.sh`** - Complete pipeline script

## 🚀 Quick Start

### 1. Extract Latent Concepts

```bash
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --threshold 0.01 \
    --save_metadata
```

### 2. Analyze Concepts

```bash
uv run python sae/analyze_latent_concepts.py \
    --input_path results/latent_concepts/layer_12.jsonl \
    --statistics_path results/latent_concepts/layer_12_statistics.json \
    --output_path results/latent_concepts/layer_12_analysis.json
```

### 3. Run Complete Pipeline

```bash
bash sae/run_latent_concepts_analysis.sh
```

## 📊 Output Format

### JSONL Format (save_latent_concepts.py)

```json
{
    "docid": 0,
    "ids": [1, 5, 12, 45],
    "weight": [0.85, 0.42, 0.18, 0.05]
}
```

### Statistics Format

```json
{
    "total_tokens": 100000,
    "total_features": 16384,
    "dead_features": 2345,
    "avg_active_features": 45.2,
    "most_active_features": [...],
    "least_active_features": [...]
}
```

## 🎯 Use Cases

### 1. Feature Interpretability

Identify what each SAE feature represents by examining its activation contexts.

### 2. Feature Co-occurrence Analysis

Discover which features frequently activate together, suggesting functional relationships.

### 3. Specialization Analysis

Determine how specialized each feature is (rare vs. common activations).

### 4. Downstream Task Analysis

Analyze which features are important for specific tasks or document types.

### 5. Ablation Studies

Use extracted features to guide targeted interventions.

## 🔧 Key Features

### Configurable Thresholds

- **Low threshold (0.001)**: More features, comprehensive but noisier
- **High threshold (0.1)**: Fewer features, cleaner but may miss weak activations
- **Default (0.01)**: Balanced approach

### Batch Processing

Support for large datasets with configurable batch sizes and max batches.

### Multiple Layers

Easy processing of multiple layers with simple loop.

### Comprehensive Analysis

- Feature frequency statistics
- Activation strength distributions
- Co-occurrence patterns
- Token-level analysis
- Specialization metrics

## 📈 Analysis Capabilities

### Feature Frequency Analysis

- How often each feature is activated
- Most/least frequently activated features
- Dead feature identification

### Activation Strength Analysis

- Global statistics (mean, std, min, max)
- Features with highest average activation
- Distribution analysis

### Token Pattern Analysis

- Distribution of active features per token
- Tokens with most active features
- Tokens with highest total activation

### Co-occurrence Analysis

- Feature pairs that frequently co-activate
- Potential feature interactions
- Feature clustering based on co-activation

## 🧪 Example Workflows

### Basic Analysis

```python
from save_latent_concepts import load_concepts

# Load concepts
concepts = load_concepts("results/latent_concepts/layer_12.jsonl")

# Analyze specific feature
for concept in concepts:
    if 1234 in concept["ids"]:
        idx = concept["ids"].index(1234)
        print(f"Token {concept['docid']}: {concept['weight'][idx]:.4f}")
```

### Advanced Analysis

```python
from analyze_latent_concepts import (
    analyze_feature_frequency,
    analyze_activation_strength,
    analyze_token_patterns
)

# Run comprehensive analysis
freq_analysis = analyze_feature_frequency(concepts)
strength_analysis = analyze_activation_strength(concepts)
token_analysis = analyze_token_patterns(concepts)
```

## 🛠️ Customization

### Adding New Analysis Functions

Extend `analyze_latent_concepts.py` with new analysis functions:

```python
def custom_analysis(concepts: list[dict]) -> dict:
    """Your custom analysis logic."""
    # Implementation here
    return results
```

### Custom Thresholds

Adjust thresholds based on your needs:

```bash
# More permissive
uv run python sae/save_latent_concepts.py --threshold 0.001 ...

# More strict
uv run python sae/save_latent_concepts.py --threshold 0.1 ...
```

### Performance Optimization

For large datasets:

```bash
uv run python sae/save_latent_concepts.py \
    --batch_size 8192 \
    --max_batches 100 \
    --num_workers 4
```

## 📚 References

- [SAELens Documentation](https://github.com/jbloomAus/SAELens)
- [Sparse Autoencoders for Interpretability](https://transformer-circuits.pub/)
- [Feature Visualization Techniques](https://distill.pub/2017/feature-visualization/)

## 🤝 Contributing

To extend this toolkit:

1. Add new analysis functions to `analyze_latent_concepts.py`
2. Create new visualization scripts
3. Add support for different SAE architectures
4. Improve documentation and examples

## 📄 License

This code is part of the InterpGR project for interpretability research.

## 🎉 Next Steps

1. **Extract concepts** from your trained SAE
2. **Analyze patterns** in the extracted features
3. **Investigate specific features** for interpretability
4. **Use insights** for model understanding and improvement
5. **Share findings** with the research community
