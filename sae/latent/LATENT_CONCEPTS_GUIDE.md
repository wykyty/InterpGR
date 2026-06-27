# Latent Concepts Extraction and Analysis Guide

This guide explains how to extract and analyze latent concepts from trained Sparse Autoencoders (SAEs) for interpretability research.

## Overview

The latent concepts extraction pipeline consists of:

1. **`save_latent_concepts.py`** - Extracts feature activations from trained SAE and saves as JSONL
2. **`analyze_latent_concepts.py`** - Analyzes saved concepts and generates reports

## Quick Start

### Step 1: Extract Latent Concepts

After training your SAE, extract latent concepts from the activation cache:

```bash
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --threshold 0.01 \
    --save_metadata
```

**Parameters:**
- `--checkpoint_dir`: Path to trained SAE checkpoint directory
- `--cache_dir`: Directory containing cached activations
- `--output_path`: Path to save extracted concepts (JSONL format)
- `--threshold`: Minimum activation value to consider a feature active (default: 0.01)
- `--save_metadata`: Save additional statistics and metadata

### Step 2: Analyze Extracted Concepts

Run analysis on the saved concepts:

```bash
uv run python sae/analyze_latent_concepts.py \
    --input_path results/latent_concepts/layer_12.jsonl \
    --statistics_path results/latent_concepts/layer_12_statistics.json \
    --output_path results/latent_concepts/layer_12_analysis.json
```

## Output Format

### JSONL Format (save_latent_concepts.py)

Each line contains a JSON object with:

```json
{
    "docid": 0,                    // Token/document index
    "ids": [1, 5, 12, 45],       // List of active feature indices
    "weight": [0.85, 0.42, 0.18, 0.05]  // Corresponding activation weights
}
```

### Statistics Format (with --save_metadata)

```json
{
    "total_tokens": 100000,
    "total_features": 16384,
    "dead_features": 2345,
    "dead_ratio": 0.143,
    "avg_active_features": 45.2,
    "most_active_features": [
        {"index": 1234, "frequency": 0.85},
        ...
    ],
    "least_active_features": [
        {"index": 5678, "frequency": 0.0001},
        ...
    ]
}
```

## Analysis Features

The analysis script provides:

1. **Feature Frequency Analysis**
   - How often each feature is activated across all tokens
   - Most/least frequently activated features

2. **Activation Strength Analysis**
   - Global statistics (mean, std, min, max)
   - Features with highest average activation strength

3. **Token Pattern Analysis**
   - Distribution of active features per token
   - Tokens with most active features
   - Tokens with highest total activation

4. **Co-occurrence Analysis**
   - Feature pairs that frequently co-activate
   - Potential feature interactions

## Use Cases

### 1. Feature Interpretability

Identify what each SAE feature represents:

```python
# Load concepts
concepts = load_concepts("results/latent_concepts/layer_12.jsonl")

# Find tokens where feature 1234 is most active
feature_1234_activations = []
for concept in concepts:
    if 1234 in concept["ids"]:
        idx = concept["ids"].index(1234)
        feature_1234_activations.append((concept["docid"], concept["weight"][idx]))

# Sort by activation strength
feature_1234_activations.sort(key=lambda x: x[1], reverse=True)
```

### 2. Feature Clustering

Cluster features based on co-activation patterns:

```python
from analyze_latent_concepts import analyze_activation_patterns

# Get co-occurrence data
co_analysis = analyze_activation_patterns(concepts)

# Build feature co-occurrence matrix
# (Implementation depends on your clustering approach)
```

### 3. Downstream Task Analysis

Analyze which features are important for specific tasks:

```python
# Filter concepts for specific document types
relevant_concepts = [c for c in concepts if c["docid"] in target_doc_ids]

# Analyze feature patterns in relevant documents
feature_freq = analyze_feature_frequency(relevant_concepts)
```

### 4. Ablation Studies

Use extracted concepts to guide feature ablation:

```python
# Identify features to ablate
features_to_ablate = [feat for feat, freq in feature_freq["sorted_features"][:10]]

# Use these in your ablation experiments
```

## Advanced Usage

### Processing Large Datasets

For large datasets, use batch processing:

```bash
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --batch_size 8192 \
    --max_batches 100  # Process only first 100 batches
```

### Custom Thresholds

Adjust threshold based on your needs:

```bash
# Lower threshold - more features considered active
uv run python sae/save_latent_concepts.py \
    --threshold 0.001 \
    ...

# Higher threshold - only strongly activated features
uv run python sae/save_latent_concepts.py \
    --threshold 0.1 \
    ...
```

### Multiple Layers

Process multiple layers:

```bash
for layer in 0 6 12 18 23; do
    uv run python sae/save_latent_concepts.py \
        --checkpoint_dir checkpoints/dsi_sae_semantic/layer_${layer} \
        --cache_dir data/activation_cache_train \
        --output_path results/latent_concepts/layer_${layer}.jsonl \
        --save_metadata
done
```

## Integration with Visualization

The extracted concepts can be integrated with visualization tools:

1. **TensorBoard** - Log feature activations over time
2. **Weights & Biases** - Track feature statistics
3. **Custom Visualizations** - Build interactive dashboards

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `--batch_size`
   - Use `--max_batches` to process subsets

2. **CUDA Errors**
   - Ensure GPU memory is sufficient
   - Try CPU processing with `--device cpu`

3. **Missing Checkpoints**
   - Verify checkpoint directory structure
   - Check that SAE training completed successfully

### Performance Tips

1. **Use DataLoader workers**: Set `--num_workers 4` for faster data loading
2. **Pin memory**: Enabled by default for GPU processing
3. **Batch processing**: Larger batches are more efficient but use more memory

## Example Workflow

```bash
# 1. Train SAE (if not already done)
uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --layer 12 \
    --save_dir checkpoints/dsi_sae_semantic

# 2. Extract latent concepts
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --threshold 0.01 \
    --save_metadata

# 3. Analyze concepts
uv run python sae/analyze_latent_concepts.py \
    --input_path results/latent_concepts/layer_12.jsonl \
    --statistics_path results/latent_concepts/layer_12_statistics.json \
    --output_path results/latent_concepts/layer_12_analysis.json

# 4. View results
cat results/latent_concepts/layer_12_analysis.json | python -m json.tool
```

## Next Steps

After extracting and analyzing latent concepts:

1. **Feature Interpretation**: Manually inspect top features to understand their meaning
2. **Circuit Analysis**: Trace how features compose to form circuits
3. **Intervention Experiments**: Use features for targeted interventions
4. **Model Editing**: Modify features to change model behavior

## References

- [SAELens Documentation](https://github.com/jbloomAus/SAELens)
- [Sparse Autoencoders for Interpretability](https://transformer-circuits.pub/)
- [Feature Visualization Techniques](https://distill.pub/2017/feature-visualization/)