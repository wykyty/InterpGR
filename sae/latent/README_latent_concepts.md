# Latent Concepts Extraction and Analysis

This directory contains tools for extracting and analyzing latent concepts from trained Sparse Autoencoders (SAEs) for interpretability research.

## 📁 Files Overview

### Core Scripts

1. **`save_latent_concepts.py`** - Extracts feature activations from trained SAE
   - Loads cached activations and trained SAE
   - Encodes activations through SAE to get feature representations
   - Saves active features and their weights as JSONL

2. **`analyze_latent_concepts.py`** - Comprehensive analysis of extracted concepts
   - Feature frequency analysis
   - Activation strength statistics
   - Co-occurrence patterns
   - Token-level analysis

3. **`feature_analysis_example.py`** - Example usage for feature analysis
   - Demonstrates how to analyze specific features
   - Shows how to find co-activated features
   - Includes specialization analysis

### Documentation

4. **`LATENT_CONCEPTS_GUIDE.md`** - Detailed usage guide
5. **`README_latent_concepts.md`** - This file

## 🚀 Quick Start

### Prerequisites

1. **Trained SAE**: You need a trained SAE checkpoint
2. **Cached Activations**: Activation cache from `cache_activations.py`

### Step-by-Step Workflow

```bash
# 1. Extract latent concepts
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --threshold 0.01 \
    --save_metadata

# 2. Analyze extracted concepts
uv run python sae/analyze_latent_concepts.py \
    --input_path results/latent_concepts/layer_12.jsonl \
    --statistics_path results/latent_concepts/layer_12_statistics.json \
    --output_path results/latent_concepts/layer_12_analysis.json

# 3. Run feature analysis examples
uv run python sae/feature_analysis_example.py
```

## 📊 Output Format

### JSONL Format

Each line in the output JSONL file represents a token's latent concept:

```json
{
    "docid": 0,
    "ids": [1, 5, 12, 45],
    "weight": [0.85, 0.42, 0.18, 0.05]
}
```

- `docid`: Token/document index
- `ids`: List of active feature indices
- `weight`: Corresponding activation weights

### Statistics Format

When using `--save_metadata`, additional statistics are saved:

```json
{
    "total_tokens": 100000,
    "total_features": 16384,
    "dead_features": 2345,
    "dead_ratio": 0.143,
    "avg_active_features": 45.2,
    "most_active_features": [...],
    "least_active_features": [...]
}
```

## 🎯 Use Cases

### 1. Feature Interpretability

```python
from save_latent_concepts import load_concepts

concepts = load_concepts("results/latent_concepts/layer_12.jsonl")

# Find contexts where feature 1234 is most active
for concept in concepts:
    if 1234 in concept["ids"]:
        idx = concept["ids"].index(1234)
        print(f"Token {concept['docid']}: {concept['weight'][idx]:.4f}")
```

### 2. Feature Co-occurrence Analysis

```python
from analyze_latent_concepts import analyze_activation_patterns

co_analysis = analyze_activation_patterns(concepts)
print("Top co-occurring feature pairs:")
for pair in co_analysis["top_co_occurring_pairs"][:5]:
    print(f"  Features {pair['features']}: {pair['count']} times")
```

### 3. Specialization Analysis

```python
from feature_analysis_example import analyze_feature_specialization

specialization = analyze_feature_specialization(concepts, feature_id=1234)
print(f"Feature 1234 activation rate: {specialization['activation_rate']:.2%}")
print(f"Specialization score: {specialization['specialization_score']:.2f}")
```

## 🔧 Advanced Configuration

### Threshold Selection

The `--threshold` parameter controls which features are considered active:

- **Lower threshold (0.001)**: More features, noisier but comprehensive
- **Higher threshold (0.1)**: Fewer features, cleaner but may miss weak activations
- **Default (0.01)**: Balanced approach

### Batch Processing

For large datasets, use batch processing:

```bash
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --batch_size 8192 \
    --max_batches 100
```

### Multiple Layers

Process multiple layers for comprehensive analysis:

```bash
for layer in 0 6 12 18 23; do
    uv run python sae/save_latent_concepts.py \
        --checkpoint_dir checkpoints/dsi_sae_semantic/layer_${layer} \
        --cache_dir data/activation_cache_train \
        --output_path results/latent_concepts/layer_${layer}.jsonl \
        --save_metadata
done
```

## 📈 Analysis Features

### Feature Frequency Analysis
- How often each feature is activated across all tokens
- Most/least frequently activated features
- Dead feature identification

### Activation Strength Analysis
- Global statistics (mean, std, min, max)
- Features with highest average activation
- Activation distribution analysis

### Token Pattern Analysis
- Distribution of active features per token
- Tokens with most active features
- Tokens with highest total activation

### Co-occurrence Analysis
- Feature pairs that frequently co-activate
- Potential feature interactions
- Feature clustering based on co-activation

## 🧪 Example Analysis

### Finding Specialized Features

```python
# Load concepts
concepts = load_concepts("results/latent_concepts/layer_12.jsonl")

# Analyze specialization for each feature
specialization_scores = []
for feat_id in range(16384):  # Assuming 16384 features
    spec = analyze_feature_specialization(concepts, feat_id)
    specialization_scores.append((feat_id, spec["specialization_score"]))

# Sort by specialization
specialization_scores.sort(key=lambda x: x[1], reverse=True)

# Top specialized features
print("Most specialized features:")
for feat_id, score in specialization_scores[:10]:
    print(f"  Feature {feat_id}: {score:.2f}")
```

### Feature Context Analysis

```python
# Get contexts for a specific feature
feature_contexts = get_feature_contexts(concepts, feature_id=1234, top_k=10)

print("Contexts where feature 1234 is most active:")
for ctx in feature_contexts:
    print(f"  Token {ctx['docid']}: {ctx['activation']:.4f}")
```

## 🔍 Interpretability Workflow

1. **Extract Concepts**: Run `save_latent_concepts.py` to get feature activations
2. **Analyze Patterns**: Use `analyze_latent_concepts.py` to identify patterns
3. **Feature Investigation**: Use `feature_analysis_example.py` for specific features
4. **Hypothesis Formation**: Based on patterns, form hypotheses about feature function
5. **Validation**: Test hypotheses through intervention experiments

## 🛠️ Troubleshooting

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