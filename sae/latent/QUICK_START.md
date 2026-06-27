# Quick Start Guide for Latent Concepts Analysis

This guide provides a quick start for extracting and analyzing latent concepts from trained SAEs.

## Prerequisites

1. **Trained SAE**: Ensure you have a trained SAE checkpoint
2. **Cached Activations**: Activation cache from `cache_activations.py`
3. **Python Environment**: With required dependencies

## Quick Start

### Option 1: Run the Complete Pipeline

```bash
# Run the complete analysis pipeline
bash sae/run_latent_concepts_analysis.sh
```

### Option 2: Step-by-Step Execution

#### Step 1: Extract Latent Concepts

```bash
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --threshold 0.01 \
    --save_metadata
```

**Parameters explained:**
- `--checkpoint_dir`: Path to your trained SAE checkpoint
- `--cache_dir`: Directory containing cached activations
- `--output_path`: Where to save the extracted concepts
- `--threshold`: Minimum activation to consider a feature active (0.01 is a good starting point)
- `--save_metadata`: Save additional statistics (recommended)

#### Step 2: Analyze the Concepts

```bash
uv run python sae/analyze_latent_concepts.py \
    --input_path results/latent_concepts/layer_12.jsonl \
    --statistics_path results/latent_concepts/layer_12_statistics.json \
    --output_path results/latent_concepts/layer_12_analysis.json
```

#### Step 3: Explore Features

```bash
uv run python sae/feature_analysis_example.py
```

## Understanding the Output

### JSONL File Structure

Each line in `layer_12.jsonl` represents one token's latent concept:

```json
{
    "docid": 0,
    "ids": [1, 5, 12, 45],
    "weight": [0.85, 0.42, 0.18, 0.05]
}
```

- **`docid`**: Token index
- **`ids`**: Which features are active for this token
- **`weight`**: How strongly each feature is activated

### Statistics File

`layer_12_statistics.json` contains summary statistics:

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

## Common Use Cases

### 1. Find What a Specific Feature Represents

```python
import json

# Load concepts
concepts = []
with open("results/latent_concepts/layer_12.jsonl") as f:
    for line in f:
        concepts.append(json.loads(line))

# Find tokens where feature 1234 is most active
feature_1234_contexts = []
for concept in concepts:
    if 1234 in concept["ids"]:
        idx = concept["ids"].index(1234)
        feature_1234_contexts.append((concept["docid"], concept["weight"][idx]))

# Sort by activation strength
feature_1234_contexts.sort(key=lambda x: x[1], reverse=True)

print("Top 5 contexts for feature 1234:")
for docid, weight in feature_1234_contexts[:5]:
    print(f"  Token {docid}: {weight:.4f}")
```

### 2. Find Features That Work Together

```python
from collections import defaultdict

# Count feature co-occurrences
co_occurrence = defaultdict(int)
for concept in concepts:
    features = concept["ids"]
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            co_occurrence[(f1, f2)] += 1

# Find most common pairs
sorted_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)
print("Most common feature pairs:")
for (f1, f2), count in sorted_pairs[:10]:
    print(f"  Features {f1} & {f2}: {count} times")
```

### 3. Identify Specialized Features

```python
# Calculate how specialized each feature is
feature_activation_rate = {}
for concept in concepts:
    for feat_id in concept["ids"]:
        feature_activation_rate[feat_id] = feature_activation_rate.get(feat_id, 0) + 1

# Convert to rate
total_tokens = len(concepts)
specialization = {
    feat: count / total_tokens 
    for feat, count in feature_activation_rate.items()
}

# Most specialized (rarely activated) features
sorted_specialization = sorted(specialization.items(), key=lambda x: x[1])
print("Most specialized features:")
for feat, rate in sorted_specialization[:10]:
    print(f"  Feature {feat}: {rate:.4%} activation rate")
```

## Advanced Usage

### Process Multiple Layers

```bash
for layer in 0 6 12 18 23; do
    uv run python sae/save_latent_concepts.py \
        --checkpoint_dir checkpoints/dsi_sae_semantic/layer_${layer} \
        --cache_dir data/activation_cache_train \
        --output_path results/latent_concepts/layer_${layer}.jsonl \
        --save_metadata
done
```

### Adjust Threshold

```bash
# Lower threshold - more features, noisier
uv run python sae/save_latent_concepts.py \
    --threshold 0.001 \
    ...

# Higher threshold - fewer features, cleaner
uv run python sae/save_latent_concepts.py \
    --threshold 0.1 \
    ...
```

### Process Large Datasets

```bash
uv run python sae/save_latent_concepts.py \
    --batch_size 8192 \
    --max_batches 100 \  # Process only first 100 batches
    ...
```

## Troubleshooting

### Common Issues

1. **"Checkpoint not found"**
   - Verify the checkpoint directory exists
   - Check that SAE training completed successfully

2. **"CUDA out of memory"**
   - Reduce `--batch_size` (try 2048 or 1024)
   - Use `--device cpu` for CPU processing

3. **"No active features"**
   - Lower the `--threshold` value
   - Check if the SAE was trained properly

### Performance Tips

1. **Use GPU**: Significantly faster than CPU
2. **Increase batch size**: Until you hit memory limits
3. **Use multiple workers**: `--num_workers 4` for faster data loading

## Next Steps

After extracting latent concepts:

1. **Examine the analysis**: Look at `layer_12_analysis.json`
2. **Investigate specific features**: Use `feature_analysis_example.py`
3. **Form hypotheses**: What might each feature represent?
4. **Validate with interventions**: Use the SAE inference tools for ablation studies

## Example Workflow

```bash
# 1. Extract concepts
uv run python sae/save_latent_concepts.py \
    --checkpoint_dir checkpoints/dsi_sae_semantic/layer_12 \
    --cache_dir data/activation_cache_train \
    --output_path results/latent_concepts/layer_12.jsonl \
    --threshold 0.01 \
    --save_metadata

# 2. Analyze
uv run python sae/analyze_latent_concepts.py \
    --input_path results/latent_concepts/layer_12.jsonl \
    --statistics_path results/latent_concepts/layer_12_statistics.json

# 3. Explore
uv run python sae/feature_analysis_example.py

# 4. View results
cat results/latent_concepts/layer_12_analysis.json | python -m json.tool
```

## Need Help?

- Check the detailed guide: `LATENT_CONCEPTS_GUIDE.md`
- Look at example code: `feature_analysis_example.py`
- Review the source code for customization