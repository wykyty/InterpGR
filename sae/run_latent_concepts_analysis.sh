#!/bin/bash

# Script to run latent concepts extraction and analysis
# Usage: bash sae/run_latent_concepts_analysis.sh

set -e  # Exit on error

# Configuration
CHECKPOINT_DIR="out/sae_train_8x/layer_12"
CACHE_DIR="data/activation_cache_dev"
OUTPUT_DIR="results/latent_concepts"
LAYER=12

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "LATENT CONCEPTS EXTRACTION AND ANALYSIS"
echo "=========================================="

# Step 1: Extract latent concepts
echo ""
echo "Step 1: Extracting latent concepts..."
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Cache: $CACHE_DIR"
echo "  Output: $OUTPUT_DIR/layer_${LAYER}.jsonl"

uv run python sae/save_latent_concepts.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --cache_dir $CACHE_DIR \
    --output_path $OUTPUT_DIR/layer_${LAYER}.jsonl \
    --threshold 0.01 \
    --save_metadata \
    --batch_size 4096

echo ""
echo "✅ Latent concepts extracted successfully!"

# Step 2: Analyze concepts
echo ""
echo "Step 2: Analyzing latent concepts..."

uv run python sae/analyze_latent_concepts.py \
    --input_path $OUTPUT_DIR/layer_${LAYER}.jsonl \
    --statistics_path $OUTPUT_DIR/layer_${LAYER}_statistics.json \
    --output_path $OUTPUT_DIR/layer_${LAYER}_analysis.json

echo ""
echo "✅ Analysis completed!"

# Step 3: Run feature analysis example
echo ""
echo "Step 3: Running feature analysis example..."

uv run python sae/feature_analysis_example.py

echo ""
echo "=========================================="
echo "ANALYSIS COMPLETE!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Latent concepts: $OUTPUT_DIR/layer_${LAYER}.jsonl"
echo "  - Statistics: $OUTPUT_DIR/layer_${LAYER}_statistics.json"
echo "  - Analysis: $OUTPUT_DIR/layer_${LAYER}_analysis.json"
echo ""
echo "Next steps:"
echo "  1. Examine the analysis results"
echo "  2. Investigate specific features using feature_analysis_example.py"
echo "  3. Use the insights for interpretability research"