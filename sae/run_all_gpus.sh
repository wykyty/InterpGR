#!/bin/bash
# Train SAE on a single layer using 8 GPUs for parallel activation collection.
# Usage: bash sae/run_all_gpus.sh [LAYER] [CHECKPOINT] [DATA_PATH] [SAVE_DIR]

LAYER="${1:-0}"
CHECKPOINT="${2:-out/dsi-semantic/49.pt}"
DATA_PATH="${3:-dataset/nq320k/dev.json}"
SAVE_DIR="${4:-checkpoints/dsi_sae_semantic}"
PYTHON="/home/zyq/wyk/InterpGR/.venv/bin/python"

echo "=== Multi-GPU SAE Training (single layer) ==="
echo "Layer:      $LAYER"
echo "Checkpoint: $CHECKPOINT"
echo "Data:       $DATA_PATH"
echo "Save dir:   $SAVE_DIR"
echo "GPUs:       8 (parallel activation collection)"
echo ""

mkdir -p "$SAVE_DIR"
$PYTHON sae/train_semantic.py \
    --checkpoint "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --save_dir "$SAVE_DIR" \
    --layers "$LAYER" \
    --n_gpus 8
