#!/bin/bash
cd "$(dirname "$0")"
mkdir -p log

CACHE_DIR="data/activation_cache"
SAVE_DIR="out/dsi_sae_semantic"
PYTHON=".venv/bin/python"

echo "=== Training SAE on layers 0-23, 8 GPUs in parallel ==="
echo "Cache: $CACHE_DIR"
echo "Save:  $SAVE_DIR"

for round in 0 1 2; do
    PIDS=()
    for i in 0 1 2 3 4 5 6 7; do
        LAYER=$((round * 8 + i))
        if [ $LAYER -ge 24 ]; then break; fi
        LOG="log/train_sae_layer${LAYER}.log"
        echo "GPU $i -> layer $LAYER (log: $LOG)"
        CUDA_VISIBLE_DEVICES=$i uv run python sae/train_semantic.py \
            --cache_dir "$CACHE_DIR" \
            --layer "$LAYER" \
            --save_dir "$SAVE_DIR" \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done

    echo "Round $((round+1))/3: waiting for layers $((round*8))-$(( (round+1)*8-1 < 23 ? (round+1)*8-1 : 23 ))..."
    for idx in "${!PIDS[@]}"; do
        wait "${PIDS[$idx]}"
        STATUS=$?
        LAYER=$((round * 8 + idx))
        if [ $STATUS -ne 0 ]; then
            echo "  Layer $LAYER FAILED (exit $STATUS)"
        else
            echo "  Layer $LAYER done."
        fi
    done
done

echo ""
echo "All 24 layers done! Check $SAVE_DIR"
