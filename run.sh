#!/bin/bash

# 如果任何命令失败，立即停止脚本
# set -e

echo "=========================================="
echo " 阶段 1: 缓存激活"
echo "=========================================="

layers=(10 11 13 19 21 22 23)
mkdir -p log/cache

gpu_id=1
for layer in "${layers[@]}"; do
    echo "正在启动 GPU $gpu_id 缓存 Layer $layer ..."
    
    CUDA_VISIBLE_DEVICES=$gpu_id uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer $layer > "log/cache/layer${layer}.log" 2>&1 &

    ((gpu_id++))    
done

wait
echo "✅ 缓存完毕"

echo "=========================================="
echo " 阶段 2: 训练 4x"
echo "=========================================="

mkdir -p log/sae_train_4x

gpu_id=1
for layer in "${layers[@]}"; do
    echo "正在启动 GPU $gpu_id 训练 Layer $layer ..."

    CUDA_VISIBLE_DEVICES=$gpu_id uv run python sae/train_sae.py \
        --cache_dir data/activation_cache_train \
        --lr 4e-4 \
        --d_sae 4096 \
        --batch_size 32768 \
        --layer $layer \
        --save_dir out/sae_train_4x > "log/sae_train_4x/layer${layer}.log" 2>&1 &

    ((gpu_id++))
done

wait
echo "✅ 4x训练完毕"

echo "=========================================="
echo " 阶段 3: 训练 8x"
echo "=========================================="

mkdir -p log/sae_train_8x

gpu_id=1
for layer in "${layers[@]}"; do
    echo "正在启动 GPU $gpu_id 训练 Layer $layer ..."

    CUDA_VISIBLE_DEVICES=$gpu_id uv run python sae/train_sae.py \
        --cache_dir data/activation_cache_train \
        --lr 4e-4 \
        --d_sae 8192 \
        --batch_size 32768 \
        --layer $layer \
        --save_dir out/sae_train_8x > "log/sae_train_8x/layer${layer}.log" 2>&1 &

    ((gpu_id++))
done

wait
echo "✅ 8x训练完毕"

echo "=========================================="
echo " 阶段 4: 评估 4x"
echo "=========================================="

mkdir -p log/sae_eval_4x

gpu_id=1
for layer in "${layers[@]}"; do
    echo "正在启动 GPU $gpu_id 评估 Layer $layer ..."

    CUDA_VISIBLE_DEVICES=$gpu_id uv run python sae/eval_sae.py \
        --cache_dir data/activation_cache_train \
        --checkpoint_dir out/sae_train_4x/layer_${layer} \
        --downstream \
        --genret_ckpt out/dsi-semantic-bert/99.pt \
        --hook_layer $layer > "log/sae_eval_4x/layer_${layer}.log" 2>&1 &

    ((gpu_id++))
done

wait
echo "✅ 4x评估完毕"

echo "=========================================="
echo " 阶段 5: 评估 8x"
echo "=========================================="

mkdir -p log/sae_eval_8x

gpu_id=1
for layer in "${layers[@]}"; do
    echo "正在启动 GPU $gpu_id 评估 Layer $layer ..."

    CUDA_VISIBLE_DEVICES=$gpu_id uv run python sae/eval_sae.py \
        --cache_dir data/activation_cache_train \
        --checkpoint_dir out/sae_train_8x/layer_${layer} \
        --downstream \
        --genret_ckpt out/dsi-semantic-bert/99.pt \
        --hook_layer $layer > "log/sae_eval_8x/layer_${layer}.log" 2>&1 &

    ((gpu_id++))
done

wait
echo "✅ 8x评估完毕"

echo "=========================================="
echo "🎉 所有任务完成！"
echo "=========================================="