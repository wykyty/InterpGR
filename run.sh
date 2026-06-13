#!/bin/bash
# cd "$(dirname "$0")"
# mkdir -p log
# uv run torchrun --nproc_per_node=8 baseline.py > log/train_semantic_bert_3.log 2>&1 &
# echo "PID: $!, log: log/train_semantic_bert_3.log"

CUDA_VISIBLE_DEVICES=0 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 14 \
    --save_dir out/sae_train_4x > log/sae_train_4x/layer14.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 14 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer14.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 15 \
    --save_dir out/sae_train_4x > log/sae_train_4x/layer15.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 15 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer15.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 16 \
    --save_dir out/sae_train_4x > log/sae_train_4x/layer16.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 16 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer16.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 17 \
    --save_dir out/sae_train_4x > log/sae_train_4x/layer17.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 17 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer17.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 18 \
    --save_dir out/sae_train_4x > log/sae_train_4x/layer17.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 18 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer17.log 2>&1 &

