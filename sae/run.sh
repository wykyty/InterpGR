# activation

# CUDA_VISIBLE_DEVICES=0 uv run python sae/cache_activations.py \
#         --checkpoint out/dsi-semantic-bert/99.pt \
#         --data_path dataset/nq320k/dev.json \
#         --cache_dir data/activation_cache_dev \
#         --n_gpus 1


uv run torchrun --nproc_per_node=8 sae/train_sae.py \
        --cache_dir data/activation_cache_dev \
        --lr 3e-4 \
        --layer 12 --save_dir out/sae_semantic_dev

uv run torchrun --nproc_per_node=8 sae/train_sae.py \
        --cache_dir data/activation_cache_train \
        --lr 3e-4 \
        --total_steps 30000 \
        --layer 2 --save_dir out/sae_semantic_train2 > log/train_semantic_bert_3.log 2>&1 &

# 使用gemini优化版本，单卡跑3w步，吃满batch_size
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 3e-4 \
    --total_steps 30000 \
    --batch_size 16384 \
    --layer 2 \
    --save_dir out/sae_semantic_train3 > log/sae_semantic_train3.log 2>&1 &

export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_dev \
    --lr 3e-4 \
    --total_steps 30000 \
    --batch_size 16384 \
    --layer 12 \
    --save_dir out/sae_dev > log/sae_dev.log 2>&1 &


## OOM
# export HF_ENDPOINT=https://hf-mirror.com
# CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
#     --cache_dir data/activation_cache_train \
#     --lr 6e-4 \
#     --total_steps 30000 \
#     --batch_size 49152 \
#     --layer 2 \
#     --save_dir out/sae_semantic_train4 > log/sae_semantic_train4.log 2>&1 &

# 降低一下
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 2 \
    --save_dir out/sae_semantic_train4 > log/sae_semantic_train4.log 2>&1 &

export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache \
    --lr 4e-4 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_semantic_train5 > log/sae_semantic_train5.log 2>&1 &

