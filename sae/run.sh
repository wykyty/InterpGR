# activation

# CUDA_VISIBLE_DEVICES=0 uv run python sae/cache_activations.py \
#         --checkpoint out/dsi-semantic-bert/99.pt \
#         --data_path dataset/nq320k/dev.json \
#         --cache_dir data/activation_cache_dev \
#         --n_gpus 1


CUDA_VISIBLE_DEVICES=0 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache \
        --n_gpus 1 \
        --layer 20

uv run torchrun --nproc_per_node=8 sae/train_sae.py \
        --cache_dir data/activation_cache_dev \
        --lr 3e-4 \
        --layer 12 --save_dir out/sae_semantic_dev

uv run torchrun --nproc_per_node=8 sae/train_sae.py \
        --cache_dir data/activation_cache_train \
        --lr 3e-4 \
        --total_steps 30000 \
        --layer 2 --save_dir out/sae_semantic_train2 > log/train_semantic_bert_3.log 2>&1 &

# 1
# 使用gemini优化版本，单卡跑3w步，吃满batch_size
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 3e-4 \
    --total_steps 30000 \
    --batch_size 16384 \
    --layer 2 \
    --save_dir out/sae_semantic_train3 > log/sae_semantic_train3.log 2>&1 &

# 2 
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_dev \
    --lr 3e-4 \
    --total_steps 30000 \
    --batch_size 16384 \
    --layer 12 \
    --save_dir out/sae_dev > log/sae_dev.log 2>&1 &


# 3 OOM
# export HF_ENDPOINT=https://hf-mirror.com
# CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
#     --cache_dir data/activation_cache_train \
#     --lr 6e-4 \
#     --total_steps 30000 \
#     --batch_size 49152 \
#     --layer 2 \
#     --save_dir out/sae_semantic_train4 > log/sae_semantic_train4.log 2>&1 &


# 4   可以
# 降低一下
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 2 \
    --save_dir out/sae_semantic_train4 > log/sae_semantic_train4.log 2>&1 &

# 5 可以
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=2 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train6 > log/sae_train6.log 2>&1 &



CUDA_VISIBLE_DEVICES=1 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train7 > log/sae_train7.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train8 > log/sae_train8.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train9 > log/sae_train9.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train10 > log/sae_train10.log 2>&1 &





#--------------------------------------------

# 评估一下 2
uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_semantic_train2/layer_2 \
    --eval_batch_size 2048 \
    --eval_batches 100

# 评估一下 5
uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_semantic_train5/layer_12 \
    --eval_batch_size 2048 \
    --eval_batches 100

# 评估
uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_semantic_train/layer_2 \
    --eval_batch_size 2048 \
    --eval_batches 100

uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train7/layer_12 \
    --eval_batch_size 2048 \
    --eval_batches 100

uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train8/layer_12 \
    --eval_batch_size 2048 \
    --eval_batches 100

CUDA_VISIBLE_DEVICES=1 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_semantic_train5/layer_12 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 12

CUDA_VISIBLE_DEVICES=1 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_semantic/layer_12 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 12

