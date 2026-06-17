# activation

# CUDA_VISIBLE_DEVICES=0 uv run python sae/cache_activations.py \
#         --checkpoint out/dsi-semantic-bert/99.pt \
#         --data_path dataset/nq320k/dev.json \
#         --cache_dir data/activation_cache_dev \
#         --n_gpus 1

layer=3
for gpu_id in {1..7}; do
    CUDA_VISIBLE_DEVICES=$gpu_id uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer $layer > "log/cache/layer${layer}.log" 2>&1 &
    layer=$((layer+1))
done


CUDA_VISIBLE_DEVICES=1 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 3 > log/cache/layer3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 4 > log/cache/layer4.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 5 > log/cache/layer5.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 6 > log/cache/layer6.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 7 > log/cache/layer7.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 8 > log/cache/layer8.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/train.json \
        --cache_dir data/activation_cache_train \
        --n_gpus 1 \
        --layer 9 > log/cache/layer9.log 2>&1 &

#--------------------------------------------------------------------------

# train
CUDA_VISIBLE_DEVICES=1 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --batch_size 32768 \
    --layer 3 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --batch_size 32768 \
    --layer 4 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer4.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --batch_size 32768 \
    --layer 5 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer5.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --batch_size 32768 \
    --layer 6 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer6.log 2>&1 &


CUDA_VISIBLE_DEVICES=5 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --batch_size 32768 \
    --layer 7 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer7.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --batch_size 32768 \
    --layer 8 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer8.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --batch_size 32768 \
    --layer 9 \
    --save_dir out/sae_train_8x > log/sae_train_8x/layer9.log 2>&1 &




#--------------------------------------------

# 评估
CUDA_VISIBLE_DEVICES=1 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train_4x/layer_3 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 3 > log/sae_eval_4x/layer_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train_4x/layer_4 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 4 > log/sae_eval_4x/layer_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train_4x/layer_5 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 5 > log/sae_eval_4x/layer_5.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train_4x/layer_6 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 6 > log/sae_eval_4x/layer_6.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train_4x/layer_7 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 7 > log/sae_eval_4x/layer_7.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train_4x/layer_8 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 8 > log/sae_eval_4x/layer_8.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train_4x/layer_9 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 9 > log/sae_eval_4x/layer_9.log 2>&1 &