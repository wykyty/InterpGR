CUDA_VISIBLE_DEVICES=1 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train13/layer_12 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 12 > log/sae_eval13.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train14/layer_12 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 12 > log/sae_eval14.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train15/layer_12 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 12 > log/sae_eval15.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train16/layer_20 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 20 > log/sae_eval16.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train17/layer_20 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 20 > log/sae_eval17.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 uv run python sae/eval_sae.py \
    --cache_dir data/activation_cache_train \
    --checkpoint_dir out/sae_train18/layer_20 \
    --downstream \
    --genret_ckpt out/dsi-semantic-bert/99.pt \
    --hook_layer 20 > log/sae_eval18.log 2>&1 &
