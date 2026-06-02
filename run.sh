# activation

CUDA_VISIBLE_DEVICES=0 uv run python sae/cache_activations.py \
        --checkpoint out/dsi-semantic-bert/99.pt \
        --data_path dataset/nq320k/dev.json \
        --cache_dir data/activation_cache_dev \
        --n_gpus 1

