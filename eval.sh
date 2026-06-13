CUDA_VISIBLE_DEVICES=0 uv run python sae/eval_sae2.py \
    --inference_dir out/sae_train_4x/layer_14/inference \
    --downstream \
    --hook_layer 14 > log/sae_eval_4x_14_Jump.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 uv run python sae/eval_sae.py \
    --checkpoint_dir out/sae_train_4x/layer_14 \
    --downstream \
    --hook_layer 15 > log/sae_eval_4x_15.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 uv run python sae/eval_sae.py \
    --checkpoint_dir out/sae_train_4x/layer_15 \
    --downstream \
    --hook_layer 16 > log/sae_eval_4x_16.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 uv run python sae/eval_sae.py \
    --checkpoint_dir out/sae_train_4x/layer_16 \
    --downstream \
    --hook_layer 17 > log/sae_eval_4x_17.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 uv run python sae/eval_sae.py \
    --checkpoint_dir out/sae_train_4x/layer_18 \
    --downstream \
    --hook_layer 18 > log/sae_eval_4x_18.log 2>&1 &

