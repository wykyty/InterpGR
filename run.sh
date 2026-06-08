CUDA_VISIBLE_DEVICES=1 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train13 > log/sae_train13.log 2>&1 &



#---------------

CUDA_VISIBLE_DEVICES=2 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train14 > log/sae_train14.log 2>&1 &


#--------------

CUDA_VISIBLE_DEVICES=3 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 16384 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 12 \
    --save_dir out/sae_train15 > log/sae_train15.log 2>&1 &


#--------------- layer 20 ---------------

CUDA_VISIBLE_DEVICES=4 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 4096 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 20 \
    --save_dir out/sae_train16 > log/sae_train16.log 2>&1 &


#--------------- layer 20 ---------------

CUDA_VISIBLE_DEVICES=5 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 8192 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 20 \
    --save_dir out/sae_train17 > log/sae_train17.log 2>&1 &


#--------------- layer 20 ---------------

CUDA_VISIBLE_DEVICES=6 uv run python sae/train_sae.py \
    --cache_dir data/activation_cache_train \
    --lr 4e-4 \
    --d_sae 16384 \
    --total_steps 30000 \
    --batch_size 32768 \
    --layer 20 \
    --save_dir out/sae_train18 > log/sae_train18.log 2>&1 &


