#!/bin/bash
# Muon 135M Model Training

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 135m \
    --optimizer muon \
    --num_gpus 4 \
    --lr_matrix 0.01 \
    --lr_adam 0.001 \
    --num_steps 20000 \
    --batch_size 128 \
    --total_batch_size 512 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --save_every 10000 \
    --eval_every 1000 \
    --compute_dd \
    "$@"