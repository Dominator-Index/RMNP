#!/bin/bash
# RMNP-v2 (shape-aware) 350M Model Training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 350m \
    --optimizer RMNP_v2 \
    --num_gpus 8 \
    --lr_matrix 0.005 \
    --lr_adam 0.005 \
    --num_steps 60000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 6000 \
    --weight_decay 0.1 \
    --save_every 5000 \
    --eval_every 1000 \
    "$@"