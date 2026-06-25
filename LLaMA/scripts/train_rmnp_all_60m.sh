#!/bin/bash
# RMNP-All 60M Model Training (embed/lm_head included in RMNP)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 60m \
    --optimizer rmnp_all \
    --num_gpus 8 \
    --lr_matrix 0.01 \
    --lr_adam 0.01 \
    --num_steps 10000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 1000 \
    --weight_decay 0.1 \
    --save_every 5000 \
    --eval_every 1000 \
    "$@"
