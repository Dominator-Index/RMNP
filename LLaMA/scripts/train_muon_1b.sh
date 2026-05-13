#!/bin/bash
# Muon 1B Model Training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 1b \
    --optimizer muon \
    --num_gpus 8 \
    --lr_matrix 0.01 \
    --lr_adam 0.001 \
    --num_steps 90000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 9000 \
    --weight_decay 0.1 \
    --save_every 10000 \
    --eval_every 1000 \
    "$@"
