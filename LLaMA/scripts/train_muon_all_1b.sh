#!/bin/bash
# Muon-All 1B Model Training (embed/lm_head included in Muon)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 1b \
    --optimizer muon_all \
    --num_gpus 8 \
    --lr_matrix 0.005 \
    --lr_adam 6e-4 \
    --num_steps 90000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 9000 \
    --weight_decay 0.1 \
    --save_every 5000 \
    --eval_every 1000 \
    "$@"
