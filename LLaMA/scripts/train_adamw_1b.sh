#!/bin/bash
# AdamW 1B Model Training

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 1b \
    --optimizer adamw \
    --num_gpus 8 \
    --lr 6e-4 \
    --num_steps 90000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 9000 \
    --weight_decay 0.1 \
    --save_every 10000 \
    --eval_every 1000 \
    "$@"
