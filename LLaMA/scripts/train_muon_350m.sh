#!/bin/bash
# Muon 350M Model Training

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"

exec "$SCRIPT_DIR/train_universal.sh" \
    --model_size 350m \
    --optimizer muon \
    --num_gpus 8 \
    --lr_matrix 0.004 \
    --lr_adam 0.001 \
    --num_steps 60000 \
    --batch_size 64 \
    --total_batch_size 512 \
    --warmup_steps 6000 \
    --weight_decay 0.1 \
    --save_every 5000 \
    --eval_every 1000 \
    `# --compute_dd disabled for public release` \
    "$@"