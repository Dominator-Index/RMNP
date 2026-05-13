#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-60}
GRAD_ACC=${GRAD_ACC:-2}

LR=${LR:-1e-3}
RMNP_LR=${RMNP_LR:-6.67e-3}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.95}
WD=${WD:-1e-1}
RMNP_WD=${RMNP_WD:-0.0}
MAX_ITERS=${MAX_ITERS:-40000}
WARMUP=${WARMUP:-4000}
GRAD_CLIP=${GRAD_CLIP:-1.0}
STREAMING_TIMEOUT=${STREAMING_TIMEOUT:-7200}
STREAMING_RETRIES=${STREAMING_RETRIES:-10}

WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}
DATASET=${DATASET:-"Skylion007/openwebtext"}
# For FineWeb-Edu: DATASET=karpathy/fineweb-edu-100b-shuffle
OUTPUT_DIR="Output"
mkdir -p ${OUTPUT_DIR}
RUN_NAME="rmnp-large-lr${LR}-rlr${RMNP_LR}-b1_${BETA1}-b2_${BETA2}-wd${WD}-rwd${RMNP_WD}-it${MAX_ITERS}"
LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "rmnp-large lr=${LR} rlr=${RMNP_LR} wd=${WD} iters=${MAX_ITERS} gpus=${GPUS}" | tee ${LOG_FILE}

torchrun --standalone --nproc_per_node=${GPUS} \
    MARS/train_rmnp_streaming.py \
    config/train_gpt2_large_rmnp_streaming.py \
    --batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACC} \
    --learning_rate=${LR} \
    --rmnp_learning_rate=${RMNP_LR} \
    --beta1=${BETA1} \
    --beta2=${BETA2} \
    --weight_decay=${WD} \
    --rmnp_weight_decay=${RMNP_WD} \
    --max_iters=${MAX_ITERS} \
    --lr_decay_iters=${MAX_ITERS} \
    --warmup_iters=${WARMUP} \
    --grad_clip=${GRAD_CLIP} \
    --streaming_timeout=${STREAMING_TIMEOUT} \
    --streaming_max_retries=${STREAMING_RETRIES} \
    --streaming_dataset=${DATASET} \
    --wandb_project=${WANDB_PROJECT} \
    --wandb_run_name=${RUN_NAME} \
    "$@" \
    2>&1 | tee -a ${LOG_FILE}
