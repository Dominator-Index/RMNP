#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-15}
GRAD_ACC=${GRAD_ACC:-4}

LR=${LR:-3e-3}
RMNP_LR=${RMNP_LR:-4e-3}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.95}
WD=${WD:-1e-1}
RMNP_WD=${RMNP_WD:-0.0}
MAX_ITERS=${MAX_ITERS:-10000}
WARMUP=${WARMUP:-1000}
GRAD_CLIP=${GRAD_CLIP:-1.0}
STREAMING_TIMEOUT=${STREAMING_TIMEOUT:-72000}
STREAMING_RETRIES=${STREAMING_RETRIES:-100}
DATASET=${DATASET:-"Skylion007/openwebtext"}
# For FineWeb-Edu: DATASET=karpathy/fineweb-edu-100b-shuffle
WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}

OUTPUT_DIR="Output"
mkdir -p ${OUTPUT_DIR}
DATASET_SHORT=$(echo ${DATASET} | sed 's/.*\///g' | sed 's/-.*//g')
RUN_NAME="rmnp-small-${DATASET_SHORT}-lr${LR}-rlr${RMNP_LR}-b1_${BETA1}-b2_${BETA2}-wd${WD}-rwd${RMNP_WD}-it${MAX_ITERS}"
LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "rmnp-small lr=${LR} rlr=${RMNP_LR} wd=${WD} iters=${MAX_ITERS} gpus=${GPUS}" | tee ${LOG_FILE}

torchrun --standalone --nproc_per_node=${GPUS} \
    RMNP/train_rmnp_streaming.py \
    config/train_gpt2_small_rmnp_streaming_owt.py \
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
