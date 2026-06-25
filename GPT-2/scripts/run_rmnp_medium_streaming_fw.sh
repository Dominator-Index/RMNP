#!/bin/bash

# Change to project root directory (one level up from scripts/)
cd "$(dirname "$0")/.." || exit 1

# HuggingFace Token configuration
# Set HF_TOKEN in your environment for faster downloads / higher API limits
export HF_TOKEN="${HF_TOKEN:-}"

# CUDA device configuration
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 ./script.sh (to use GPUs 0-3)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
# Default parameters
BATCH_SIZE=${BATCH_SIZE:-15}
GRAD_ACC=${GRAD_ACC:-4}
GPUS=${GPUS:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-480}

if [ "$GRAD_ACC" == "4" ] && [ "$GPUS" != "8" ]; then
  GRAD_ACC=$((GLOBAL_BATCH_SIZE / (BATCH_SIZE * GPUS)))
  AUTO_CALC=true
else
  AUTO_CALC=false
fi

LR=${LR:-1.5e-3}
RMNP_LR=${RMNP_LR:-2e-3}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.95}
WD=${WD:-1e-1}
RMNP_WD=${RMNP_WD:-0.0}
MAX_ITERS=${MAX_ITERS:-20000}
WARMUP=${WARMUP:-2000}
GRAD_CLIP=${GRAD_CLIP:-1.0}
STREAMING_TIMEOUT=${STREAMING_TIMEOUT:-7200}
STREAMING_RETRIES=${STREAMING_RETRIES:-10}
# Dataset: "karpathy/fineweb-edu-100b-shuffle" or "Skylion007/openwebtext"
DATASET=${DATASET:-"karpathy/fineweb-edu-100b-shuffle"}
WANDB_PROJECT=${WANDB_PROJECT:-"mars-fw"}
OUTPUT_DIR="Output"
mkdir -p ${OUTPUT_DIR}
DATASET_SHORT=$(echo ${DATASET} | sed 's/.*\///g' | sed 's/-.*//g')
RUN_NAME="rmnp-medium-${DATASET_SHORT}-lr${LR}-rlr${RMNP_LR}-b1_${BETA1}-b2_${BETA2}-wd${WD}-rwd${RMNP_WD}-it${MAX_ITERS}"
LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "Starting training with the following parameters:" | tee ${LOG_FILE}
if [ "$AUTO_CALC" == "true" ]; then
  echo "  [AUTO-CALCULATED] GRAD_ACC_STEPS: ${GRAD_ACC} (to maintain global batch size = ${GLOBAL_BATCH_SIZE})" | tee -a ${LOG_FILE}
fi
echo "  BATCH_SIZE: ${BATCH_SIZE}" | tee -a ${LOG_FILE}
echo "  GRAD_ACC_STEPS: ${GRAD_ACC}" | tee -a ${LOG_FILE}
echo "  NUM_GPUS: ${GPUS}" | tee -a ${LOG_FILE}
echo "  GLOBAL_BATCH_SIZE: $((BATCH_SIZE * GPUS * GRAD_ACC))" | tee -a ${LOG_FILE}
echo "  LR: ${LR}" | tee -a ${LOG_FILE}
echo "  RMNP_LR: ${RMNP_LR}" | tee -a ${LOG_FILE}
echo "  WEIGHT_DECAY: ${WD}" | tee -a ${LOG_FILE}
echo "  RMNP_WEIGHT_DECAY: ${RMNP_WD}" | tee -a ${LOG_FILE}
echo "  MAX_ITERS: ${MAX_ITERS}" | tee -a ${LOG_FILE}
echo "  DATASET: ${DATASET}" | tee -a ${LOG_FILE}
echo "  RUN_NAME: ${RUN_NAME}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}
# Run training
torchrun --standalone --nproc_per_node=${GPUS} \
      RMNP/train_rmnp_streaming.py \
      config/train_gpt2_medium_rmnp_streaming_fw.py \
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
echo "" | tee -a ${LOG_FILE}
echo "Training completed. Log saved to: ${LOG_FILE}" | tee -a ${LOG_FILE}
