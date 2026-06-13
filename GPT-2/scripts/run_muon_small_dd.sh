#!/bin/bash
# GPT-2 Small + Muon with Diagonal Dominance monitoring
# Runs train_muon_streaming_test.py (DD-enabled version)

cd "$(dirname "$0")/.." || exit 1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-120}
GRAD_ACC=${GRAD_ACC:-1}

LR=${LR:-3e-3}
MUON_LR=${MUON_LR:-2e-2}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.95}
WD=${WD:-1e-1}
MUON_WD=${MUON_WD:-0.0}
MAX_ITERS=${MAX_ITERS:-10000}
WARMUP=${WARMUP:-1000}
GRAD_CLIP=${GRAD_CLIP:-1.0}
DATASET=${DATASET:-"Skylion007/openwebtext"}
WANDB_PROJECT=${WANDB_PROJECT:-"mars-owt"}
STREAMING_TIMEOUT=${STREAMING_TIMEOUT:-72000}
STREAMING_RETRIES=${STREAMING_RETRIES:-100}

DATASET_SHORT=$(echo ${DATASET} | sed 's/.*\///g' | sed 's/-.*//g')
RUN_NAME="muon-small-dd-${DATASET_SHORT}-lr${LR}-mlr${MUON_LR}-wd${WD}-it${MAX_ITERS}"

OUTPUT_DIR="Output"
mkdir -p ${OUTPUT_DIR}
LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "=== GPT-2 Small + Muon + DD ===" | tee ${LOG_FILE}
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a ${LOG_FILE}
echo "  GPUS: ${GPUS}, BATCH_SIZE: ${BATCH_SIZE}, GRAD_ACC: ${GRAD_ACC}" | tee -a ${LOG_FILE}
echo "  LR: ${LR}, MUON_LR: ${MUON_LR}, WD: ${WD}" | tee -a ${LOG_FILE}
echo "  MAX_ITERS: ${MAX_ITERS}, WARMUP: ${WARMUP}" | tee -a ${LOG_FILE}
echo "  DATASET: ${DATASET}" | tee -a ${LOG_FILE}
echo "  RUN_NAME: ${RUN_NAME}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

torchrun --standalone --nproc_per_node=${GPUS} \
    RMNP/train_muon_streaming_test.py \
    config/train_gpt2_small_muon_streaming.py \
    --batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=${GRAD_ACC} \
    --learning_rate=${LR} \
    --muon_learning_rate=${MUON_LR} \
    --beta1=${BETA1} \
    --beta2=${BETA2} \
    --weight_decay=${WD} \
    --muon_weight_decay=${MUON_WD} \
    --max_iters=${MAX_ITERS} \
    --lr_decay_iters=${MAX_ITERS} \
    --warmup_iters=${WARMUP} \
    --grad_clip=${GRAD_CLIP} \
    --streaming_timeout=${STREAMING_TIMEOUT} \
    --streaming_max_retries=${STREAMING_RETRIES} \
    --streaming_dataset=${DATASET} \
    --wandb_log=True \
    --wandb_project=${WANDB_PROJECT} \
    --wandb_run_name=${RUN_NAME} \
    "$@" \
    2>&1 | tee -a ${LOG_FILE}

echo "Training completed. Log: ${LOG_FILE}" | tee -a ${LOG_FILE}
