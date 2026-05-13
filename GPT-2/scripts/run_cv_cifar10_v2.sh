#!/bin/bash

# CV Training Script for CIFAR10 with ResNet18 — RMNP-v2 variant
# Same setup as run_cv_cifar10.sh but uses the shape-aware RMNP-v2 optimizer
# (transposes tall matrices before row L2 normalization).
# Best LR from README Table 14 (adam lr fixed at 0.006)
# Batch size 128 x 4 GPUs = 512 effective
# Runs AdamW -> Muon -> RMNP-v2 sequentially

cd "$(dirname "$0")/.." || exit 1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

DATASET="cifar10"
NET="resnet18"
TRAIN_BSZ="512"
EVAL_BSZ="100"
NEPOCH="200"
SCHEDULER="cosine"
SEED="0"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"
BETA1="0.9"
BETA2="0.999"
WANDB_PROJECT=${WANDB_PROJECT:-"mars-cv"}

# ============================================
# AdamW (lr=0.002, best=94.02%)
# ============================================
echo "========== AdamW =========="
OPTIM="adamw"
LR="0.002"
WD="0.01"
WANDB_NAME="${DATASET}_${NET}_${OPTIM}_lr${LR}_bs${TRAIN_BSZ}_wd${WD}"
python MARS/train_CV.py \
    --cuda "${CUDA_DEVICES}" \
    --dataset ${DATASET} \
    --net ${NET} \
    --optim ${OPTIM} \
    --train_bsz ${TRAIN_BSZ} \
    --eval_bsz ${EVAL_BSZ} \
    --lr ${LR} \
    --wd ${WD} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --Nepoch ${NEPOCH} \
    --scheduler ${SCHEDULER} \
    --seed ${SEED} \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${WANDB_NAME}" \
    "$@"

# ============================================
# Muon (mlr=0.04, alr=0.006, best=94.65%)
# ============================================
echo "========== Muon =========="
OPTIM="muon"
LR="0.04"
ADAMW_LR="0.006"
WD="0.0"
WANDB_NAME="${DATASET}_${NET}_${OPTIM}_lr${LR}_alr${ADAMW_LR}_bs${TRAIN_BSZ}_wd${WD}"
python MARS/train_CV.py \
    --cuda "${CUDA_DEVICES}" \
    --dataset ${DATASET} \
    --net ${NET} \
    --optim ${OPTIM} \
    --train_bsz ${TRAIN_BSZ} \
    --eval_bsz ${EVAL_BSZ} \
    --lr ${LR} \
    --adamw_lr ${ADAMW_LR} \
    --wd ${WD} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --Nepoch ${NEPOCH} \
    --scheduler ${SCHEDULER} \
    --seed ${SEED} \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${WANDB_NAME}" \
    "$@"

# ============================================
# RMNP-v2 (mlr=0.006, alr=0.006, mirrors RMNP setup)
# ============================================
echo "========== RMNP-v2 =========="
OPTIM="rmnp_v2"
LR="0.006"
ADAMW_LR="0.006"
WD="0.0"
WANDB_NAME="${DATASET}_${NET}_${OPTIM}_lr${LR}_alr${ADAMW_LR}_bs${TRAIN_BSZ}_wd${WD}"
python MARS/train_CV.py \
    --cuda "${CUDA_DEVICES}" \
    --dataset ${DATASET} \
    --net ${NET} \
    --optim ${OPTIM} \
    --train_bsz ${TRAIN_BSZ} \
    --eval_bsz ${EVAL_BSZ} \
    --lr ${LR} \
    --adamw_lr ${ADAMW_LR} \
    --wd ${WD} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --Nepoch ${NEPOCH} \
    --scheduler ${SCHEDULER} \
    --seed ${SEED} \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${WANDB_NAME}" \
    "$@"

echo "========== All done =========="
