#!/bin/bash
# CIFAR-10 ResNet-18 Diagonal Dominance Monitoring
# Best LR from README Table 14 (adam lr fixed at 0.006)
# Runs Muon with DD metrics logged to wandb

cd "$(dirname "$0")/.." || exit 1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

DATASET="cifar10"
NET="resnet18"
TRAIN_BSZ="512"
EVAL_BSZ="100"
NEPOCH="200"
SCHEDULER="cosine"
SEED="0"
BETA1="0.9"
BETA2="0.999"
DD_EVERY="${DD_EVERY:-10}"
WANDB_PROJECT=${WANDB_PROJECT:-"mars-cv-dd"}

# ============================================
# Muon with DD monitoring (mlr=0.04, alr=0.006)
# ============================================
echo "========== Muon + DD =========="
OPTIM="muon"
LR="${MUON_LR:-0.04}"
ADAMW_LR="${ADAMW_LR:-0.006}"
WD="0.0"
WANDB_NAME="${DATASET}_${NET}_${OPTIM}_lr${LR}_alr${ADAMW_LR}_dd"

python RMNP/train_CV_dd.py \
    --cuda "${CUDA_VISIBLE_DEVICES}" \
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
    --dd_every ${DD_EVERY} \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_name "${WANDB_NAME}" \
    "$@"

echo "========== All done =========="
