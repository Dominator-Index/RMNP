#!/bin/bash
# Universal training driver for LLaMA pretraining.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
MODEL_SIZE="60m"
OPTIMIZER="muon"
NUM_GPUS="8"
LEARNING_RATE="0.001"
LR_MATRIX=""
LR_ADAM=""
NUM_STEPS="20000"
BATCH_SIZE="32"
TOTAL_BATCH_SIZE="512"
WARMUP_STEPS="2000"
WEIGHT_DECAY="1e-5"
SAVE_EVERY="10000"
EVAL_EVERY="1000"
COMPUTE_DD="false"
WANDB_PROJECT="${WANDB_PROJECT:-mars-c4}"
WANDB_NAME=""
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)       MODEL_SIZE="$2";        shift 2 ;;
        --optimizer)        OPTIMIZER="$2";         shift 2 ;;
        --num_gpus)         NUM_GPUS="$2";          shift 2 ;;
        --lr)               LEARNING_RATE="$2";     shift 2 ;;
        --lr_matrix)        LR_MATRIX="$2";         shift 2 ;;
        --lr_adam)          LR_ADAM="$2";           shift 2 ;;
        --num_steps)        NUM_STEPS="$2";         shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";        shift 2 ;;
        --total_batch_size) TOTAL_BATCH_SIZE="$2";  shift 2 ;;
        --warmup_steps)     WARMUP_STEPS="$2";      shift 2 ;;
        --weight_decay)     WEIGHT_DECAY="$2";      shift 2 ;;
        --save_every)       SAVE_EVERY="$2";        shift 2 ;;
        --eval_every)       EVAL_EVERY="$2";        shift 2 ;;
        --save_dir)         SAVE_DIR="$2";          shift 2 ;;
        --wandb_name)       WANDB_NAME="$2";        shift 2 ;;
        --continue_from)    CONTINUE_FROM="$2";     shift 2 ;;
        --compute_dd)       COMPUTE_DD="true";      shift 1 ;;
        --help|-h)
            echo "Usage: $0 --model_size {60m|135m|350m|1b} --optimizer {muon|muon_all|RMNP|rmnp_all|adamw} --lr_matrix LR --lr_adam LR [options]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate
if [[ ! "$MODEL_SIZE" =~ ^(60m|130m|135m|350m|1b)$ ]]; then
    echo "Error: invalid model_size '$MODEL_SIZE'"; exit 1
fi
if [[ ! "$OPTIMIZER" =~ ^(muon|muon_all|RMNP|rmnp_all|adamw)$ ]]; then
    echo "Error: invalid optimizer '$OPTIMIZER'"; exit 1
fi
if [[ "$OPTIMIZER" != "adamw" ]] && [[ -z "$LR_MATRIX" || -z "$LR_ADAM" ]]; then
    echo "Error: $OPTIMIZER requires --lr_matrix and --lr_adam"; exit 1
fi

GRAD_ACCUM=$(( TOTAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS) ))
if [[ $(( GRAD_ACCUM * BATCH_SIZE * NUM_GPUS )) -ne $TOTAL_BATCH_SIZE ]]; then
    echo "Error: total_batch_size not divisible by batch_size * num_gpus"; exit 1
fi

case $MODEL_SIZE in
    60m)        MODEL_CONFIG="$PROJECT_DIR/configs/llama_60m.json" ;;
    130m|135m)  MODEL_CONFIG="$PROJECT_DIR/configs/llama_135m.json" ;;
    350m)       MODEL_CONFIG="$PROJECT_DIR/configs/llama_350m.json" ;;
    1b)         MODEL_CONFIG="$PROJECT_DIR/configs/llama_1b.json" ;;
esac

[[ -z "$WANDB_NAME" ]] && WANDB_NAME="llama-${MODEL_SIZE}-${OPTIMIZER}-matrix${LR_MATRIX}-adam${LR_ADAM}"
[[ -z "$SAVE_DIR" ]]   && SAVE_DIR="$PROJECT_DIR/checkpoints/llama_${MODEL_SIZE}-${TIMESTAMP}-${OPTIMIZER}"

mkdir -p "$SAVE_DIR"
echo "model=$MODEL_SIZE opt=$OPTIMIZER lr_matrix=$LR_MATRIX lr_adam=$LR_ADAM steps=$NUM_STEPS gpus=$NUM_GPUS"

ARGS=(
    --model_config "$MODEL_CONFIG"
    --optimizer "$OPTIMIZER"
    --lr "$LEARNING_RATE"
    --num_training_steps "$NUM_STEPS"
    --batch_size "$BATCH_SIZE"
    --total_batch_size "$TOTAL_BATCH_SIZE"
    --gradient_accumulation "$GRAD_ACCUM"
    --warmup_steps "$WARMUP_STEPS"
    --weight_decay "$WEIGHT_DECAY"
    --save_every "$SAVE_EVERY"
    --eval_every "$EVAL_EVERY"
    --save_dir "$SAVE_DIR"
    --wandb_name "$WANDB_NAME"
    --dtype "bfloat16"
    --max_length 256
    --scheduler "cosine"
    --grad_clipping 1.0
)
[[ -n "$LR_MATRIX" ]]         && ARGS+=(--lr_matrix "$LR_MATRIX")
[[ -n "$LR_ADAM" ]]           && ARGS+=(--lr_adam "$LR_ADAM")
[[ "$COMPUTE_DD" == "true" ]] && ARGS+=(--compute_dd)
[[ -n "$CONTINUE_FROM" ]]     && ARGS+=(--continue_from "$CONTINUE_FROM")

cd "$PROJECT_DIR"
exec torchrun --standalone --nproc_per_node=$NUM_GPUS torchrun_main.py "${ARGS[@]}"
