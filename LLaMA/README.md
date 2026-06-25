# LLaMA Pre-training with Muon and RMNP Optimizers

A clean and efficient PyTorch training pipeline for LLaMA models using Muon and RMNP optimizers (plus their `_all` variants). Supports distributed training across multiple model sizes (60M, 135M, 350M, 1B parameters) on streaming C4.

## Environment Setup

### Prerequisites
- CUDA-capable GPUs (4+ GPUs recommended)
- Python 3.9+
- Conda package manager

### Installation

1. **Clone and enter the directory:**
```bash
cd LLaMA_PreTraining
```

2. **Create conda environment:**
```bash
conda create -n llama_training python=3.9
conda activate llama_training
```

3. **Install core dependencies:**
```bash
# PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Essential packages
pip install transformers datasets wandb loguru tqdm
pip install huggingface_hub tokenizers
pip install numpy scipy matplotlib seaborn
pip install muon-optimizer

# Optional: Install from requirements files
# pip install -r requirements_pip.txt
```

4. **Configure environment variables:**

⚠️ **Required Configuration:** You must set the following environment variables before training:

**Option 1: Using environment variables directly**
```bash
# HuggingFace Token (Required for dataset access)
export HF_TOKEN="your_huggingface_token_here"

# WandB API Key (Required for experiment tracking)
export WANDB_API_KEY="your_wandb_api_key_here"

# WandB Project Name (Optional, defaults to 'llama-pretraining')
export WANDB_PROJECT="your_project_name_here"
```

**Option 2: Using .env file (recommended)**
```bash
# Copy the template and fill in your values
cp .env.template .env
# Edit .env file with your actual tokens
nano .env
# Load environment variables
source .env
```

**How to get these tokens:**
- **HF_TOKEN:** Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)
- **WANDB_API_KEY:** Get from [WandB Settings](https://wandb.ai/settings)

## Quick Start

The pipeline provides ready-to-use training scripts for every optimizer and model size.

### Available Training Scripts

There are five optimizers, and each one has a script for every model size (60M, 135M, 350M, and 1B).

- **AdamW** (baseline): `scripts/train_adamw_<size>.sh`
- **Muon**: `scripts/train_muon_<size>.sh`
- **Muon-All** (embedding and lm_head also go through Muon): `scripts/train_muon_all_<size>.sh`
- **RMNP**: `scripts/train_RMNP_<size>.sh`
- **RMNP-All** (embedding and lm_head also go through RMNP): `scripts/train_rmnp_all_<size>.sh`

Here `<size>` is `60m`, `135m`, `350m`, or `1b`. Every script streams C4, so no data preparation is needed. The exact command and recipe for each one is in the Running Training Scripts section below, and all of the learning rates are collected in the Hyperparameters Reference table near the end.

### Data

Training streams **C4** (`allenai/c4`, English) straight from the HuggingFace Hub, so you do not need to download or tokenize anything in advance. If you start hitting rate limits, set `HF_TOKEN`.

### Running Training Scripts

All commands assume you are in the `LLaMA/` directory:

```bash
cd path/to/RMNP/LLaMA
chmod +x scripts/*.sh
```

Each `scripts/train_<opt>_<size>.sh` is a small wrapper around `train_universal.sh` that fills in the recipe for one model size. Every script runs on 8 GPUs with batch size 64 and a total batch of 512, which works out to gradient accumulation 1. The sequence length is 256 and training runs in bfloat16. To change a setting, append the matching flag and it overrides the default. For example, this runs a shorter Muon job with a different matrix learning rate:

```bash
bash scripts/train_muon_60m.sh --lr_matrix 0.02 --num_steps 5000
```

The scripts are grouped by optimizer, then by model size. There are five optimizers in total. AdamW is the baseline and takes a single learning rate through `--lr`. Muon and RMNP run the matrix optimizer on the 2D weights only, so the embedding and lm_head still use AdamW. The `_all` variants send the embedding and lm_head through the matrix optimizer as well. The comment above each command lists the exact learning rates and step count, and warmup is always 10% of the steps.

#### 1. AdamW  (baseline, single learning rate)
```bash
# 60M   lr 1e-3, steps 10000
bash scripts/train_adamw_60m.sh
# 135M  lr 1e-3, steps 20000
bash scripts/train_adamw_135m.sh
# 350M  lr 1e-3, steps 60000
bash scripts/train_adamw_350m.sh
# 1B    lr 6e-4, steps 90000
bash scripts/train_adamw_1b.sh
```

#### 2. Muon  (matrix on 2D weights, embed/lm_head via AdamW)
```bash
# 60M   lr_matrix 0.01,  lr_adam 1e-3, steps 10000
bash scripts/train_muon_60m.sh
# 135M  lr_matrix 0.01,  lr_adam 1e-3, steps 20000
bash scripts/train_muon_135m.sh
# 350M  lr_matrix 0.004, lr_adam 1e-3, steps 60000
bash scripts/train_muon_350m.sh
# 1B    lr_matrix 0.001, lr_adam 6e-4, steps 90000
bash scripts/train_muon_1b.sh
```

#### 3. Muon-All  (embed/lm_head also through Muon)
```bash
# 60M   lr_matrix 0.03,  lr_adam 1e-3, steps 10000
bash scripts/train_muon_all_60m.sh
# 135M  lr_matrix 0.01,  lr_adam 1e-3, steps 20000
bash scripts/train_muon_all_135m.sh
# 350M  lr_matrix 0.01,  lr_adam 1e-3, steps 60000
bash scripts/train_muon_all_350m.sh
# 1B    lr_matrix 0.005, lr_adam 6e-4, steps 90000
bash scripts/train_muon_all_1b.sh
```

#### 4. RMNP  (matrix on 2D weights, embed/lm_head via AdamW)

For RMNP, `lr_adam` is kept equal to `lr_matrix`.
```bash
# 60M   lr_matrix = lr_adam 0.005, steps 10000
bash scripts/train_RMNP_60m.sh
# 135M  lr_matrix = lr_adam 0.03,  steps 20000
bash scripts/train_RMNP_135m.sh
# 350M  lr_matrix = lr_adam 0.005, steps 60000
bash scripts/train_RMNP_350m.sh
# 1B    lr_matrix = lr_adam 0.005, steps 90000
bash scripts/train_RMNP_1b.sh
```

#### 5. RMNP-All  (embed/lm_head also through RMNP)
```bash
# 60M   lr_matrix = lr_adam 0.01,  steps 10000
bash scripts/train_rmnp_all_60m.sh
# 135M  lr_matrix = lr_adam 0.02,  steps 20000
bash scripts/train_rmnp_all_135m.sh
# 350M  lr_matrix = lr_adam 0.005, steps 60000
bash scripts/train_rmnp_all_350m.sh
# 1B    lr_matrix = lr_adam 0.005, steps 90000
bash scripts/train_rmnp_all_1b.sh
```

**Override a default or resume from a checkpoint**

```bash
# Shorter run with a different matrix learning rate
bash scripts/train_muon_60m.sh --lr_matrix 0.02 --num_steps 5000

# Continue from a saved checkpoint
bash scripts/train_RMNP_135m.sh --continue_from ./checkpoints/previous_run/model_10000
```

### Training Configuration

All scripts use these defaults (override by appending flags):
- **GPUs:** 8 (`--num_gpus 8`)
- **Total Batch Size:** 512 (`--batch_size 64` × 8 GPUs → gradient accumulation 1)
- **Sequence Length:** 256 tokens
- **Mixed Precision:** bfloat16
- **Warmup:** 10% of `--num_steps`

### Output Structure

Training outputs are saved in `checkpoints/` with timestamped directories:
```
checkpoints/
└── llama_60m-2026-01-29-14-30-00-muon-lr0.001/
    ├── model_5000/          # Intermediate checkpoint
    ├── model_10000/         # Intermediate checkpoint  
    ├── model_final/         # Final model
    ├── training_params.json # Training configuration
    └── training_command.log # Command used
```

### Monitoring

- **WandB Integration:** Automatic experiment tracking
- **Logging:** Detailed console output with step timing
- **Metrics:** Loss, perplexity, throughput, and gradient norms

## Advanced Usage

### Custom Training
```bash
# Use the universal script for full customization
bash scripts/train_universal.sh \
    --model_size 135m \
    --optimizer RMNP \
    --num_gpus 8 \
    --lr_matrix 0.03 \
    --lr_adam 0.03 \
    --num_steps 20000 \
    --batch_size 64 \
    --total_batch_size 512
```

### Multi-Node Training
Modify the torchrun parameters in scripts:
```bash
torchrun --nnodes=2 --nproc_per_node=4 --master_addr=node1 --master_port=29500 ...
```

## Optimizer Details

### AdamW
- **Use:** the baseline. It takes one learning rate, which you set with `--lr`.

### Muon
- **Use:** it runs the matrix update on the 2D weights and AdamW on everything else. Set `--lr_matrix` for the matrix part and `--lr_adam` for the AdamW part.
- **`muon_all` variant:** it sends the embedding and lm_head through Muon as well.

### RMNP
- **Use:** the same split as Muon, except the matrix step is RMNP. Set `--lr_matrix` and `--lr_adam`. In all of our recipes we keep `--lr_adam` equal to `--lr_matrix`.
- **`rmnp_all` variant:** it sends the embedding and lm_head through RMNP as well.

## Hyperparameters Reference

Every run shares the same setup. It uses 8 GPUs, a per-GPU batch size of 64, a total batch of 512, a sequence length of 256, and bfloat16. The weight decay is `0.1` and the warmup is 10% of the steps. The learning rates below are the exact values baked into each script.

For Muon and Muon-All, the two numbers are written as `lr` / `matrix_lr`. The first one is the AdamW learning rate on the 1D parameters, and the second one is the matrix learning rate on the 2D weights. For RMNP and RMNP-All we keep the AdamW learning rate equal to the matrix learning rate, so a single number is shown.

| Model | Steps | AdamW `lr` | Muon `lr` / `matrix_lr` | Muon-All `lr` / `matrix_lr` | RMNP `lr` = `matrix_lr` | RMNP-All `lr` = `matrix_lr` |
|:------|:-----:|:----------:|:-----------------------:|:---------------------------:|:-----------------------:|:---------------------------:|
| LLaMA 60M  | 10000 | 1e-3 | 1e-3 / 0.01  | 1e-3 / 0.03  | 0.005 | 0.01  |
| LLaMA 135M | 20000 | 1e-3 | 1e-3 / 0.01  | 1e-3 / 0.01  | 0.03  | 0.02  |
| LLaMA 350M | 60000 | 1e-3 | 1e-3 / 0.004 | 1e-3 / 0.01  | 0.005 | 0.005 |
| LLaMA 1B   | 90000 | 6e-4 | 6e-4 / 0.001 | 6e-4 / 0.005 | 0.005 | 0.005 |

## Troubleshooting

**Common Issues:**
1. **CUDA OOM:** Reduce `--batch_size` or enable `--activation_checkpointing`
2. **Missing muon:** Install with `pip install muon-optimizer`
3. **WandB errors:** Ensure `WANDB_API_KEY` is set correctly in environment variables
4. **HuggingFace access errors:** Ensure `HF_TOKEN` is set correctly for dataset access
5. **Missing environment variables:** All required tokens must be set before starting training

**GPU Requirements:**
- 60M model: ~2GB per GPU
- 130M model: ~4GB per GPU  
- 350M model: ~8GB per GPU

## Environment Variables Reference

| Variable | Required | Description | How to Obtain |
|----------|----------|-------------|---------------|
| `HF_TOKEN` | **Yes** | HuggingFace access token for datasets | [HuggingFace Settings](https://huggingface.co/settings/tokens) |
| `WANDB_API_KEY` | **Yes** | WandB API key for experiment tracking | [WandB Settings](https://wandb.ai/settings) |
| `WANDB_PROJECT` | No | WandB project name (default: llama-pretraining) | Set to your project name |

## Contact

Questions and feedback are welcome. Feel free to email Zhuoli Ouyang at oyzl2004@gmail.com or Zhuoli.Ouyang@dartmouth.edu, or our collaborator Shenyang Deng at shenyang.deng.gr@dartmouth.edu.