# GPT-2 Pre-training with RMNP, Muon, and AdamW Optimizers

A comprehensive PyTorch training pipeline for GPT-2 models with multiple optimizer implementations (RMNP, Muon, AdamW). Supports distributed training across GPT-2 model sizes (Small 125M, Medium 355M, Large 770M).

## Environment Setup

### Prerequisites
- CUDA-capable GPUs (8 GPUs recommended for default configs)
- Python 3.12.0
- Conda package manager

### Installation

1. **Clone and enter the directory:**
```bash
cd RMNP
```

2. **Create conda environment:**
```bash
conda create -n rmnp_training python=3.12
conda activate rmnp_training
```

3. **Install core dependencies:**
```bash
# PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Essential packages
pip install transformers datasets wandb tiktoken numpy
pip install huggingface_hub tokenizers
pip install flash_attn einops

# Optional: Install from requirements files
# pip install -r requirements.txt
```

4. **Configure environment variables:**

**Required Configuration:** You must set the following environment variables before training:

**Option 1: Using environment variables directly**
```bash
# HuggingFace Token (Required for streaming dataset access)
export HF_TOKEN="your_huggingface_token_here"

# WandB API Key (Required for experiment tracking)
export WANDB_API_KEY="your_wandb_api_key_here"

# WandB Project Name (Optional, defaults to 'rmnp')
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

### Data Preparation

Prepare the [OpenWebText](https://huggingface.co/datasets/openwebtext) data following [nanoGPT](https://github.com/karpathy/nanoGPT/):

```bash
python data/openwebtext/prepare.py
```

For streaming datasets (OpenWebText), no manual data preparation is needed.

## Quick Start

The pipeline provides ready-to-use training scripts for different optimizer and model size combinations:

### Available Training Scripts

1. **RMNP Optimizer:**
   - `scripts/run_rmnp_small.sh` - GPT-2 Small (125M)
   - `scripts/run_rmnp_medium.sh` - GPT-2 Medium (355M)
   - `scripts/run_rmnp_large.sh` - GPT-2 Large (770M)

2. **Muon Optimizer:**
   - `scripts/run_muon_small.sh` - GPT-2 Small
   - `scripts/run_muon_medium.sh` - GPT-2 Medium
   - `scripts/run_muon_large.sh` - GPT-2 Large

3. **AdamW Baseline:**
   - `scripts/run_adamw_small.sh` - GPT-2 Small
   - `scripts/run_adamw_medium.sh` - GPT-2 Medium
   - `scripts/run_adamw_large.sh` - GPT-2 Large

4. **Streaming Data (OpenWebText):**
   - `scripts/run_*_streaming.sh` - OpenWebText streaming

### Running Training Scripts

**Basic usage:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Train with RMNP optimizer (GPT-2 Small)
./scripts/run_rmnp_small.sh

# Train with Muon optimizer (GPT-2 Medium)
./scripts/run_muon_medium.sh

# Train with streaming data (OpenWebText)
./scripts/run_rmnp_large_streaming.sh
```

**With custom parameters:**
```bash
# Override default parameters
LR=3e-3 MAX_ITERS=50000 ./scripts/run_rmnp_small.sh

# Use different number of GPUs
GPUS=4 ./scripts/run_rmnp_medium_streaming.sh
```

**Using torchrun directly:**
```bash
torchrun --standalone --nproc_per_node=4 \
    MARS/train_rmnp.py \
    config/train_gpt2_small_rmnp.py \
    --batch_size=15 \
    --gradient_accumulation_steps=8
```

### Training Configuration

All scripts use optimized defaults:
- **Total Batch Size:** 480 (distributed across GPUs)
- **Sequence Length:** 1024 tokens
- **Gradient Accumulation:** Auto-calculated based on GPU count
- **Distributed Training:** 4 GPUs (adjustable in scripts)

### Output Structure

Training outputs are saved in `Output/` directory:
```
Output/
└── rmnp-small-openwebtext-lr3e-3-rlr2e-2-..._20260129_143000.log
```

### Monitoring

- **WandB Integration:** Automatic experiment tracking
- **Logging:** Detailed console output with step timing
- **Metrics:** Loss, perplexity, throughput, and gradient norms

## Optimizer Details

### RMNP Optimizer
- **Type:** Grouped parameter optimizer with Newton-Schulz preconditioning
- **Key Features:** Separate learning rates for matrix and 1D parameters
- **Hyperparameters:** `learning_rate`, `rmnp_learning_rate`, `beta1`, `beta2`, `weight_decay`
- **Best for:** Fine-grained parameter optimization

### Muon Optimizer
- **Type:** Mixed optimizer with Newton-Schulz updates
- **Hyperparameters:** `learning_rate`, `muon_learning_rate`, `beta1`, `beta2`, `weight_decay`
- **Best for:** General pre-training tasks

### AdamW Baseline
- **Type:** Standard adaptive gradient method
- **Hyperparameters:** `learning_rate`, `beta1`, `beta2`, `weight_decay`
- **Best for:** Baseline comparison

## Hyperparameters Reference

| Model | Size | lr (AdamW) | lr (Muon) | lr (RMNP) | wd (AdamW) | wd (Muon) | wd (RMNP) |
|:-----:|:----:|:----------:|:---------:|:----------:|:----------:|:---------:|:----------:|
| GPT-2 Small | 125M | 6e-4 | 2e-2 | 2e-2 | 1e-1 | 0.0 | 0.0 |
| GPT-2 Medium | 355M | 3e-4 | 1e-2 | 1e-2 | 1e-1 | 0.0 | 0.0 |
| GPT-2 Large | 770M | 2e-4 | 6.67e-3 | 6.67e-3 | 1e-1 | 0.0 | 0.0 |

## Troubleshooting

**Common Issues:**
1. **CUDA OOM:** Reduce `--batch_size` or increase `--gradient_accumulation_steps`
2. **WandB errors:** Ensure `WANDB_API_KEY` is set correctly in environment variables
3. **HuggingFace access errors:** Ensure `HF_TOKEN` is set correctly for dataset access
4. **Missing environment variables:** All required tokens must be set before starting training

**GPU Requirements:**
- GPT-2 Small: ~8GB per GPU
- GPT-2 Medium: ~16GB per GPU
- GPT-2 Large: ~24GB per GPU

## Environment Variables Reference

| Variable | Required | Description | How to Obtain |
|----------|----------|-------------|---------------|
| `HF_TOKEN` | **Yes** (streaming) | HuggingFace access token for datasets | [HuggingFace Settings](https://huggingface.co/settings/tokens) |
| `WANDB_API_KEY` | **Yes** | WandB API key for experiment tracking | [WandB Settings](https://wandb.ai/settings) |
| `WANDB_PROJECT` | No | WandB project name (default: rmnp) | Set to your project name |
| `CUDA_VISIBLE_DEVICES` | No | GPU devices to use (default: 0,1,2,3) | Set to available GPUs |

## Acknowledgements

This repo is built upon [nanoGPT](https://github.com/karpathy/nanoGPT/), [levanter](https://github.com/stanford-crfm/levanter/) and [Sophia](https://github.com/Liuhong99/Sophia).

For support or contributions, please refer to the training logs and WandB runs for debugging information.
