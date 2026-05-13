# LLaMA Pre-training with Muon and RMNP Optimizers

A clean and efficient PyTorch training pipeline for LLaMA models using Muon and RMNP optimizers. Supports distributed training across multiple model sizes (60M, 130M, 350M parameters).

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

The pipeline provides ready-to-use training scripts for different optimizer and model size combinations:

### Available Training Scripts

1. **Muon Optimizer:**
   - `scripts/train_muon_60m.sh` - 60M parameters
   - `scripts/train_muon_130m.sh` - 130M parameters  
   - `scripts/train_muon_350m.sh` - 350M parameters

2. **RMNP Optimizer:**
   - `scripts/train_RMNP_60m.sh` - 60M parameters
   - `scripts/train_RMNP_130m.sh` - 130M parameters
   - `scripts/train_RMNP_350m.sh` - 350M parameters

### Running Training Scripts

**Basic usage:**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Train with Muon optimizer (60M model)
./scripts/train_muon_60m.sh

# Train with RMNP optimizer (130M model)  
./scripts/train_RMNP_130m.sh
```

**With custom parameters:**
```bash
# Override default parameters
./scripts/train_muon_60m.sh --num_steps 50000 --lr 0.002

# Continue from checkpoint
./scripts/train_RMNP_130m.sh --continue_from ./checkpoints/previous_run/model_10000
```

### Training Configuration

All scripts use optimized defaults:
- **Total Batch Size:** 512 (distributed across GPUs)
- **Sequence Length:** 256 tokens
- **Mixed Precision:** bfloat16
- **Gradient Accumulation:** Auto-calculated
- **Distributed Training:** 4 GPUs (adjustable in scripts)

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
./scripts/train_universal.sh \
    --model_size 130m \
    --optimizer RMNP \
    --lr_matrix 0.005 \
    --lr_adam 0.001 \
    --num_steps 20000 \
    --batch_size 24
```

### Multi-Node Training
Modify the torchrun parameters in scripts:
```bash
torchrun --nnodes=2 --nproc_per_node=4 --master_addr=node1 --master_port=29500 ...
```

## Optimizer Details

### Muon Optimizer
- **Type:** Mixed AdamW optimizer
- **Usage:** Single learning rate (`--lr`)
- **Best for:** General pre-training tasks

### RMNP Optimizer  
- **Type:** Grouped parameter optimizer
- **Usage:** Dual learning rates (`--lr_matrix`, `--lr_adam`)
- **Best for:** Fine-tuned parameter optimization

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

For support or contributions, please refer to the training logs and WandB runs for debugging information.