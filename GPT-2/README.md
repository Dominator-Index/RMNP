# GPT-2 Pre-training with RMNP, Muon, and AdamW Optimizers

A comprehensive PyTorch training pipeline for GPT-2 models with multiple optimizer implementations (RMNP, Muon, AdamW). Supports distributed training across GPT-2 model sizes (Small 125M, Medium 355M, Large 770M, and XL 1.5B on FineWeb-Edu).

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

The scripts come in two groups. Every group has an AdamW, a Muon, and an RMNP version.

**Streaming (no data prep).** These are the recommended scripts, and they cover every model size.
- `scripts/run_<opt>_<size>_streaming_owt.sh` streams OpenWebText. It covers Small, Medium, and Large.
- `scripts/run_<opt>_<size>_streaming_fw.sh` streams FineWeb-Edu-100B. It covers Small, Medium, Large, and **XL (1.5B)**.

So, for example, GPT-2 XL on FineWeb-Edu runs with `scripts/run_adamw_xl_streaming_fw.sh`, `scripts/run_muon_xl_streaming_fw.sh`, or `scripts/run_rmnp_xl_streaming_fw.sh`.

**Pre-tokenized OpenWebText.** These need `python data/openwebtext/prepare.py` first, and they cover Small, Medium, and Large only.
- `scripts/run_<opt>_<size>.sh` for `<opt>` in `adamw`, `muon`, `rmnp` and `<size>` in `small`, `medium`, `large`.

The exact command for every optimizer, dataset, and size is in the Running Training Scripts section below.

### Running Training Scripts

All commands below assume that you are in the `GPT-2/` directory. First move into it and make the scripts executable.

```bash
cd path/to/RMNP/GPT-2
chmod +x scripts/*.sh
```

**Setting hyperparameters.** Each streaming script (`*_streaming_owt.sh` and `*_streaming_fw.sh`) already carries the right defaults for its model size. Just run it and you get the exact recipe listed below. To change a value, put the variable in front of the command. For example, this trains GPT-2 Small with a different learning rate and leaves everything else alone:

```bash
LR=2e-3 bash scripts/run_adamw_small_streaming_fw.sh
```

The variables you can set this way are `GPUS`, `BATCH_SIZE`, `GRAD_ACC`, `LR`, `MUON_LR` or `RMNP_LR`, `WD`, `MUON_WD` or `RMNP_WD`, `MAX_ITERS`, `WARMUP`, `GRAD_CLIP`, `DATASET`, `STREAMING_TIMEOUT`, `STREAMING_RETRIES`, and `WANDB_PROJECT`. You can also add normal `--flag=value` arguments after the script name, and they go straight to `torchrun`.

The scripts are grouped by dataset, then optimizer, then model size. The two datasets never mix: OpenWebText scripts end in `_owt`, and FineWeb-Edu-100B scripts end in `_fw`. Every command below spells out its full recipe, so you can copy one, paste it, and run it as is.

---

## Dataset 1: OpenWebText (`_owt`)

These scripts stream OpenWebText, so you do not need to prepare any data first. A pre-tokenized variant also exists, and it is described in the note at the end of this section.

### 1.1 AdamW on OpenWebText
```bash
# Small  (125M)
GPUS=8 BATCH_SIZE=15 LR=6e-4 WD=1e-1 MAX_ITERS=10000 WARMUP=1000 bash scripts/run_adamw_small_streaming_owt.sh
# Medium (355M)
GPUS=8 BATCH_SIZE=15 LR=3e-4 WD=1e-1 MAX_ITERS=20000 WARMUP=2000 bash scripts/run_adamw_medium_streaming_owt.sh
# Large  (770M)
GPUS=8 BATCH_SIZE=15 LR=2e-4 WD=1e-1 MAX_ITERS=40000 WARMUP=4000 bash scripts/run_adamw_large_streaming_owt.sh
```

### 1.2 Muon on OpenWebText
```bash
# Small  (125M)
GPUS=8 BATCH_SIZE=15 LR=3e-3   MUON_LR=2e-2    WD=1e-1 MAX_ITERS=10000 WARMUP=1000 bash scripts/run_muon_small_streaming_owt.sh
# Medium (355M)
GPUS=8 BATCH_SIZE=15 LR=1.5e-3 MUON_LR=1e-2    WD=1e-1 MAX_ITERS=20000 WARMUP=2000 bash scripts/run_muon_medium_streaming_owt.sh
# Large  (770M)
GPUS=8 BATCH_SIZE=15 LR=1e-3   MUON_LR=6.67e-3 WD=1e-1 MAX_ITERS=40000 WARMUP=4000 bash scripts/run_muon_large_streaming_owt.sh
```

### 1.3 RMNP on OpenWebText
```bash
# Small  (125M)
GPUS=8 BATCH_SIZE=15 LR=3e-3   RMNP_LR=4e-3 WD=1e-1 MAX_ITERS=10000 WARMUP=1000 bash scripts/run_rmnp_small_streaming_owt.sh
# Medium (355M)
GPUS=8 BATCH_SIZE=15 LR=1.5e-3 RMNP_LR=5e-3 WD=1e-1 MAX_ITERS=20000 WARMUP=2000 bash scripts/run_rmnp_medium_streaming_owt.sh
# Large  (770M)
GPUS=8 BATCH_SIZE=15 LR=1e-3   RMNP_LR=3e-3 WD=1e-1 MAX_ITERS=40000 WARMUP=4000 bash scripts/run_rmnp_large_streaming_owt.sh
```

> **Pre-tokenized OpenWebText.** If you prefer the pre-tokenized pipeline, first run `python data/openwebtext/prepare.py`. After the data is ready, launch `bash scripts/run_<opt>_<size>.sh`, where `<opt>` is `adamw`, `muon`, or `rmnp`, and `<size>` is `small`, `medium`, or `large`. These scripts wrap a fixed 8-GPU `torchrun` call, and they do not read the environment variables listed above. So if you want to change a hyperparameter here, call `torchrun` directly as shown in the "Using torchrun directly" section.

---

## Dataset 2: FineWeb-Edu-100B (`_fw`)

These scripts stream FineWeb-Edu-100B, so again you do not need to prepare any data. This dataset also adds the **XL (1.5B)** model, which the OpenWebText recipe does not cover.

### 2.1 AdamW on FineWeb-Edu
```bash
# Small  (125M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=6e-4 WD=1e-1 MAX_ITERS=10000 WARMUP=1000 bash scripts/run_adamw_small_streaming_fw.sh
# Medium (355M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=3e-4 WD=1e-1 MAX_ITERS=20000 WARMUP=2000 bash scripts/run_adamw_medium_streaming_fw.sh
# Large  (770M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=2e-4 WD=1e-1 MAX_ITERS=40000 WARMUP=4000 bash scripts/run_adamw_large_streaming_fw.sh
# XL     (1.5B)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=2e-4 WD=1e-1 MAX_ITERS=50000 WARMUP=5000 bash scripts/run_adamw_xl_streaming_fw.sh
```

### 2.2 Muon on FineWeb-Edu
```bash
# Small  (125M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=3e-3   MUON_LR=2e-2    WD=1e-1 MAX_ITERS=10000 WARMUP=1000 bash scripts/run_muon_small_streaming_fw.sh
# Medium (355M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=1.5e-3 MUON_LR=1e-2    WD=1e-1 MAX_ITERS=20000 WARMUP=2000 bash scripts/run_muon_medium_streaming_fw.sh
# Large  (770M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=1e-3   MUON_LR=6.67e-3 WD=1e-1 MAX_ITERS=40000 WARMUP=4000 bash scripts/run_muon_large_streaming_fw.sh
# XL     (1.5B)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=1e-3   MUON_LR=6.67e-3 WD=1e-1 MAX_ITERS=50000 WARMUP=5000 bash scripts/run_muon_xl_streaming_fw.sh
```

### 2.3 RMNP on FineWeb-Edu
```bash
# Small  (125M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=3e-3   RMNP_LR=3e-3 WD=1e-1 MAX_ITERS=10000 WARMUP=1000 bash scripts/run_rmnp_small_streaming_fw.sh
# Medium (355M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=1.5e-3 RMNP_LR=2e-3 WD=1e-1 MAX_ITERS=20000 WARMUP=2000 bash scripts/run_rmnp_medium_streaming_fw.sh
# Large  (770M)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=1e-3   RMNP_LR=3e-3 WD=1e-1 MAX_ITERS=40000 WARMUP=4000 bash scripts/run_rmnp_large_streaming_fw.sh
# XL     (1.5B)
GPUS=8 BATCH_SIZE=15 GRAD_ACC=4  LR=1e-3   RMNP_LR=2e-3 WD=1e-1 MAX_ITERS=50000 WARMUP=5000 bash scripts/run_rmnp_xl_streaming_fw.sh
```

---

> The global batch size stays at 480, which equals `BATCH_SIZE × GPUS × GRAD_ACC`. With `GPUS=8` the defaults already give 480, so you do not need to change anything. If you change `GPUS`, the streaming scripts recompute `GRAD_ACC` for you and keep the global batch at 480. When a GPU runs out of memory, lower `BATCH_SIZE` and raise `GRAD_ACC` by the same factor, so that the product stays the same.

**Examples that override the defaults**

```bash
# Sweep the RMNP matrix LR on FineWeb-Edu Small
RMNP_LR=2e-3 bash scripts/run_rmnp_small_streaming_fw.sh

# Run Muon Medium on 4 GPUs. GRAD_ACC is recomputed so the global batch stays 480.
GPUS=4 bash scripts/run_muon_medium_streaming_owt.sh

# Pass an extra flag straight to torchrun. Anything after the script name is forwarded.
bash scripts/run_adamw_small_streaming_fw.sh --min_lr=1e-5
```

### Using torchrun directly

```bash
torchrun --standalone --nproc_per_node=8 \
    RMNP/train_rmnp.py \
    config/train_gpt2_small_rmnp.py \
    --batch_size=15 \
    --gradient_accumulation_steps=4
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

These are the exact learning rates baked into the configs and scripts. The two datasets use the same AdamW and Muon learning rates, but RMNP uses a slightly different matrix learning rate on each one, so they are listed separately below.

For Muon and RMNP, two learning rates are shown as `lr` / `matrix_lr`. The first one is the AdamW learning rate applied to the 1D parameters (LayerNorm and biases), and the second one is the matrix learning rate applied to the 2D weights. Across every run the weight decay is `1e-1` on the AdamW part and `0.0` on the matrix part, the warmup is 10% of the steps, and the global batch size is 480.

### OpenWebText (`_owt`)

| Model | Size | Steps | AdamW `lr` | Muon `lr` / `matrix_lr` | RMNP `lr` / `matrix_lr` |
|:------|:----:|:-----:|:----------:|:-----------------------:|:-----------------------:|
| GPT-2 Small  | 125M | 10000 | 6e-4 | 3e-3 / 2e-2    | 3e-3 / 4e-3 |
| GPT-2 Medium | 355M | 20000 | 3e-4 | 1.5e-3 / 1e-2  | 1.5e-3 / 5e-3 |
| GPT-2 Large  | 770M | 40000 | 2e-4 | 1e-3 / 6.67e-3 | 1e-3 / 3e-3 |

### FineWeb-Edu-100B (`_fw`)

| Model | Size | Steps | AdamW `lr` | Muon `lr` / `matrix_lr` | RMNP `lr` / `matrix_lr` |
|:------|:----:|:-----:|:----------:|:-----------------------:|:-----------------------:|
| GPT-2 Small  | 125M | 10000 | 6e-4 | 3e-3 / 2e-2    | 3e-3 / 3e-3 |
| GPT-2 Medium | 355M | 20000 | 3e-4 | 1.5e-3 / 1e-2  | 1.5e-3 / 2e-3 |
| GPT-2 Large  | 770M | 40000 | 2e-4 | 1e-3 / 6.67e-3 | 1e-3 / 3e-3 |
| GPT-2 XL     | 1.5B | 50000 | 2e-4 | 1e-3 / 6.67e-3 | 1e-3 / 2e-3 |

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

## Contact

Questions and feedback are welcome. Feel free to email Zhuoli Ouyang at oyzl2004@gmail.com or Zhuoli.Ouyang@dartmouth.edu, or our collaborator Shenyang Deng at shenyang.deng.gr@dartmouth.edu.
