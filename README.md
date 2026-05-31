# RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

This repository contains the official implementation of **RMNP (Row-Momentum Normalized Preconditioning)**, a scalable matrix-based optimizer for large model pre-training. RMNP replaces the Newton–Schulz (NS) iteration used by **Muon** with a simple per-row $\ell_2$ normalization of the momentum buffer, which is provably equivalent to Muon's orthogonalization step under the row-wise block-diagonal dominance regime that we observe to hold (and grow stronger) for transformer gradient momentum matrices in practice.

## Algorithms

### Muon

```math
\begin{aligned}
        &\textbf{Input:} \ \eta,\ \mu,\ K \ \text{(NS steps)} \\
        &\textbf{Initialize:}\ \theta_0,\ m_0 \leftarrow 0 \\
        &\textbf{for } t=1 \text{ to } T \text{ do} \\
        &\quad g_t \leftarrow \nabla f(\theta_{t-1}) \\
        &\quad m_t \leftarrow \mu\, m_{t-1} + g_t \\
        &\quad \color{red}{O_t \leftarrow \text{NewtonSchulz}(m_t,\ K)} \\
        &\quad \theta_t \leftarrow \theta_{t-1} - \eta\, O_t \\
        &\textbf{end for}
    \end{aligned}
```

### RMNP (Ours)

```math
\begin{aligned}
        &\textbf{Input:} \ \eta,\ \mu,\ \epsilon \\
        &\textbf{Initialize:}\ W_0,\ M_0 \leftarrow 0 \\
        &\textbf{for } t=1 \text{ to } T \text{ do} \\
        &\quad G_t \leftarrow \nabla_W f_t(W_{t-1}) \\
        &\quad M_t \leftarrow \mu\, M_{t-1} + (1-\mu)\, G_t \\
        &\quad \color{blue}{R_t \leftarrow \mathrm{RowNormalize}(M_t;\ \epsilon)} \\
        &\quad W_t \leftarrow W_{t-1} - \eta\, R_t \\
        &\textbf{end for}
    \end{aligned}
```

with
```math
\bigl[\mathrm{RowNormalize}(M;\epsilon)\bigr]_{i,:} \;=\; \frac{M_{i,:}}{\lVert M_{i,:} \rVert_2 + \epsilon}.
```

### Diagonal-Dominance Monitoring

To verify the condition under which $\mathrm{RowNormalize}(M_t) \approx U V^\top$ holds, we monitor the row-wise diagonal-dominance ratio of the Gram matrix $V_t V_t^\top$ throughout training. For row $i$ we define

```math
r_i \;=\; \frac{\bigl|(V_t V_t^\top)_{ii}\bigr|}{\tfrac{1}{m-1}\sum_{j\neq i}\bigl|(V_t V_t^\top)_{ij}\bigr|},
```

and aggregate across rows to obtain $r_{\text{avg}}$, $r_{\min}$, $r_{\max}$. Averaging these three statistics over all matrix parameters in the network gives the global metrics $\overline{r_{\text{avg}}}$, $\overline{r_{\min}}$, $\overline{r_{\max}}$. A value $r_i > 1$ means the diagonal entry exceeds the mean off-diagonal magnitude in row $i$ — the larger, the closer $V_t V_t^\top$ is to a diagonal matrix.

![Global diagonal-dominance ratios $\overline{r_{\text{avg}}}, \overline{r_{\min}}, \overline{r_{\max}}$ across GPT-2 (Small/Medium/Large, top) and LLaMA (60M/130M/350M, bottom). X-axis: relative training progress (%); y-axis: log scale; red dashed line $y=1$ marks the dominance threshold.](assets/diagonal_dominance_ratio.png)

**Observations.** Across all six configurations and the full training trajectory: $\overline{r_{\min}}$ stays comfortably above the $y=1$ threshold, $\overline{r_{\text{avg}}}$ consistently exceeds $5$, and $\overline{r_{\max}}$ reaches the order of tens. More importantly, **diagonal dominance strengthens monotonically as model size grows** — GPT-2 Large and LLaMA 350M exhibit visibly higher $\overline{r}$ across all three statistics than their smaller counterparts. This indicates that the row-wise block-diagonal dominance underlying RMNP is not an artefact of small scale; it becomes *more* pronounced as models scale, making RMNP an increasingly favorable replacement for Muon's NS iteration at scale.

**Key idea.** When $M_t$ is row-diagonally dominant (empirically observed and strengthening with scale), the leading singular directions of $M_t$ align with its rows, and the orthogonal factor from $M_t = U\Sigma V^\top$ satisfies $U V^\top \approx \mathrm{RowNormalize}(M_t)$. RMNP therefore matches Muon's update direction while replacing the iterative NS polynomial (multiple matmuls per step) with a single elementwise normalization — yielding lower wall-clock cost and friendlier scaling to large hidden dimensions.

## Main Results

### Perplexity

![Final validation perplexity (lower is better) across three pretraining settings. **Left:** LLaMA on C4 — 60M (1B tokens), 130M (2B), 350M (6B), 1B (9B). **Middle:** GPT-2 on FineWeb-Edu-100B — Small (125M), Medium (355M), Large (770M), XL (1.5B). **Right:** GPT-2 on OpenWebText — Small (5B tokens), Medium (10B), Large (20B). RMNP attains the lowest perplexity in every cell.](assets/main_results_bar.png)

RMNP matches or exceeds Muon's perplexity across **every** model scale and dataset, consistent with the diagonal-dominance trend reported above.

### Preconditioner Wall-Clock Time

<p align="center">
  <img src="assets/time_scaling.png" alt="Wall-clock time for 100 preconditioning steps of RMNP vs. Muon as GPT-2 model size scales from 60M to 1.5B." width="55%">
</p>

RMNP's row normalization is **13×–44× faster** than Muon's Newton–Schulz orthogonalization on GPT-2 models from 60M to 1.5B (measured over 100 steps with batch size 16 on a single RTX Pro 6000 GPU), and the gap widens with model size: as Newton–Schulz becomes the dominant bottleneck at scale, RMNP's lightweight preconditioner becomes increasingly attractive for very large models.

## Repository Layout

```
RMNP/
├── GPT-2/        # GPT-2 (125M / 355M / 770M / XL) pre-training pipeline
│   ├── MARS/             # model & training entrypoints (train_{adamw,muon,rmnp}.py)
│   ├── config/           # per-(size, optimizer) training configs
│   ├── scripts/          # ready-to-run shell launchers
│   └── data/             # OpenWebText preparation (nanoGPT-style)
└── LLaMA/        # LLaMA (60M / 135M / 350M / 1B) pre-training pipeline
    ├── optimizers/       # RMNP_optimizer.py, muon_optimizer.py, *_v2 variants
    ├── configs/          # llama_{60m,135m,350m,1b}.json model configs
    ├── scripts/          # per-(size, optimizer) launchers
    └── torchrun_main.py  # distributed training entrypoint
```

Both sub-projects ship three optimizer baselines — **AdamW**, **Muon**, **RMNP** — so that results can be reproduced under matched data, schedule, and hyperparameters.

## Installation

We recommend Python **3.12** with CUDA-capable GPUs. Create a fresh environment and install the pinned dependencies:

```bash
git clone https://github.com/Dominator-Index/RMNP.git
cd RMNP

conda create -n rmnp python=3.12 -y
conda activate rmnp

pip install -r requirements.txt
```

> `flash-attn` requires a working CUDA toolchain and may take several minutes to build; if it fails, install it separately with `pip install flash-attn --no-build-isolation` after `torch` is in place. The setup is fully compatible with the upstream [MARS](https://github.com/AGI-Arena/MARS) repository, so its install instructions also work.

## Quick Start

Each sub-project is self-contained; see its local README for dataset preparation and per-script hyperparameters:

- [`GPT-2/README.md`](GPT-2/README.md) — GPT-2 pre-training on **OpenWebText** (Small / Medium / Large) and **FineWeb-Edu** (Small / Medium / Large / XL).
- [`LLaMA/README.md`](LLaMA/README.md) — LLaMA pre-training (60M – 1B) with `torchrun`.

Once the environment is ready, launch a run with:

```bash
# GPT-2 Small with RMNP
cd GPT-2
export HF_TOKEN=...        # for streaming datasets
export WANDB_API_KEY=...
bash scripts/run_rmnp_small.sh

# LLaMA 60M with RMNP
cd LLaMA
bash scripts/train_RMNP_60m.sh
```

## Using the RMNP Optimizer in Your Own Code

A standalone optimizer package lives at [`rmnp/`](rmnp/). Install via PyPI:

```bash
pip install rmnp
```

Use it like any `torch.optim.Optimizer`. Following Muon's convention, route 2D weight matrices through `RMNP` and keep 1D/0D parameters plus the embedding/head on AdamW:

```python
import torch
from rmnp import RMNP

matrix_params = [p for p in model.parameters() if p.ndim >= 2]
other_params  = [p for p in model.parameters() if p.ndim < 2]

rmnp_opt  = RMNP(matrix_params, lr=2e-2, momentum=0.95, weight_decay=0.0)
adamw_opt = torch.optim.AdamW(other_params, lr=3e-4, weight_decay=0.1)

# In the training loop: call .step() on both, .zero_grad() on both.
```

Distributed training works out of the box: when `WORLD_SIZE > 1`, updates are sharded across ranks and synchronized via `all_reduce`.
## Citation

If you find this work useful, please cite:

```bibtex
@article{deng2026rmnp,
  title   = {RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization},
  author  = {Deng, Shenyang and Ouyang, Zhuoli and Pang, Tianyu and Liu, Zihang and Jin, Ruochen and Yu, Shuhua and Yang, Yaoqing},
  journal = {arXiv preprint arXiv:2603.20527},
  year    = {2026}
}
```

## Acknowledgements

This repository is built upon [MARS](https://github.com/AGI-Arena/MARS) and [GaLore](https://github.com/jiaweizzhao/GaLore). We thank the authors for open-sourcing their codebases.

