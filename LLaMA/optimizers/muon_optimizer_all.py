# Muon optimizer variant: apply Muon to ALL 2D+ parameters including embed and lm_head
# Standard Muon excludes embed/lm_head (uses AdamW for them).
# This variant includes them in the Muon group for ablation study.

import torch
from muon import MuonWithAuxAdam


def get_muon_all_optimizer(model, lr_muon=0.005, lr_adamw=0.001, weight_decay=0.1):
    """
    Returns a MuonWithAuxAdam optimizer where ALL 2D+ parameters use Muon,
    including embed_tokens and lm_head.
    Only 1D/0D parameters (e.g., LayerNorm) use AdamW.
    """
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2]
    adam_params = [p for n, p in model.named_parameters() if p.ndim < 2]
    param_groups = [
        dict(params=muon_params, use_muon=True, lr=lr_muon, weight_decay=weight_decay, momentum=0.95),
        dict(params=adam_params, use_muon=False, lr=lr_adamw, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay)
    ]
    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer
