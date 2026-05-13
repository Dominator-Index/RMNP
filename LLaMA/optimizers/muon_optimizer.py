# Muon optimizer integration for distributed training
# Source: https://github.com/KellerJordan/Muon
# Usage: Muon for hidden layers, AdamW for others

import torch
from muon import MuonWithAuxAdam

def get_muon_optimizer(model, lr_muon=0.005, lr_adamw=0.001, weight_decay=0.1):
    """
    Returns a MuonWithAuxAdam optimizer for distributed training.
    - Muon is applied to hidden layers (ndim >= 2)
    - AdamW is applied to all other parameters
    """
    hidden_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'embed' not in n and 'lm_head' not in n]
    other_params = [p for n, p in model.named_parameters() if p.ndim < 2 or 'embed' in n or 'lm_head' in n]
    param_groups = [
        dict(params=hidden_params, use_muon=True, lr=lr_muon, weight_decay=weight_decay, momentum=0.95),
        dict(params=other_params, use_muon=False, lr=lr_adamw, betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay)
    ]
    optimizer = MuonWithAuxAdam(param_groups)
    return optimizer

def get_muon_optimizer_with_monitoring(model, lr_muon=0.005, lr_adamw=0.001, weight_decay=0.1, 
                                     enable_monitoring=True, logger=None):
    """
    Returns a MuonWithAuxAdam optimizer with momentum monitoring capabilities.
    """
    from .muon_with_monitoring import get_muon_optimizer_with_monitoring as get_monitored
    return get_monitored(model, lr_muon, lr_adamw, weight_decay, enable_monitoring, logger)

# Example usage in your training script:
# from muon_optimizer import get_muon_optimizer
# optimizer = get_muon_optimizer(model)
# (all other training code remains unchanged)

# For distributed training, MuonWithAuxAdam supports torch.distributed.
