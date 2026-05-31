"""
RMNP: Row-Momentum Normalized Preconditioning optimizer.

A scalable matrix-based optimizer that replaces Muon's Newton-Schulz
orthogonalization with a single row-wise L2 normalization of the
momentum buffer, motivated by the row-wise block-diagonal dominance
of transformer gradient momentum matrices.

Reference:
    Deng et al., "RMNP: Row-Momentum Normalized Preconditioning for
    Scalable Matrix-Based Optimization", arXiv:2603.20527, 2026.

This file is adapted from the Muon optimizer
(KellerJordan/modded-nanogpt), keeping the same API surface so that
RMNP can drop in wherever Muon is used.
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


@torch.compile
def row_normalize(G: torch.Tensor) -> torch.Tensor:
    """Normalize each row of a 2D matrix to unit L2 norm."""
    return F.normalize(G, p=2, dim=-1)


class RMNP(torch.optim.Optimizer):
    """RMNP — Row-Momentum Normalized Preconditioning.

    The update rule (per matrix parameter):
        m_t = mu * m_{t-1} + g_t
        u_t = g_t + mu * m_t                          # Nesterov (optional)
        r_t = row_normalize(u_t) * max(1, sqrt(m/n))  # row-wise L2 + Muon scale
        W_t = (1 - lr * wd) * W_{t-1} - lr * r_t

    Notes:
        - Intended for 2D weight matrices only. Pass 1D/0D parameters
          (biases, LayerNorm scales) and embedding/head matrices to a
          separate AdamW group, following Muon's convention.
        - Supports distributed training: when WORLD_SIZE > 1, work is
          sharded across ranks and synchronized with all_reduce.

    Args:
        params: Iterable of 2D parameters.
        lr: Learning rate.
        momentum: Momentum coefficient mu.
        nesterov: Whether to apply Nesterov momentum (recommended).
        weight_decay: Decoupled L2 weight decay.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            total_params = sum(p.numel() for p in group['params'])
            device = group['params'][0].device
            updates_flat = torch.zeros(total_params, device=device, dtype=torch.bfloat16)

            curr_idx = 0
            for i, p in enumerate(group['params']):
                if i % self.world_size == self.rank:
                    g = p.grad
                    assert g is not None, "RMNP requires gradients for all params"
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    g = row_normalize(g)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx:curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx + p.numel()].view_as(p.data).type_as(p.data)
                p.data.mul_(1.0 - lr * weight_decay).add_(g, alpha=-lr)
                curr_idx += p.numel()

        return loss
