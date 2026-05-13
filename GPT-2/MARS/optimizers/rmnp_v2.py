"""
Adapted from KellerJordan/modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py

RMNP-v2 optimizer: Shape-aware Row-Normalized Nesterov with Polynomial Scaling.

Same as RMNP, but adds a transpose check (matching Muon's convention) so that
tall matrices (m > n in PyTorch shape) are normalized along the longer axis.
Concretely, if the gradient has PyTorch shape (m, n) with m > n, we transpose
to a wide shape, apply row L2 normalization, then transpose back; this is
equivalent to column-normalizing the original tensor and aligns with the
diagonal approximation of the natural Newton-Schulz form for tall matrices
(M(M^T M)^{-1/2} rather than (MM^T)^{-1/2} M).

All hyperparameters and the post-normalization scaling factor
max(1, sqrt(m/n))**0.5 are unchanged from RMNP.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import os

@torch.compile
def row_normalize(G):
    """
    Shape-aware row/column L2 normalization.

    If G has PyTorch shape (m, n) with m <= n (wide), normalizes each row to
    unit L2 norm. If m > n (tall), transposes G so the longer dim becomes
    dim=-1, applies the same row L2 normalization, and transposes back; this
    is equivalent to column-normalizing the original tensor.

    Args:
        G: Input tensor (2D)

    Returns:
        Normalized tensor of the same shape as G.
    """
    if G.size(0) > G.size(1):
        return F.normalize(G.T, p=2, dim=-1).T
    return F.normalize(G, p=2, dim=-1)

class RMNP_v2(torch.optim.Optimizer):
    """
    RMNP-v2 - Shape-aware Row-Normalized Nesterov with Polynomial Scaling

    Same as RMNP but routes tall matrices (m > n in PyTorch shape) through a
    transpose-row_normalize-transpose path, matching Muon's transpose
    convention. For wide matrices (m <= n) the behavior is identical to RMNP.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        weight_decay: L2 weight decay. (default: 0.0)
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(self.world_size) == int(self.rank):
                    g = p.grad
                    assert g is not None
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    # Shape-aware row/column normalization (transpose for tall matrices)
                    g = row_normalize(g)
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.mul_(1.-lr*weight_decay).add_(g, alpha=-lr)
                curr_idx += p.numel()
