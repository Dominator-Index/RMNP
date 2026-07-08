"""
Adapted from KellerJordan/modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py

RMNP optimizer: Row-Normalized Nesterov with Polynomial Scaling.
Similar to Muon but uses row normalization instead of Newton-Schulz orthogonalization.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
import os

@torch.compile
def row_normalize(G):
    """
    Row normalization: normalize each row of G to unit L2 norm.

    Design note (intentional difference from Muon):
        Muon's Newton-Schulz step transposes any tall matrix (m > n) so that the
        iteration runs on the smaller min(m, n) dimension. RMNP does NOT transpose.
        We always normalize G in its native (out_features, in_features) layout, i.e.
        we normalize each *row* (each output neuron's fan-in update) regardless of
        whether the matrix is tall or wide.

        This keeps the normalized dimension consistent across every layer (always the
        output/row dimension). Empirically we found that this native, no-transpose
        variant trains better than the transpose-aligned (min(m, n)) variant that
        would exactly mirror Muon, so the no-transpose behavior below is deliberate
        and should be kept.

    Args:
        G: Input tensor to normalize (2D), shape (m, n).

    Returns:
        Row-normalized tensor where each of the m rows has unit L2 norm.
    """
    return F.normalize(G, p=2, dim=-1)

class RMNP(torch.optim.Optimizer):
    """
    RMNP - Row-Normalized Nesterov with Polynomial Scaling

    Similar to Muon but uses row normalization instead of Newton-Schulz orthogonalization.
    Each row of the gradient update is normalized to unit L2 norm.

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
                    # Use row normalization instead of Newton-Schulz orthogonalization
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
