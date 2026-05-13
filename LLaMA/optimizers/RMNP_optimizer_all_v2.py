# RMNP-All-v2 optimizer: apply RMNP-v2 to ALL 2D+ parameters including embed and lm_head
# Same algorithm as RMNP_optimizer_all.py but uses the shape-aware row/column
# normalization (transpose tall matrices before row L2 normalization).

import torch
import torch.nn.functional as F
import math


def _row_normalize_shape_aware(g, eps=1e-12):
    """Shape-aware row/column L2 normalization.

    If g has PyTorch shape (..., m, n) with m <= n, normalizes along the last
    dim (each row to unit L2). If m > n, transposes the last two dims first,
    normalizes, then transposes back; equivalent to column-normalizing.
    """
    if g.size(-2) > g.size(-1):
        return F.normalize(g.transpose(-1, -2), p=2, dim=-1, eps=eps).transpose(-1, -2)
    return F.normalize(g, p=2, dim=-1, eps=eps)


class RMNP_All_v2_Grouped(torch.optim.Optimizer):
    """
    RMNP-All-v2 optimizer: shape-aware RMNP for ALL 2D+ parameters,
    including embed_tokens and lm_head. Only 1D/0D parameters use Adam.
    """

    def __init__(self, param_groups, lr_rmnp=0.005, lr_adam=0.001, momentum=0.95, beta=0.95,
                 weight_decay=0.0, betas=(0.9, 0.95), eps=1e-10):
        defaults = dict(lr_rmnp=lr_rmnp, lr_adam=lr_adam, momentum=momentum, beta=beta,
                       weight_decay=weight_decay, betas=betas, eps=eps)
        super(RMNP_All_v2_Grouped, self).__init__(param_groups, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group.get('momentum', 0.95)
            beta = group.get('beta', 0.95)
            weight_decay = group.get('weight_decay', 0.0)
            betas = group.get('betas', (0.9, 0.95))
            eps = group.get('eps', 1e-10)
            is_rmnp = group.get('is_rmnp', True)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_state = self.state.setdefault(p, {})

                if is_rmnp and grad.dim() >= 2:
                    # RMNP-v2 for 2D+ parameters
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']

                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    # Shape-aware row/column normalization
                    normed = _row_normalize_shape_aware(nesterov_buf)

                    scale = max(1, math.sqrt(grad.size(-2) / grad.size(-1)))
                    normed = normed * scale

                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)

                    p.data.add_(normed, alpha=-lr)
                    param_state['momentum_buffer'] = buf

                else:
                    # Adam for 1D/0D parameters
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0

                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1

                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])

                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom

                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)

                    p.data.add_(adam_update, alpha=-step_size)

        return loss


def get_rmnp_all_v2_optimizer(model, lr_rmnp=0.005, lr_adam=0.001, weight_decay=0.1, momentum=0.95, beta=0.95):
    """
    Returns an RMNP-All-v2 optimizer where ALL 2D+ parameters use RMNP-v2,
    including embed_tokens and lm_head.
    Only 1D/0D parameters (e.g., LayerNorm) use Adam.
    """
    rmnp_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    adam_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]

    param_groups = [
        dict(params=rmnp_params, lr=lr_rmnp, lr_rmnp=lr_rmnp, lr_adam=lr_adam,
             weight_decay=weight_decay, momentum=momentum, beta=beta, is_rmnp=True),
        dict(params=adam_params, lr=lr_adam, lr_rmnp=lr_rmnp, lr_adam=lr_adam,
             weight_decay=weight_decay, momentum=momentum, beta=beta, is_rmnp=False)
    ]
    optimizer = RMNP_All_v2_Grouped(param_groups)
    return optimizer
