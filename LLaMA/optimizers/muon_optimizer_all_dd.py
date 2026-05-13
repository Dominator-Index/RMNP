"""
Muon-All optimizer with diagonal dominance (DD) monitoring.
Like muon_optimizer_dd but applies Muon to ALL 2D+ params including embed/lm_head.
DD ratio metrics are computed for every 2D parameter.
"""

import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration (copied from official muon package for consistency)."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def compute_diagonal_dominance(g):
    """
    Compute diagonal dominance metrics for the Gram matrix g @ g^T.

    Args:
        g: tensor of shape (m, n)

    Returns:
        dict with ratio metrics (ratio_to_max and ratio_to_avg, each with avg/min/max).
    """
    gram = g.float() @ g.float().T
    m = gram.shape[0]
    if m < 2:
        keys = ["ratio_to_max_avg", "ratio_to_max_min", "ratio_to_max_max",
                "ratio_to_avg_avg", "ratio_to_avg_min", "ratio_to_avg_max"]
        return {k: 0.0 for k in keys}

    diag = torch.diag(gram)

    gram_abs = torch.abs(gram)
    mask = torch.eye(m, dtype=torch.bool, device=gram.device)
    gram_for_max = gram_abs.clone()
    gram_for_max[mask] = float('-inf')
    off_diag_max_per_row = gram_for_max.max(dim=1).values
    off_diag_avg_per_row = (gram_abs.sum(dim=1) - diag.abs()) / (m - 1 + 1e-10)

    ratio_to_max = diag / (off_diag_max_per_row + 1e-10)
    ratio_to_avg = diag / (off_diag_avg_per_row + 1e-10)

    return {
        "ratio_to_max_avg": ratio_to_max.mean().item(),
        "ratio_to_max_min": ratio_to_max.min().item(),
        "ratio_to_max_max": ratio_to_max.max().item(),
        "ratio_to_avg_avg": ratio_to_avg.mean().item(),
        "ratio_to_avg_min": ratio_to_avg.min().item(),
        "ratio_to_avg_max": ratio_to_avg.max().item(),
    }


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Adam update."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonAllWithDD(torch.optim.Optimizer):
    """
    Muon-All with DD monitoring.
    Muon for ALL 2D+ params (including embed/lm_head), AdamW for 1D only.
    DD ratio metrics computed for every 2D parameter.

    Attributes:
        dd_per_param (dict): Per-parameter DD ratio metrics for all 2D params.
    """
    def __init__(self, param_groups, param_names=None):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

        self.param_names = param_names or {}
        self.dd_per_param = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.dd_per_param = {}
        param_idx = 0

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                beta = group["momentum"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size())

                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        grad = p.grad
                        momentum_buf = state["momentum_buffer"]

                        momentum_buf.lerp_(grad, 1 - beta)
                        update = grad.lerp_(momentum_buf, beta)

                        if update.ndim == 4:
                            update = update.view(len(update), -1)

                        # DD metric computation (before Newton-Schulz)
                        pname = self.param_names.get(id(p), f"param_{param_idx}")
                        dd_metrics = compute_diagonal_dominance(update)
                        self.dd_per_param[pname] = dd_metrics
                        param_idx += 1

                        # Newton-Schulz orthogonalization
                        update = zeropower_via_newtonschulz5(update, steps=5)
                        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                    dist.all_gather(
                        params_pad[base_i:base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()])
            else:
                # AdamW for 1D params only (no DD needed)
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        # Gather DD metrics from all ranks
        if dist.get_world_size() > 1:
            all_dd = [None] * dist.get_world_size()
            dist.all_gather_object(all_dd, self.dd_per_param)
            merged = {}
            for rank_dd in all_dd:
                merged.update(rank_dd)
            self.dd_per_param = merged

        return loss


def get_muon_all_optimizer_dd(model, lr_muon=0.005, lr_adamw=0.001, weight_decay=0.1):
    """
    Returns a MuonAllWithDD optimizer.
    Muon for ALL 2D+ params (including embed/lm_head), AdamW for 1D only.
    DD ratio metrics computed for every parameter.
    """
    muon_params = []
    adam_params = []
    param_names = {}

    for n, p in model.named_parameters():
        if p.ndim >= 2:
            muon_params.append(p)
            param_names[id(p)] = n
        else:
            adam_params.append(p)

    param_groups = [
        dict(params=muon_params, use_muon=True, lr=lr_muon,
             weight_decay=weight_decay, momentum=0.95),
        dict(params=adam_params, use_muon=False, lr=lr_adamw,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay),
    ]
    return MuonAllWithDD(param_groups, param_names=param_names)
