"""
Muon optimizer with diagonal dominance (DD) monitoring.
Reimplements MuonWithAuxAdam from the official muon package, inserting
DD metric computation on the Gram matrix g @ g^T of each parameter's
gradient (after momentum/Nesterov, before Newton-Schulz orthogonalization).

Three DD metrics are computed per parameter:
  1. u metric:          u_i = diag_i - sum_{j!=i} |off_diag_{ij}|  (Gershgorin)
  2. ratio_to_max:      diag_i / max_{j!=i}(|off_diag_{ij}|)
  3. ratio_to_avg:      diag_i / avg_{j!=i}(|off_diag_{ij}|)
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
        dict with 9 scalar metrics (3 types x avg/min/max over rows).
    """
    gram = g.float() @ g.float().T  # (m, m), use float32 for precision
    m = gram.shape[0]
    if m < 2:
        keys = ["u_avg", "u_min", "u_max",
                "ratio_to_max_avg", "ratio_to_max_min", "ratio_to_max_max",
                "ratio_to_avg_avg", "ratio_to_avg_min", "ratio_to_avg_max"]
        return {k: 0.0 for k in keys}

    diag = torch.diag(gram)
    off_diag_sum = torch.sum(torch.abs(gram), dim=1) - torch.abs(diag)
    u = diag - off_diag_sum

    gram_abs = torch.abs(gram)
    mask = torch.eye(m, dtype=torch.bool, device=gram.device)
    gram_for_max = gram_abs.clone()
    gram_for_max[mask] = float('-inf')
    off_diag_max_per_row = gram_for_max.max(dim=1).values
    off_diag_avg_per_row = (gram_abs.sum(dim=1) - diag.abs()) / (m - 1 + 1e-10)

    ratio_to_max = diag / (off_diag_max_per_row + 1e-10)
    ratio_to_avg = diag / (off_diag_avg_per_row + 1e-10)

    return {
        "u_avg": u.mean().item(), "u_min": u.min().item(), "u_max": u.max().item(),
        "ratio_to_max_avg": ratio_to_max.mean().item(),
        "ratio_to_max_min": ratio_to_max.min().item(),
        "ratio_to_max_max": ratio_to_max.max().item(),
        "ratio_to_avg_avg": ratio_to_avg.mean().item(),
        "ratio_to_avg_min": ratio_to_avg.min().item(),
        "ratio_to_avg_max": ratio_to_avg.max().item(),
    }


def adam_update(grad, buf1, buf2, step, betas, eps):
    """Adam update (same as official muon package)."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdamDD(torch.optim.Optimizer):
    """
    MuonWithAuxAdam with Diagonal Dominance monitoring.

    Replicates the official MuonWithAuxAdam optimizer logic exactly,
    but inlines muon_update() so that DD metrics can be computed on
    the gradient Gram matrix g @ g^T after momentum/Nesterov and
    before Newton-Schulz orthogonalization.

    Attributes:
        dd_per_param (dict): Per-parameter DD ratio metrics (hidden + embed/lm_head).
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
        # DD metrics storage
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

                        # --- Inlined muon_update (same math as official) ---
                        # Momentum: momentum = beta * momentum + (1-beta) * grad
                        momentum_buf.lerp_(grad, 1 - beta)
                        # Nesterov: update = (1-beta) * grad + beta * momentum  (in-place on grad)
                        update = grad.lerp_(momentum_buf, beta)

                        if update.ndim == 4:
                            update = update.view(len(update), -1)

                        # === DD metric computation (before Newton-Schulz) ===
                        pname = self.param_names.get(id(p), f"param_{param_idx}")
                        dd_metrics = compute_diagonal_dominance(update)
                        self.dd_per_param[pname] = dd_metrics
                        param_idx += 1

                        # Newton-Schulz orthogonalization
                        update = zeropower_via_newtonschulz5(update, steps=5)
                        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                        # Weight decay + apply update
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                    dist.all_gather(
                        params_pad[base_i:base_i + dist.get_world_size()],
                        params_pad[base_i + dist.get_rank()])
            else:
                # AdamW branch (identical to official) + DD for embed/lm_head
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1

                    # DD monitoring for embed/lm_head (2D params in AdamW group)
                    if p.grad.ndim >= 2:
                        pname = self.param_names.get(id(p), f"adam_param_{param_idx}")
                        dd_metrics = compute_diagonal_dominance(p.grad)
                        self.dd_per_param[pname] = dd_metrics
                        param_idx += 1

                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        # Gather DD metrics from all ranks to rank 0
        if dist.get_world_size() > 1:
            all_dd = [None] * dist.get_world_size()
            dist.all_gather_object(all_dd, self.dd_per_param)
            # Merge: each rank has different params, combine them all
            merged = {}
            for rank_dd in all_dd:
                merged.update(rank_dd)
            self.dd_per_param = merged

        return loss


def get_muon_optimizer_dd(model, lr_muon=0.005, lr_adamw=0.001, weight_decay=0.1):
    """
    Returns a MuonWithAuxAdamDD optimizer with DD monitoring.
    Same parameter grouping logic as get_muon_optimizer:
      - Muon for hidden 2D+ params (excluding embed/lm_head)
      - AdamW for the rest
    """
    hidden_params = []
    other_params = []
    param_names = {}

    for n, p in model.named_parameters():
        if p.ndim >= 2 and 'embed' not in n and 'lm_head' not in n:
            hidden_params.append(p)
            param_names[id(p)] = n
        else:
            other_params.append(p)
            if p.ndim >= 2:
                param_names[id(p)] = n

    param_groups = [
        dict(params=hidden_params, use_muon=True, lr=lr_muon,
             weight_decay=weight_decay, momentum=0.95),
        dict(params=other_params, use_muon=False, lr=lr_adamw,
             betas=(0.9, 0.95), eps=1e-10, weight_decay=weight_decay),
    ]
    return MuonWithAuxAdamDD(param_groups, param_names=param_names)
