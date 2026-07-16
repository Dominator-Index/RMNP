"""
p-RMNP optimizer: SVD-free general-p extension of RMNP.

Forked from optimizers/rmnp.py (which is adapted from KellerJordan/modded-nanogpt).
Math spec: Paper-Optimizer/HTRMNP/extended_rmnp_from_sgd_to_rmnp.md (Sec. 9.1/9.2)
and Paper-Optimizer/HTRMNP/p_rmnp_algorithm.md.

Row transform (the only change to the update rule relative to RMNP):

    R_p(H)_{i,:} = H_{i,:} / max(||H_{i,:}||_2, eps)^{1 - 1/p}

  p = 1        -> identity on the update carrier (momentum-SGD direction)
  p = infinity -> unit row normalization (original RMNP)

The adaptive variant selects a per-parameter p* by maximizing the SMuon
tightness criterion J(p) = N(p)^2 / (D(p) + 1e-12) with the spectral
statistics replaced by their row analogues (no SVD anywhere):

    sigma_i(M)  ->  a_i = ||H_{i,:}||_2   (instantaneous, momentum already smooths it)
    C_ii        ->  c_i = <G_{i,:}, u_i>  (EMA)
    B_ii        ->  b_i = ||u_i A||_2^2   (EMA)

Like RMNP, this optimizer never transposes: rows are always the output
(fan-out) dimension, regardless of matrix shape.
"""

import math
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist


def row_power(G, p, eps=1e-12):
    """
    p-RMNP row transform: divide each row by ||row||_2^{1 - 1/p}.

    p = 1 returns G unchanged; p = inf reduces to unit row normalization
    (identical to RMNP's row_normalize). Row norms are accumulated in fp32.
    Exactly-zero rows stay exactly zero.

    NOTE: intentionally NOT @torch.compile'd — under adaptive p the exponent
    changes during training, which would trigger repeated recompilation.
    """
    if math.isinf(p):
        return F.normalize(G, p=2, dim=-1)
    if p == 1.0:
        return G
    row_norms = G.float().norm(dim=-1, keepdim=True)
    denom = row_norms.clamp_min(eps).pow(1.0 - 1.0 / p)
    return G / denom.to(G.dtype)


def polynomial_scale(m, n, p, mode="interpolate"):
    """
    Aspect-ratio scale s_p(m, n) applied after the row transform.

    'interpolate' : max(1, m/n)^{(1 - 1/p)/2}  — recommended; equals 1 at p=1
                    (no extra scaling of the momentum-SGD carrier) and equals
                    the original RMNP scale max(1, m/n)^{1/2} at p=inf.
    'legacy'      : max(1, m/n)^{1/2} for every p — mechanical extension of the
                    original code (keeps a layerwise aspect scale even at p=1).
    'none'        : 1 — pure mixed-norm LMO direction.
    """
    base = max(1.0, m / n)
    if mode == "none":
        return 1.0
    if mode == "legacy":
        return base ** 0.5
    if mode == "interpolate":
        beta = 1.0 if math.isinf(p) else (1.0 - 1.0 / p)
        return base ** (0.5 * beta)
    raise ValueError(f"Unknown scale_mode: {mode}")


def strict_lmo_normalize(R, a, p):
    """
    Optional strict mixed-norm LMO normalization: divide the row-transformed
    direction R by Z_p(H) = (sum_i a_i^{1 + 1/p})^{1/(p+1)}, so that
    ||R / Z_p||_{p+1, 2} = 1 exactly. Computed in log-space for stability.
    a is the vector of carrier row norms (fp32).
    """
    if math.isinf(p):
        # ||.||_{inf,2} ball: every nonzero row already has unit norm.
        return R
    q = 1.0 + 1.0 / p
    a_pos = a[a > 0]
    if a_pos.numel() == 0:
        return R
    log_z = torch.logsumexp(q * a_pos.log(), dim=0) / (p + 1.0)
    return R / log_z.exp().to(R.dtype)


class PRMNP(torch.optim.Optimizer):
    """
    PRMNP - p-RMNP: RMNP with a general row-power exponent and an optional
    layerwise adaptive p* selector (SMuon-style, but SVD-free).

    The step() skeleton (distributed round-robin sharding, flat bf16 update
    buffer, all_reduce, decoupled weight decay) is copied line-for-line from
    RMNP; the only change is the row transform and its scaling.

    Arguments (beyond RMNP's lr / momentum / nesterov / weight_decay):
        p: fixed row-power exponent; any value <= 0 means p = infinity
           (exact RMNP endpoint). Ignored when adaptive=True.
        scale_mode: 'interpolate' (default) | 'legacy' | 'none', see
           polynomial_scale().
        strict_lmo: divide by Z_p(H) so updates lie on the unit
           ||.||_{p+1,2} ball (default False: the shared positive scalar is
           absorbed into the learning rate, as optimizers usually do).
        adaptive: enable the per-parameter p* selector; p* is stored in
           self.state[param]['p_star'] and refreshed by update_p_state().
        pmin, pmax: continuous search interval for p* (SMuon defaults).
        init_p: 'pmax' | 'pmin' | float — p used before the first
           update_p_state() call (SMuon uses pmax, i.e. start at the
           RMNP-like endpoint).
        stat_momentum: EMA coefficient for the c_i / b_i statistics
           (a_i is used instantaneously, matching SMuon's treatment of the
           momentum singular values).
        eps: row-norm clamp for the power denominators.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.,
                 p=0.0, scale_mode="interpolate", strict_lmo=False,
                 adaptive=False, pmin=1.02, pmax=50.0, init_p="pmax",
                 stat_momentum=0.95, eps=1e-12):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.fixed_p = math.inf if p <= 0 else float(p)
        self.scale_mode = scale_mode
        self.strict_lmo = strict_lmo
        self.adaptive = adaptive
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        if init_p == "pmax":
            self.init_p_value = self.pmax
        elif init_p == "pmin":
            self.init_p_value = self.pmin
        else:
            self.init_p_value = float(init_p)
        self.stat_momentum = float(stat_momentum)
        self.eps = eps
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    def _effective_p(self, state):
        if self.adaptive:
            return state.get('p_star', self.init_p_value)
        return self.fixed_p

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            device = group['params'][0].device if group['params'] else 'cuda'
            updates_flat = torch.zeros(total_params, device=device, dtype=torch.bfloat16)
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
                    # p-RMNP row transform in place of RMNP's row_normalize
                    p_eff = self._effective_p(state)
                    if self.strict_lmo:
                        a = g.float().norm(dim=-1)
                        g = row_power(g, p_eff, self.eps)
                        g = strict_lmo_normalize(g, a, p_eff)
                    else:
                        g = row_power(g, p_eff, self.eps)
                    g = g * polynomial_scale(g.size(0), g.size(1), p_eff, self.scale_mode)
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

    @torch.no_grad()
    def update_p_state(self, activations, use_gram=False):
        """
        Refresh per-parameter p* from row statistics. Call after backward
        (with unscaled gradients) and before step().

        activations: dict {parameter -> recorded input activation or Gram
        matrix} as produced by p_rmnp_utils.wrap_model.ActivationRecorder.
        Parameters without a recorded activation keep their current p*
        (or init_p before the first refresh).

        Only parameters owned by this rank (same round-robin sharding as
        step()) are processed, so no extra communication is needed: each
        rank selects p* exactly for the parameters whose updates it computes.
        The carrier preview below replays step()'s momentum arithmetic
        out-of-place and never writes to the momentum buffer.
        """
        if not self.adaptive:
            return
        # lazy import: fixed-p usage must not require scipy
        from p_rmnp_utils.row_selector import row_curvature, select_p_from_row_statistics

        for group in self.param_groups:
            momentum = group['momentum']
            nesterov = group['nesterov']
            for i, p in enumerate(group['params']):
                if i % int(self.world_size) != int(self.rank):
                    continue
                state = self.state[p]
                state.setdefault('p_star', self.init_p_value)
                act = None if activations is None else activations.get(p)
                if act is None:
                    continue
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                g32 = g.float()
                # preview the update carrier without mutating the buffer:
                # step() does buf = momentum*buf + g, then (nesterov) H = g + momentum*buf
                buf = state.get('momentum_buffer')
                if buf is None:
                    buf_preview = g32.clone()
                else:
                    buf_preview = buf.float().mul(momentum).add_(g32)
                H = g32.add(buf_preview, alpha=momentum) if nesterov else g32

                a = H.norm(dim=-1)
                u = H / a.clamp_min(self.eps).unsqueeze(-1)
                c = (g32 * u).sum(dim=-1)
                b = row_curvature(u, act, use_gram=use_gram)

                # EMA on the noisy statistics only; a is instantaneous
                # (the momentum carrier is already an EMA upstream).
                beta = self.stat_momentum
                if 'c_ema' not in state or state['c_ema'].shape != c.shape:
                    state['c_ema'] = c.detach().clone()
                    state['b_ema'] = b.detach().clone()
                else:
                    state['c_ema'].lerp_(c, 1.0 - beta)
                    state['b_ema'].lerp_(b, 1.0 - beta)

                state['p_star'] = select_p_from_row_statistics(
                    a, state['c_ema'], state['b_ema'], self.pmin, self.pmax
                )

    def p_star_summary(self):
        """Return (mean, min, max, count) of p* over this rank's parameters."""
        values = [s['p_star'] for s in self.state.values() if 'p_star' in s]
        if not values:
            return None
        return (sum(values) / len(values), min(values), max(values), len(values))
