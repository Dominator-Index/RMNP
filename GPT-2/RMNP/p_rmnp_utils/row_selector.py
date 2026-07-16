"""
SVD-free layerwise p* selector for p-RMNP.

Row analogue of the SMuon tightness criterion (Massena et al., Prop. 2;
reference implementation: schatten-muon smuon/svs/tightness_exact.py):

    p* = argmax_{p in [pmin, pmax]}  N(p)^2 / (D(p) + 1e-12)

    N(p) = sum_i c_i * a_i^{1/p}      (alignment term)
    D(p) = sum_i b_i * a_i^{2/p}      (curvature term)

with the spectral statistics replaced by their row analogues:

    sigma_i(M) -> a_i = ||H_{i,:}||_2            carrier row norms
    C_ii       -> c_i = <G_{i,:}, u_i>           grad/carrier row alignment
    B_ii       -> b_i = ||u_i A||^2 = u_i Q u_i^T  activation row curvature

Because rows are indexed by the (fixed) output neuron, no SVD is needed
anywhere and the exact (non-separable) denominator is directly computable.

Deliberately kept identical to the SMuon reference behavior for a minimal
diff: the objective uses N(p)^2 without the positive-part clamp [N]_+
(strict non-negative-step-size theory would use [N]_+^2 / D; the two agree
whenever N(p) > 0), and p* is found by a bounded continuous scalar search
(scipy minimize_scalar, Brent-type), NOT a discrete candidate grid.
"""

import torch


def _orient_activation(n_expected, act, use_gram):
    """
    Return the activation as an (n_expected, samples) matrix, or validate an
    (n_expected, n_expected) Gram. Linear inputs are recorded as
    (..., in_features); anything with a trailing in_features axis is accepted.
    """
    shape = tuple(act.shape)
    if use_gram:
        if act.dim() == 2 and shape == (n_expected, n_expected):
            return act
        raise ValueError(
            f"use_gram=True expects a ({n_expected}, {n_expected}) Gram matrix, got {shape}."
        )
    if act.dim() == 2 and shape[0] == n_expected and shape[1] != n_expected:
        return act  # already (n, samples)
    if shape[-1] == n_expected:
        return act.reshape(-1, n_expected).transpose(0, 1)
    raise ValueError(
        f"Activation shape {shape} has no axis matching the layer input dim {n_expected}."
    )


def row_curvature(u_rows, act, use_gram=False):
    """
    b_i = ||u_i A||^2 (raw activations) or u_i Q u_i^T (Gram Q = A^T A).

    u_rows: (m, n) unit carrier rows, fp32.
    Returns a length-m fp32 vector, clamped at 0.
    """
    n = u_rows.size(-1)
    act = _orient_activation(n, act, use_gram).to(device=u_rows.device, dtype=torch.float32)
    if use_gram:
        b = ((u_rows @ act) * u_rows).sum(dim=-1)
    else:
        ua = u_rows @ act  # (m, samples)
        b = (ua * ua).sum(dim=-1)
    return b.clamp_min(0.0)


def _compute_ND_row(p, c, b, a):
    """N(p) = sum_i c_i a_i^{1/p}; D(p) = sum_i b_i a_i^{2/p}."""
    p_safe = min(max(p, 1.0 + 1e-9), 1.0e4)
    inv_p = 1.0 / p_safe
    a_pow = a ** inv_p
    N = (c * a_pow).sum()
    D = (b * a_pow * a_pow).sum()
    return N, D


def select_p_from_row_statistics(a, c_ema, b_ema, pmin, pmax):
    """
    Continuous bounded search for p* maximizing J(p) = N(p)^2 / (D(p) + 1e-12).

    Mirrors the SMuon reference call exactly:
        minimize_scalar(objective, bounds=(pmin, pmax), method="bounded")
    which is a derivative-free bounded Brent-type search evaluating real p
    inside the interval (the result is a real number such as 3.2746, not a
    member of any candidate grid). Statistics are moved to CPU float64 once
    so the ~tens of objective evaluations are cheap and synchronization-free.
    """
    from scipy.optimize import minimize_scalar  # lazy: fixed-p path needs no scipy

    a64 = a.detach().to("cpu", torch.float64)
    c64 = c_ema.detach().to("cpu", torch.float64)
    b64 = b_ema.detach().to("cpu", torch.float64)

    def objective(p):
        N, D = _compute_ND_row(p, c64, b64, a64)
        return -((N * N) / (D + 1e-12)).item()

    res = minimize_scalar(objective, bounds=(pmin, pmax), method="bounded")
    return float(res.x)
