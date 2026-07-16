"""
Unit tests for the p-RMNP optimizer and its adaptive p* selector.

Run from the RMNP/ directory (so that `optimizers` and `p_rmnp_utils`
resolve as top-level packages, exactly as in training):

    cd GPT-2/RMNP && python p_rmnp_utils/test_p_rmnp.py

Requires torch (+ scipy for the selector tests). The regression test
against the original RMNP optimizer requires CUDA (RMNP hardcodes a cuda
update buffer); it is skipped on CPU-only machines.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from optimizers.p_rmnp import PRMNP, row_power, polynomial_scale, strict_lmo_normalize

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS" if cond else "FAIL"), name)


def test_row_power_endpoints():
    torch.manual_seed(0)
    H = torch.randn(8, 5, dtype=torch.float32)

    # p = 1: identity on the carrier
    check("row_power p=1 is identity", torch.equal(row_power(H, 1.0), H))

    # p = inf: unit rows, matches F.normalize (RMNP's row_normalize)
    R_inf = row_power(H, math.inf)
    import torch.nn.functional as F
    check("row_power p=inf == F.normalize", torch.equal(R_inf, F.normalize(H, p=2, dim=-1)))

    # finite p: output row lengths equal a_i^{1/p}
    for p in [1.5, 2.0, 4.0, 16.0]:
        a = H.norm(dim=-1)
        out_norms = row_power(H, p).norm(dim=-1)
        check(f"row_power p={p} row lengths == a^(1/p)",
              torch.allclose(out_norms, a ** (1.0 / p), rtol=1e-5, atol=1e-6))

    # zero rows stay exactly zero
    Hz = H.clone()
    Hz[3] = 0.0
    check("row_power keeps zero rows zero", torch.equal(row_power(Hz, 4.0)[3], torch.zeros(5)))


def test_polynomial_scale():
    m, n = 3072, 768  # tall, m/n = 4
    check("scale interpolate p=1 -> 1.0", polynomial_scale(m, n, 1.0, "interpolate") == 1.0)
    check("scale interpolate p=inf == legacy",
          abs(polynomial_scale(m, n, math.inf, "interpolate") - polynomial_scale(m, n, 2.0, "legacy")) < 1e-12)
    check("scale legacy == max(1,m/n)^0.5", abs(polynomial_scale(m, n, 7.0, "legacy") - 2.0) < 1e-12)
    check("scale none == 1", polynomial_scale(m, n, 7.0, "none") == 1.0)
    # wide matrices are never scaled
    check("scale wide == 1", polynomial_scale(768, 3072, math.inf, "interpolate") == 1.0)
    # interpolate exponent between the endpoints
    s4 = polynomial_scale(m, n, 4.0, "interpolate")
    check("scale interpolate p=4 == (m/n)^{(1-1/4)/2}", abs(s4 - 4.0 ** (0.375)) < 1e-12)


def test_strict_lmo():
    torch.manual_seed(1)
    H = torch.randn(6, 10, dtype=torch.float32)
    for p in [1.5, 3.0, 8.0]:
        a = H.norm(dim=-1)
        R = strict_lmo_normalize(row_power(H, p), a, p)
        # mixed norm ||R||_{p+1,2} should be exactly 1
        mixed = (R.norm(dim=-1) ** (p + 1.0)).sum() ** (1.0 / (p + 1.0))
        check(f"strict LMO unit ||.||_{{p+1,2}} at p={p}", abs(mixed.item() - 1.0) < 1e-5)


def _run_optimizer_steps(opt_cls, kwargs, params_init, grads, steps=3):
    params = [torch.nn.Parameter(x.clone()) for x in params_init]
    opt = opt_cls(params, **kwargs)
    for s in range(steps):
        for prm, g in zip(params, grads):
            prm.grad = g[s].clone()
        opt.step()
    return [prm.data.clone() for prm in params]


def test_regression_vs_rmnp():
    """PRMNP(p<=0 i.e. inf, scale_mode='legacy') must reproduce RMNP exactly."""
    if not torch.cuda.is_available():
        print("SKIP regression vs RMNP (needs CUDA: RMNP hardcodes a cuda buffer)")
        return
    from optimizers.rmnp import RMNP
    torch.manual_seed(2)
    dev = "cuda"
    shapes = [(16, 8), (8, 16), (12, 12)]
    params_init = [torch.randn(*s, device=dev) for s in shapes]
    grads = [torch.stack([torch.randn(*s, device=dev) for _ in range(3)]) for s in shapes]
    kwargs = dict(lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.1)
    out_rmnp = _run_optimizer_steps(RMNP, kwargs, params_init, grads)
    out_prmnp = _run_optimizer_steps(
        PRMNP, dict(kwargs, p=0.0, scale_mode="legacy", strict_lmo=False), params_init, grads
    )
    ok = all(torch.allclose(x, y, rtol=1e-5, atol=1e-6) for x, y in zip(out_rmnp, out_prmnp))
    max_diff = max((x - y).abs().max().item() for x, y in zip(out_rmnp, out_prmnp))
    check(f"PRMNP(p=inf, legacy) == RMNP (max diff {max_diff:.2e})", ok)


def test_selector_sanity():
    """M=G case: c_i = a_i > 0. Uniform b with heterogeneous a should give an
    interior or boundary p*, and scaling the direction by any positive scalar
    must not change p* (scale invariance)."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("SKIP selector tests (scipy not installed)")
        return
    from p_rmnp_utils.row_selector import select_p_from_row_statistics, _compute_ND_row

    torch.manual_seed(3)
    a = torch.tensor([4.0, 2.0, 1.0, 0.5])
    c = a.clone()          # M = G: c_i = a_i, all positive
    b = torch.ones(4)
    p1 = select_p_from_row_statistics(a, c, b, 1.02, 50.0)
    check(f"selector returns real p* in bounds (got {p1:.4f})", 1.02 <= p1 <= 50.0)

    # J(p*) must be >= J at interior probe points. The exact bounds are
    # excluded: bounded Brent (like SMuon's) never evaluates the endpoints,
    # so when the optimum sits on a bound, res.x lands within xatol of it.
    def J(p):
        N, D = _compute_ND_row(p, c.double(), b.double(), a.double())
        return ((N * N) / (D + 1e-12)).item()
    check("selector value dominates interior probes",
          all(J(p1) >= J(q) - 1e-9 for q in [1.5, 2.0, 5.0, 10.0, 49.0]))
    # this configuration's optimum is at the lower bound; Brent should land
    # within its default tolerance of it
    check(f"boundary optimum lands near pmin (got {p1:.6f})", p1 - 1.02 < 1e-3)

    # scale invariance: c -> s*c leaves p* unchanged (s cancels in N^2/D? no —
    # s^2 in numerator only; but argmax is unchanged since J scales by s^2)
    p2 = select_p_from_row_statistics(a, 3.7 * c, b, 1.02, 50.0)
    check(f"selector invariant to positive scaling of c ({p1:.4f} vs {p2:.4f})",
          abs(p1 - p2) < 1e-3)

    # curvature concentrated on large-a rows should push p* upward (more
    # row-scale compression) relative to curvature on small-a rows
    b_hi = torch.tensor([10.0, 1.0, 1.0, 0.1])
    b_lo = torch.tensor([0.1, 1.0, 1.0, 10.0])
    p_hi = select_p_from_row_statistics(a, c, b_hi, 1.02, 50.0)
    p_lo = select_p_from_row_statistics(a, c, b_lo, 1.02, 50.0)
    check(f"curvature on large-a rows raises p* ({p_hi:.3f} > {p_lo:.3f})", p_hi > p_lo)


def test_row_curvature():
    from p_rmnp_utils.row_selector import row_curvature
    torch.manual_seed(4)
    m, n, bsz = 6, 10, 32
    u = torch.nn.functional.normalize(torch.randn(m, n), dim=-1)
    act = torch.randn(bsz, n)                     # (samples, in_features)
    b_raw = row_curvature(u, act, use_gram=False)
    gram = act.reshape(-1, n).t() @ act.reshape(-1, n)
    b_gram = row_curvature(u, gram, use_gram=True)
    check("row_curvature raw == gram", torch.allclose(b_raw, b_gram, rtol=1e-4, atol=1e-4))
    # 3D activations (batch, seq, n) accepted
    act3 = act.reshape(4, 8, n)
    b_3d = row_curvature(u, act3, use_gram=False)
    check("row_curvature accepts (b, t, n)", torch.allclose(b_raw, b_3d, rtol=1e-5, atol=1e-5))


def test_update_p_state():
    """update_p_state must set p_star without touching the momentum buffer,
    and adaptive step() must consume the stored p_star."""
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("SKIP update_p_state test (scipy not installed)")
        return
    torch.manual_seed(5)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, n, bsz = 8, 6, 16
    w = torch.nn.Parameter(torch.randn(m, n, device=dev))
    opt = PRMNP([w], lr=0.02, adaptive=True, pmin=1.02, pmax=50.0, stat_momentum=0.9)

    # seed a momentum buffer via one plain step
    w.grad = torch.randn(m, n, device=dev)
    if dev == "cuda":
        opt.step()
    else:
        # step() needs a cuda-capable flat buffer only when params are cuda;
        # on cpu the device follows the param, so this still runs
        opt.step()
    buf_before = opt.state[w]["momentum_buffer"].clone()

    w.grad = torch.randn(m, n, device=dev)
    act = torch.randn(bsz, n, device=dev)
    opt.update_p_state({w: act}, use_gram=False)

    st = opt.state[w]
    check("update_p_state stores p_star", "p_star" in st and 1.02 <= st["p_star"] <= 50.0)
    check("update_p_state stores c_ema/b_ema", "c_ema" in st and "b_ema" in st)
    check("update_p_state does not mutate momentum buffer",
          torch.equal(buf_before, st["momentum_buffer"]))

    # params without recorded activation keep init_p
    w2 = torch.nn.Parameter(torch.randn(m, n, device=dev))
    opt2 = PRMNP([w2], lr=0.02, adaptive=True, init_p="pmax", pmax=50.0)
    w2.grad = torch.randn(m, n, device=dev)
    opt2.update_p_state({}, use_gram=False)
    check("missing activation -> p_star stays init_p (pmax)",
          opt2.state[w2].get("p_star") == 50.0)

    # EMA: second call moves c_ema toward the new c
    c1 = st["c_ema"].clone()
    w.grad = torch.randn(m, n, device=dev)
    opt.update_p_state({w: act}, use_gram=False)
    check("second update_p_state changes c_ema (EMA active)",
          not torch.equal(c1, st["c_ema"]))

    summary = opt.p_star_summary()
    check("p_star_summary returns stats", summary is not None and summary[3] == 1)


if __name__ == "__main__":
    test_row_power_endpoints()
    test_polynomial_scale()
    test_strict_lmo()
    test_regression_vs_rmnp()
    test_selector_sanity()
    test_row_curvature()
    test_update_p_state()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print("FAILED:", name)
        sys.exit(1)
