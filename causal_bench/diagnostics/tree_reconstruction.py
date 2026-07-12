"""Broadcast-on-tree reconstruction — the statistical-mechanics-faithful phase
transition behind #122 (real belief propagation, Kesten-Stigum threshold,
finite-size scaling). Synthetic; purely a statistical-physics object.

Why this exists: the Gaussian VP-SDE probe (`hierarchy_probe.py`) is a *crossover
detector* — its likelihood-ratio is a single scalar with a fixed Gaussian-CDF
crossover and **no genuine phase transition** (no finite-size-scaling knob). A
real transition needs the **tree recursion**: a root spin is broadcast down a
``b``-ary tree through a binary symmetric channel (flip prob ε), and the root is
reconstructed from the leaves by BP. Reconstruction is possible iff the *reduced
control parameter* ``b·λ² > 1`` with ``λ = 1 − 2ε`` — the **Kesten-Stigum
threshold** ``ε* = (1 − 1/√b)/2``. As tree depth → ∞ the crossover sharpens into a
true transition; that sharpening (finite-size scaling) is the honest evidence.

**"BP" here = belief propagation** — the message-passing / cavity *inference*
recursion that computes the posterior over the root from the leaves. NOT
backpropagation; there are no gradients or neural nets anywhere in this module.

BP is computed by **population dynamics / density evolution**: track the
distribution of the BP field (log-odds) at a node *given its own spin = +1* (by
symmetry). Leaf field = +∞ (observed); the edge channel attenuates a field by
``atanh(λ tanh(h))``; a node sums its children's attenuated fields. The
reconstruction **magnetization** ``m = E[tanh(h)]`` is the order parameter — > 0
below KS, → 0 above. numpy only, vectorized.

**Forward-channel variants (for the diffusion connection, #131 Part B).** The
q-ary tools here assume the **uniform** categorical channel (a token is kept w.p.
θ else resampled uniformly), whose exact BP denoiser is the recursion below. A
**D3PM** (Discrete Denoising Diffusion Probabilistic Models, Austin et al. 2021)
*absorbing/masking* channel instead replaces tokens by a [MASK] symbol — so its
exact BP needs a different **leaf-message** model: an unmasked leaf is a *clean*
fully-informative observation (a delta), a masked leaf is *fully uninformative*
(flat), rather than the uniform channel's graded per-leaf corruption. Same tree
recursion, different leaf likelihood + noise schedule — still belief propagation,
not a trained net. (A *trained* D3PM learns the denoiser with cross-entropy and
runs no BP; BP is our exact-analysis oracle for the synthetic case.)
"""
from __future__ import annotations

import numpy as np

_CLIP = 1.0 - 1e-9


def ks_threshold(branching: int) -> float:
    """Kesten-Stigum reconstruction threshold for a ``b``-ary tree + binary
    symmetric channel: reconstruction possible iff ``ε < (1 − 1/√b)/2``
    (equivalently ``b·(1−2ε)² > 1``)."""
    return 0.5 * (1.0 - 1.0 / np.sqrt(branching))


def bp_magnetization(depth: int, branching: int, epsilon: float, *,
                     pop: int = 3000, seed: int = 0, leaf_field: float = 16.0) -> float:
    """Reconstruction magnetization ``m = E[tanh(h)]`` at the root of a depth-
    ``depth`` ``b``-ary broadcast tree with channel flip ``ε``, via BP population
    dynamics (the field distribution conditioned on the node's spin = +1). ``m``
    is the order parameter: it converges to a positive fixed point below the KS
    threshold and decays to 0 above it as ``depth`` grows."""
    rng = np.random.default_rng(seed)
    lam = 1.0 - 2.0 * epsilon
    h = np.full(pop, leaf_field)                       # leaves (spin +1): field +∞
    for _ in range(depth):
        acc = np.zeros(pop)
        for _b in range(branching):
            hc = h[rng.integers(0, pop, size=pop)].copy()
            # given this node's spin +1, each child spin flips w.p. ε
            hc[rng.random(pop) < epsilon] *= -1.0
            acc += np.arctanh(np.clip(lam * np.tanh(hc), -_CLIP, _CLIP))
        h = acc
    return float(np.mean(np.tanh(h)))


def reconstruction_scan(branching: int, *, depth: int = 12,
                        eps_grid: np.ndarray | None = None, pop: int = 3000,
                        seed: int = 0) -> dict:
    """Scan channel noise ε: reconstruction magnetization + susceptibility
    ``−dm/dε``. The **susceptibility peak** locates the transition ``ε*`` — the
    order-parameter-based localization (no accuracy threshold). Returns
    ``{eps, magnetization, susceptibility, eps_star, control_at_star}`` where
    ``control = b·(1−2ε)²`` (≈ 1 at the transition)."""
    eps_grid = np.linspace(0.02, 0.45, 24) if eps_grid is None else np.asarray(eps_grid, float)
    m = np.array([bp_magnetization(depth, branching, float(e), pop=pop, seed=seed)
                  for e in eps_grid])
    # susceptibility = −dm/dε at midpoints; peak = transition
    de = np.diff(eps_grid)
    susc = -np.diff(m) / de
    mid = 0.5 * (eps_grid[:-1] + eps_grid[1:])
    j = int(np.argmax(susc))
    eps_star = float(mid[j])
    return {"eps": eps_grid, "magnetization": m, "susceptibility": susc,
            "eps_star": eps_star,
            "control_at_star": float(branching * (1.0 - 2.0 * eps_star) ** 2)}


def finite_size_widths(branching: int, *, depths=(3, 6, 12),
                       eps_grid: np.ndarray | None = None, pop: int = 2500,
                       seed: int = 0) -> dict:
    """Finite-size scaling: transition **width** vs tree depth. Width = ``1 / max
    susceptibility`` (a sharper peak ⇒ narrower transition). A genuine phase
    transition **sharpens** (width shrinks) as depth grows — the evidence that
    distinguishes a transition from a mere crossover. Returns
    ``{depths, widths, eps_stars}``."""
    eps_grid = np.linspace(0.02, 0.45, 24) if eps_grid is None else np.asarray(eps_grid, float)
    widths, stars = [], []
    for d in depths:
        r = reconstruction_scan(branching, depth=int(d), eps_grid=eps_grid,
                                pop=pop, seed=seed)
        widths.append(1.0 / float(np.max(r["susceptibility"])))
        stars.append(r["eps_star"])
    return {"depths": np.asarray(depths), "widths": np.asarray(widths),
            "eps_stars": np.asarray(stars)}


# ─── Canonical KS diagnostic: fixed point + linear stability ──────────────────
def bp_fixed_point_magnetization(branching: int, epsilon: float, *, pop: int = 4000,
                                 seed: int = 0, max_iter: int = 80, tol: float = 1e-3,
                                 leaf_field: float = 16.0) -> float:
    """The **canonical** reconstruction order parameter: iterate the BP density-
    evolution recursion to its **fixed point** (depth → ∞) and return the
    magnetization ``m*``. ``m* > 0`` ⟺ reconstruction possible; ``m* → 0`` ⟺ not.
    Unlike the finite-depth susceptibility peak this has no depth bias — it lands
    on the true transition."""
    rng = np.random.default_rng(seed)
    lam = 1.0 - 2.0 * epsilon
    h = np.full(pop, leaf_field)
    prev = 1.0
    for it in range(max_iter):
        acc = np.zeros(pop)
        for _b in range(branching):
            hc = h[rng.integers(0, pop, size=pop)].copy()
            hc[rng.random(pop) < epsilon] *= -1.0
            acc += np.arctanh(np.clip(lam * np.tanh(hc), -_CLIP, _CLIP))
        h = acc
        m = float(np.mean(np.tanh(h)))
        if it > 8 and abs(m - prev) < tol:
            break
        prev = m
    return max(m, 0.0)


def reconstruction_threshold(branching: int, *, m_floor: float = 0.03,
                             lo: float = 0.001, hi: float = 0.499, iters: int = 16,
                             pop: int = 3000, seed: int = 0) -> float:
    """Empirical reconstruction threshold ``ε_c`` by bisection on the fixed-point
    magnetization (``m*`` is monotone-decreasing in ε): the largest ε at which
    ``m* > m_floor``. For a binary symmetric channel + small branching this
    coincides with the Kesten-Stigum threshold ``ks_threshold(b)`` (a canonical
    self-consistency check); a gap would signal the harder robust-reconstruction
    (1RSB) regime at larger branching."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bp_fixed_point_magnetization(branching, mid, pop=pop, seed=seed) > m_floor:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def linear_stability_multiplier(branching: int, epsilon: float, *, m_in: float = 0.01,
                                pop: int = 20000, seed: int = 0) -> float:
    """Measure the recursion's **linear-stability multiplier** at the uninformative
    (``m=0``) fixed point: seed a tiny magnetization, apply ONE density-evolution
    step, return ``m_out / m_in``. The KS threshold *is* where this crosses 1, and
    it should equal ``b·λ² = branching·(1−2ε)²`` — deriving the threshold from the
    recursion itself, not from an operational peak."""
    rng = np.random.default_rng(seed)
    lam = 1.0 - 2.0 * epsilon
    h = np.full(pop, np.arctanh(np.clip(m_in, -_CLIP, _CLIP)))   # E[tanh h] = m_in
    acc = np.zeros(pop)
    for _b in range(branching):
        hc = h[rng.integers(0, pop, size=pop)].copy()
        hc[rng.random(pop) < epsilon] *= -1.0
        acc += np.arctanh(np.clip(lam * np.tanh(hc), -_CLIP, _CLIP))
    return float(np.mean(np.tanh(acc)) / m_in)


# ─── q-ary (Potts) BP density evolution — shared primitive for #131 ───────────
def qary_ks_threshold(branching: int) -> float:
    """Kesten-Stigum threshold in the Potts *closeness* parameter θ (channel
    ``M_ij = (1−θ)/q + θ·δ_ij``, second eigenvalue λ = θ): reconstruction by the
    linear/BP-from-scratch estimator iff ``b·θ² > 1`` ⟺ ``θ > 1/√b``. (For binary,
    θ = 1 − 2ε, recovering ``b(1−2ε)²=1``.)"""
    return 1.0 / np.sqrt(branching)


def _qary_overlap(H: np.ndarray, q: int) -> float:
    """Potts overlap of a field population ``H`` (pop, q) conditioned on the true
    symbol = 0: ``(q·E[belief(0)] − 1)/(q−1)`` ∈ [0,1] (0 = uniform, 1 = certain)."""
    Hm = H - H.max(1, keepdims=True)
    b = np.exp(Hm)
    b /= b.sum(1, keepdims=True)
    return float((q * b[:, 0].mean() - 1.0) / (q - 1))


def qary_bp_magnetization(depth: int, branching: int, q: int, theta: float, *,
                          init: str = "planted", pop: int = 4000, seed: int = 0,
                          leaf_field: float = 16.0) -> float:
    """q-ary Potts BP density evolution (population dynamics), conditioned on the
    node's true symbol = 0. Returns the reconstruction **overlap** after ``depth``
    levels. ``init='planted'`` starts from observed leaves (detects the
    *reconstruction* threshold — does an informative fixed point exist); ``init=
    'uninformative'`` starts near the trivial fixed point (detects the *KS/
    algorithmic* threshold — is BP-from-scratch stable). For q ≥ 5 the two can
    differ (the hard-phase gap, #131 Part A); binary (q=2) reproduces the BSC
    recursion.

    Channel ``M_ij = (1−θ)/q + θ·δ_ij``: with prob θ the symbol is kept, else
    resampled uniformly over all q symbols."""
    rng = np.random.default_rng(seed)
    if init == "planted":
        H = np.zeros((pop, q)); H[:, 0] = leaf_field           # observed = true = 0
    else:
        H = 1e-3 * rng.standard_normal((pop, q))               # near trivial fixed point
        H[:, 0] += 1e-2                                         # tiny bias toward truth
    qidx = np.arange(q)
    for _ in range(depth):
        acc = np.zeros((pop, q))
        for _b in range(branching):
            h = H[rng.integers(0, pop, size=pop)]              # (pop, q) sampled messages
            # child true symbol given node=0: keep 0 w.p. θ, else uniform over q
            keep = rng.random(pop) < theta
            c = np.where(keep, 0, rng.integers(0, q, size=pop))
            # relabel the (symbol-0-conditioned) field to be conditioned on symbol c
            roll = (qidx[None, :] - c[:, None]) % q
            h = np.take_along_axis(h, roll, axis=1)
            # channel attenuation: h'_a = log[(1−θ)/q · Σ_b e^{h_b} + θ e^{h_a}]
            hmax = h.max(1, keepdims=True)
            e = np.exp(h - hmax)
            S = e.sum(1, keepdims=True)
            acc += np.log((1.0 - theta) / q * S + theta * e + 1e-300) + hmax
        H = acc - acc.max(1, keepdims=True)                    # normalize (log-domain)
    return _qary_overlap(H, q)


def has_reconstruction_gap(branching: int, q: int, theta: float, *, depth: int = 20,
                           pop: int = 5000, seed: int = 0, planted_floor: float = 0.05,
                           uninform_ceil: float = 0.03) -> bool:
    """#131 Part A: is ``(branching, q, θ)`` in the **hard phase** — reconstruction
    possible from an informative start but NOT by BP-from-scratch? True when the
    planted overlap survives (> ``planted_floor``) while the uninformative one
    collapses (< ``uninform_ceil``). This is the algorithmic-vs-information-theoretic
    gap (`ε_KS < ε < ε_recon`), which opens for q ≥ 5 / large branching and is
    absent for the binary symmetric channel."""
    mp = qary_bp_magnetization(depth, branching, q, theta, init="planted",
                               pop=pop, seed=seed)
    mu = qary_bp_magnetization(depth, branching, q, theta, init="uninformative",
                               pop=pop, seed=seed)
    return bool(mp > planted_floor and mu < uninform_ceil)


def _qary_threshold_bisect(branching: int, q: int, init: str, *, depth: int, pop: int,
                           seed: int, floor: float, iters: int) -> float:
    """Bisect the closeness θ (overlap is increasing in θ) for the θ at which the
    ``init``-started BP overlap crosses ``floor``."""
    lo, hi = 1.0 / q, 0.999
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if qary_bp_magnetization(depth, branching, q, mid, init=init, pop=pop, seed=seed) > floor:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def reconstruction_gap(branching: int, q: int, *, depth: int = 26, pop: int = 5000,
                       seed: int = 0, floor: float = 0.05, iters: int = 15,
                       tol: float = 0.02) -> dict:
    """#131 Part A (finished): map the KS-vs-reconstruction gap boundaries by
    bisecting **both** BP initializations in the Potts closeness θ.

    - ``theta_recon`` (planted start) — the *information-theoretic* threshold: the
      smallest θ at which an informative fixed point survives.
    - ``theta_ks`` (uninformative start) — the *algorithmic* KS threshold: the
      smallest θ at which BP-from-scratch reconstructs. Should ≈ ``1/√b``.

    ``has_gap`` ⟺ ``theta_recon < theta_ks − tol`` — the **hard phase** where
    reconstruction is possible but not by BP-from-ignorance (opens for q ≥ 5;
    absent for the binary symmetric channel). Returns the two empirical thresholds,
    the analytic KS ``1/√b``, ``has_gap`` and ``gap_width``."""
    th_recon = _qary_threshold_bisect(branching, q, "planted", depth=depth, pop=pop,
                                      seed=seed, floor=floor, iters=iters)
    th_ks = _qary_threshold_bisect(branching, q, "uninformative", depth=depth, pop=pop,
                                   seed=seed, floor=floor, iters=iters)
    return {"theta_recon": th_recon, "theta_ks": th_ks,
            "theta_ks_analytic": 1.0 / np.sqrt(branching),
            "has_gap": bool(th_recon < th_ks - tol),
            "gap_width": float(max(0.0, th_ks - th_recon))}
