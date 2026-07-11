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

BP is computed by **population dynamics / density evolution**: track the
distribution of the BP field (log-odds) at a node *given its own spin = +1* (by
symmetry). Leaf field = +∞ (observed); the edge channel attenuates a field by
``atanh(λ tanh(h))``; a node sums its children's attenuated fields. The
reconstruction **magnetization** ``m = E[tanh(h)]`` is the order parameter — > 0
below KS, → 0 above. numpy only, vectorized.
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
