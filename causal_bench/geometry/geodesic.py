"""Geodesic distance + geodesic positivity region R (#159, gated).

The GEODESIC upgrade of the flat Euclidean positivity R (`sampling.stratified`,
#164): "in region R" is a geodesic ε-neighbourhood on the estimated manifold,
not a Euclidean ball. The load-bearing fact — **geodesic distance ≥ Euclidean
distance always** — means the manifold can only reveal points to be FARTHER, so
the failure this fixes is one-directional: two clouds Euclidean-near but on
different manifold sheets get *manufactured* positivity by a Euclidean R, and the
geodesic R correctly reports NO overlap. (There is no opposite error to fix:
nothing Euclidean-far is geodesically near.) This is the concrete mechanism
behind "manifold-aware propensity prevents FALSE positivity."

Two geodesic estimators:
  - **dijkstra** — exact shortest path on the k-NN distance graph (the Isomap
    geodesic; the ground-truth oracle). Disconnected sheets → ``inf``.
  - **heat** — Varadhan's formula ``d² ≈ −4t·log h_t`` with the heat kernel
    ``h_t = exp(−tL)`` applied matrix-free by #155's `heat_apply` (reuses the
    spectral machinery, GPU-able). Approximate; validated to rank-correlate with
    dijkstra on connected manifolds.

Design-only / gated like the rest of #99 — validated on synthetic manifolds with
known geodesic structure, dormant until the detector fires on real embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─── geodesic distance ──────────────────────────────────────────────────────────

def geodesic_distance_dijkstra(X, *, k: int = 15, sources=None) -> np.ndarray:
    """Exact graph geodesic (shortest path on the symmetric k-NN distance graph).
    `sources=None` → full (n, n) matrix; else (len(sources), n). Disconnected
    components are ``inf`` — the honest signal that R is unreachable."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra
    Dm = kneighbors_graph(X, k, mode="distance", include_self=False)
    Dm = Dm.maximum(Dm.T)
    return dijkstra(Dm, directed=False, indices=sources)


def geodesic_distance_heat(L, source: int, *, t: float = 0.05, order: int = 40) -> np.ndarray:
    """Varadhan heat-geodesic from node `source`: ``d ≈ sqrt(−4t·log(h_t/max))``
    with ``h_t = exp(−tL)`` via #155's `heat_apply` (matrix-free, numpy or cupy L).
    Approximate — the dijkstra path is the oracle; this reuses the spectral stack
    and is what runs matrix-free on the GPU at scale."""
    from .spectral import heat_apply, _is_cupy
    xp = __import__("cupy") if _is_cupy(L) else np
    delta = xp.zeros(L.shape[0])
    delta[source] = 1.0
    u = heat_apply(L, delta, t, order=order)
    u = xp.clip(u, 1e-30, None)
    return xp.sqrt(xp.maximum(-4.0 * t * xp.log(u / u.max()), 0.0))


# ─── geodesic positivity region R ───────────────────────────────────────────────

@dataclass
class GeodesicPositivityR:
    """Geodesic analogue of `sampling.stratified.PositivityR`. `coverage` = frac
    of targets with a particle within *geodesic* `radius`; `uncovered` flags the
    targets no particle geodesically reaches; `in_region` marks particles inside
    R; `mass_in_region` is their weight share (unweighted particle share if no
    `log_w`)."""
    radius: float
    coverage: float
    uncovered: np.ndarray        # bool over targets
    in_region: np.ndarray        # bool over particles (par_idx order)
    mass_in_region: float
    method: str


def _target_particle_geodesic(X, tgt_idx, par_idx, k, method) -> np.ndarray:
    """(n_targets, n_particles) geodesic distances."""
    if method == "dijkstra":
        D = geodesic_distance_dijkstra(X, k=k, sources=tgt_idx)     # (n_tgt, n)
        return np.asarray(D)[:, par_idx]
    if method == "heat":
        from .spectral import build_knn_laplacian
        L = build_knn_laplacian(X, k=k)
        rows = [np.asarray(geodesic_distance_heat(L, int(ti)))[par_idx] for ti in tgt_idx]
        return np.vstack(rows)
    raise ValueError(f"method must be 'dijkstra' or 'heat', got {method!r}")


def geodesic_positivity_overlap(X, particle_mask, target_mask, *, radius: float,
                                k: int = 15, method: str = "dijkstra",
                                log_w=None) -> GeodesicPositivityR:
    """Coverage/mass of the rare region R (targets) by the particle cloud under
    the geodesic metric. `X` holds ALL points (particles ∪ targets) so a single
    manifold graph carries both; the boolean masks pick each set out of it."""
    X = np.asarray(X, dtype=float)
    par_idx = np.nonzero(np.asarray(particle_mask, bool))[0]
    tgt_idx = np.nonzero(np.asarray(target_mask, bool))[0]
    d_tp = _target_particle_geodesic(X, tgt_idx, par_idx, k, method)   # (n_tgt, n_par)

    covered = d_tp.min(axis=1) <= radius if par_idx.size else np.zeros(tgt_idx.size, bool)
    coverage = float(covered.mean()) if tgt_idx.size else 1.0
    in_region = d_tp.min(axis=0) <= radius if tgt_idx.size else np.zeros(par_idx.size, bool)

    if log_w is None:
        mass = float(in_region.mean()) if par_idx.size else 0.0
    else:
        lw = np.asarray(log_w, dtype=float)[par_idx]
        m = np.max(lw)
        w = np.exp(lw - m) if np.isfinite(m) else np.zeros_like(lw)
        w = w / w.sum() if w.sum() > 0 else w
        mass = float(w[in_region].sum())
    return GeodesicPositivityR(radius=radius, coverage=coverage, uncovered=~covered,
                               in_region=in_region, mass_in_region=mass, method=method)


# ─── synthetic manifolds with known geodesic structure ──────────────────────────

def make_two_sheets(n: int = 150, gap: float = 0.3, seed: int = 0):
    """Two parallel unit sheets separated by `gap` in z. With `gap` >> intra-sheet
    spacing the k-NN graph keeps them DISCONNECTED (geodesic ∞ between sheets)
    while they stay Euclidean-near — the manufactured-positivity trap. Returns
    (X, sheetA_mask, sheetB_mask)."""
    rng = np.random.default_rng(seed)
    A = np.column_stack([rng.uniform(0, 1, (n, 2)), np.zeros(n)])
    B = np.column_stack([rng.uniform(0, 1, (n, 2)), np.full(n, gap)])
    X = np.vstack([A, B])
    a_mask = np.zeros(2 * n, bool); a_mask[:n] = True
    return X, a_mask, ~a_mask


def make_swiss_roll(n: int = 400, seed: int = 0):
    """Connected 2-D swiss roll in 3-D — a manifold where geodesic ≫ Euclidean
    across folds. Returns X (n, 3) and the intrinsic arclength coordinate t."""
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(0, 1, n))
    h = rng.uniform(0, 1, n)
    X = np.column_stack([t * np.cos(t), h * 21, t * np.sin(t)])
    return X, t
