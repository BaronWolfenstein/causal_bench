"""Geometry / stratification detectors — the GATE for the manifold-aware
propensity (#99). The spec's "what to measure first (cheap)": before paying for
any Riemannian machinery, measure whether the flat-embedding approximation
actually fails. DESIGN-ONLY / GATED and validated on synthetic curved-vs-flat-
vs-stratified data — so the moment real MOTOR/CLMBR embeddings arrive, pointing
these at the real manifold says immediately whether to go Riemannian.

Detectors (Axis A curvature + Axis B STRUCT-S):
  * local_intrinsic_dim            — Axis A: ID << ambient => a low-dim manifold
  * geodesic_euclidean_divergence  — Axis A: geodesic >> Euclidean => curvature
  * spectral_stratification        — Axis B (S2): near-zero eigenvalue count +
                                     spectral gap => disjoint sheets (reuses
                                     spectral.lanczos_smallest)
`screen` runs all three and returns the raw signals plus a coarse verdict.
"""
from __future__ import annotations

import numpy as np

from .spectral import build_knn_laplacian, lanczos_smallest


def local_intrinsic_dim(X, discard_frac: float = 0.1):
    """TwoNN intrinsic-dimension estimate (Facco et al.).  For each point,
    μ = r₂/r₁ (2nd/1st NN distance) ~ Pareto(d); fit −log(1−F) = d·log μ through
    the origin.  Returns (d_hat, per-point μ spread) — d_hat ≈ ambient on flat
    full-dim data, ≈ manifold dim on a curved sheet."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)                          # cols: self, 1st, 2nd
    r1, r2 = dist[:, 1], dist[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = mu[mu > 1.0 + 1e-9]
    mu.sort()
    n = len(mu)
    keep = int(n * (1 - discard_frac))                 # drop unstable tail (F->1)
    x = np.log(mu[:keep])
    y = -np.log(1.0 - (np.arange(1, n + 1) / (n + 1))[:keep])
    d_hat = float(np.sum(x * y) / np.sum(x * x))        # slope through origin
    return d_hat, float(np.std(np.log(mu)))


def geodesic_euclidean_divergence(X, k: int = 12, n_src: int = 80, seed: int = 0):
    """Curvature DIAGNOSTIC (reported, not gated): does the geodesic/Euclidean
    ratio GROW with distance?  On a curved manifold far-apart points' geodesics
    wrap (ratio rises with distance); on flat data the ratio is ~constant (a
    fixed graph-discretization factor — a raw median ratio is NOT a clean signal
    in high-D, which is why this is a diagnostic).  Returns
    (growth = mean ratio in the top vs bottom Euclidean tercile, frac_finite).
    NaN if too few connected far pairs (e.g. disjoint sheets)."""
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    n = X.shape[0]
    G = kneighbors_graph(X, k, mode="distance", include_self=False).maximum(
        kneighbors_graph(X, k, mode="distance", include_self=False).T)
    src = np.random.default_rng(seed).choice(n, size=min(n_src, n), replace=False)
    geo = shortest_path(G, method="D", directed=False, indices=src)
    euc = np.sqrt(((X[src][:, None, :] - X[None, :, :]) ** 2).sum(-1))
    fin = np.isfinite(geo) & (euc > 1e-9)
    frac_finite = float(fin.mean())
    e, ratio = euc[fin], geo[fin] / euc[fin]
    if e.size < 50:
        return float("nan"), frac_finite
    lo, hi = np.quantile(e, [1 / 3, 2 / 3])
    growth = float(np.mean(ratio[e >= hi]) / max(np.mean(ratio[e <= lo]), 1e-9))
    return growth, frac_finite


def spectral_stratification(X, k: int = 12, n_eig: int = 10, tol: float = 1e-6,
                            gap_sig: float = 4.0):
    """STRUCT-S / S2 via the eigengap heuristic on the normalized k-NN Laplacian.

    Two counts:
      * `n_components` = #eigenvalues below `tol` (true numerical zeros) = fully
        DISCONNECTED sheets.  Misses weakly-bridged sheets (one exact zero, a
        tiny algebraic-connectivity λ₁, then a jump).
      * `n_sheets` = the eigengap heuristic (von Luxburg): the largest relative
        gap `λ_{i+1}/λ_i` in the low spectrum marks the sheet boundary, IF that
        gap is significant (max gap ≫ median gap).  Catches BOTH clean-disjoint
        and weakly-bridged; a connected structureless manifold stays at 1.
    Returns (n_components, n_sheets, gap_significance, eigenvalues)."""
    L = build_knn_laplacian(X, k=k, normalized=True)
    vals, _ = lanczos_smallest(L, k=min(n_eig, X.shape[0] - 1))
    vals = np.sort(np.real(vals))
    n_zero = int(np.sum(vals < tol))
    # eigengap = largest ADDITIVE gap in the LOW spectrum (sheet boundaries live
    # there); ratios are avoided because the trivial ~0 eigenvalue blows them up.
    d_all = np.diff(vals)
    win = d_all[:min(6, len(d_all))]               # low-spectrum gaps only
    kstar = int(np.argmax(win)) + 1                # #eigenvalues before the largest gap
    sig = float(win.max() / (np.median(np.abs(d_all)) + 1e-12))   # vs typical gap
    n_sheets = kstar if sig > gap_sig else 1
    return n_zero, n_sheets, sig, vals


def screen(X, ambient_dim=None, k: int = 12):
    """Run all three detectors and return signals + a coarse verdict."""
    if ambient_dim is None:
        ambient_dim = X.shape[1]
    d_hat, mu_spread = local_intrinsic_dim(X)
    geo_growth, geo_frac_finite = geodesic_euclidean_divergence(X, k=k)
    n_comp, n_sheets, gap_sig, _ = spectral_stratification(X, k=k)
    # GATES (robust): curved = intrinsic dim materially below ambient;
    # stratified = >1 sheet via the eigengap heuristic (catches clean-disjoint AND
    # weakly-bridged; a connected structureless manifold stays at 1). geodesic
    # growth is a reported diagnostic only (noisy in high-D).
    curved = d_hat < 0.85 * ambient_dim
    stratified = n_sheets >= 2
    return dict(
        intrinsic_dim=d_hat, ambient_dim=float(ambient_dim), mu_spread=mu_spread,
        geodesic_growth=geo_growth, geodesic_frac_finite=geo_frac_finite,
        n_components=n_comp, n_sheets=n_sheets, gap_significance=gap_sig,
        curved=bool(curved), stratified=bool(stratified),
        flat_approx_fails=bool(curved or stratified),
    )
