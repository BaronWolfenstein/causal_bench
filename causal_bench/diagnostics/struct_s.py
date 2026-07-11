"""STRUCT-S — stratification detector for the manifold/discreteness design axis
(docs/superpowers/specs/2026-07-10-manifold-aware-propensity-design.md).

Decides whether a frozen-encoder embedding is a single smooth manifold or continuous
sheets joined by (event-driven) discrete jumps — i.e. whether a hybrid jump-diffusion
generator could ever be warranted. This module implements the CPU tests:

- S2 ``spectral_component_count`` — eigengap heuristic on the k-NN-graph Laplacian.
- S3 ``local_id_heterogeneity`` — dispersion of the per-point local intrinsic dim.
- S4 ``density_gap`` — low-density separators via the MST edge-length gap.
- S1 ``event_aligned_bimodality`` — the *decisive* test: displacement bimodality
  (via ``standardized_bimodality``, the size-invariant Z-Dip principle, #107) AND
  alignment of that split with intercurrent-event markers.

S2-S4 are embedding-only and **corroborate / rule out** (flag or clear a *candidate*
stratification). **S1 is decisive** but needs displacements + intercurrent-event
markers: the algorithm is CPU/synthetic-validatable and lives here, while feeding it
*real* temporal MEDS trajectory displacements + ICE timestamps is the on-box step.
Only S1 confirms the *event-driven jump* structure that licenses the jump term.
numpy/scipy/sklearn only.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, laplacian, minimum_spanning_tree
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def spectral_component_count(Z: np.ndarray, *, n_neighbors: int = 8,
                             max_k: int = 10) -> dict:
    """S2 — number of near-disconnected sheets via the **eigengap heuristic** on the
    normalized Laplacian of a **Gaussian (RBF) affinity** graph with a *local* scale
    ``σ`` (median distance to the ``n_neighbors``-th NN). Full affinity avoids k-NN
    fragmentation; the local ``σ`` makes cross-sheet affinity ~0 for well-separated
    sheets and small-but-nonzero for bridged ones — so the eigengap (not a
    zero-threshold) sets ``n_strata``. Dense eigenvalues (robust for diagnostic-sized
    cohorts; O(n³) — large-graph sparse solve is the scaling follow-up).
    Returns ``{n_strata, eigenvalues, gaps}``."""
    Z = np.asarray(Z, float)
    n = len(Z)
    tiny = np.finfo(float).tiny
    kk = min(n_neighbors + 1, n)
    dk, _ = NearestNeighbors(n_neighbors=kk).fit(Z).kneighbors(Z)
    sigma = float(np.median(dk[:, -1])) or 1.0     # local scale
    D = pairwise_distances(Z)
    W = np.exp(-(D ** 2) / (2.0 * sigma ** 2 + tiny))
    np.fill_diagonal(W, 0.0)
    L = np.asarray(laplacian(W, normed=True))
    vals = np.sort(np.linalg.eigvalsh(L))[: max_k + 1]
    gaps = np.diff(vals)
    n_strata = int(np.argmax(gaps) + 1)            # eigengap: strata before the gap
    return {"n_strata": n_strata, "eigenvalues": vals, "gaps": gaps}


def local_id_heterogeneity(Z: np.ndarray, *, k: int = 10) -> dict:
    """S3 — per-point **local intrinsic dimension** (Levina-Bickel kNN MLE) and its
    dispersion. A single smooth manifold has ~constant local ID; high dispersion
    (``cv``/``iqr``) signals strata of differing dimension. Returns
    ``{local_id, cv, iqr}``."""
    Z = np.asarray(Z, float)
    n = len(Z)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    dist, _ = nn.kneighbors(Z)                      # (n, k+1); col 0 is self (dist 0)
    r = dist[:, 1:]                                 # (n, k) NN distances, ascending
    eps = np.finfo(float).tiny
    # MLE: d_hat(x) = [ (1/(k-1)) Σ_{j=1..k-1} ln(r_k / r_j) ]^{-1}
    logratio = np.log((r[:, -1:] + eps) / (r[:, :-1] + eps))   # (n, k-1)
    mean_logratio = logratio.mean(axis=1)
    local_id = 1.0 / np.where(mean_logratio > eps, mean_logratio, np.nan)
    local_id = np.nan_to_num(local_id, nan=np.nanmedian(local_id))
    cv = float(np.std(local_id) / (np.mean(local_id) + eps))
    q75, q25 = np.percentile(local_id, [75, 25])
    return {"local_id": local_id, "cv": cv, "iqr": float(q75 - q25)}


def density_gap(Z: np.ndarray, *, quantile: float = 0.9, threshold: float = 3.0,
                min_frac: float = 0.1) -> dict:
    """S4 — low-density **separators** between high-density regions. A minimum
    spanning tree has one long edge bridging otherwise-dense clusters;
    ``gap_ratio = max_edge / quantile(edges, q)`` is large when such a separator
    exists (~1-2 for a single connected support). To avoid flagging a lone tail
    **outlier** (whose long MST edge merely isolates *one* point), the long edge only
    counts as a gap if cutting it splits the MST into **balanced** pieces — the
    smaller side ≥ ``min_frac`` of the points (a real cluster split, not an outlier).
    Returns ``{gap_ratio, has_gap, max_edge, ref_edge, min_side_frac}``."""
    Z = np.asarray(Z, float)
    n = len(Z)
    tiny = np.finfo(float).tiny
    mst = minimum_spanning_tree(pairwise_distances(Z)).tocoo()
    if mst.data.size == 0:
        return {"gap_ratio": 1.0, "has_gap": False, "max_edge": 0.0,
                "ref_edge": 0.0, "min_side_frac": 0.0}
    max_edge = float(mst.data.max())
    ref_edge = float(np.quantile(mst.data, quantile))
    gap_ratio = max_edge / (ref_edge + tiny)
    # cut the single largest edge; measure how balanced the two sides are
    keep = mst.data < max_edge
    pruned = coo_matrix((mst.data[keep], (mst.row[keep], mst.col[keep])), shape=(n, n))
    _, labels = connected_components(pruned + pruned.T, directed=False)
    sizes = np.sort(np.bincount(labels))
    min_side_frac = float(sizes[-2] / n) if len(sizes) >= 2 else 0.0
    has_gap = bool(gap_ratio > threshold and min_side_frac >= min_frac)
    return {"gap_ratio": gap_ratio, "has_gap": has_gap, "max_edge": max_edge,
            "ref_edge": ref_edge, "min_side_frac": min_side_frac}


# ------------------------------------------------------------------ S1 (decisive)
def _to_magnitude(displacements: np.ndarray) -> np.ndarray:
    """Reduce displacement vectors to a 1-D magnitude for the bimodality test
    (a scalar displacement is passed through)."""
    d = np.asarray(displacements, float)
    return d if d.ndim == 1 else np.linalg.norm(d.reshape(len(d), -1), axis=1)


def _gmm_bic_gap(x: np.ndarray, *, seed: int) -> float:
    """BIC(1-component) − BIC(2-component) for 1-D ``x``. Positive ⇒ a two-mode
    fit is preferred (bimodal); ~0 or negative ⇒ unimodal. BIC's complexity
    penalty already accounts for n, so the gap is a sound bimodality statistic."""
    from sklearn.mixture import GaussianMixture
    x = x.reshape(-1, 1)
    b1 = GaussianMixture(1, random_state=seed).fit(x).bic(x)
    b2 = GaussianMixture(2, random_state=seed, n_init=2).fit(x).bic(x)
    return float(b1 - b2)


def standardized_bimodality(x: np.ndarray, *, n_null: int = 40,
                            seed: int = 0) -> dict:
    """Size-invariant bimodality score (the **Z-Dip** principle, issue #107):
    standardize the GMM-BIC gap against a **unimodal** Gaussian null of the same n.
    ``z_bimodal`` is comparable across sample sizes with a universal threshold,
    unlike the raw gap (or raw Hartigan dip) which drifts with n. Returns
    ``{z_bimodal, bic_gap}``. (#107's route swaps the GMM gap for Hartigan's dip
    as the underlying statistic; the standardization here is the same idea.)"""
    x = _to_magnitude(x)
    eps = np.finfo(float).tiny
    gap = _gmm_bic_gap(x, seed=seed)
    rng = np.random.default_rng(seed)
    mu, sd = float(x.mean()), float(x.std()) + eps
    null = np.array([_gmm_bic_gap(rng.normal(mu, sd, size=len(x)),
                                  seed=int(rng.integers(1_000_000)))
                     for _ in range(n_null)])
    z = (gap - null.mean()) / (null.std() + eps)
    return {"z_bimodal": float(z), "bic_gap": gap}


def event_aligned_bimodality(displacements: np.ndarray, ice_flags: np.ndarray, *,
                             z_threshold: float = 3.0, align_threshold: float = 0.7,
                             n_null: int = 40, seed: int = 0) -> dict:
    """S1 — the *decisive* test: is the displacement distribution **bimodal** AND
    is that split **aligned with intercurrent events**? Only both together license
    the event-driven jump term (a bimodal split unrelated to ICEs is not).

    ``displacements`` — per-sample scalar or vector displacement (reduced to
    magnitude). ``ice_flags`` — per-sample 0/1 intercurrent-event marker. The
    algorithm is CPU/synthetic-validatable; feeding it *real* MEDS trajectory
    displacements + ICE timestamps is the on-box step. Alignment = direction-
    agnostic AUC of ICE-status predicted by displacement magnitude (0.5 = none,
    1.0 = the high-displacement mode is exactly the ICE cohort)."""
    from sklearn.metrics import roc_auc_score
    x = _to_magnitude(displacements)
    ice = np.asarray(ice_flags).astype(int)
    bm = standardized_bimodality(x, n_null=n_null, seed=seed)
    bimodal = bm["z_bimodal"] > z_threshold
    if len(np.unique(ice)) < 2:
        align = 0.5
    else:
        auc = roc_auc_score(ice, x)
        align = float(max(auc, 1.0 - auc))            # direction-agnostic
    ice_aligned = align > align_threshold
    return {
        "s1_confirms": bool(bimodal and ice_aligned),
        "bimodal": bool(bimodal),
        "z_bimodal": bm["z_bimodal"],
        "ice_alignment": align,
        "ice_aligned": bool(ice_aligned),
        "bic_gap": bm["bic_gap"],
    }


# ------------------------------------------------ S5 (hierarchical-compositionality)
def hierarchical_levels(Z: np.ndarray, *, k_grid=(2, 3, 4, 6, 8, 12, 16, 24),
                        n_grid: int = 18) -> dict:
    """S5 — hierarchical/compositional structure via the **diffusion phase-transition**
    depth probe (Sclocchi-Favero-Wyart; `diagnostics/hierarchy_probe.py`, #122).
    Where S2-S4 ask "is the embedding stratified into sheets/jumps?", S5 asks "is
    it a **tree of latents?**" — a staircase in the transition ``t*(k)`` across
    clustering granularities signals hierarchy depth. Returns ``{estimated_levels,
    hierarchical, t_star_of_k}`` (``hierarchical`` ⟺ > 1 distinct scale)."""
    from causal_bench.diagnostics.hierarchy_probe import depth_scan
    r = depth_scan(np.asarray(Z, float), k_grid=k_grid, n_grid=n_grid)
    return {"estimated_levels": r["estimated_levels"],
            "hierarchical": bool(r["estimated_levels"] > 1),
            "t_star_of_k": r["t_star_of_k"]}


def struct_s_screen(Z: np.ndarray, *, n_neighbors: int = 10, k: int = 10,
                    displacements: np.ndarray | None = None,
                    ice_flags: np.ndarray | None = None, n_null: int = 40,
                    run_s5: bool = True) -> dict:
    """Run S2-S4 (+ optional S1, S5) and summarize. ``candidate_stratified`` is
    True if S2 finds > 1 stratum OR S4 finds a density gap (S3 corroborates). If
    ``displacements`` **and** ``ice_flags`` are supplied, also run the decisive S1
    (event-aligned bimodality). If ``run_s5`` (default), run S5 (diffusion
    phase-transition depth probe) → ``S5_levels`` / ``S5_hierarchical``."""
    s2 = spectral_component_count(Z, n_neighbors=n_neighbors)
    s3 = local_id_heterogeneity(Z, k=k)
    s4 = density_gap(Z)
    candidate = bool(s2["n_strata"] > 1 or s4["has_gap"])
    out = {
        "candidate_stratified": candidate,
        "needs_S1_to_confirm": True,
        "S1_confirms": None,
        "S2_n_strata": s2["n_strata"],
        "S3_local_id_cv": s3["cv"],
        "S4_gap_ratio": s4["gap_ratio"],
        "S4_has_gap": s4["has_gap"],
    }
    if displacements is not None and ice_flags is not None:
        s1 = event_aligned_bimodality(displacements, ice_flags, n_null=n_null)
        out["needs_S1_to_confirm"] = False
        out["S1_confirms"] = s1["s1_confirms"]
        out["S1_z_bimodal"] = s1["z_bimodal"]
        out["S1_ice_alignment"] = s1["ice_alignment"]
    if run_s5:
        s5 = hierarchical_levels(Z)
        out["S5_levels"] = s5["estimated_levels"]
        out["S5_hierarchical"] = s5["hierarchical"]
    return out
