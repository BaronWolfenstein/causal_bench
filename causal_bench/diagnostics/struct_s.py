"""STRUCT-S — stratification detector for the manifold/discreteness design axis
(docs/superpowers/specs/2026-07-10-manifold-aware-propensity-design.md).

Decides whether a frozen-encoder embedding is a single smooth manifold or continuous
sheets joined by (event-driven) discrete jumps — i.e. whether a hybrid jump-diffusion
generator could ever be warranted. This module implements the CPU, embedding-only
tests **S2-S4**:

- S2 ``spectral_component_count`` — eigengap heuristic on the k-NN-graph Laplacian.
- S3 ``local_id_heterogeneity`` — dispersion of the per-point local intrinsic dim.
- S4 ``density_gap`` — low-density separators via the MST edge-length gap.

**S1** (event-aligned displacement bimodality) is the *decisive* test but needs real
temporal MEDS trajectories with intercurrent-event timestamps — it is on-box-gated and
NOT here. S2-S4 therefore **corroborate / rule out**: they can flag a *candidate*
stratification (or clear it), but only S1 confirms the *event-driven jump* structure
that licenses the jump term. numpy/scipy/sklearn only.
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


def struct_s_screen(Z: np.ndarray, *, n_neighbors: int = 10, k: int = 10) -> dict:
    """Run S2-S4 and summarize. ``candidate_stratified`` is True if S2 finds > 1
    stratum OR S4 finds a density gap (S3 heterogeneity is reported as corroboration).
    Always sets ``needs_S1_to_confirm=True``: S2-S4 cannot establish the *event-driven
    jump* structure that licenses a jump-diffusion — only the on-box S1 can."""
    s2 = spectral_component_count(Z, n_neighbors=n_neighbors)
    s3 = local_id_heterogeneity(Z, k=k)
    s4 = density_gap(Z)
    candidate = bool(s2["n_strata"] > 1 or s4["has_gap"])
    return {
        "candidate_stratified": candidate,
        "needs_S1_to_confirm": True,
        "S2_n_strata": s2["n_strata"],
        "S3_local_id_cv": s3["cv"],
        "S4_gap_ratio": s4["gap_ratio"],
        "S4_has_gap": s4["has_gap"],
    }
