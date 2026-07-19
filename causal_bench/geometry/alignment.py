"""Multimodal manifold alignment via the generalized eigenproblem (#160, gated).

Fuse two embedding graphs (e.g. EHR + imaging) into ONE shared latent frame so a
single propensity/score `g` lives on the aligned manifold — the "one g on the
aligned manifold" requirement from the #99 design note. Build a joint block
Laplacian `L_joint` with cross-links at a set of known correspondences, and solve
the generalized eigenproblem

    L_joint f = λ D_joint f            (Ham–Lee–Saul spectral alignment)

for the low-frequency coordinates `f`. Restricting `f` to the A-rows and B-rows
gives the two clouds' coordinates in the SAME eigenbasis — corresponding points
coincide, so the alignment generalizes from a few anchors to all points. Needs
the eigen*vectors* (no Chebyshev shortcut), so it uses `eigsh` / #155's Lanczos.

Well-posedness caveat (known, not a bug): spectral alignment needs a
NON-degenerate low spectrum. Symmetric manifolds (a uniform square, a sphere)
have eigenvalue degeneracy → eigenvector rotation ambiguity, so exact
point-matching is ill-posed there; the aligned coordinates are only defined up to
a rotation within each degenerate eigenspace. Validated here on a curve/strip
with an ordered spectrum, where the first coordinate cleanly recovers the
intrinsic parameter for both views.

Design-only / gated: validated on synthetic warped copies with KNOWN
correspondence, dormant until a real second modality lands (the MitralVision
imaging-encoder trigger). Gate is a second modality, NOT real region R.
"""
from __future__ import annotations

import numpy as np


def _knn_weights(X, k: int = 15, sigma=None):
    """Symmetric Gaussian k-NN weight matrix (the `W` inside #155's Laplacian)."""
    from sklearn.neighbors import kneighbors_graph
    import scipy.sparse as sp
    Dm = kneighbors_graph(X, k, mode="distance", include_self=False)
    Dm = Dm.maximum(Dm.T)
    if sigma is None:
        sigma = float(np.median(Dm.data))
    W = Dm.copy()
    W.data = np.exp(-(Dm.data ** 2) / (2.0 * sigma ** 2))
    return W.tocsr()


def joint_laplacian(Wa, Wb, corr, mu: float):
    """Block graph [Wa, 0; 0, Wb] plus cross-edges of weight `mu` at each
    correspondence (i, j) (i in A, j in B). Returns (L_joint, D_joint) for the
    generalized eigenproblem, both CSR."""
    import scipy.sparse as sp
    na = Wa.shape[0]
    W = sp.block_diag([Wa, Wb]).tolil()
    for i, j in corr:
        W[i, na + j] += mu
        W[na + j, i] += mu
    W = W.tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = sp.diags(deg)
    L = (D - W).tocsr()
    return L, D.tocsr()


def align(Xa, Xb, corr, *, d: int = 2, k: int = 15, mu: float = 1.0):
    """Align two clouds into a shared d-dim frame. `corr` is a list of (i, j)
    anchor pairs. Returns (Fa, Fb, evals): Fa (na, d) and Fb (nb, d) are the two
    clouds in the SAME joint eigenbasis (corresponding points coincide); evals
    are the d smallest non-trivial generalized eigenvalues."""
    from scipy.sparse.linalg import eigsh
    Wa, Wb = _knn_weights(Xa, k), _knn_weights(Xb, k)
    na = Wa.shape[0]
    L, D = joint_laplacian(Wa, Wb, corr, mu)
    # generalized L f = λ D f; shift-invert just above 0 isolates the smallest
    # eigenpairs robustly (L is singular at exactly 0 -> use a tiny positive sigma)
    vals, vecs = eigsh(L, k=d + 1, M=D, sigma=1e-8, which="LM")
    order = np.argsort(vals)
    vecs = vecs[:, order]
    F = vecs[:, 1:d + 1]                    # drop the trivial constant eigenvector
    return F[:na], F[na:], np.sort(vals)[1:d + 1]


def alignment_error(Fa, Fb) -> float:
    """Mean distance between corresponding points in the shared frame, normalized
    by the cloud scale (row i of Fa ↔ row i of Fb). ~0 = the two views coincide;
    the honest "did corresponding points land on top of each other" metric
    (robust to dense sampling, unlike exact nearest-neighbour)."""
    Fa, Fb = np.asarray(Fa), np.asarray(Fb)
    scale = float(np.std(np.vstack([Fa, Fb]))) + 1e-12
    return float(np.mean(np.linalg.norm(Fa - Fb, axis=1)) / scale)


def retrieval_accuracy(Fa, Fb, k: int = 10) -> float:
    """Fraction of A-points whose true correspondent is among its k nearest
    B-points in the shared frame. Density-robust (adjacent manifold points crowd
    each other, so top-k retrieval, not exact rank-1, is the meaningful check)."""
    from scipy.spatial import cKDTree
    nn = np.asarray(cKDTree(Fb).query(Fa, k=k)[1]).reshape(Fa.shape[0], -1)
    return float(np.mean([i in nn[i] for i in range(Fa.shape[0])]))


def make_aligned_pair(n: int = 200, seed: int = 0, warp: bool = True):
    """Two views of one intrinsic manifold with KNOWN correspondence (row i ↔ row
    i): A is a 1-D-dominant strip (t + thin transverse noise → ordered,
    non-degenerate spectrum); B is a rotation (+ optional nonlinear lift to 3-D)
    of it. Same manifold, different ambient coordinates. Returns (Xa, Xb)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)
    Xa = np.column_stack([t, 0.03 * rng.standard_normal(n)])
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    XY = Xa @ R
    Xb = np.column_stack([XY, np.sin(3.0 * t)]) if warp else XY
    return Xa, Xb
