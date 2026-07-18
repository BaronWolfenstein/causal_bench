"""Graph-Laplacian spectral machinery for the manifold-aware propensity (#99).

DESIGN-ONLY / GATED: this is the computational realization from
`docs/superpowers/specs/2026-07-10-manifold-aware-propensity-design.md`
("one Laplacian, two solvers").  It is validated on synthetic manifolds with
KNOWN geometry and is NOT wired to any estimator — it exists so that, the moment
real MOTOR/CLMBR embeddings unblock the gate, the detector + heat-kernel `g` can
run on the real manifold.

Every geometric object is a spectral filter `h(L)` of one k-NN graph Laplacian:
  heat kernel / metric g   h(λ) = exp(-t λ)
  Whittle-Matérn covariance h(λ) = (κ² + λ)^(-ν/2)   (ν tunes edge-case roughness)
Chebyshev applies `h(L)` matrix-free (O(edges), no eigenvectors); Lanczos returns
top-k eigenpairs when the coordinates themselves are needed (multimodal
alignment).  CPU (scipy sparse) or GPU (cupy sparse) is chosen by the array type
of `L` — the same "namespace by array" seam as the SMC backend.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# k-NN graph Laplacian
# ---------------------------------------------------------------------------
def build_knn_laplacian(X, k: int = 15, sigma=None, normalized: bool = True):
    """Symmetric k-NN graph Laplacian (scipy CSR).  Gaussian edge weights with a
    self-tuning bandwidth (median k-NN distance).  normalized=True gives the
    symmetric normalized Laplacian `I − D^{-1/2} W D^{-1/2}` (spectrum ⊂ [0, 2],
    the discrete Laplace–Beltrami analogue); False gives `D − W`."""
    from sklearn.neighbors import kneighbors_graph
    import scipy.sparse as sp
    n = X.shape[0]
    Dm = kneighbors_graph(X, k, mode="distance", include_self=False)
    Dm = Dm.maximum(Dm.T)                          # symmetrize (mutual union)
    if sigma is None:
        sigma = float(np.median(Dm.data))
    W = Dm.copy()
    W.data = np.exp(-(Dm.data ** 2) / (2.0 * sigma ** 2))
    deg = np.asarray(W.sum(axis=1)).ravel()
    if normalized:
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        Dh = sp.diags(dinv)
        L = sp.eye(n) - Dh @ W @ Dh
    else:
        L = sp.diags(deg) - W
    return L.tocsr()


def to_gpu(L):
    """scipy CSR -> cupy CSR (for GPU Chebyshev filtering)."""
    import cupyx.scipy.sparse as csp
    return csp.csr_matrix(L)


def _is_cupy(L):
    return "cupy" in type(L).__module__


# ---------------------------------------------------------------------------
# Chebyshev filter apply — h(L) @ signal, matrix-free
# ---------------------------------------------------------------------------
def lambda_max(L, iters: int = 30):
    """Largest eigenvalue via power iteration (matrix-free)."""
    xp = __import__("cupy") if _is_cupy(L) else np
    v = xp.asarray(np.random.default_rng(0).standard_normal(L.shape[0]))
    v = v / xp.linalg.norm(v)
    for _ in range(iters):
        v = L @ v
        nv = xp.linalg.norm(v)
        if float(nv) == 0:
            break
        v = v / nv
    return float(v @ (L @ v))


def _cheb_coeffs(h, order: int, lmax: float):
    """Chebyshev coefficients of `h` on [0, lmax] (host numpy; tiny)."""
    m = order + 1
    theta = np.pi * (np.arange(m) + 0.5) / m
    lam = (np.cos(theta) + 1.0) * lmax / 2.0                    # [0, lmax]
    hv = h(lam)
    return np.array([(2.0 / m) * np.sum(hv * np.cos(k * theta)) for k in range(order + 1)])


def cheb_apply(L, signal, h, order: int = 40, lmax=None):
    """Apply `h(L)` to `signal` via the Chebyshev recurrence on `L̃ = 2L/lmax − I`.
    O(order · nnz), no eigen-decomposition.  `h` is a numpy-vectorized function of
    the eigenvalue.  `signal` and `L` share device (numpy/cupy)."""
    if lmax is None:
        lmax = lambda_max(L)                                     # matrix-free power iter
    c = _cheb_coeffs(h, order, lmax)
    xp = __import__("cupy") if _is_cupy(L) else np
    c = xp.asarray(c)
    a = 2.0 / lmax
    t0 = signal
    t1 = a * (L @ signal) - signal                              # L̃ @ signal
    out = 0.5 * c[0] * t0 + c[1] * t1
    for k in range(2, order + 1):
        t2 = 2.0 * (a * (L @ t1) - t1) - t0                     # 2 L̃ T_{k-1} − T_{k-2}
        out = out + c[k] * t2
        t0, t1 = t1, t2
    return out


def heat_apply(L, signal, t: float, order: int = 40, lmax=None):
    """Heat-kernel filter `exp(−t L) @ signal` (the metric-g / Layer-3 corruption)."""
    return cheb_apply(L, signal, lambda lam: np.exp(-t * lam), order, lmax)


def matern_apply(L, signal, kappa: float = 1.0, nu: float = 1.5,
                 order: int = 40, lmax=None):
    """Whittle–Matérn filter `(κ² + L)^(−ν/2) @ signal` — the propensity covariance
    applied to data (never materializes the dense covariance)."""
    return cheb_apply(L, signal, lambda lam: (kappa ** 2 + lam) ** (-nu / 2.0), order, lmax)


# ---------------------------------------------------------------------------
# Lanczos — top-k eigenpairs (needed for the low-frequency coordinates)
# ---------------------------------------------------------------------------
def lanczos_smallest(L, k: int):
    """Top-k smallest eigenpairs of `L` (the low-frequency manifold coordinates /
    truncated Matérn basis / alignment `f`).  There is no Chebyshev shortcut for
    the eigenvectors themselves."""
    if _is_cupy(L):
        from cupyx.scipy.sparse.linalg import eigsh
    else:
        from scipy.sparse.linalg import eigsh
    vals, vecs = eigsh(L, k=k, which="SA")                      # smallest algebraic
    order = np.argsort(vals.get() if _is_cupy(L) else vals)
    return vals[order], vecs[:, order]
