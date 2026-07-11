"""Tangent-space-penalty DSM + gap-sampler — CPU prototype (PR #99 spec).

Instantiates the manifold-aware score-training loss from the manifold-aware
propensity design (docs/superpowers/specs/2026-07-10-manifold-aware-propensity-
design.md) on a synthetic curved 1-manifold:

    L = L_DSM  +  λ · E_gap[ || (I − P_x̃) ( s_θ(x̃) + (x̃ − x̃_proj)/σ_reg² ) ||² ]

L_DSM is ordinary denoising score matching on the *observed* (gapped) data. The
second term is evaluated on **gap-sampled** points — near-manifold points in an
under-covered region the DSM data never reaches — and penalizes the component of
the score NORMAL to the manifold that fails to point back toward it. So the
gap-sampler supplies the coordinates where the tangent penalty is enforced; the
two pieces are one mechanism.

Closed-form insight: a projector obeys ``||N v||² = Σ_j (n_j·v)²``, so the
penalty is a SUM of scalar terms — one per normal-basis vector — and the whole
objective stays a single ridge least-squares in the linear score weights, at ANY
codimension. No torch, no SGD, deterministic.

Manifolds (all analytic, for validation): ``ArcManifold`` (1-mfld in R², codim 1),
``SwissRoll`` (2-mfld in R³, codim 1), ``Helix`` (curve in R³, codim 2 — the
multi-normal case), ``Plane`` (flat 2-mfld in R⁵, codim 3 — the negative control
where the penalty should be harmless). ``estimate_local_normals`` reads the normal
basis off a point cloud by local PCA — the data-driven metric that lets the same
penalty run on REAL embeddings with no analytic manifold.

The score model is a random-Fourier-feature linear map (RBF-kernel ridge on the
velocity field) — a universal approximator that keeps everything analytic and
CPU-validatable, matching the numpy-core philosophy of the generative package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------- the manifold
class ArcManifold:
    """A 1-manifold: an arc of a circle of radius ``R`` with polar angle in
    ``[lo, hi]``, embedded in R². The normal at a point is the (unit) radial
    direction; the tangent is perpendicular to it."""

    def __init__(self, R: float = 1.0, lo: float = 0.0, hi: float = np.pi):
        self.R, self.lo, self.hi = float(R), float(lo), float(hi)

    def point(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        return self.R * np.column_stack([np.cos(theta), np.sin(theta)])

    def sample(self, n: int, rng: np.random.Generator,
               gaps: Sequence[Tuple[float, float]] = ()) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform angles in ``[lo, hi]`` minus any ``gaps`` intervals (rejection
        sampling). Returns ``(angles, points)``."""
        keep: List[float] = []
        while len(keep) < n:
            cand = rng.uniform(self.lo, self.hi, size=n)
            for g in gaps:
                cand = cand[(cand < g[0]) | (cand > g[1])]
            keep.extend(cand.tolist())
        ang = np.asarray(keep[:n])
        return ang, self.point(ang)

    def project(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest arc point (``feet``) and unit tangent there. For a circle the
        nearest point is the radial projection, angle-clamped to ``[lo, hi]``."""
        X = np.asarray(X, dtype=float)
        ang = np.clip(np.arctan2(X[:, 1], X[:, 0]), self.lo, self.hi)
        feet = self.point(ang)
        tang = np.column_stack([-np.sin(ang), np.cos(ang)])   # d/dθ, unit
        return feet, tang

    def normal(self, feet: np.ndarray) -> np.ndarray:
        """Unit normal (outward radial) at arc points ``feet``."""
        return feet / np.linalg.norm(feet, axis=1, keepdims=True)

    def gap_sample(self, gap: Tuple[float, float], *, n: int, sigma: float,
                   rng: np.random.Generator):
        """Near-manifold noised points whose feet fall in the angular ``gap``.
        Returns ``(x_noised, feet, unit_normals, r)`` with ``r = (x̃−foot)/σ²``."""
        ang = rng.uniform(gap[0], gap[1], size=n)
        feet = self.point(ang)
        x_noised = feet + sigma * rng.normal(size=feet.shape)
        normals = self.normal(feet)
        r = (x_noised - feet) / (sigma ** 2)
        return x_noised, feet, normals, r


def offmanifold_dist(X: np.ndarray, man: ArcManifold) -> np.ndarray:
    """Euclidean distance from each row of ``X`` to the manifold."""
    feet, _ = man.project(X)
    return np.linalg.norm(X - feet, axis=1)


# --------------------------------------------------------- random Fourier features
@dataclass
class RFF:
    """Random Fourier features for the RBF kernel: φ(x) = √(2/D)·cos(xW + b)."""
    n_features: int
    dim: int
    scale: float = 1.0
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(scale=self.scale, size=(self.dim, self.n_features))
        self.b = rng.uniform(0.0, 2 * np.pi, size=self.n_features)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.sqrt(2.0 / self.n_features) * np.cos(X @ self.W + self.b)


# --------------------------------------------------------------------- DSM pieces
def dsm_target(x0: np.ndarray, x_noised: np.ndarray, sigma: float) -> np.ndarray:
    """Denoising-score-matching target: ∇ log p_σ(x̃|x0) = −(x̃ − x0)/σ²."""
    return -(np.asarray(x_noised) - np.asarray(x0)) / (sigma ** 2)


def fit_linear_score(Phi: np.ndarray, target: np.ndarray, *,
                     Phi_pen: Optional[np.ndarray] = None,
                     normals: Optional[np.ndarray] = None,
                     r_pen: Optional[np.ndarray] = None,
                     lam: float = 0.0, ridge: float = 1e-4) -> np.ndarray:
    """Ridge-fit a 2-output linear score ``Wout`` (shape ``(2, d)``) so that
    ``ŝ = Phi @ Woutᵀ`` matches the DSM ``target`` (shape ``(n, 2)``), plus the
    tangent penalty on the ``Phi_pen`` points.

    ``normals`` gives, per gap point i, an orthonormal basis of the NORMAL space:
    shape ``(n, D)`` for a single normal (codimension-1) or ``(n, c, D)`` for a
    rank-``c`` normal space (codimension ≥ 2). Since a projector obeys
    ``||N v||² = Σ_j (n_j·v)²``, the penalty is a SUM of scalar terms — one per
    (point, normal-basis-vector):
        λ · Σ_j ( n_{i,j}·(Wout φ_i) + n_{i,j}·r_i )²
    each linear in the stacked weights ``w = [Wout[0], …, Wout[D-1]]``. So the
    whole objective stays one ridge system of size ``D·d`` for any codimension —
    a curve in R³ (c=2) or a low-dim manifold in a high-D embedding just stacks
    more penalty rows."""
    d = Phi.shape[1]
    D = target.shape[1]                                   # ambient dimension
    G = Phi.T @ Phi                                        # d×d, shared by all rows
    A = np.zeros((D * d, D * d))
    for k in range(D):
        A[k * d:(k + 1) * d, k * d:(k + 1) * d] = G       # block-diagonal DSM
    rhs = np.concatenate([Phi.T @ target[:, k] for k in range(D)])

    if lam > 0 and Phi_pen is not None:
        N = np.asarray(normals, dtype=float)
        if N.ndim == 2:                                   # (n, D) → (n, 1, D)
            N = N[:, None, :]
        C_blocks, d_blocks = [], []
        for j in range(N.shape[1]):                       # one block per normal-basis vector
            nj = N[:, j, :]                               # (n, D)
            C_blocks.append(np.hstack([nj[:, [k]] * Phi_pen for k in range(D)]))
            d_blocks.append(np.sum(nj * r_pen, axis=1))   # n_{i,j} · r_i
        C = np.vstack(C_blocks)
        d_pen = np.concatenate(d_blocks)
        A += lam * (C.T @ C)
        rhs -= lam * (C.T @ d_pen)

    A += ridge * np.eye(D * d)
    w = np.linalg.solve(A, rhs)
    return np.vstack([w[k * d:(k + 1) * d] for k in range(D)])   # (D, d)


def score_fn(Wout: np.ndarray, rff: RFF):
    """Return a callable X → score(X) for the fitted weights."""
    return lambda X: rff.transform(X) @ Wout.T


def denoise(x_noised: np.ndarray, Wout: np.ndarray, rff: RFF,
            sigma: float) -> np.ndarray:
    """One Tweedie denoising step: x̂0 = x̃ + σ²·ŝ(x̃)."""
    s = rff.transform(x_noised) @ Wout.T
    return np.asarray(x_noised) + (sigma ** 2) * s


# ------------------------------------------------------------------- gap sampler
def gap_noise_points(man, gap: Tuple[float, float], *, n: int,
                     sigma: float, rng: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``n`` near-manifold noised points whose feet fall inside ``gap`` — the
    under-covered region the DSM data never reaches. Returns
    ``(x_noised, feet, unit_normals, r)`` where ``r = (x_noised − feet)/σ²`` is
    the restoring term fed to the tangent penalty (σ_reg = σ).

    Off-support × near-M × within-stratum: the STRUCT-S battery decides WHICH
    gaps are gaps-to-fill (vs jumps-to-respect) upstream; here the gap is given.
    Delegates to the manifold's ``gap_sample`` so it works for any manifold
    (``ArcManifold`` in R², ``SwissRoll`` in R³)."""
    return man.gap_sample(gap, n=n, sigma=sigma, rng=rng)


# --------------------------------------------------------------- 2-manifold in R³
class SwissRoll:
    """The canonical Swiss-roll surface — a curved 2-manifold in R³, still
    codimension-1 (so the scalar tangent-penalty closed form applies unchanged):

        P(t, h) = (t·cos t,  h,  t·sin t),   t ∈ [t_lo, t_hi],  h ∈ [h_lo, h_hi].

    The ``h`` axis is orthogonal to the spiral, so nearest-point projection
    decouples: ``h* = clip(y)`` and ``t*`` is a 1-D search over the spiral — no
    scipy/KD-tree needed, everything numpy/analytic."""

    def __init__(self, t_lo: float = 1.5 * np.pi, t_hi: float = 4.0 * np.pi,
                 h_lo: float = 0.0, h_hi: float = 3.0, grid: int = 1500):
        self.t_lo, self.t_hi = float(t_lo), float(t_hi)
        self.h_lo, self.h_hi = float(h_lo), float(h_hi)
        self._tg = np.linspace(self.t_lo, self.t_hi, grid)          # 1-D t search grid
        self._xz = np.column_stack([self._tg * np.cos(self._tg),
                                    self._tg * np.sin(self._tg)])    # spiral in xz

    def point(self, t: np.ndarray, h: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        h = np.asarray(h, dtype=float)
        return np.column_stack([t * np.cos(t), h, t * np.sin(t)])

    def normal(self, t: np.ndarray) -> np.ndarray:
        """Unit normal n = normalize(∂P/∂t × ∂P/∂h); lies in the xz-plane."""
        t = np.asarray(t, dtype=float)
        nx = -(np.sin(t) + t * np.cos(t))
        nz = np.cos(t) - t * np.sin(t)
        n = np.column_stack([nx, np.zeros_like(t), nz])
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def sample(self, n: int, rng: np.random.Generator,
               gaps: Sequence[Tuple[float, float]] = ()) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform ``(t, h)`` with ``t`` avoiding ``gaps`` intervals. Returns
        ``(params, points)`` where ``params`` is the ``(n, 2)`` ``[t, h]`` array."""
        keep: List[float] = []
        while len(keep) < n:
            cand = rng.uniform(self.t_lo, self.t_hi, size=n)
            for g in gaps:
                cand = cand[(cand < g[0]) | (cand > g[1])]
            keep.extend(cand.tolist())
        t = np.asarray(keep[:n])
        h = rng.uniform(self.h_lo, self.h_hi, size=n)
        return np.column_stack([t, h]), self.point(t, h)

    def project(self, X: np.ndarray, chunk: int = 1000):
        """Nearest surface point (``feet``). ``h`` clips; ``t`` is the nearest
        spiral parameter over the 1-D grid (chunked to bound memory)."""
        X = np.asarray(X, dtype=float)
        h = np.clip(X[:, 1], self.h_lo, self.h_hi)
        xz = X[:, [0, 2]]
        tstar = np.empty(len(X))
        for i in range(0, len(X), chunk):
            block = xz[i:i + chunk]
            d2 = ((block[:, None, :] - self._xz[None, :, :]) ** 2).sum(-1)  # (b, grid)
            tstar[i:i + chunk] = self._tg[np.argmin(d2, axis=1)]
        return self.point(tstar, h), tstar

    def gap_sample(self, gap: Tuple[float, float], *, n: int, sigma: float,
                   rng: np.random.Generator):
        """Near-surface noised points whose feet have ``t`` inside ``gap``.
        Returns ``(x_noised, feet, unit_normals, r)``."""
        t = rng.uniform(gap[0], gap[1], size=n)
        h = rng.uniform(self.h_lo, self.h_hi, size=n)
        feet = self.point(t, h)
        x_noised = feet + sigma * rng.normal(size=feet.shape)
        normals = self.normal(t)
        r = (x_noised - feet) / (sigma ** 2)
        return x_noised, feet, normals, r


# ----------------------------------------------- 1-manifold in R³ (codimension 2)
class Helix:
    """A helix — a curve (1-manifold) in R³, so **codimension 2**: the normal
    space at each point is a 2-D plane, and the tangent penalty no longer
    collapses to a scalar. This exercises the general normal-basis path in
    ``fit_linear_score`` (the on-mission case: a low-dim manifold in a higher-D
    ambient space, like a patient-embedding manifold).

        P(t) = (R·cos t,  R·sin t,  b·t),   t ∈ [t_lo, t_hi].

    The unit normal basis is the Frenet principal-normal + binormal:
        N(t) = (−cos t, −sin t, 0),   B(t) = (b·sin t, −b·cos t, R)/√(R²+b²),
    both unit, orthogonal to each other and to the tangent."""

    def __init__(self, R: float = 1.0, b: float = 0.15,
                 t_lo: float = 0.0, t_hi: float = 4.0 * np.pi, grid: int = 3000):
        self.R, self.b = float(R), float(b)
        self.t_lo, self.t_hi = float(t_lo), float(t_hi)
        self.s = np.sqrt(self.R ** 2 + self.b ** 2)        # ||P'(t)||, constant
        self._tg = np.linspace(self.t_lo, self.t_hi, grid)
        self._pg = self.point(self._tg)                    # grid of curve points

    def point(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.column_stack([self.R * np.cos(t), self.R * np.sin(t), self.b * t])

    def tangent(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        tg = np.column_stack([-self.R * np.sin(t), self.R * np.cos(t),
                              self.b * np.ones_like(t)])
        return tg / self.s

    def frenet_normals(self, t: np.ndarray) -> np.ndarray:
        """Orthonormal normal basis ``(n, 2, 3)`` — [principal normal, binormal]."""
        t = np.asarray(t, dtype=float)
        N = np.column_stack([-np.cos(t), -np.sin(t), np.zeros_like(t)])
        B = np.column_stack([self.b * np.sin(t), -self.b * np.cos(t),
                             self.R * np.ones_like(t)]) / self.s
        return np.stack([N, B], axis=1)                    # (n, 2, 3)

    def sample(self, n: int, rng: np.random.Generator,
               gaps: Sequence[Tuple[float, float]] = ()) -> Tuple[np.ndarray, np.ndarray]:
        keep: List[float] = []
        while len(keep) < n:
            cand = rng.uniform(self.t_lo, self.t_hi, size=n)
            for g in gaps:
                cand = cand[(cand < g[0]) | (cand > g[1])]
            keep.extend(cand.tolist())
        t = np.asarray(keep[:n])
        return t, self.point(t)

    def project(self, X: np.ndarray, chunk: int = 1000):
        """Nearest curve point (``feet``): 1-D search over ``t`` (chunked)."""
        X = np.asarray(X, dtype=float)
        tstar = np.empty(len(X))
        for i in range(0, len(X), chunk):
            block = X[i:i + chunk]
            d2 = ((block[:, None, :] - self._pg[None, :, :]) ** 2).sum(-1)
            tstar[i:i + chunk] = self._tg[np.argmin(d2, axis=1)]
        return self.point(tstar), tstar

    def gap_sample(self, gap: Tuple[float, float], *, n: int, sigma: float,
                   rng: np.random.Generator):
        """Near-curve noised points whose feet have ``t`` inside ``gap``. Returns
        ``(x_noised, feet, normal_basis (n,2,3), r)``."""
        t = rng.uniform(gap[0], gap[1], size=n)
        feet = self.point(t)
        x_noised = feet + sigma * rng.normal(size=feet.shape)
        normals = self.frenet_normals(t)
        r = (x_noised - feet) / (sigma ** 2)
        return x_noised, feet, normals, r


# ---------------------------------------------- learned metric (data-driven normals)
def estimate_local_normals(X_ref: np.ndarray, query: np.ndarray, *, k: int = 30,
                           intrinsic_dim: int = 1) -> np.ndarray:
    """Estimate the NORMAL-space basis at each ``query`` point by local PCA on its
    ``k`` nearest neighbours in the reference cloud ``X_ref`` — the data-driven
    stand-in for an analytic ``frenet_normals``. The top ``intrinsic_dim`` local
    principal directions span the estimated tangent space; the remaining
    ``D − intrinsic_dim`` span the estimated normal space.

    Returns ``(n_query, D − intrinsic_dim, D)``, ready to pass as ``normals`` to
    ``fit_linear_score``. This is what makes the penalty work on REAL embeddings:
    no analytic manifold needed — the geometry is read off the point cloud. numpy
    brute-force k-NN (no sklearn), CPU."""
    X_ref = np.asarray(X_ref, dtype=float)
    query = np.asarray(query, dtype=float)
    D = X_ref.shape[1]
    k = min(k, len(X_ref))
    out = np.empty((len(query), D - intrinsic_dim, D))
    for i in range(len(query)):
        d2 = ((X_ref - query[i]) ** 2).sum(1)
        idx = np.argpartition(d2, k - 1)[:k]
        nbr = X_ref[idx]
        nbr = nbr - nbr.mean(0)
        _, _, Vt = np.linalg.svd(nbr, full_matrices=True)
        out[i] = Vt[intrinsic_dim:]                    # bottom rows = normal space
    return out


# --------------------------------------------------- flat manifold (negative control)
class Plane:
    """A 2-flat (flat 2-manifold) embedded in R^D — **codimension D−2**, and
    FLAT (zero curvature). The control case for the gated design: on a flat
    manifold the plain Euclidean DSM is already correct (the normal field is
    constant), so the tangent penalty should be a near-no-op. Random orthonormal
    tangent/normal frame; analytic projection."""

    def __init__(self, dim: int = 5, intrinsic: int = 2, span: float = 3.0,
                 seed: int = 0):
        self.dim, self.intrinsic, self.span = dim, intrinsic, float(span)
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        self.T = Q[:intrinsic]                          # (intrinsic, D) tangent basis
        self.N = Q[intrinsic:]                          # (D-intrinsic, D) normal basis
        self.origin = rng.normal(size=dim)

    def point(self, coords: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(coords, dtype=float) @ self.T

    def sample(self, n: int, rng: np.random.Generator,
               gaps: Sequence[Tuple[float, float]] = ()):
        keep: List = []
        while len(keep) < n:
            c0 = rng.uniform(-self.span, self.span, size=n)
            for g in gaps:
                c0 = c0[(c0 < g[0]) | (c0 > g[1])]       # gap on the first tangent coord
            keep.extend(c0.tolist())
        c0 = np.asarray(keep[:n])
        rest = rng.uniform(-self.span, self.span, size=(n, self.intrinsic - 1))
        coords = np.column_stack([c0, rest])
        return coords, self.point(coords)

    def normal_basis(self, n: int) -> np.ndarray:
        """Constant orthonormal normal basis, tiled to ``(n, D−intrinsic, D)``."""
        return np.tile(self.N[None], (n, 1, 1))

    def project(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        d = X - self.origin
        feet = self.origin + (d @ self.T.T) @ self.T    # drop normal components
        return feet, None

    def gap_sample(self, gap: Tuple[float, float], *, n: int, sigma: float,
                   rng: np.random.Generator):
        c0 = rng.uniform(gap[0], gap[1], size=n)
        rest = rng.uniform(-self.span, self.span, size=(n, self.intrinsic - 1))
        coords = np.column_stack([c0, rest])
        feet = self.point(coords)
        x_noised = feet + sigma * rng.normal(size=feet.shape)
        normals = self.normal_basis(n)
        r = (x_noised - feet) / (sigma ** 2)
        return x_noised, feet, normals, r
