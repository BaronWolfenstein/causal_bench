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

Closed-form insight: for a 1-manifold in R², the normal space is 1-D, so the
normal projector is N = n nᵀ and ``(I−P)v = n (n·v)``. The penalty per point is
therefore the SCALAR ``(n·s + n·r)²`` — the whole objective stays a single ridge
least-squares in the linear score weights. No torch, no SGD, deterministic.

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

    Penalty per gap point i (unit normal ``n_i``, restoring term ``r_i``):
        λ · ( n_i·(Wout φ_i) + n_i·r_i )²
    which is linear in the stacked weights ``w = [Wout[0], …, Wout[D-1]]`` — one
    ridge system of size ``D·d``. Works for any ambient dimension D (D=2 arc in
    R², D=3 Swiss roll in R³): the manifold is codimension-1 so ``n_i`` is a
    single unit vector and the penalty stays a scalar."""
    d = Phi.shape[1]
    D = target.shape[1]                                   # ambient dimension
    G = Phi.T @ Phi                                        # d×d, shared by all rows
    A = np.zeros((D * d, D * d))
    for k in range(D):
        A[k * d:(k + 1) * d, k * d:(k + 1) * d] = G       # block-diagonal DSM
    rhs = np.concatenate([Phi.T @ target[:, k] for k in range(D)])

    if lam > 0 and Phi_pen is not None:
        # scalar-per-point design row c_i = [n_i0·φ_i, …, n_i(D-1)·φ_i]; target −n_i·r_i
        C = np.hstack([normals[:, [k]] * Phi_pen for k in range(D)])
        d_pen = np.sum(normals * r_pen, axis=1)           # n_i · r_i
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
