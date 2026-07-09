"""Zero-flow conditional-independence test (Wang, Wang, Liu, Suzuki 2026,
"Zero-Flow Encoders", arXiv:2602.00797) — numpy/sklearn, CPU, no torch.

Rectified-flow view: the optimal velocity of a rectified flow with *independent*
coupling is v*(x, t) = E[x1 - x0 | x_t = x]; at t=0.5, x_0.5 = ½(x0 + x1). The
**zero-flow criterion**: v* ≡ 0 at t=0.5 iff the two distributions are equal. We
don't train a flow — we estimate that conditional expectation with a cross-fitted
regressor and use the mean squared magnitude of its prediction as a
distribution-difference statistic. For two samples with equal means/marginals it
still has power, because Cov(v, x_0.5) = ½(Σ₁ − Σ₀): a covariance/copula gap makes
the velocity linearly detectable.

CI test for X ⫫ Y | Z (residualized form — high power, fast):
  1. residualize rX = X − Ê[X|Z], rY = Y − Ê[Y|Z] (cross-fitted; a flexible
     regressor absorbs nonlinear Z effects). Under H0, rX ⫫ rY.
  2. test rX ⫫ rY with the zero-flow statistic between the residual joint and its
     Y-permuted product; a permutation p-value calibrates it.

Returns a `CITestResult` whose fields (verdict / test / effective_n) map 1:1 onto
SGA's `EmpiricalCIResult` — wiring a verdict onto a KG claim edge is a plain
field copy. No torch, no GPU.

Extension: a torch neural-velocity backend (for high-dim / embedding-space CI)
drops into `velocity_factory` with no API change — specced in
docs/superpowers/specs/2026-07-08-zero-flow-torch-extension.md, deferred until a
high-dim × large-n regime warrants it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


@dataclass(frozen=True)
class CITestResult:
    verdict: str          # "supports" (CI holds) | "refutes" (CI violated) | "underpowered"
    test: str             # "zero-flow-ci"
    effective_n: int
    p_value: float
    statistic: float


def _col(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _default_nuisance():
    """Flexible regressor for residualizing on Z (absorbs nonlinear Z effects)."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=60, max_depth=8, random_state=0, n_jobs=1)


def _default_velocity():
    """Velocity model for the zero-flow statistic. Linear is exact for the
    covariance/copula signal (Cov(v, x_0.5) = ½ΔΣ) and is instant; pass a
    nonlinear regressor to detect higher-order dependence."""
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def _crossfit_residual(V, Z, factory: Callable, cv: int,
                       rng: np.random.Generator) -> np.ndarray:
    from sklearn.model_selection import KFold
    V = _col(V).ravel()
    Z = _col(Z)
    cv_eff = max(2, min(cv, len(V) // 2))
    pred = np.zeros(len(V))
    for tr, te in KFold(n_splits=cv_eff, shuffle=True, random_state=0).split(Z):
        reg = factory()
        reg.fit(Z[tr], V[tr])
        pred[te] = reg.predict(Z[te])
    return V - pred


def zero_flow_statistic(A, B, *, regressor_factory: Callable = _default_velocity,
                        cv: int = 2, rng: Optional[np.random.Generator] = None) -> float:
    """Zero-flow statistic between samples A and B: mean squared magnitude of the
    cross-fitted velocity E[x1−x0 | x_0.5] under independent coupling. ≈0 when A
    and B share a distribution; grows with any distributional gap (incl. equal
    marginals but different covariance)."""
    from sklearn.model_selection import KFold
    rng = rng or np.random.default_rng(0)
    A, B = _col(A), _col(B)
    n = min(len(A), len(B))
    a = A[rng.permutation(len(A))[:n]]      # independent coupling
    b = B[rng.permutation(len(B))[:n]]
    pooled = np.vstack([a, b])
    mu, sd = pooled.mean(0), pooled.std(0) + 1e-8
    a, b = (a - mu) / sd, (b - mu) / sd
    x_half = 0.5 * (a + b)
    v = b - a
    cv_eff = max(2, min(cv, n // 2))
    pred = np.zeros_like(v)
    for tr, te in KFold(n_splits=cv_eff, shuffle=True, random_state=0).split(x_half):
        reg = regressor_factory()
        reg.fit(x_half[tr], v[tr])
        pred[te] = reg.predict(x_half[te]).reshape(len(te), -1)
    return float(np.mean(np.sum(pred ** 2, axis=1)))


def zero_flow_ci_test(X, Y, Z, *, n_perm: int = 100, alpha: float = 0.05,
                      cv: int = 2, min_n: int = 50,
                      nuisance_factory: Callable = _default_nuisance,
                      velocity_factory: Callable = _default_velocity,
                      rng: Optional[np.random.Generator] = None) -> CITestResult:
    """Test X ⫫ Y | Z. Small p ⇒ CI violated ("refutes"); else "supports";
    n < min_n ⇒ "underpowered"."""
    rng = rng or np.random.default_rng(0)
    n = len(_col(X))
    if n < min_n:
        return CITestResult("underpowered", "zero-flow-ci", n, float("nan"), float("nan"))

    rX = _crossfit_residual(X, Z, nuisance_factory, cv, rng)
    rY = _crossfit_residual(Y, Z, nuisance_factory, cv, rng)
    joint = np.column_stack([rX, rY])

    def product():
        return np.column_stack([rX, rY[rng.permutation(n)]])   # break rX–rY

    t_obs = zero_flow_statistic(product(), joint, regressor_factory=velocity_factory,
                                cv=cv, rng=rng)
    t_null: List[float] = [
        zero_flow_statistic(product(), product(), regressor_factory=velocity_factory,
                            cv=cv, rng=rng)
        for _ in range(n_perm)
    ]
    p = (1.0 + np.sum(np.asarray(t_null) >= t_obs)) / (1.0 + n_perm)
    verdict = "refutes" if p < alpha else "supports"
    return CITestResult(verdict, "zero-flow-ci", n, float(p), float(t_obs))


def markov_blanket(target: int, data: np.ndarray, *, alpha: float = 0.05,
                   n_perm: int = 60, rng: Optional[np.random.Generator] = None) -> list:
    """Recover the Markov blanket of column `target`: variables NOT conditionally
    independent of it given all the others. O(p) CI tests."""
    rng = rng or np.random.default_rng(0)
    data = np.asarray(data, dtype=float)
    p = data.shape[1]
    others = [j for j in range(p) if j != target]
    blanket = []
    for j in others:
        Z = data[:, [c for c in others if c != j]]
        res = zero_flow_ci_test(data[:, target], data[:, j], Z,
                                n_perm=n_perm, alpha=alpha, rng=rng)
        if res.verdict == "refutes":
            blanket.append(j)
    return blanket
