"""Point-treatment binary-outcome DGP for exp33 (Donsker learner benchmark).

No censoring machinery: the estimand is the plain ATE, so the empirical-
process and remainder terms of AIPW/TMLE are directly computable against
the exposed truth (`true_g`, `true_Q`, `true_tau`).

Two nuisance-surface variants:
- "jumpy": threshold gate on W1 (LVEDD-style) in BOTH g0 and Q0 — cadlag
  with genuine jumps: inside LTB/HAL's function class, outside HAR's
  square-integrable-derivative condition.
- "smooth": the same structural strength via tanh, inside every class.

Positivity is healthy by construction (g0 in [0.1, 0.9]); positivity
stress is exp2's job.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit

SURFACES = ("jumpy", "smooth")
GATE = 0.6          # threshold on W1
_TAU_MC_N = 2_000_000
_TAU_MC_SEED = 20260702

# Correlated covariates: W ~ N(0, S), unit variances, mild correlation.
_CHOL = np.linalg.cholesky(
    np.array([[1.0, 0.3, 0.2, 0.0],
              [0.3, 1.0, 0.3, 0.1],
              [0.2, 0.3, 1.0, 0.2],
              [0.0, 0.1, 0.2, 1.0]]))


def _draw_W(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, 4)) @ _CHOL.T


def _gate_term(W: np.ndarray, surface: str) -> np.ndarray:
    """The W1 feature: hard indicator (jumpy) or tanh ramp (smooth)."""
    if surface == "jumpy":
        return (W[:, 0] >= GATE).astype(float)
    # tanh(2x) spans ~[-1,1]; rescale to [0,1] so both variants share range
    return 0.5 * (1.0 + np.tanh(2.0 * (W[:, 0] - GATE)))


def true_g(W: np.ndarray, surface: str) -> np.ndarray:
    """P(A=1 | W), bounded in [0.1, 0.9] by construction."""
    s = _gate_term(W, surface)
    lin = -0.3 + 0.7 * W[:, 1] - 0.5 * W[:, 2] + 1.4 * s
    return 0.1 + 0.8 * expit(lin)


def true_Q(a: int, W: np.ndarray, surface: str) -> np.ndarray:
    """E[Y | A=a, W]."""
    s = _gate_term(W, surface)
    lin = (-0.8 + 0.6 * W[:, 1] + 0.4 * W[:, 2] * W[:, 3]
           + 1.1 * s + a * (-0.9 + 0.8 * s - 0.3 * W[:, 3]))
    return expit(lin)


@lru_cache(maxsize=None)
def true_tau(surface: str) -> float:
    """ATE by Monte Carlo integration over the W distribution (cached)."""
    rng = np.random.default_rng(_TAU_MC_SEED)
    W = _draw_W(_TAU_MC_N, rng)
    return float(np.mean(true_Q(1, W, surface) - true_Q(0, W, surface)))


def draw_point_treatment(n: int, surface: str, seed: int) -> pd.DataFrame:
    """One simulated trial: columns W1..W4, A, Y."""
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
    rng = np.random.default_rng(seed)
    W = _draw_W(n, rng)
    A = rng.binomial(1, true_g(W, surface))
    pY = np.where(A == 1, true_Q(1, W, surface), true_Q(0, W, surface))
    Y = rng.binomial(1, pY)
    return pd.DataFrame({
        "W1": W[:, 0], "W2": W[:, 1], "W3": W[:, 2], "W4": W[:, 3],
        "A": A.astype(int), "Y": Y.astype(int),
    })
