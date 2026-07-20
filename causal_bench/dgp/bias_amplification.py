"""Bias-amplification DGP + the outcome-adaptive guard (#174, ENCIRCLE).

Conditioning a propensity/adjustment set on a near-INSTRUMENT — a variable that
predicts treatment but not outcome — *amplifies* the bias from any residual
UNMEASURED confounder rather than reducing it (Pearl 2010; Wooldridge 2009; Myers
et al. 2011). External controls always carry residual unmeasured confounding, and
a frozen-encoder embedding is instrument-rich, so "adjust for everything observed"
is actively harmful, not merely inefficient.

DGP (linear-Gaussian, where the amplification result is exact):
  U ~ N(0,1)   UNMEASURED confounder → both A and Y
  Z ~ N(0,1)   instrument (observed) → A only, NOT Y
  X ~ N(0,1)   measured confounder (observed) → both A and Y
  A = 1{ α_z·Z + α_u·U + α_x·X + logistic noise }
  Y = τ·A + β_u·U + β_x·X + ε           (Z absent from Y)

With τ = 0 the true ATE is exactly 0, so any nonzero estimate *is* the bias.
Adjusting for {X} leaves residual U-bias; adjusting for {X, Z} leaves the SAME
U-bias but amplified — including the instrument shrinks the residual variance of A,
inflating the confounding bias. The `outcome_adaptive_screen` guard keeps only
covariates associated with Y given A (Z fails: Z ⊥ Y | A, X), recovering {X}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BiasAmpConfig:
    alpha_z: float = 2.0     # instrument → treatment (STRONG — drives amplification)
    alpha_u: float = 1.0     # unmeasured confounder → treatment
    alpha_x: float = 1.0     # measured confounder → treatment
    beta_u: float = 2.0      # unmeasured confounder → outcome
    beta_x: float = 1.0      # measured confounder → outcome
    tau: float = 0.0         # true ATE (0 → estimate == bias, the clean null)
    sigma_y: float = 1.0


def draw_bias_amplification(n: int, seed: int, config: BiasAmpConfig = BiasAmpConfig()) -> pd.DataFrame:
    """Observed columns Z, X, A, Y. U is the unmeasured confounder — generated
    but NEVER returned (that is what makes it unmeasured)."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(n)               # UNMEASURED
    Z = rng.standard_normal(n)               # instrument (observed)
    X = rng.standard_normal(n)               # measured confounder (observed)
    logit_a = config.alpha_z * Z + config.alpha_u * U + config.alpha_x * X
    A = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_a))).astype(float)
    Y = (config.tau * A + config.beta_u * U + config.beta_x * X
         + config.sigma_y * rng.standard_normal(n))
    return pd.DataFrame({"Z": Z, "X": X, "A": A, "Y": Y})


def true_tau(config: BiasAmpConfig = BiasAmpConfig()) -> float:
    return config.tau


def regression_adjustment_ate(df: pd.DataFrame, adjustment_cols) -> float:
    """ATE = the OLS coefficient on A in `Y ~ A + adjustment_cols`. The adjustment
    SET is exactly `adjustment_cols` — the knob whose (mis)choice this DGP probes."""
    n = len(df)
    cols = [df["A"].to_numpy()] + [df[c].to_numpy() for c in adjustment_cols]
    Xmat = np.column_stack([np.ones(n), *cols])
    beta, *_ = np.linalg.lstsq(Xmat, df["Y"].to_numpy(), rcond=None)
    return float(beta[1])                    # coefficient on A


def outcome_adaptive_screen(df: pd.DataFrame, covariates, *, t_thresh: float = 1.96):
    """The guard: keep covariate `c` only if it is associated with Y in the
    *covariate–outcome* model `Y ~ covariates` (|t| on its coefficient >
    `t_thresh`). A pure instrument is dropped; a confounder / outcome-predictor is
    kept. Returns the bias-amplification-safe adjustment set.

    CRUCIAL — do NOT condition on the treatment A in this screen. A is a common
    effect of the instrument and the unmeasured confounder (Z → A ← U), so
    conditioning on A opens a **collider** path Z→A←U→Y that manufactures a
    spurious Z–Y association and the screen would (wrongly) keep the instrument.
    Screening on the treatment-free outcome model avoids that trap."""
    cols = list(covariates)
    n = len(df)
    Xmat = np.column_stack([np.ones(n), *[df[c].to_numpy() for c in cols]])  # NO A
    y = df["Y"].to_numpy()
    beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    resid = y - Xmat @ beta
    dof = max(n - Xmat.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(Xmat.T @ Xmat)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    return [c for i, c in enumerate(cols) if abs(tvals[1 + i]) > t_thresh]  # cols start at idx 1
