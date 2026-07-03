"""Shared measurement-error corrections for covariate error-in-variables.

Regression calibration for a covariate measured with classical additive error,
`w_obs = w_true + ε`, `ε ~ N(0, σ_x²)`, `ε ⟂ Z` for error-free covariates `Z`.
Used by the covariate-measurement-error sensitivity experiments (exp31 on an OLS
estimand, exp32 on the TMLE clever covariate) so the correction is defined once.
"""
from __future__ import annotations

import numpy as np


def regression_calibrate(w_obs, Z, sigma_x: float, *,
                         return_residual_variance: bool = False):
    """E[w_true | w_obs, Z] under classical additive error (Carroll et al.).

    Solves the normal equations for the linear predictor of the latent `w_true`
    from `(w_obs, Z)`, using observed moments plus the reliability-study `σ_x²`:
    `var(w_true) = var(w_obs) − σ_x²`; `cov(w_true, w_obs) = var(w_true)`;
    `cov(w_true, Z) = cov(w_obs, Z)` (error ⟂ Z).

    Parameters
    ----------
    w_obs : (n,) array — the error-laden covariate.
    Z : (n,) or (n, k) array — error-free conditioning covariates.
    sigma_x : classical additive-error SD (from a reliability study).
    return_residual_variance : also return `τ²_resid = Var(w_true | w_obs, Z)`
        — the conditional variance RC cannot recover, the O(σ_x²) driver of the
        residual bias.

    Returns ``w_hat`` (the calibrated covariate), or ``(w_hat, tau2_resid)``.
    """
    w_obs = np.asarray(w_obs, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    P = np.column_stack([w_obs, Z])
    mu = P.mean(0)
    Sigma = np.cov(P, rowvar=False, ddof=0)
    var_w_true = max(w_obs.var() - sigma_x**2, 1e-6)      # floored
    c = np.empty(P.shape[1])
    c[0] = var_w_true                                     # cov(w_true, w_obs)
    c[1:] = Sigma[0, 1:]                                  # cov(w_true, Z)=cov(w_obs, Z)
    coef = np.linalg.solve(Sigma + 1e-9 * np.eye(len(c)), c)
    w_hat = mu[0] + (P - mu) @ coef                       # mean(w_true)=mean(w_obs)
    if return_residual_variance:
        return w_hat, float(var_w_true - c @ coef)        # var − cᵀΣ⁻¹c
    return w_hat
