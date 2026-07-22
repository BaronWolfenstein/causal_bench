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


def regression_calibrate_hetero(w_obs, Z, sigma_x, *, ridge: float = 1e-9):
    """Regression calibration with **heteroskedastic** classical error.

    `regression_calibrate` assumes a single scalar `σ_x`, so the calibration posterior
    variance is the SAME for every unit. That matters more than it looks: a correction
    that is uniform across units is invisible to TMLE targeting, whose Newton step is
    invariant to rescaling the clever covariate (see #182). Only when `σ_x` VARIES across
    units — different sites, devices, or a reliability sub-study with unequal replicate
    counts — does the posterior become genuinely unit-specific.

    Model: `w_obs_i = w_true_i + ε_i`, `ε_i ~ N(0, σ_x,i²)`, `ε ⟂ Z`, and
    `w_true | Z ~ N(μ(Z), σ²_{w|z})`. Then the posterior is Gaussian with a per-unit
    shrinkage factor toward the regression line:

        λ_i   = σ²_{w|z} / (σ²_{w|z} + σ_x,i²)
        m_i   = μ̂(Z_i) + λ_i · (w_obs_i − μ̂(Z_i))
        τ²_i  = λ_i · σ_x,i²

    `μ̂(Z)` comes from regressing `w_obs` on `Z` (unbiased for `μ(Z)` since the error has
    mean zero), and `σ²_{w|z}` from that fit's residual variance minus the average error
    variance. Precisely-measured units shrink little (λ→1, τ²→0); poorly-measured units
    shrink hard and keep a large posterior variance.

    Returns ``(m, tau2)`` — both length-n.
    """
    w_obs = np.asarray(w_obs, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    s2 = np.asarray(sigma_x, dtype=float) ** 2
    if s2.ndim == 0:
        s2 = np.full(w_obs.shape, float(s2))

    D = np.column_stack([np.ones(len(w_obs)), Z])
    coef = np.linalg.solve(D.T @ D + ridge * np.eye(D.shape[1]), D.T @ w_obs)
    mu_z = D @ coef
    resid_var = float(np.var(w_obs - mu_z, ddof=D.shape[1]))
    var_w_given_z = max(resid_var - float(np.mean(s2)), 1e-6)   # peel off error variance

    lam = var_w_given_z / (var_w_given_z + s2)                  # per-unit shrinkage
    m = mu_z + lam * (w_obs - mu_z)
    tau2 = lam * s2
    return m, tau2
