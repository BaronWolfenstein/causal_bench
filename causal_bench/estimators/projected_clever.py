"""Projected clever covariate — EIF projection onto the observed-data space (#182).

exp32 (#66) corrects a mismeasured confounder by **regression calibration**: it plugs
``m = E[W_true | W_obs, Z]`` into the data so the propensity ``g`` and hence the clever
covariate ``H(A,W) = A/g(W) − (1−A)/(1−g(W))`` are rebuilt on a de-attenuated confounder.
exp32's own docstring concedes this is a *first-order* correction that does not
blanket-preserve the TMLE second-order remainder.

The reason is Jensen. ``H`` is **nonlinear** in ``W``, so

    E[ H(A, W_true) | W_obs, A ]   ≠   H(A, E[W_true | W_obs, A])
            the projection                     the plug-in

The left-hand side — the *projected* clever covariate — is the actual estimating function
on the observed data; the right-hand side approximates it. Under a logistic
``g(w) = sigma(a + b·w)`` both ``1/g`` and ``1/(1−g)`` are convex in ``w``, so the plug-in
**systematically understates** the clever covariate's magnitude, by a gap that grows with
the measurement error.

This is not a bespoke construction: it is the operation **A-TMLE** formalises — project
the efficient influence function onto a sub-tangent-space, target within the projected
model, and retain asymptotic linearity. A-TMLE projects onto the tangent space of a
data-adaptively selected working model; here the subspace is the *observed-data* tangent
space, integrating out the unobserved true covariate. See ``reference_atmle_subgroup_qiu``
(arXiv:2605.15483) for the version already assessed in this project; its results on valid
inference after projection are the relevant theory.

SCOPE. This addresses **estimation given an adjustment set**. It does NOT repair
measurement bias in covariate *selection* — attenuation makes true confounders look weak
(false negatives) and can promote noise-driven ones, which no estimating function fixes.
Those are separate failure modes and must be composed, not conflated (#182).

The calibration posterior ``W_true | W_obs, Z ~ N(m, tau^2)`` is supplied by
``measurement_error.regression_calibrate(..., return_residual_variance=True)``.
"""
from __future__ import annotations

import numpy as np

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _clip(g, eps: float):
    return np.clip(np.asarray(g, float), eps, 1.0 - eps)


def plugin_clever_covariate(A, g_fn, w1_mean, *, eps: float = 1e-6):
    """exp32's arm: ``H(A, m)`` — the clever covariate evaluated AT the calibrated mean."""
    A = np.asarray(A)
    g = _clip(g_fn(np.asarray(w1_mean, float)), eps)
    return np.where(A == 1, 1.0 / g, -1.0 / (1.0 - g))


def projected_clever_covariate(A, g_fn, w1_mean, w1_sd, *, n_quad: int = 32,
                               eps: float = 1e-6):
    """``E[H(A, W_true) | W_obs, A]`` by Gauss-Hermite quadrature over the calibration
    posterior ``W_true ~ N(w1_mean, w1_sd**2)``.

    ``g_fn`` maps an array of candidate ``w1`` values to propensities, holding every other
    covariate fixed per row: given a grid of shape ``(n, n_quad)`` it must return
    propensities of the same shape (broadcasting a 1-D input of shape ``(n,)`` too).
    Because ``H`` depends on ``W`` only through ``g``, integrating ``g`` is sufficient."""
    A = np.asarray(A)
    m = np.asarray(w1_mean, float)
    sd = np.asarray(w1_sd, float)
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)   # ∫f(x)e^{-x²/2}dx
    grid = m[:, None] + sd[:, None] * nodes[None, :]              # (n, n_quad)
    g = _clip(g_fn(grid), eps)
    w = weights[None, :] / _SQRT_2PI                              # normalise to E_N(0,1)
    e_inv_g = (w / g).sum(axis=1)
    e_inv_1mg = (w / (1.0 - g)).sum(axis=1)
    return np.where(A == 1, e_inv_g, -e_inv_1mg)


def jensen_gap_report(*, sigma_x: float, n: int = 2000, slope: float = 0.8,
                      intercept: float = -0.3, seed: int = 0,
                      n_quad: int = 48) -> dict:
    """Quantify the plug-in's shortfall on a self-contained classical-error setup.

    A latent ``W1_true ~ N(0,1)`` drives treatment through ``g = sigma(a + b·W1_true)``;
    the analyst sees ``W1_obs = W1_true + N(0, sigma_x^2)``. With standard-normal prior and
    classical error the posterior is exactly ``N(m, tau^2)`` with
    ``m = W1_obs/(1+sigma_x^2)`` and ``tau^2 = sigma_x^2/(1+sigma_x^2)``.

    ``projected_error`` / ``plugin_error`` score each against a high-resolution Monte-Carlo
    evaluation of the same conditional expectation, so "which is closer to the truth" is
    measured, not asserted."""
    rng = np.random.default_rng(seed)
    w1_true = rng.normal(size=n)
    g_true = 1.0 / (1.0 + np.exp(-(intercept + slope * w1_true)))
    A = (rng.random(n) < g_true).astype(int)
    w1_obs = w1_true + sigma_x * rng.normal(size=n)

    denom = 1.0 + sigma_x ** 2
    m = w1_obs / denom
    tau = np.full(n, np.sqrt(sigma_x ** 2 / denom))

    def g_fn(w1):
        return 1.0 / (1.0 + np.exp(-(intercept + slope * np.asarray(w1, float))))

    proj = projected_clever_covariate(A, g_fn, m, tau, n_quad=n_quad)
    plug = plugin_clever_covariate(A, g_fn, m)

    draws = m[:, None] + tau[:, None] * rng.normal(size=(n, 4000))
    gv = _clip(g_fn(draws), 1e-6)
    mc = np.where(A[:, None] == 1, 1.0 / gv, -1.0 / (1.0 - gv)).mean(axis=1)

    gap = np.abs(proj) - np.abs(plug)
    return {
        "sigma_x": sigma_x,
        "mean_abs_gap": float(np.mean(gap)),
        "relative_gap": float(np.mean(gap) / np.mean(np.abs(plug))),
        "projected_error": float(np.mean(np.abs(proj - mc))),
        "plugin_error": float(np.mean(np.abs(plug - mc))),
        "mean_abs_plugin": float(np.mean(np.abs(plug))),
        "n": n,
    }
