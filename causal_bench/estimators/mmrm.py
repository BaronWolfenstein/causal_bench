"""MMRM — mixed model for repeated measures, fitted by REML (#183 / exp43).

The regulatory-standard analysis for continuous longitudinal endpoints: visit treated as
categorical, an **unstructured** within-subject covariance, REML estimation, and the
treatment x visit interaction giving the per-visit effect. It uses every observed visit
through the likelihood rather than imputing, which is what makes it valid under MAR and
is why it displaced LOCF.

Implemented directly rather than via `statsmodels.MixedLM`, which fits *random effects*
(compound symmetry at best) and has no native unstructured residual covariance across
visits — a random-intercept model is not an MMRM and should not be labelled one.

Model. For subject i observed at visit set o_i,

    y_i = X_i beta + e_i,     e_i ~ N(0, Sigma[o_i, o_i])

with Sigma a full T x T unstructured covariance shared across subjects. Dropout enters
only by shrinking o_i — no imputation — which is precisely the MAR-validity mechanism.

REML. Sigma is parameterised by a log-Cholesky factor (guaranteeing positive
definiteness), and the restricted log-likelihood profiles out beta:

    -2 l_R(Sigma) = sum_i [ log|Sigma_i| + r_i' Sigma_i^-1 r_i ] + log| sum_i X_i' Sigma_i^-1 X_i |

The final term is the REML adjustment (the GLS information log-det) that ML omits and
that removes the downward bias in the variance estimate. The residual quadratic form is
accumulated in one pass via  sum_i r_i' Sigma_i^-1 r_i = y'V^-1 y - beta' X'V^-1 y,
which holds because beta is the GLS solution.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _chol_from_params(params: np.ndarray, T: int) -> np.ndarray:
    """Lower-triangular factor with positive diagonal from T(T+1)/2 unconstrained params."""
    L = np.zeros((T, T))
    L[np.tril_indices(T)] = params
    d = np.diag_indices(T)
    L[d] = np.exp(np.clip(L[d], -12.0, 12.0))
    return L


def _gls_pass(Sigma, ys, Xs, obs):
    """One pass: accumulate X'V^-1X, X'V^-1y, y'V^-1y and sum log|Sigma_i|."""
    p = Xs[0].shape[1]
    XtVX = np.zeros((p, p))
    XtVy = np.zeros(p)
    ytVy = 0.0
    logdet = 0.0
    for y_i, X_i, o in zip(ys, Xs, obs):
        S = Sigma[np.ix_(o, o)]
        c = cho_factor(S, lower=True)
        Si_X = cho_solve(c, X_i)
        Si_y = cho_solve(c, y_i)
        XtVX += X_i.T @ Si_X
        XtVy += X_i.T @ Si_y
        ytVy += float(y_i @ Si_y)
        logdet += 2.0 * float(np.sum(np.log(np.diag(c[0]))))
    return XtVX, XtVy, ytVy, logdet


def _neg2_reml(params, ys, Xs, obs, T):
    L = _chol_from_params(params, T)
    Sigma = L @ L.T + 1e-10 * np.eye(T)
    try:
        XtVX, XtVy, ytVy, logdet = _gls_pass(Sigma, ys, Xs, obs)
        beta = np.linalg.solve(XtVX, XtVy)
        quad = ytVy - float(beta @ XtVy)          # = sum_i r_i' Sigma_i^-1 r_i
        sign, ld_info = np.linalg.slogdet(XtVX)   # REML adjustment (ML omits this)
        if sign <= 0 or not np.isfinite(quad):
            return 1e12
        return logdet + quad + ld_info
    except Exception:
        return 1e12


def fit_mmrm(y, subject, visit, X, *, n_visits=None, maxiter: int = 500):
    """Fit an MMRM by REML.

    Parameters
    ----------
    y, subject, visit : 1-D arrays, one row per observed subject-visit. `visit` must be
        integer-coded 0..T-1. Missing visits are simply absent (monotone or intermittent).
    X : (n_obs, p) design matrix — build treatment, visit dummies and their interaction
        with `mmrm_design`.

    Returns ``{beta, vcov, se, Sigma, neg2_reml, converged, n_subjects}``. `vcov` is the
    GLS information inverse ``(sum_i X_i' Sigma_i^-1 X_i)^-1``.
    """
    y = np.asarray(y, float)
    subject = np.asarray(subject)
    visit = np.asarray(visit, int)
    X = np.asarray(X, float)
    T = int(n_visits if n_visits is not None else visit.max() + 1)

    ys, Xs, obs = [], [], []
    for s in np.unique(subject):
        m = subject == s
        order = np.argsort(visit[m])
        ys.append(y[m][order])
        Xs.append(X[m][order])
        obs.append(visit[m][order])

    # start from an independence model scaled to the marginal variance
    v0 = max(float(np.var(y, ddof=1)), 1e-6)
    p0 = np.zeros(T * (T + 1) // 2)
    p0[np.cumsum(np.arange(1, T + 1)) - 1] = 0.5 * np.log(v0)   # diagonal entries

    res = minimize(_neg2_reml, p0, args=(ys, Xs, obs, T), method="Nelder-Mead",
                   options={"maxiter": maxiter * len(p0), "fatol": 1e-6, "xatol": 1e-5})
    L = _chol_from_params(res.x, T)
    Sigma = L @ L.T
    XtVX, XtVy, _, _ = _gls_pass(Sigma, ys, Xs, obs)
    vcov = np.linalg.inv(XtVX)
    beta = vcov @ XtVy
    return {"beta": beta, "vcov": vcov, "se": np.sqrt(np.diag(vcov)), "Sigma": Sigma,
            "neg2_reml": float(res.fun), "converged": bool(res.success),
            "n_subjects": len(ys)}


def mmrm_design(A, visit, T: int):
    """Design matrix for the standard MMRM mean model: visit-specific intercepts plus a
    treatment x visit interaction, so ``beta[T + t]`` is the treatment effect AT visit t
    (no proportionality assumed across visits)."""
    A = np.asarray(A, float)
    visit = np.asarray(visit, int)
    n = len(visit)
    X = np.zeros((n, 2 * T))
    X[np.arange(n), visit] = 1.0                      # visit-specific intercepts
    X[np.arange(n), T + visit] = A                    # treatment effect per visit
    return X


def treatment_effect_at(fit: dict, t: int, T: int) -> tuple:
    """(estimate, se) of the treatment effect at visit ``t`` from a `mmrm_design` fit."""
    j = T + t
    return float(fit["beta"][j]), float(fit["se"][j])
