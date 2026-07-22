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

    # Initialise from the empirical covariance of COMPLETE cases (falling back to a
    # scaled identity). Starting at a diagonal wastes most of the optimiser's work
    # rediscovering the correlation structure that is sitting in the data.
    full = [k for k, o in enumerate(obs) if len(o) == T]
    if len(full) > T + 1:
        Yc = np.stack([ys[k] for k in full])
        S0 = np.cov(Yc, rowvar=False) + 1e-6 * np.eye(T)
    else:
        S0 = max(float(np.var(y, ddof=1)), 1e-6) * np.eye(T)
    try:
        L0 = np.linalg.cholesky(S0)
    except np.linalg.LinAlgError:
        L0 = np.sqrt(max(float(np.var(y, ddof=1)), 1e-6)) * np.eye(T)
    p0 = L0[np.tril_indices(T)].copy()
    d = np.cumsum(np.arange(1, T + 1)) - 1                      # diagonal positions
    p0[d] = np.log(np.clip(np.diag(L0), 1e-8, None))            # log-Cholesky diagonal

    # L-BFGS-B (quasi-Newton, finite-difference gradients) rather than Nelder-Mead:
    # the objective is smooth in the log-Cholesky parameters, and simplex methods
    # degrade badly as T grows (T(T+1)/2 parameters). Nelder-Mead is kept as a
    # fallback for the rare non-convergence.
    res = minimize(_neg2_reml, p0, args=(ys, Xs, obs, T), method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 200 * len(p0)})
    if not res.success:
        res = minimize(_neg2_reml, res.x, args=(ys, Xs, obs, T), method="Nelder-Mead",
                       options={"maxiter": maxiter * len(p0), "fatol": 1e-6,
                                "xatol": 1e-5})
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


# ── Kenward-Roger small-sample adjustment ────────────────────────────────────
# With an unstructured Sigma the naive variance (X'V^-1X)^-1 ignores that Sigma itself is
# estimated, so it is biased DOWN and intervals under-cover in small samples. Kenward-Roger
# both inflates the variance and supplies a Satterthwaite-type df for it.
#
# Everything below is written in the DIRECT-ENTRY parameterisation theta = vech(Sigma),
# where Sigma is LINEAR in theta. Two consequences make this far cheaper than it looks:
# dSigma/dtheta_k is a single-entry (symmetrised) matrix, and d2Sigma/dtheta_j dtheta_k = 0,
# so KR's second-derivative term R_jk VANISHES. No Jacobian from the optimiser's
# log-Cholesky parameterisation is needed either: the information depends only on the
# fitted Sigma and the design, not on how the fit was parameterised.
def _dsigma_basis(T: int):
    """d Sigma / d theta_k for theta = vech(Sigma): symmetrised single-entry matrices."""
    out = []
    for a in range(T):
        for b in range(a + 1):
            E = np.zeros((T, T))
            E[a, b] = 1.0
            E[b, a] = 1.0                      # symmetric; equals 1 on the diagonal
            out.append(E)
    return out


def kr_adjust(Sigma, Xs, obs, contrast):
    """Kenward-Roger adjusted variance and denominator df for a scalar contrast.

    Returns ``{var_naive, var_kr, df, se_naive, se_kr}``. ``contrast`` is the vector l
    with estimand l'beta.
    """
    T = Sigma.shape[0]
    dS = _dsigma_basis(T)
    q, p = len(dS), Xs[0].shape[1]
    l = np.asarray(contrast, float)

    # per-subject accumulators
    M = np.zeros((p, p))
    B = [np.zeros((p, p)) for _ in range(q)]              # KR's P_j = X'V^-1 Vdot_j V^-1 X
    C = [[np.zeros((p, p)) for _ in range(q)] for _ in range(q)]   # KR's Q_jk
    S = np.zeros((q, q))                                  # tr(A Vdot_j A Vdot_k)
    for X_i, o in zip(Xs, obs):
        A = np.linalg.inv(Sigma[np.ix_(o, o)])
        AX = A @ X_i
        M += X_i.T @ AX
        AD = [A @ dS[k][np.ix_(o, o)] for k in range(q)]  # A Vdot_k
        for j in range(q):
            B[j] += AX.T @ dS[j][np.ix_(o, o)] @ AX
            for k in range(q):
                C[j][k] += AX.T @ dS[j][np.ix_(o, o)] @ A @ dS[k][np.ix_(o, o)] @ AX
                S[j, k] += float(np.trace(AD[j] @ AD[k]))
    Phi = np.linalg.inv(M)

    # REML expected information: I_jk = 1/2 tr(P Vdot_j P Vdot_k), P = V^-1 - V^-1 X Phi X'V^-1
    I = np.empty((q, q))
    for j in range(q):
        for k in range(q):
            I[j, k] = 0.5 * (S[j, k] - np.trace(Phi @ C[j][k]) - np.trace(Phi @ C[k][j])
                             + np.trace(Phi @ B[j] @ Phi @ B[k]))
    # Guard: with q = T(T+1)/2 covariance parameters and few subjects the information
    # is near-singular, and the KR adjustment degenerates (observed at T=6/n=30: a
    # 1.84x variance inflation and df = inf). Flag rather than silently return it.
    q_over_n = q / max(len(Xs), 1)
    cond = float(np.linalg.cond(I)) if np.all(np.isfinite(I)) else float("inf")
    W = np.linalg.pinv(I)                                 # asymptotic cov of theta-hat

    # KR adjusted variance (R_jk = 0 because Sigma is linear in theta)
    mid = np.zeros((p, p))
    for j in range(q):
        for k in range(q):
            mid += W[j, k] * (C[j][k] - B[j] @ Phi @ B[k])
    Phi_A = Phi + 2.0 * Phi @ mid @ Phi

    var_naive = float(l @ Phi @ l)
    var_kr = float(l @ Phi_A @ l)
    # Satterthwaite-type df on the ADJUSTED variance: g_k = d(l'Phi l)/d theta_k
    g = np.array([float(l @ Phi @ B[k] @ Phi @ l) for k in range(q)])
    denom = float(g @ W @ g)
    df = float(2.0 * var_kr ** 2 / denom) if denom > 1e-18 else float("inf")
    degenerate = (not np.isfinite(df)) or cond > 1e10 or q_over_n > 0.4
    return {"var_naive": var_naive, "var_kr": max(var_kr, 1e-12), "df": max(df, 1.0),
            "se_naive": float(np.sqrt(max(var_naive, 0.0))),
            "se_kr": float(np.sqrt(max(var_kr, 1e-12))),
            "kr_degenerate": bool(degenerate), "info_cond": cond,
            "q_over_n": float(q_over_n)}


def fit_mmrm_kr(y, subject, visit, X, *, contrast, n_visits=None):
    """`fit_mmrm` plus the Kenward-Roger adjustment for one scalar contrast."""
    fit = fit_mmrm(y, subject, visit, X, n_visits=n_visits)
    T = fit["Sigma"].shape[0]
    subject = np.asarray(subject); visit = np.asarray(visit, int); X = np.asarray(X, float)
    Xs, obs = [], []
    for s in np.unique(subject):
        m = subject == s
        order = np.argsort(visit[m])
        Xs.append(X[m][order]); obs.append(visit[m][order])
    kr = kr_adjust(fit["Sigma"], Xs, obs, contrast)
    fit.update(kr)
    fit["estimate"] = float(np.asarray(contrast, float) @ fit["beta"])
    return fit
