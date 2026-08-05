"""Probabilistic quantitative bias analysis (QBA) for unmeasured confounding (exp49).

Fox / MacLehose / Lash (IJE 2023;52:1624) probabilistic QBA, closed-form v1 (the
n-independent linear/omitted-variable form). A measured covariate set X, an
UNMEASURED binary confounder U (drives treatment A and outcome Y), continuous
linear outcome so tau is the true ATE. The naive estimate (U omitted) is biased;
QBA samples the bias parameters from priors and reports systematic- and total-error
intervals.

Closed form: for a linear outcome the omitted-variable bias is exact,
    naive_beta_A = oracle_beta_A + beta_U * gamma,
where beta_U is the U->Y coefficient and gamma is the A-coefficient of (U ~ A + X).
So adjusted = naive - beta_U*gamma; sampling (beta_U, gamma) from priors gives the
systematic-error distribution, and subtracting a random-error draw gives total error.

The EXPERIMENT (beyond a point bias-adjustment): calibration + robustness to
bias-parameter MISSPECIFICATION -- does the total-error interval cover the truth
under correct priors, and how does coverage degrade as the priors are wrong?
Self-validating: (i) oracle recovers tau; (ii) correct-prior total interval ~ nominal
coverage; (iii) misspecified priors undercover.
"""
from __future__ import annotations

import numpy as np


def simulate(n, conf=1.2, tau=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    U = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.6 * X[:, 0])))).astype(float)  # unmeasured, X-correlated
    e = 1.0 / (1.0 + np.exp(-(-0.3 + 0.4 * X[:, 0] + conf * U + 0.3 * X[:, 1])))
    A = (rng.random(n) < e).astype(float)
    Y = tau * A + 0.8 * X[:, 0] + 0.5 * X[:, 1] + 1.5 * U + rng.normal(size=n)
    return dict(X=X, U=U, A=A, Y=Y, tau=float(tau))


def _ols(D, y):
    XtXi = np.linalg.inv(D.T @ D)
    b = XtXi @ D.T @ y
    resid = y - D @ b
    s2 = (resid @ resid) / (len(y) - D.shape[1])
    return b, np.sqrt(np.diag(XtXi) * s2)


def naive_oracle(sim):
    X, A, Y, U = sim["X"], sim["A"], sim["Y"], sim["U"]
    n = len(A); one = np.ones((n, 1))
    bn, sen = _ols(np.hstack([one, A[:, None], X]), Y)          # Y ~ A + X   (U omitted)
    bo, _ = _ols(np.hstack([one, A[:, None], X, U[:, None]]), Y)  # Y ~ A + X + U (oracle)
    gamma = _ols(np.hstack([one, A[:, None], X]), U)[0][1]        # A-coef of U ~ A + X
    return dict(naive=float(bn[1]), se_naive=float(sen[1]),
                oracle=float(bo[1]), beta_U=float(bo[-1]), gamma=float(gamma))


def qba(naive, se_naive, beta_U_prior, gamma_prior, S=100_000, seed=0):
    """beta_U_prior = (min, mode, max) trapezoidal-ish (triangular); gamma_prior = (mean, sd)."""
    rng = np.random.default_rng(seed)
    bU = rng.triangular(*beta_U_prior, S)
    g = rng.normal(gamma_prior[0], gamma_prior[1], S)
    syst = naive - bU * g
    total = syst - rng.normal(0, se_naive, S)
    return syst, total


def _priors(no, mode):
    """Bias-parameter priors centered correctly, or misspecified."""
    g = (no["gamma"], 0.03)
    if mode == "correct":
        return (no["beta_U"] - 0.5, no["beta_U"], no["beta_U"] + 0.5), g
    if mode == "misspec_null":      # assume U is harmless
        return (-0.15, 0.0, 0.15), g
    if mode == "misspec_half":      # underestimate U's effect by half
        c = 0.5 * no["beta_U"]
        return (c - 0.4, c, c + 0.4), g
    raise ValueError(mode)


def calibrate(n=1500, conf=1.2, tau=1.0, n_reps=200, mode="correct", seed=0):
    cov_s, cov_t, nb, adj = [], [], [], []
    for r in range(n_reps):
        no = naive_oracle(simulate(n, conf=conf, tau=tau, seed=seed + r))
        bU_p, g_p = _priors(no, mode)
        syst, total = qba(no["naive"], no["se_naive"], bU_p, g_p, seed=r)
        ls, hs = np.percentile(syst, [2.5, 97.5]); cov_s.append(ls <= tau <= hs)
        lt, ht = np.percentile(total, [2.5, 97.5]); cov_t.append(lt <= tau <= ht)
        nb.append(no["naive"] - tau); adj.append(float(np.median(syst)) - tau)
    return {"mode": mode, "coverage_syst": float(np.mean(cov_s)),
            "coverage_total": float(np.mean(cov_t)), "naive_bias": float(np.mean(nb)),
            "adjusted_bias": float(np.mean(adj))}


def report_rows(n=1500, conf=1.2, n_reps=200, seed=0):
    return [calibrate(n=n, conf=conf, n_reps=n_reps, mode=m, seed=seed)
            for m in ("correct", "misspec_half", "misspec_null")]
