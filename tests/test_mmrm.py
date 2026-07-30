"""REML MMRM (#183 / exp43). Validates the fit against known truth before it is used to
demonstrate anything: it must recover beta and an UNSTRUCTURED Sigma on complete data,
keep recovering beta when visits are missing (the MAR-validity mechanism — dropout
shrinks the observed set rather than being imputed), and show the REML variance
correction relative to ML.
"""
import numpy as np
import pytest

from causal_bench.estimators.mmrm import (
    fit_mmrm, mmrm_design, treatment_effect_at, _chol_from_params, _neg2_reml, _gls_pass,
)

T = 3
TRUE_EFF = np.array([0.0, 0.4, 0.9])          # treatment effect per visit (grows)
TRUE_MEAN = np.array([1.0, 1.2, 1.5])         # visit-specific intercepts


def _sigma_truth():
    # unstructured-ish: unequal variances, decaying correlation
    sd = np.array([1.0, 1.3, 1.6])
    R = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.7], [0.4, 0.7, 1.0]])
    return np.outer(sd, sd) * R


def _simulate(n=600, seed=0, drop_frac=0.0):
    rng = np.random.default_rng(seed)
    Sigma = _sigma_truth()
    A = rng.integers(0, 2, n)
    E = rng.multivariate_normal(np.zeros(T), Sigma, size=n)
    Y = TRUE_MEAN[None, :] + A[:, None] * TRUE_EFF[None, :] + E
    subj, vis, yy, aa = [], [], [], []
    for i in range(n):
        last = T - 1
        if drop_frac > 0 and rng.random() < drop_frac:
            last = rng.integers(0, T - 1)          # monotone dropout after `last`
        for t in range(last + 1):
            subj.append(i); vis.append(t); yy.append(Y[i, t]); aa.append(A[i])
    return (np.array(yy), np.array(subj), np.array(vis), np.array(aa))


def test_chol_parameterisation_is_positive_definite():
    L = _chol_from_params(np.array([0.1, -0.3, 0.2, 0.5, -0.1, 0.0]), 3)
    S = L @ L.T
    assert np.all(np.linalg.eigvalsh(S) > 0)
    assert np.allclose(S, S.T)


def test_recovers_treatment_effects_on_complete_data():
    y, subj, vis, A = _simulate(n=800, seed=1)
    fit = fit_mmrm(y, subj, vis, mmrm_design(A, vis, T), n_visits=T)
    for t in range(T):
        est, se = treatment_effect_at(fit, t, T)
        assert abs(est - TRUE_EFF[t]) < 4 * se          # within MC error of truth
    assert fit["n_subjects"] == 800


def test_recovers_the_unstructured_covariance():
    y, subj, vis, A = _simulate(n=800, seed=2)
    fit = fit_mmrm(y, subj, vis, mmrm_design(A, vis, T), n_visits=T)
    S, S_true = fit["Sigma"], _sigma_truth()
    assert np.max(np.abs(S - S_true)) < 0.35            # unequal variances recovered
    # the off-diagonal structure is genuinely non-compound-symmetric
    assert S[0, 1] > S[0, 2]                            # decaying correlation preserved


def test_still_recovers_effects_under_missing_visits():
    # Dropout shrinks each subject's observed set; the likelihood uses what is there.
    # This is the MAR-validity mechanism and must not bias beta.
    y, subj, vis, A = _simulate(n=1200, seed=3, drop_frac=0.4)
    assert len(y) < 1200 * T                            # genuinely incomplete
    fit = fit_mmrm(y, subj, vis, mmrm_design(A, vis, T), n_visits=T)
    est, se = treatment_effect_at(fit, T - 1, T)
    assert abs(est - TRUE_EFF[-1]) < 4 * se


def test_reml_adjustment_is_actually_present():
    # REML differs from ML precisely by the log|X'V^-1X| term. Dropping it must change
    # the objective — otherwise we would be fitting ML and calling it REML.
    y, subj, vis, A = _simulate(n=200, seed=4)
    X = mmrm_design(A, vis, T)
    ys, Xs, obs = [], [], []
    for s in np.unique(subj):
        m = subj == s
        ys.append(y[m]); Xs.append(X[m]); obs.append(vis[m])
    params = np.array([0.0, 0.2, 0.0, 0.1, 0.1, 0.0])
    reml = _neg2_reml(params, ys, Xs, obs, T)
    Sigma = _chol_from_params(params, T) @ _chol_from_params(params, T).T
    XtVX, XtVy, ytVy, logdet = _gls_pass(Sigma, ys, Xs, obs)
    beta = np.linalg.solve(XtVX, XtVy)
    ml = logdet + (ytVy - beta @ XtVy)                  # ML objective: no info log-det
    assert reml > ml                                     # adjustment is positive here
    assert reml == pytest.approx(ml + np.linalg.slogdet(XtVX)[1])
