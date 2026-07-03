import numpy as np
from sklearn.linear_model import LogisticRegression

from causal_bench.dgp.point_treatment import draw_point_treatment, true_tau
from causal_bench.estimators.point import (
    fit_nuisances, oracle_nuisances, point_aipw, point_tmle)

W_COLS = ["W1", "W2", "W3", "W4"]


def _sim(seed, n=700, surface="smooth"):
    df = draw_point_treatment(n=n, surface=surface, seed=seed)
    return df[W_COLS].values, df["A"].values.astype(float), df["Y"].values.astype(float)


def test_oracle_aipw_and_tmle_recover_truth():
    tau0 = true_tau("smooth")
    pts_a, pts_t = [], []
    for seed in range(40):
        W, A, Y = _sim(seed)
        nf = oracle_nuisances(W, "smooth")
        pts_a.append(point_aipw(A, Y, nf).point)
        pts_t.append(point_tmle(A, Y, nf).point)
    assert abs(np.mean(pts_a) - tau0) < 0.02
    assert abs(np.mean(pts_t) - tau0) < 0.02


def test_oracle_ci_covers():
    tau0 = true_tau("smooth")
    cover = 0
    for seed in range(60):
        W, A, Y = _sim(seed)
        r = point_aipw(A, Y, oracle_nuisances(W, "smooth"))
        cover += (r.ci_lower <= tau0 <= r.ci_upper)
    assert cover / 60 > 0.85   # ~95% nominal, MC slack


def test_crossfit_toggle_changes_nuisances_but_not_shape():
    W, A, Y = _sim(0)
    g_l = LogisticRegression(max_iter=1000)
    q_l = LogisticRegression(max_iter=1000)
    nf_off = fit_nuisances(W, A, Y, g_l, q_l, crossfit=False, random_state=0)
    nf_on = fit_nuisances(W, A, Y, g_l, q_l, crossfit=True, random_state=0)
    assert nf_off.g.shape == nf_on.g.shape == (700,)
    assert not np.allclose(nf_off.g, nf_on.g)      # OOF differs from in-sample
    g_new, Q1_new, Q0_new = nf_on.predict(W[:5])
    assert g_new.shape == Q1_new.shape == Q0_new.shape == (5,)


def test_tmle_point_within_bounds():
    W, A, Y = _sim(1)
    g_l = LogisticRegression(max_iter=1000)
    q_l = LogisticRegression(max_iter=1000)
    nf = fit_nuisances(W, A, Y, g_l, q_l, crossfit=False, random_state=0)
    r = point_tmle(A, Y, nf)
    assert -1.0 <= r.point <= 1.0
    assert r.se > 0 and r.ci_lower < r.point < r.ci_upper
    assert abs(float(np.mean(r.ic))) < 1e-8    # IC centered
