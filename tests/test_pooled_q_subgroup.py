"""Tests for the pooled-Q subgroup estimator (#77 event-rate functional)."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from causal_bench.estimators.pooled_q_subgroup import PooledQSubgroupEstimator


def _make_survival_df(n=1200, thr=1.1, seed=0, informative_censoring=False):
    """Synthetic single-arm survival data with a small COVARIATE-DEFINED subgroup.

    Realistic ENCIRCLE regime (subgroups are covariate thresholds, e.g. LVEDD>75):
    subgroup 1 = the small upper-W1 tail `1{W1 > thr}`, and the event risk is strongly
    W1-driven — so a pooled Q(W) genuinely borrows the map into the sparse tail, which
    is where subgroup-only overfits. (An S ⊥ W random level shift is the *worst* case
    for pooled-Q and is deliberately NOT used here.) Events → T_obs in (0, horizon),
    Delta=1; non-events → admin-censored at horizon. Optional informative dropout
    exercises IPCW.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, 4))
    S = (W[:, 0] > thr).astype(int)                       # small covariate-defined subgroup
    horizon = 1.0
    lin = 1.6 * W[:, 0] - 0.8 * W[:, 1] + 0.6 * W[:, 2] - 0.5   # strong W1 signal
    p = expit(lin)
    Y = (rng.random(n) < p).astype(int)

    T_obs = np.where(Y == 1, rng.uniform(0.05, 0.95, n), horizon)
    Delta = Y.astype(float)

    if informative_censoring:
        # some W-dependent pre-horizon dropout among the non-events
        drop = (Y == 0) & (rng.random(n) < expit(0.5 * W[:, 0] - 1.5))
        T_obs = np.where(drop, rng.uniform(0.05, 0.95, n), T_obs)
        Delta = np.where(drop, 0.0, Delta)

    df = pd.DataFrame({
        "W1": W[:, 0], "W2": W[:, 1], "W3": W[:, 2], "W4": W[:, 3],
        "T_obs": T_obs, "Delta": Delta, "subgroup_label": S,
        "A": 1.0,  # single-arm
    })
    true_rate = {s: float(p[S == s].mean()) for s in (0, 1)}
    return df, true_rate, horizon


def test_returns_one_result_per_subgroup_with_valid_cis():
    df, _, horizon = _make_survival_df(seed=1)
    res = PooledQSubgroupEstimator().estimate(df, horizon=horizon)
    assert len(res) == 2
    for r in res:
        assert np.isfinite(r.point_estimate) and np.isfinite(r.standard_error)
        assert r.standard_error > 0
        assert r.ci_lower <= r.point_estimate <= r.ci_upper
        assert 0.0 <= r.point_estimate <= 1.0
        assert r.ic is not None and r.ic.shape[0] == len(df)


def test_recovers_true_subgroup_rate():
    """Both subgroups' point estimates land within ~2.5 SE of the true rate."""
    df, true_rate, horizon = _make_survival_df(n=2000, seed=2)
    res = {r.estimand: r for r in PooledQSubgroupEstimator().estimate(df, horizon=horizon)}
    for s in (0, 1):
        r = res[f"rate|S={s}"]
        assert abs(r.point_estimate - true_rate[s]) < 2.5 * r.standard_error + 0.02


def test_pooled_beats_subgroup_only_rmse_on_small_covariate_subgroup():
    """The exp34/#189 win: on a SMALL covariate-defined subgroup with a strong W→Y
    signal, pooling the outcome map lowers RMSE-to-truth vs fitting within the subgroup.
    (RMSE over seeds — the robust demonstration; for a binary outcome the per-run
    IC-SE alone is dominated by the irreducible binomial variance and does not separate.)
    """
    err_pooled, err_only = [], []
    for sd in range(40):
        df, tr, h = _make_survival_df(n=1000, thr=1.4, seed=sd)     # n_s ≈ 80
        if (df["subgroup_label"] == 1).sum() < 5:
            continue
        p = {r.estimand: r for r in PooledQSubgroupEstimator(pooled=True).estimate(df, horizon=h)}
        o = {r.estimand: r for r in PooledQSubgroupEstimator(pooled=False).estimate(df, horizon=h)}
        if "rate|S=1" not in p or "rate|S=1" not in o:
            continue
        err_pooled.append(p["rate|S=1"].point_estimate - tr[1])
        err_only.append(o["rate|S=1"].point_estimate - tr[1])
    rmse_pooled = np.sqrt(np.mean(np.square(err_pooled)))
    rmse_only = np.sqrt(np.mean(np.square(err_only)))
    assert rmse_pooled < rmse_only, f"pooled {rmse_pooled:.4f} !< only {rmse_only:.4f}"


def test_ipcw_path_runs_with_informative_censoring():
    df, _, horizon = _make_survival_df(seed=4, informative_censoring=True)
    res = PooledQSubgroupEstimator().estimate(df, horizon=horizon)
    assert len(res) == 2
    assert all(np.isfinite(r.point_estimate) and r.standard_error > 0 for r in res)


def test_missing_subgroup_column_raises():
    df, _, horizon = _make_survival_df(seed=5)
    with pytest.raises(ValueError):
        PooledQSubgroupEstimator(subgroup_col="nope").estimate(df, horizon=horizon)


def test_single_outcome_class_subgroup_does_not_crash():
    """A small covariate-defined subgroup can be entirely events; the within-subgroup
    (pooled=False) outcome fit must not crash (LogisticRegression raises on <2 classes) —
    the outcome regression is the constant class and psi_s recovers it (=1 here)."""
    df, _, horizon = _make_survival_df(n=1500, thr=1.1, seed=3)
    m = df["subgroup_label"] == 1
    df.loc[m, "T_obs"] = 0.5                       # force subgroup 1 to be ALL events
    df.loc[m, "Delta"] = 1.0
    for pooled in (True, False):
        res = {r.estimand: r for r in
               PooledQSubgroupEstimator(pooled=pooled).estimate(df, horizon=horizon)}
        assert "rate|S=1" in res
        r = res["rate|S=1"]
        assert np.isfinite(r.point_estimate) and np.isfinite(r.standard_error)
        assert r.point_estimate > 0.98            # all-events subgroup -> rate ~ 1


# --------------------------------------------------------------- RMST (#189)

def _make_rmst_df(n=3000, thr=1.0, seed=0, horizon=2.0, informative_censoring=False,
                  non_ph=False, truth_n=300_000):
    """Single-arm survival data with a small covariate-defined subgroup and a computable
    RMST truth = E[min(T, tau) | S=s] from a large clean reference draw. PH (exponential)
    by default; `non_ph` gives subgroup-dependent Weibull shapes (crossing hazards);
    `informative_censoring` drops subjects on a within-subgroup-varying covariate (W2)."""
    rng = np.random.default_rng(seed)

    def draw(m, rs):
        W = rs.normal(size=(m, 4))
        S = (W[:, 0] > thr).astype(int)
        lam = np.exp(-0.2 + 0.4 * W[:, 0] - 0.9 * W[:, 1] + 0.5 * W[:, 2] + 0.5 * S)
        if non_ph:
            shape = np.where(S == 1, 1.7, 0.8)              # crossing hazards
            T = (-np.log(rs.random(m))) ** (1.0 / shape) / lam
        else:
            T = rs.exponential(1.0 / lam)
        return W, S, T

    W, S, T = draw(n, rng)
    T_obs = np.minimum(T, horizon)
    Delta = (T <= horizon).astype(float)
    if informative_censoring:
        C = rng.exponential(1.0 / np.exp(-1.6 + 0.6 * W[:, 1]))   # ~20% dropout, varies within subgroup
        drop = C < T_obs
        T_obs = np.where(drop, C, T_obs)
        Delta = np.where(drop, 0.0, Delta)

    df = pd.DataFrame({"W1": W[:, 0], "W2": W[:, 1], "W3": W[:, 2], "W4": W[:, 3],
                       "T_obs": T_obs, "Delta": Delta, "subgroup_label": S, "A": 1.0})
    Wt, St, Tt = draw(truth_n, np.random.default_rng(seed + 7))
    mm = np.minimum(Tt, horizon)
    truth = {s: float(mm[St == s].mean()) for s in (0, 1)}
    return df, truth, horizon


def _km_rmst(T_obs, Delta, horizon):
    """Naive per-subgroup Kaplan-Meier RMST (no covariate adjustment, no IPCW)."""
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time
    kmf = KaplanMeierFitter().fit(T_obs, (Delta == 1))
    return float(restricted_mean_survival_time(kmf, t=horizon))


def test_rmst_returns_one_result_per_subgroup_with_valid_cis():
    df, _, h = _make_rmst_df(seed=1)
    res = PooledQSubgroupEstimator(n_grid=20).estimate(df, horizon=h, estimand="subgroup_rmst")
    assert len(res) == 2
    for r in res:
        assert r.estimand.startswith("rmst|S=")
        assert np.isfinite(r.point_estimate) and r.standard_error > 0
        assert 0.0 <= r.point_estimate <= h
        assert r.ci_lower <= r.point_estimate <= r.ci_upper
        assert r.ic is not None and r.ic.shape[0] == len(df)


def test_rmst_recovers_true_subgroup_rmst():
    """Both subgroups' RMST land within ~2.5 SE of the MC truth E[min(T,tau)|S=s]."""
    df, truth, h = _make_rmst_df(n=5000, seed=3, horizon=2.0)
    res = {r.estimand: r for r in
           PooledQSubgroupEstimator(n_grid=25).estimate(df, horizon=h, estimand="subgroup_rmst")}
    for s in (0, 1):
        r = res[f"rmst|S={s}"]
        assert abs(r.point_estimate - truth[s]) < 2.5 * r.standard_error + 0.02


def test_rmst_recovers_under_non_ph_crossing_hazards():
    """The discrete-time hazard's interval baselines absorb a non-PH (crossing) shape."""
    df, truth, h = _make_rmst_df(n=6000, seed=11, horizon=2.0, non_ph=True)
    res = {r.estimand: r for r in
           PooledQSubgroupEstimator(n_grid=25).estimate(df, horizon=h, estimand="subgroup_rmst")}
    for s in (0, 1):
        r = res[f"rmst|S={s}"]
        assert abs(r.point_estimate - truth[s]) < 2.5 * r.standard_error + 0.02


def test_rmst_ipcw_beats_naive_km_under_informative_censoring():
    """The #189 payoff: under informative censoring on a within-subgroup-varying covariate,
    pooled-Q's IPCW-adjusted RMST is far less biased than a naive per-subgroup KM RMST,
    which ignores the covariate-dependent censoring. RMSE-to-truth over seeds."""
    e_ipcw, e_km = [], []
    for sd in range(30):
        df, truth, h = _make_rmst_df(n=4000, seed=sd, horizon=2.0, informative_censoring=True)
        r = {x.estimand: x for x in
             PooledQSubgroupEstimator(n_grid=25).estimate(df, horizon=h, estimand="subgroup_rmst")}
        m = (df["subgroup_label"] == 0).values
        e_ipcw.append(r["rmst|S=0"].point_estimate - truth[0])
        e_km.append(_km_rmst(df["T_obs"].values[m], df["Delta"].values[m], h) - truth[0])
    rmse_ipcw = np.sqrt(np.mean(np.square(e_ipcw)))
    rmse_km = np.sqrt(np.mean(np.square(e_km)))
    assert rmse_ipcw < rmse_km, f"IPCW {rmse_ipcw:.4f} !< KM {rmse_km:.4f}"
    assert abs(np.mean(e_ipcw)) < abs(np.mean(e_km))          # and less biased


def test_rmst_coverage_is_near_nominal():
    """IC-based 95% CIs cover the truth ~95% of the time (clean PH), over seeds."""
    hit = tot = 0
    for sd in range(30):
        df, truth, h = _make_rmst_df(n=3000, seed=100 + sd, horizon=2.0)
        res = {r.estimand: r for r in
               PooledQSubgroupEstimator(n_grid=20).estimate(df, horizon=h, estimand="subgroup_rmst")}
        for s in (0, 1):
            r = res[f"rmst|S={s}"]
            tot += 1
            hit += int(r.ci_lower <= truth[s] <= r.ci_upper)
    assert hit / tot >= 0.88, f"coverage {hit/tot:.2f} below 0.88 over {tot} intervals"


def test_rmst_rp_spline_nuisance_recovers_truth():
    """The RP-spline (flexsurvspline) nuisance backend (#188) is debiased by our own
    per-subgroup TMLE to the same targeted RMST. Skipped when rpy2/flexsurv is absent
    (the estimator then silently falls back to the logistic-hazard nuisance)."""
    from causal_bench.estimators.rp_spline_nuisance import _flexsurv_available
    if not _flexsurv_available():
        pytest.skip("rpy2 / flexsurv R package not available")
    df, truth, h = _make_rmst_df(n=4000, seed=11, horizon=2.0, non_ph=True)
    res = {r.estimand: r for r in PooledQSubgroupEstimator(
        nuisance="rp_spline", n_grid=20).estimate(df, horizon=h, estimand="subgroup_rmst")}
    for s in (0, 1):
        r = res[f"rmst|S={s}"]
        assert abs(r.point_estimate - truth[s]) < 2.5 * r.standard_error + 0.02


def test_rmst_rp_spline_falls_back_when_unavailable(monkeypatch):
    """When predict_rp_survival returns None (R stack missing / fit failed), the estimator
    falls back to the logistic-hazard nuisance rather than crashing."""
    import causal_bench.estimators.rp_spline_nuisance as rp
    monkeypatch.setattr(rp, "predict_rp_survival", lambda *a, **k: None)
    df, truth, h = _make_rmst_df(n=2500, seed=2, horizon=2.0)
    res = {r.estimand: r for r in PooledQSubgroupEstimator(
        nuisance="rp_spline", n_grid=20).estimate(df, horizon=h, estimand="subgroup_rmst")}
    assert len(res) == 2
    for s in (0, 1):
        r = res[f"rmst|S={s}"]
        assert np.isfinite(r.point_estimate) and r.standard_error > 0
