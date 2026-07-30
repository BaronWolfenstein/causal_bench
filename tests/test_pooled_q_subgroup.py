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
