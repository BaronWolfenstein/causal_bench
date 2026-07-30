"""Operating characteristics for the pooled-Q subgroup RMST estimator (#189/#77/#188).

Self-validating: every comparison is against a Monte-Carlo truth drawn from the same
generative process (event rate P(T<=tau|S=s) and RMST E[min(T,tau)|S=s]). Three findings,
each a table over seeds:

  (A) IPCW vs naive KM under WITHIN-SUBGROUP informative censoring — the clear win: KM
      ignores the covariate-dependent censoring and biases; pooled-Q's IPCW corrects it.
  (B) Borrowing: pooled-Q vs subgroup-only, reported for BOTH estimands — the borrowing
      win is real for the event RATE (#77) but ATTENUATED for RMST, because the per-
      subgroup one-step targeting makes the RMST point estimate first-order insensitive
      to initial-nuisance quality and time-integration averages the residual variance.
  (C) Non-PH: logistic-hazard vs RP-spline (flexsurvspline, #188) nuisance under crossing
      hazards — both are debiased to truth; RP is the smooth non-PH option (skipped when
      rpy2/flexsurv is absent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_bench.estimators.pooled_q_subgroup import PooledQSubgroupEstimator


def make_data(n, thr, seed, horizon=2.0, informative_censoring=False, non_ph=False,
              truth_n=300_000):
    """Single-arm survival data; subgroup 1 = the small tail 1{W1>thr}. Returns
    (df, truth_rate, truth_rmst, horizon) with MC truths for both estimands."""
    rng = np.random.default_rng(seed)

    def draw(m, rs):
        W = rs.normal(size=(m, 4))
        S = (W[:, 0] > thr).astype(int)
        lam = np.exp(-0.2 + 0.4 * W[:, 0] - 0.9 * W[:, 1] + 0.5 * W[:, 2] + 0.5 * S)
        if non_ph:
            shape = np.where(S == 1, 1.7, 0.8)                 # crossing hazards
            T = (-np.log(rs.random(m))) ** (1.0 / shape) / lam
        else:
            T = rs.exponential(1.0 / lam)
        return W, S, T

    W, S, T = draw(n, rng)
    T_obs = np.minimum(T, horizon)
    Delta = (T <= horizon).astype(float)
    if informative_censoring:
        C = rng.exponential(1.0 / np.exp(-1.6 + 0.6 * W[:, 1]))    # ~20%, varies within subgroup
        drop = C < T_obs
        T_obs = np.where(drop, C, T_obs)
        Delta = np.where(drop, 0.0, Delta)

    df = pd.DataFrame({"W1": W[:, 0], "W2": W[:, 1], "W3": W[:, 2], "W4": W[:, 3],
                       "T_obs": T_obs, "Delta": Delta, "subgroup_label": S, "A": 1.0})
    Wt, St, Tt = draw(truth_n, np.random.default_rng(seed + 7))
    rate = {s: float((Tt[St == s] <= horizon).mean()) for s in (0, 1)}
    rmst = {s: float(np.minimum(Tt, horizon)[St == s].mean()) for s in (0, 1)}
    return df, rate, rmst, horizon


def _km_rmst(T_obs, Delta, horizon):
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time
    kmf = KaplanMeierFitter().fit(T_obs, (Delta == 1))
    return float(restricted_mean_survival_time(kmf, t=horizon))


def _rmse_bias(errs):
    e = np.asarray(errs, float)
    if e.size == 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(e))


def censoring_vs_km(thrs=(1.0, 1.4, 1.8), n=4000, n_reps=30, horizon=2.0, n_grid=25, seed=0):
    """Finding (A): pooled-Q IPCW vs naive KM RMST under within-subgroup informative
    censoring, swept over subgroup size (thr)."""
    rows = []
    for thr in thrs:
        e_ipcw, e_km, ns = [], [], []
        for sd in range(n_reps):
            df, _, rmst, h = make_data(n, thr, seed + sd, horizon, informative_censoring=True)
            m = (df["subgroup_label"] == 1).values
            if m.sum() < 15:
                continue
            ns.append(int(m.sum()))
            res = {r.estimand: r for r in PooledQSubgroupEstimator(n_grid=n_grid)
                   .estimate(df, horizon=h, estimand="subgroup_rmst")}
            if "rmst|S=1" not in res:
                continue
            e_ipcw.append(res["rmst|S=1"].point_estimate - rmst[1])
            e_km.append(_km_rmst(df["T_obs"].values[m], df["Delta"].values[m], h) - rmst[1])
        ri, bi = _rmse_bias(e_ipcw)
        rk, bk = _rmse_bias(e_km)
        rows.append({"thr": thr, "n_s": int(np.median(ns)) if ns else 0, "reps": len(e_ipcw),
                     "ipcw_rmse": ri, "ipcw_bias": bi, "km_rmse": rk, "km_bias": bk})
    return rows


def borrowing(thrs=(1.0, 1.4, 1.8), n=1500, n_reps=40, horizon=2.0, n_grid=20, seed=1000):
    """Finding (B): pooled vs subgroup-only, BOTH estimands (event rate + RMST), clean."""
    rows = []
    for thr in thrs:
        er = {"pool": [], "only": []}          # event-rate errors
        em = {"pool": [], "only": []}          # rmst errors
        ns = []
        for sd in range(n_reps):
            df, rate, rmst, h = make_data(n, thr, seed + sd, horizon)
            if (df["subgroup_label"] == 1).sum() < 15:
                continue
            ns.append(int((df["subgroup_label"] == 1).sum()))
            for tag, pooled in (("pool", True), ("only", False)):
                est = PooledQSubgroupEstimator(pooled=pooled, n_grid=n_grid)
                # The within-subgroup (pooled=False) fit can hit a single-outcome-class
                # subgroup (a tiny all-events tail) and raise; skip that estimand for the
                # rep rather than distort the DGP. Pooled borrows across subgroups and is
                # unaffected — which is itself part of the borrowing story.
                try:
                    rr = {r.estimand: r for r in est.estimate(df, horizon=h)}
                    if "rate|S=1" in rr:
                        er[tag].append(rr["rate|S=1"].point_estimate - rate[1])
                except Exception:
                    pass
                try:
                    mm = {r.estimand: r for r in est.estimate(df, horizon=h, estimand="subgroup_rmst")}
                    if "rmst|S=1" in mm:
                        em[tag].append(mm["rmst|S=1"].point_estimate - rmst[1])
                except Exception:
                    pass
        row = {"thr": thr, "n_s": int(np.median(ns)) if ns else 0, "reps": len(em["pool"])}
        row["rate_pool_rmse"], _ = _rmse_bias(er["pool"])
        row["rate_only_rmse"], _ = _rmse_bias(er["only"])
        row["rmst_pool_rmse"], _ = _rmse_bias(em["pool"])
        row["rmst_only_rmse"], _ = _rmse_bias(em["only"])
        rows.append(row)
    return rows


def non_ph_nuisance(n=4000, n_reps=15, horizon=2.0, n_grid=20, seed=2000):
    """Finding (C): logistic vs RP-spline nuisance under crossing hazards. RP rows are
    empty (skipped) when rpy2/flexsurv is unavailable."""
    from causal_bench.estimators.rp_spline_nuisance import _flexsurv_available
    have_rp = _flexsurv_available()
    rows = []
    for nz in (["logistic", "rp_spline"] if have_rp else ["logistic"]):
        errs = {0: [], 1: []}
        for sd in range(n_reps):
            df, _, rmst, h = make_data(n, 1.0, seed + sd, horizon, non_ph=True)
            res = {r.estimand: r for r in PooledQSubgroupEstimator(nuisance=nz, n_grid=n_grid)
                   .estimate(df, horizon=h, estimand="subgroup_rmst")}
            for s in (0, 1):
                if f"rmst|S={s}" in res:
                    errs[s].append(res[f"rmst|S={s}"].point_estimate - rmst[s])
        row = {"nuisance": nz, "reps": n_reps}
        for s in (0, 1):
            row[f"rmse_S{s}"], row[f"bias_S{s}"] = _rmse_bias(errs[s])
        rows.append(row)
    return rows, have_rp
