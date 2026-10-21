"""IPCW-augmented AIPW for a cumulative-incidence risk difference F_1(t) - F_0(t) under right censoring.

Shared by AIPWEstimator (t = horizon) and PointwiseRMSTEstimator (t on a grid). Two defects of the earlier versions are
fixed here (audit 2026-10-21):

1. Censoring. The old outcome Y_t = 1{T_obs <= t, Delta = 1} counts a patient censored before t as event-free. That
   biases F_a(t) downward whenever censoring occurs before t, and the bias does not vanish with the grid size K.
   Fix: the status at t is KNOWN iff T_obs >= t or a cause-1 event was observed (Delta = 1). Known patients are
   weighted by 1 / G(min(T_obs, t)- | A, W), with G the censoring survival from a Cox model on (A, W). Q is fit on
   the known patients only, which is consistent under conditionally independent censoring (C indep. T | A, W).
   Competing events (event_type = 2) are treated as censoring, matching the repo convention (TMLE+IPCW; see the
   exp8 docstring) and the truths in dgp.survival, which ignore cause 2.
2. Nuisance mismatch (AIPW only). The old point estimate used a SuperLearner Q in the plug-in term but a separate
   RidgeCV Q in the residual terms. With a correct propensity that is still off by
   E[(Q_SL - Q_ridge)_1 - (Q_SL - Q_ridge)_0], so double robustness was lost. Fix: ONE Q in both terms
   (cross-fitted when crossfit=True).

The influence function treats G as known (the usual IPCW-AIPW plug-in SE; slightly conservative when G is estimated).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from causal_bench.super_learner import SuperLearner

G_FLOOR = 0.05      # positivity floor on the censoring survival
G_TRUNC = 0.01      # propensity truncation


def _censored_mask(df, horizon):
    return (df["Delta"].values == 0) & (df["T_obs"].values < horizon - 1e-12)   # incl. competing events (repo convention)


def censoring_survival_fn(df, W, horizon):
    """Return s -> G(s- | A_i, W_i) (vectorized over subjects), from a Cox model for the censoring hazard."""
    cens = _censored_mask(df, horizon)
    n = len(df)
    if cens.sum() < 5:                                     # (near-)no censoring: G = 1
        return lambda s: np.ones(n)
    from lifelines import CoxPHFitter
    X = pd.DataFrame(np.column_stack([df["A"].values, W]), columns=["A"] + [f"w{j}" for j in range(W.shape[1])])
    d = X.copy(); d["T"] = df["T_obs"].values; d["C"] = cens.astype(int)
    cph = CoxPHFitter(penalizer=1e-3).fit(d, duration_col="T", event_col="C")
    H0 = cph.baseline_cumulative_hazard_
    times = H0.index.values.astype(float); h0 = H0.values[:, 0]
    risk = np.asarray(cph.predict_partial_hazard(X)).ravel()

    def G(s):
        s = np.asarray(s, float)
        k = np.searchsorted(times, s, side="left") - 1      # left limit: hazard strictly before s
        Hs = np.where(k >= 0, h0[np.clip(k, 0, None)], 0.0)
        return np.maximum(np.exp(-Hs * risk), G_FLOOR)
    return G


def ipcw_aipw_ic(df, W, A, g, t, Gfn, *, n_folds, random_state, crossfit, horizon):
    """Uncentered IC_i for F_1(t) - F_0(t); the estimate is IC.mean(). Gfn=None ⇒ no censoring correction (legacy)."""
    n = len(df)
    T = df["T_obs"].values
    ev1 = df["Delta"].values == 1
    Y = ((T <= t) & ev1).astype(float)                               # event by t
    if Gfn is None:
        known = np.ones(n, bool); w = np.ones(n)
    else:
        known = (T >= t) | ev1
        w = known / Gfn(np.minimum(T, t))
    X_AW = np.column_stack([A, W]); X_1W = np.column_stack([np.ones(n), W]); X_0W = np.column_stack([np.zeros(n), W])

    def fit_pred(tr, te):
        tr = tr[known[tr]]
        sl = SuperLearner(task="regression", n_folds=n_folds, random_state=random_state).fit(X_AW[tr], Y[tr])
        return np.clip(sl.predict(X_1W[te]), 0, 1), np.clip(sl.predict(X_0W[te]), 0, 1)

    Q1 = np.zeros(n); Q0 = np.zeros(n)
    if crossfit:
        for tr, te in KFold(n_folds, shuffle=True, random_state=random_state).split(X_AW):
            Q1[te], Q0[te] = fit_pred(tr, te)
    else:
        Q1[:], Q0[:] = fit_pred(np.arange(n), np.arange(n))
    gt = np.clip(g, G_TRUNC, 1 - G_TRUNC)
    return (Q1 - Q0) + w * ((A / gt) * (Y - Q1) - ((1 - A) / (1 - gt)) * (Y - Q0))
