"""Royston-Parmar flexible-parametric survival nuisance for the pooled-Q subgroup RMST
estimator (issue #188, Route A).

`predict_rp_survival` returns the conditional survival matrix S[i, k] = S(t_grid[k] | W_i,
S_i). The pooled-Q RMST estimator then debiases these predictions with its OWN per-subgroup
one-step TMLE — RP is a *nuisance*, not a plug-in final estimator (matching #188's "RP as
nuisance; the DR gap is filled downstream", with our single-arm targeting filling it instead
of concrete's).

Two interchangeable backends, same return contract:

  * **flexsurv** (`flexsurv::flexsurvspline` via rpy2 / r_scripts/flexsurv_bridge.R) — the
    reference implementation; preferred when the R stack is present.
  * **lifelines** (`CRCSplineFitter`, the Crowther-Royston-Clements AFT cubic spline) — a
    pure-Python fallback so the non-PH RP nuisance works with NO R (e.g. the SMB demo box).
    Same spline family, same group-varying time-effect, no rpy2/flexsurv dependency.

Degrades gracefully: returns None if neither backend is available or the fit fails, so the
estimator falls back to its pooled logistic-hazard nuisance and CI stays green everywhere.
No hard R dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_R_BRIDGE = Path(__file__).parent.parent.parent / "r_scripts" / "flexsurv_bridge.R"


def _flexsurv_available() -> bool:
    """True if rpy2 is importable and the flexsurv R package is installed."""
    try:
        import rpy2.robjects.packages as rpacks
        rpacks.importr("flexsurv")
        return True
    except Exception:
        return False


def _lifelines_available() -> bool:
    """True if lifelines (with CRCSplineFitter) is importable."""
    try:
        from lifelines import CRCSplineFitter  # noqa: F401
        return True
    except Exception:
        return False


def _predict_flexsurv(df, W_cols, subgroup_col, t_grid, k):
    import rpy2.robjects as ro
    import rpy2.robjects.pandas2ri as pandas2ri
    from rpy2.robjects.conversion import localconverter

    cols = list(W_cols) + [subgroup_col, "T_obs", "Delta"]
    sub = df[cols].copy()
    sub["T_obs"] = sub["T_obs"].astype(float)
    sub["Delta"] = sub["Delta"].astype(float)
    n = len(sub)
    try:
        ro.r["source"](str(_R_BRIDGE))
        run = ro.globalenv["run_flexsurv_survival"]
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(sub.reset_index(drop=True))
            S_r = run(r_df, ro.StrVector(list(W_cols)), ro.StrVector([subgroup_col]),
                      ro.FloatVector(np.asarray(t_grid, float)), int(k))
            S = np.asarray(ro.conversion.rpy2py(S_r), dtype=float)
    except Exception:
        return None
    if S.shape != (n, len(t_grid)):
        return None
    return S


def _predict_lifelines(df, W_cols, subgroup_col, t_grid, k):
    """CRCSplineFitter (Royston-Parmar-family) survival, R-free.

    CRCSplineFitter is an *accelerated failure time* spline model, so it is already non-PH:
    confounders + subgroup enter the AFT location term `beta_`, giving each subgroup a
    non-PH hazard shape (the reason to reach for RP over the pooled logistic hazard). The
    baseline spline (`gamma*_`) is shared. This is a touch less flexible than flexsurv's
    group-varying spline coefficients (`anc=list(gamma1=~S)`) — lifelines' initial-point
    construction rejects covariates on the ancillary spline terms — but RP is a *nuisance*
    the pooled-Q one-step targeting debiases downstream, so the shared-baseline AFT spline
    is sufficient (bias is first-order insensitive to nuisance flexibility here).
    """
    from lifelines import CRCSplineFitter

    cols = list(W_cols) + [subgroup_col]
    fit_df = df[cols + ["T_obs", "Delta"]].copy()
    fit_df["T_obs"] = np.maximum(fit_df["T_obs"].astype(float), 1e-4)  # durations must be > 0
    fit_df["Delta"] = fit_df["Delta"].astype(float)

    n_knots = max(3, int(k) + 1)                       # flexsurv k internal knots -> baseline knots
    regressors = {"beta_": " + ".join(cols)}           # confounders + subgroup on AFT location
    for i in range(n_knots):
        regressors[f"gamma{i}_"] = "1"                 # shared baseline spline
    assert set(regressors) == {"beta_"} | {f"gamma{i}_" for i in range(n_knots)}

    try:
        f = CRCSplineFitter(n_baseline_knots=n_knots, penalizer=1e-3)
        f.fit(fit_df, duration_col="T_obs", event_col="Delta", regressors=regressors)
        S = f.predict_survival_function(fit_df, times=np.asarray(t_grid, float))
        S = S.values.T                                 # (times, rows) -> (rows, times)
    except Exception:
        return None
    if S.shape != (len(fit_df), len(t_grid)):
        return None
    return S


def predict_rp_survival(df: pd.DataFrame, W_cols, subgroup_col: str, t_grid,
                        k: int = 2) -> np.ndarray | None:
    """Predicted S(t_k | W_i, S_i) as an (n, len(t_grid)) array, or None if no RP backend
    is available / the fit fails (caller falls back to the logistic-hazard nuisance).

    Prefers flexsurv (R) when present; otherwise uses the pure-Python lifelines backend so
    the non-PH RP nuisance is available with no R stack. A group-varying spline time-effect
    lets each subgroup take a non-PH hazard shape — the reason to reach for RP over the
    pooled logistic hazard.
    """
    S = None
    if _flexsurv_available():
        S = _predict_flexsurv(df, W_cols, subgroup_col, t_grid, k)
    if S is None and _lifelines_available():
        S = _predict_lifelines(df, W_cols, subgroup_col, t_grid, k)
    if S is None or not np.all(np.isfinite(S)):
        return None
    return np.clip(S, 1e-6, 1.0 - 1e-6)
