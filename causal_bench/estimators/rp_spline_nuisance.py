"""Royston-Parmar (flexsurvspline) survival nuisance for the pooled-Q subgroup RMST
estimator (issue #188, Route A).

`predict_rp_survival` fits `flexsurv::flexsurvspline` (via rpy2 / r_scripts/flexsurv_bridge.R)
and returns the conditional survival matrix S[i, k] = S(t_grid[k] | W_i, S_i). The pooled-Q
RMST estimator then debiases these predictions with its OWN per-subgroup one-step TMLE — RP
is a *nuisance*, not a plug-in final estimator (matching #188's "RP as nuisance; the DR gap is
filled downstream", with our single-arm targeting filling it instead of concrete's).

Degrades gracefully: returns None if rpy2 / the flexsurv R package is unavailable or the fit
fails, so the estimator falls back to its pooled logistic-hazard nuisance and CI stays green
on machines without the R stack. No hard R dependency.
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


def predict_rp_survival(df: pd.DataFrame, W_cols, subgroup_col: str, t_grid,
                        k: int = 2) -> np.ndarray | None:
    """Predicted S(t_k | W_i, S_i) as an (n, len(t_grid)) array, or None if the RP
    fit is unavailable / failed (caller falls back to the logistic-hazard nuisance).

    A group-varying spline time-effect (anc=list(gamma1=~S)) lets each subgroup take a
    non-PH hazard shape — the reason to reach for RP over the pooled logistic hazard.
    """
    if not _flexsurv_available():
        return None

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

    # Bridge returns a 1x1 NaN on failure; any non-conforming / non-finite result -> fallback.
    if S.shape != (n, len(t_grid)) or not np.all(np.isfinite(S)):
        return None
    return np.clip(S, 1e-6, 1.0 - 1e-6)
