"""cobalt (R) balance cross-check — package-time, guarded rpy2 bridge.

Regulator-familiar covariate balance via `cobalt::bal.tab`. This is a
*cross-check* for the final evidence package, not the primary diagnostic:
reviewers recognise cobalt's bal.tab/love.plot output, but cobalt does NOT
produce the region-resolved SMD / deep-R ESS map that catches the edge-fill
failure mode (see experiments/exp29_balance_diagnostics.py — that stays
primary). Mirrors the repo's `concrete` rpy2-bridge convention: lazy import,
graceful skip if rpy2 or the `cobalt` R package is unavailable.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_R_BRIDGE = Path(__file__).resolve().parents[2] / "r_scripts" / "cobalt_bridge.R"


def _cobalt_available() -> bool:
    try:
        import rpy2.robjects.packages as rpacks  # noqa: F401
        rpacks.importr("cobalt")
        return True
    except Exception:
        return False


def cobalt_baltab(
    target: pd.DataFrame,
    baseline: pd.DataFrame,
    weights: np.ndarray,
    covs: list[str],
) -> Optional[pd.DataFrame]:
    """`cobalt::bal.tab` per-covariate balance (mean diffs + variance ratios).

    ``weights`` aligns to the stacked (target, baseline) rows: Target Group rows
    carry weight 1.0, Baseline Cohort rows carry the propensity odds-weight.
    Returns the balance data.frame, or ``None`` (with a warning) if rpy2 or the
    ``cobalt`` R package is not installed — matching the estimator bridges.
    """
    if not _cobalt_available():
        warnings.warn(
            "cobalt R package / rpy2 not available — skipping cobalt cross-check "
            "(the primary region-resolved diagnostic in exp29 is unaffected)",
            stacklevel=2,
        )
        return None

    import rpy2.robjects as ro
    import rpy2.robjects.pandas2ri as pandas2ri
    from rpy2.robjects.conversion import localconverter

    stacked = pd.concat([target[covs], baseline[covs]], ignore_index=True)
    stacked["treat"] = np.r_[np.ones(len(target)), np.zeros(len(baseline))].astype(int)
    w = np.asarray(weights, dtype=float)
    if len(w) != len(stacked):
        raise ValueError("weights must align with stacked (target, baseline) rows")

    ro.r["source"](str(_R_BRIDGE))
    run_fn = ro.globalenv["run_cobalt_baltab"]
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(stacked)
        r_w = ro.conversion.py2rpy(pd.Series(w))
        r_covs = ro.StrVector(list(covs))
        out = run_fn(r_df, r_w, r_covs)
        return ro.conversion.rpy2py(out)
