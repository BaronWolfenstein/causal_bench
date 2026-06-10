"""concrete RMST estimator — rpy2 bridge to McCoy's concrete R package.

Python side uses rpy2 to call R. The R script r_scripts/concrete_bridge.R
is structured for reticulate compatibility so McCoy can also source it
directly in RStudio.

Gracefully returns [] if rpy2 or the concrete R package is not installed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult

# Required columns that concrete's formatArguments expects
_REQUIRED_COLS = {"T_obs", "event_type", "A"}


def _concrete_available() -> bool:
    """Return True if rpy2 is importable and the concrete R package is installed."""
    try:
        import rpy2.robjects as ro  # noqa: F401
        ro.packages.importr("concrete")
        return True
    except Exception:
        return False


def prepare_for_r(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a causal_bench DataFrame for handoff to pandas2ri / concrete.

    Applies all normalizations that concrete's formatArguments requires:
    - Validates required columns are present
    - Fills NaN in L1 with column median (pandas2ri errors on NA in numeric)
    - Upcasts float32 → float64 (avoids silent truncation)
    - Casts Delta/event_type and A to int64
    - Resets index to 0..n-1 (R doesn't handle non-default row indices)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from generate_data(), must have event_type column added.

    Returns
    -------
    pd.DataFrame ready for pandas2ri.py2rpy().

    Raises
    ------
    ValueError if any required column is missing.
    KeyError propagated if column access fails unexpectedly.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"prepare_for_r: missing required columns {missing}")

    out = df.copy()

    # Fill NaN L1 with column median (0.0 if all NaN)
    if "L1" in out.columns and out["L1"].isna().any():
        fill_val = out["L1"].median()
        if np.isnan(fill_val):
            fill_val = 0.0
        out["L1"] = out["L1"].fillna(fill_val)

    # Upcast float32 → float64 to avoid silent precision loss in pandas2ri
    for col in out.select_dtypes(include=[np.float32]).columns:
        out[col] = out[col].astype(np.float64)

    # event_type and A must be integer for concrete
    out["event_type"] = out["event_type"].astype(np.int64)
    out["A"] = out["A"].astype(np.int64)

    # Reset index — R doesn't handle non-default row indices
    out = out.reset_index(drop=True)

    return out


class ConcreteRMSTEstimator(BaseEstimator):
    """RMST estimator via McCoy's concrete R package.

    Calls concrete::formatArguments → doConcrete → targetRMST via rpy2.
    Returns [] (empty list) with a warning if rpy2 or concrete is unavailable,
    so the MC runner treats it as N/A rather than crashing.
    """

    name = "concrete_RMST"

    def __init__(self, horizon: float = 1.0):
        self._horizon = horizon

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
    ) -> list[EstimatorResult]:
        if not _concrete_available():
            warnings.warn(
                "concrete R package not available — skipping ConcreteRMSTEstimator",
                stacklevel=2,
            )
            return []

        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()
        concrete = ro.packages.importr("concrete")

        df_r = df.copy()
        df_r["event_type"] = df_r["Delta"].astype(int)
        df_r = prepare_for_r(df_r)

        r_df = pandas2ri.py2rpy(df_r)

        args = concrete.formatArguments(
            DataTable=r_df,
            EventTime=ro.StrVector(["T_obs"]),
            EventType=ro.StrVector(["event_type"]),
            Treatment=ro.StrVector(["A"]),
            Intervention=ro.IntVector([0, 1]),
            TargetTime=ro.FloatVector([horizon]),
        )
        est = concrete.doConcrete(args)
        rmst_result = concrete.targetRMST(est)

        # Parse R list output — structure depends on concrete version
        # McCoy's API: rmst_result is a named list with "Results" element
        try:
            results_r = rmst_result.rx2("Results")
            ate_r = results_r.rx2("ATE")
            point = float(np.array(ate_r.rx2("Estimate"))[0])
            se = float(np.array(ate_r.rx2("SE"))[0])
            ci_lower = point - 1.96 * se
            ci_upper = point + 1.96 * se
        except Exception as exc:
            warnings.warn(f"concrete result parsing failed: {exc}", stacklevel=2)
            return []

        return [EstimatorResult(
            name="concrete_RMST",
            estimand=estimand,
            point_estimate=point,
            standard_error=se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        )]
