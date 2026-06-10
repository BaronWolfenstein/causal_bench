"""concrete RMST estimator — rpy2 bridge to McCoy's concrete R package.

Python side sources r_scripts/concrete_bridge.R and calls
run_concrete_bridge(r_df, horizon) so all result-parsing and API-version
negotiation lives in one place (the R script).

The R bridge passes L1 as CensoringTV (not as an outcome covariate), which
is required since concrete 1.1.1.9000 (2026-06-10 commit d37e37c).

Gracefully returns [] if rpy2 or the concrete R package is not installed.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult

# Required columns that concrete's formatArguments expects
_REQUIRED_COLS = {"T_obs", "event_type", "A"}

# Absolute path to the R bridge script (same repo, fixed relative location)
_R_BRIDGE = Path(__file__).parent.parent.parent / "r_scripts" / "concrete_bridge.R"


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
    - Upcasts float32 → float64 (avoids silent truncation in pandas2ri)
    - Casts event_type and A to int64
    - Resets index to 0..n-1 (R doesn't handle non-default row indices)

    L1 is intentionally left as-is (NaN values preserved). The R bridge
    routes L1 into CensoringTV, not the outcome covariate set, so NaN rows
    are filtered there. Imputing NaN here would send imputed L1 to the
    censoring model, which is incorrect.

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
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"prepare_for_r: missing required columns {missing}")

    out = df.copy()

    # Upcast float32 → float64 to avoid silent precision loss in pandas2ri
    for col in out.select_dtypes(include=[np.float32]).columns:
        out[col] = out[col].astype(np.float64)

    # event_type and A must be integer for concrete
    out["event_type"] = out["event_type"].astype(np.int64)
    out["A"] = out["A"].astype(np.int64)

    # Reset index — R doesn't handle non-default row indices
    out = out.reset_index(drop=True)

    return out


def concrete_sensitivity(
    df: pd.DataFrame,
    horizon: float = 1.0,
    deltas: list[float] | None = None,
) -> pd.DataFrame:
    """Run concrete::senseCensoring() and return a tidy DataFrame.

    A 1-D delta-shift MAR sensitivity analysis using concrete's doubly-robust
    TMLE as the estimator. At each delta, a fraction delta of the censored
    patients are counterfactually treated as events and the risk difference
    is re-estimated. Because L1 is forwarded to CensoringTV, the baseline
    (delta=0) already uses the L1-corrected IPCW.

    Parameters
    ----------
    df      : causal_bench DataFrame (must have L1 for CensoringTV to activate).
    horizon : study horizon.
    deltas  : fractions of censored patients assumed to be events.
              Default: [0.0, 0.05, 0.10, 0.15, 0.20].

    Returns
    -------
    pd.DataFrame with columns:
        delta, estimate, se, ci_lower, ci_upper

    Raises
    ------
    RuntimeError if concrete R package is not available.
    """
    if deltas is None:
        deltas = [0.0, 0.05, 0.10, 0.15, 0.20]

    if not _concrete_available():
        raise RuntimeError(
            "concrete R package not available — cannot run sensitivity analysis. "
            "Install with: remotes::install_github('blind-contours/concrete')"
        )

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    ro.r["source"](str(_R_BRIDGE))
    run_sens = ro.globalenv["run_concrete_sensitivity"]

    df_r = df.copy()
    df_r["event_type"] = df_r["Delta"].astype(int)
    df_r = prepare_for_r(df_r)
    r_df     = pandas2ri.py2rpy(df_r)
    r_deltas = ro.FloatVector(deltas)

    try:
        result_r = run_sens(r_df, float(horizon), r_deltas)
        return pandas2ri.rpy2py(result_r).reset_index(drop=True)
    except Exception as exc:
        raise RuntimeError(f"concrete sensitivity analysis failed: {exc}") from exc


class ConcreteRMSTEstimator(BaseEstimator):
    """RMST estimator via McCoy's concrete R package.

    Sources r_scripts/concrete_bridge.R and calls run_concrete_bridge(),
    which handles API versioning and result parsing. Returns [] with a
    warning if rpy2 or concrete is unavailable, so the MC runner treats
    it as N/A rather than crashing.

    L1 (when present) is forwarded to concrete's CensoringTV argument so
    it conditions the IPCW — not the outcome hazards. This avoids the
    collider bias that arises from conditioning on a post-treatment
    time-varying variable in the outcome model.
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

        # Source the R bridge (idempotent — R caches sourced environments)
        ro.r["source"](str(_R_BRIDGE))
        run_bridge = ro.globalenv["run_concrete_bridge"]

        df_r = df.copy()
        df_r["event_type"] = df_r["Delta"].astype(int)
        df_r = prepare_for_r(df_r)
        r_df = pandas2ri.py2rpy(df_r)

        try:
            result_r = run_bridge(r_df, float(horizon))
            point = float(np.array(result_r.rx2("ATE"))[0])
            se    = float(np.array(result_r.rx2("SE"))[0])
            if not (np.isfinite(point) and np.isfinite(se)):
                warnings.warn("concrete returned non-finite ATE/SE", stacklevel=2)
                return []
        except Exception as exc:
            warnings.warn(f"concrete bridge failed: {exc}", stacklevel=2)
            return []

        ci_lower = point - 1.96 * se
        ci_upper = point + 1.96 * se

        return [EstimatorResult(
            name="concrete_RMST",
            estimand=estimand,
            point_estimate=point,
            standard_error=se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        )]
