"""concrete win ratio estimator — rpy2 bridge to concrete's targetWinRatio().

Uses the direct TMLE (method="direct") by default, which solves the win/loss
EIF estimating equations jointly and cuts WR bias ~5x vs the plug-in
(concrete PR #30 validation). The plug-in (method="plugin") is available
for comparison experiments.

Gracefully returns [] if rpy2 or concrete is unavailable.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.estimators.base import BaseEstimator
from causal_bench.estimators.concrete_rmst import _concrete_available, prepare_for_r
from causal_bench.metrics import EstimatorResult

_R_BRIDGE = Path(__file__).parent.parent.parent / "r_scripts" / "concrete_bridge.R"


class ConcreteWinRatioEstimator(BaseEstimator):
    """Win ratio estimator via McCoy's concrete R package.

    Sources r_scripts/concrete_bridge.R and calls run_concrete_win_ratio(),
    which handles both the direct TMLE (targetWinRatio) and plug-in
    (getWinRatio via doConcrete) modes. Returns [] with a warning if rpy2
    or concrete is unavailable.
    """

    def __init__(
        self,
        method: str = "direct",
        horizon: float = 1.0,
        strata_cols: list[str] | None = None,
    ):
        if method not in ("direct", "plugin"):
            raise ValueError(f"method must be 'direct' or 'plugin'; got {method!r}")
        self._method = method
        self._horizon = horizon
        self._strata_cols = strata_cols

    @property
    def name(self) -> str:
        return f"concrete_WR_{self._method}"

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "WR",
    ) -> list[EstimatorResult]:
        if not _concrete_available():
            warnings.warn(
                f"concrete R package not available — skipping {self.name}",
                stacklevel=2,
            )
            return []

        import rpy2.robjects as ro
        import rpy2.robjects.pandas2ri as pandas2ri
        from rpy2.robjects.conversion import localconverter

        ro.r["source"](str(_R_BRIDGE))
        run_win_ratio = ro.globalenv["run_concrete_win_ratio"]

        df_r = df.copy()
        df_r["event_type"] = df_r["Delta"].astype(int)
        df_r = prepare_for_r(df_r)

        r_strata = ro.StrVector(self._strata_cols) if self._strata_cols else ro.rinterface.NULL
        r_method = ro.StrVector([self._method])

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r)

        try:
            result_r = run_win_ratio(r_df, float(horizon),
                                     method=r_method, strata_cols=r_strata)
            wr  = float(np.array(result_r.rx2("WR"))[0])
            se  = float(np.array(result_r.rx2("SE"))[0])
            if not (np.isfinite(wr) and np.isfinite(se)):
                warnings.warn(f"{self.name}: concrete returned non-finite WR/SE", stacklevel=2)
                return []
        except Exception as exc:
            warnings.warn(f"{self.name} bridge failed: {exc}", stacklevel=2)
            return []

        ci_lower = float(np.array(result_r.rx2("CI_lower"))[0])
        ci_upper = float(np.array(result_r.rx2("CI_upper"))[0])

        return [EstimatorResult(
            name=self.name,
            estimand=estimand,
            point_estimate=wr,
            standard_error=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )]
