"""EffectXShift estimator — rpy2 bridge to McCoy's EffectXShift R package.

Finds the baseline-covariate rule defining the subgroup with the largest
average treatment effect relative to its complement, then estimates the
selected contrast on held-out folds via CV-TMLE / AIPW to give
repeated-sampling guarantees after subgroup selection (McCoy 2026).

Returns three EstimatorResult objects:
  effectxshift_V_ATE      — ATE in the selected high-benefit subgroup V
  effectxshift_Vc_ATE     — ATE in the complement V^c
  effectxshift_contrast   — V ATE − V^c ATE (the post-selection estimand)

Current scope (2026-06-25): single binary randomized treatment, fully
observed scalar endpoint. Not suitable for time-to-event or competing-risks
data. Gracefully returns [] if rpy2 or EffectXShift is not installed.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult

_R_BRIDGE = Path(__file__).parent.parent.parent / "r_scripts" / "effectxshift_bridge.R"


def _effectxshift_available() -> bool:
    try:
        import rpy2.robjects.packages as rpacks
        rpacks.importr("EffectXshift")
        return True
    except Exception:
        return False


class EffectXShiftEstimator(BaseEstimator):
    """Post-selection HTE subgroup estimator via McCoy's EffectXShift R package.

    Requires a binary randomized treatment and a fully observed scalar outcome.
    Do not use with time-to-event or competing-risks endpoints.
    """

    name = "effectxshift"

    def __init__(
        self,
        outcome_col: str = "Y",
        covariate_cols: list[str] | None = None,
        n_folds: int = 5,
        max_depth: int = 2,
    ):
        self._outcome_col    = outcome_col
        self._covariate_cols = covariate_cols or ["W1", "W2", "W3", "W4"]
        self._n_folds        = n_folds
        self._max_depth      = max_depth

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
    ) -> list[EstimatorResult]:
        if not _effectxshift_available():
            warnings.warn(
                "EffectXShift R package not available — skipping. "
                "Install with: remotes::install_github('mccoy-lab/EffectXshift')",
                stacklevel=2,
            )
            return []

        if self._outcome_col not in df.columns:
            warnings.warn(
                f"EffectXShiftEstimator: outcome column '{self._outcome_col}' "
                "not in DataFrame — skipping",
                stacklevel=2,
            )
            return []

        missing = [c for c in self._covariate_cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"EffectXShiftEstimator: covariate columns {missing} missing — skipping",
                stacklevel=2,
            )
            return []

        import rpy2.robjects as ro
        import rpy2.robjects.pandas2ri as pandas2ri
        from rpy2.robjects.conversion import localconverter

        ro.r["source"](str(_R_BRIDGE))
        run_bridge = ro.globalenv["run_effectxshift_bridge"]

        sub = df[[self._outcome_col, "A"] + self._covariate_cols].copy().dropna()
        if len(sub) < 50:
            warnings.warn(
                f"EffectXShiftEstimator: only {len(sub)} complete rows — skipping",
                stacklevel=2,
            )
            return []

        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df   = ro.conversion.py2rpy(sub.reset_index(drop=True))
                result = run_bridge(
                    r_df,
                    ro.StrVector([self._outcome_col]),
                    ro.StrVector(["A"]),
                    ro.StrVector(self._covariate_cols),
                    ro.IntVector([self._n_folds]),
                    ro.IntVector([self._max_depth]),
                )

            v_ate    = float(np.array(result.rx2("v_ate"))[0])
            vc_ate   = float(np.array(result.rx2("vc_ate"))[0])
            contrast = float(np.array(result.rx2("contrast"))[0])
            v_se     = float(np.array(result.rx2("v_se"))[0])
            vc_se    = float(np.array(result.rx2("vc_se"))[0])
            cont_se  = float(np.array(result.rx2("contrast_se"))[0])
            rule     = str(np.array(result.rx2("rule"))[0])

        except Exception as exc:
            warnings.warn(f"EffectXShiftEstimator bridge failed: {exc}", stacklevel=2)
            return []

        conv = {"rule": rule}

        def _result(
            name: str, point: float, se: float, label: str
        ) -> EstimatorResult | None:
            if not (np.isfinite(point) and np.isfinite(se) and se > 0):
                return None
            return EstimatorResult(
                name=name,
                estimand=label,
                point_estimate=point,
                standard_error=se,
                ci_lower=point - 1.96 * se,
                ci_upper=point + 1.96 * se,
                convergence_info=conv,
            )

        return [
            r for r in [
                _result("effectxshift_V_ATE",    v_ate,    v_se,    "HTE_V_ATE"),
                _result("effectxshift_Vc_ATE",   vc_ate,   vc_se,   "HTE_Vc_ATE"),
                _result("effectxshift_contrast", contrast, cont_se, "HTE_V_minus_Vc"),
            ]
            if r is not None
        ]
