"""BCF/BART HTE estimator — rpy2 bridge for posterior tree-based subgroup summaries.

Uses Bayesian Causal Forest (Hahn et al. 2020, 'bcf' R package) to estimate
per-patient CATEs, then fits a parsimonious rpart summary tree following
Hahn's suggestion (2026-06-25) as an alternative to EffectXShift's
selected-rule CV-TMLE framing.

Returns three EstimatorResult objects:
  bcf_bart_high_leaf   — posterior mean ATE in the highest-CATE leaf
  bcf_bart_low_leaf    — posterior mean ATE in the lowest-CATE leaf
  bcf_bart_contrast    — high_leaf ATE − low_leaf ATE (posterior SD as SE)

Falls back to vanilla BART ('BART' R package) if bcf is unavailable.
Gracefully returns [] if neither package or rpy2 is installed.

Requires a binary treatment and fully observed scalar outcome. Not suitable
for time-to-event or competing-risks endpoints.

References:
  Hahn, Murray & Carvalho (2020). Bayesian regression tree models for causal
    inference. Bayesian Analysis 15(3):965-1056.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult

_R_BRIDGE = Path(__file__).parent.parent.parent / "r_scripts" / "bcf_bart_bridge.R"


def _bcf_bart_available() -> bool:
    try:
        import rpy2.robjects.packages as rpacks
        try:
            rpacks.importr("bcf")
            return True
        except Exception:
            rpacks.importr("BART")
            return True
    except Exception:
        return False


class BCFBARTEstimator(BaseEstimator):
    """BCF/BART CATE estimator with rpart summary tree (Hahn et al. 2020).

    Returns highest-leaf ATE, lowest-leaf ATE, and the posterior contrast
    between them — the estimand Hahn proposes as an alternative to
    EffectXShift's selected V vs V^c framing.
    """

    name = "bcf_bart"

    def __init__(
        self,
        outcome_col: str = "Y",
        covariate_cols: list[str] | None = None,
        nburn: int = 500,
        nsim: int = 500,
        min_leaf_n: int = 10,
    ):
        self._outcome_col    = outcome_col
        self._covariate_cols = covariate_cols or ["W1", "W2", "W3", "W4"]
        self._nburn          = nburn
        self._nsim           = nsim
        self._min_leaf_n     = min_leaf_n

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
    ) -> list[EstimatorResult]:
        if not _bcf_bart_available():
            warnings.warn(
                "Neither 'bcf' nor 'BART' R package available — skipping BCFBARTEstimator. "
                "Install with: install.packages('bcf')",
                stacklevel=2,
            )
            return []

        if self._outcome_col not in df.columns:
            warnings.warn(
                f"BCFBARTEstimator: outcome column '{self._outcome_col}' "
                "not in DataFrame — skipping",
                stacklevel=2,
            )
            return []

        missing = [c for c in self._covariate_cols if c not in df.columns]
        if missing:
            warnings.warn(
                f"BCFBARTEstimator: covariate columns {missing} missing — skipping",
                stacklevel=2,
            )
            return []

        import rpy2.robjects as ro
        import rpy2.robjects.pandas2ri as pandas2ri
        from rpy2.robjects.conversion import localconverter

        ro.r["source"](str(_R_BRIDGE))
        run_bridge = ro.globalenv["run_bcf_bart_bridge"]

        sub = df[[self._outcome_col, "A"] + self._covariate_cols].copy().dropna()
        if len(sub) < 50:
            warnings.warn(
                f"BCFBARTEstimator: only {len(sub)} complete rows — skipping",
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
                    ro.IntVector([self._nburn]),
                    ro.IntVector([self._nsim]),
                    ro.IntVector([self._min_leaf_n]),
                )

            converged = bool(np.array(result.rx2("converged"))[0])
            if not converged:
                warnings.warn(
                    "BCFBARTEstimator: summary tree degenerated (no splits)", stacklevel=2
                )
                return []

            high_ate      = float(np.array(result.rx2("high_leaf_ate"))[0])
            high_se       = float(np.array(result.rx2("high_leaf_se"))[0])
            low_ate       = float(np.array(result.rx2("low_leaf_ate"))[0])
            low_se        = float(np.array(result.rx2("low_leaf_se"))[0])
            contrast      = float(np.array(result.rx2("contrast"))[0])
            cont_se       = float(np.array(result.rx2("contrast_se"))[0])
            top_split_var = str(np.array(result.rx2("top_split_var"))[0])

        except Exception as exc:
            warnings.warn(f"BCFBARTEstimator bridge failed: {exc}", stacklevel=2)
            return []

        conv = {"top_split_var": top_split_var}

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
                _result("bcf_bart_high_leaf", high_ate, high_se, "HTE_high_leaf_ATE"),
                _result("bcf_bart_low_leaf",  low_ate,  low_se,  "HTE_low_leaf_ATE"),
                _result("bcf_bart_contrast",  contrast, cont_se, "HTE_high_minus_low_leaf"),
            ]
            if r is not None
        ]
