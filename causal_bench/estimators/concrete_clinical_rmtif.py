"""Clinical RMT-IF estimator — rpy2 bridge to concrete::clinicalRMTIF().

clinicalRMTIF uses the multistate / illness-death engine with an analytic
adjoint-value EIF — a different estimator implementation from getRMTIF()
(which post-processes a doConcrete fit). It takes raw illness/terminal event
times directly rather than a ConcreteEst object, so it needs its own R bridge
function (run_clinical_rmtif in r_scripts/concrete_bridge.R) and its own
subject-id bookkeeping when joined into Experiment 12's estimand family.

Maps causal_bench's competing_risks DGP (event_type in {0,1,2}) to the
illness-death structure with death-priority favorability: event_type==1
(primary, non-fatal) -> illness; event_type==2 (competing, fatal) -> terminal.
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


class ClinicalRMTIFEstimator(BaseEstimator):
    """RMT-IF via concrete's clinicalRMTIF() multistate engine.

    Requires a competing-risks DGP (event_type in {0, 1, 2}); degenerates to
    a standard death-only analysis if event_type==2 never occurs.
    """

    name = "clinical_RMTIF"

    def __init__(self, horizon: float = 1.0, signif: float = 0.05):
        self._horizon = horizon
        self.signif = signif

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
    ) -> list[EstimatorResult]:
        if not _concrete_available():
            warnings.warn(
                "concrete R package not available — skipping ClinicalRMTIFEstimator",
                stacklevel=2,
            )
            return []

        if "event_type" not in df.columns:
            warnings.warn(
                "ClinicalRMTIFEstimator requires an event_type column (competing-risks DGP)",
                stacklevel=2,
            )
            return []

        import rpy2.robjects as ro
        import rpy2.robjects.pandas2ri as pandas2ri
        from rpy2.robjects.conversion import localconverter

        ro.r["source"](str(_R_BRIDGE))
        run_fn = ro.globalenv["run_clinical_rmtif"]

        df_r = prepare_for_r(df.copy())

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r)

        try:
            r_result = run_fn(r_df, float(horizon), signif=ro.FloatVector([self.signif]))
        except Exception as exc:
            warnings.warn(f"{self.name}: R bridge failed — {exc}", stacklevel=2)
            return []

        try:
            point = float(np.array(r_result.rx2("point"))[0])
            se    = float(np.array(r_result.rx2("se"))[0])
            ci_lo = float(np.array(r_result.rx2("ci_lower"))[0])
            ci_hi = float(np.array(r_result.rx2("ci_upper"))[0])
        except Exception as exc:
            warnings.warn(f"{self.name}: failed to parse R result — {exc}", stacklevel=2)
            return []

        if not (np.isfinite(point) and np.isfinite(se) and se > 0):
            return []

        return [EstimatorResult(
            name=self.name,
            estimand=estimand,
            point_estimate=point,
            standard_error=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        )]
