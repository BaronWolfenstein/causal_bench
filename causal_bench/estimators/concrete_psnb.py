"""Clinical PSNB/PSWR estimator — rpy2 bridge to concrete::clinicalPSNB().

clinicalPSNB replaces the implicit reach weights in the standard hierarchical
win ratio with a user-supplied charter vector, producing the
priority-standardized net benefit (PSNB = Σ_k α_k Δ_k) and win ratio
(PSWR) with IF-based CIs.

Uses the same illness-death mapping as ClinicalRMTIFEstimator:
  event_type==1 (primary, non-fatal) -> illness;
  event_type==2 (competing, fatal)   -> terminal / death-priority.

Returns two EstimatorResult objects per call: one for PSNB, one for PSWR.
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

_DEFAULT_CHARTER = (0.5, 0.5)  # equal weight: tier-1 illness, tier-2 death


class ClinicalPSNBEstimator(BaseEstimator):
    """Priority-standardized net benefit / win ratio via concrete's clinicalPSNB().

    charter controls the per-tier weights α (must sum to 1). The default
    (0.5, 0.5) assigns equal weight to the illness and death tiers; pass
    e.g. (0.3, 0.7) to weight death more heavily, matching the clinical
    charter used in the trial protocol.

    Returns two EstimatorResult objects: one for PSNB and one for PSWR.

    Scale (McCoy et al., arXiv:2607.22950, Def 1 / Eq 3): PSNB = Σ_k α_k Δ_k is a
    charter-weighted NET-BENEFIT (win minus loss) PROBABILITY imbalance, where
    Δ_k = E[c_k | R_k=1] with c_k ∈ {-1,0,+1}; it is dimensionless and bounded in
    [-1, 1] (a positive value favors treatment). It is NOT a time-scale quantity —
    "average extra time in a more-favorable state" is the RMT-IF estimand
    (ClinicalRMTIFEstimator), a different summary; only the illness-death event
    mapping is shared. PSWR = Σ_k α_k w_k / Σ_k α_k ℓ_k is the ratio-scale companion
    (unitless; PSWR>1 iff PSNB>0), interpreted like a win ratio but only within a
    fixed charter (not comparable across charters).

    Requires a competing-risks DGP (event_type in {0, 1, 2}).
    """

    def __init__(
        self,
        charter: tuple[float, ...] = _DEFAULT_CHARTER,
        horizon: float = 1.0,
        signif: float = 0.05,
    ):
        charter = tuple(float(w) for w in charter)
        # The illness-death mapping is 2-tier (death, illness); the R bridge hardcodes
        # n_tiers=2, so a longer charter would otherwise fail deep inside clinicalPSNB.
        if len(charter) != 2:
            raise ValueError(
                f"charter must have 2 weights for the illness-death mapping "
                f"(death, illness); got {len(charter)}"
            )
        if any(w < 0 for w in charter):
            raise ValueError(f"charter weights must be non-negative (got {charter})")
        if abs(sum(charter) - 1.0) > 1e-9:
            raise ValueError(
                f"charter weights must sum to 1 (got {sum(charter):.6f})"
            )
        self._charter = charter
        self._horizon = horizon
        self._signif = signif

    @property
    def name(self) -> str:
        return "clinical_PSNB"

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
    ) -> list[EstimatorResult]:
        if not _concrete_available():
            warnings.warn(
                "concrete R package not available — skipping ClinicalPSNBEstimator",
                stacklevel=2,
            )
            return []

        if "event_type" not in df.columns:
            warnings.warn(
                "ClinicalPSNBEstimator requires an event_type column (competing-risks DGP)",
                stacklevel=2,
            )
            return []

        import rpy2.robjects as ro
        import rpy2.robjects.pandas2ri as pandas2ri
        from rpy2.robjects.conversion import localconverter

        ro.r["source"](str(_R_BRIDGE))
        run_fn = ro.globalenv["run_clinical_psnb"]

        df_r = prepare_for_r(df.copy())

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r)

        try:
            r_result = run_fn(
                r_df,
                float(horizon),
                charter=ro.FloatVector(self._charter),
                signif=ro.FloatVector([self._signif]),
            )
        except Exception as exc:
            warnings.warn(f"{self.name}: R bridge failed — {exc}", stacklevel=2)
            return []

        def _scalar(key: str) -> float:
            try:
                return float(np.array(r_result.rx2(key))[0])
            except Exception:
                return float("nan")

        psnb       = _scalar("psnb")
        se_psnb    = _scalar("se_psnb")
        lo_psnb    = _scalar("ci_lower_psnb")
        hi_psnb    = _scalar("ci_upper_psnb")
        pswr       = _scalar("pswr")
        se_pswr    = _scalar("se_pswr")
        lo_pswr    = _scalar("ci_lower_pswr")
        hi_pswr    = _scalar("ci_upper_pswr")

        results = []
        if np.isfinite(psnb) and np.isfinite(se_psnb) and se_psnb > 0:
            results.append(EstimatorResult(
                name=f"{self.name}_PSNB",
                estimand=estimand,
                point_estimate=psnb,
                standard_error=se_psnb,
                ci_lower=lo_psnb,
                ci_upper=hi_psnb,
            ))
        else:
            warnings.warn(f"{self.name}: non-finite PSNB result", stacklevel=2)

        if np.isfinite(pswr) and np.isfinite(se_pswr) and se_pswr > 0:
            results.append(EstimatorResult(
                name=f"{self.name}_PSWR",
                estimand=estimand,
                point_estimate=pswr,
                standard_error=se_pswr,
                ci_lower=lo_pswr,
                ci_upper=hi_pswr,
            ))
        else:
            warnings.warn(f"{self.name}: non-finite PSWR result", stacklevel=2)

        return results
