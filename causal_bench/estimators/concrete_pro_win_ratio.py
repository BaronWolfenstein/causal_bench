"""concrete PRO win ratio bridge — clinicalWinRatio() with pro= tiers.

Targets concrete PR #35 (merged 2026-06-19) + PR #36 (open as of 2026-06-22).

PR #35 added the `pro` argument: continuous/ordinal PRO tiers appended at the
bottom of the win hierarchy, compared with a reach-weighted IPCW-corrected CDF.

PR #36 commits (all pending merge):
  80ab8af — crossover IPCW: `clinicalWinRatio()`/`clinicalPSNB()` gain `crossover=`
            for the hypothetical no-switching estimand (IPCW = 1/(S_dropout × S_crossover)).
            GPC rewrite: PRO block is a sequential GPC, not bilinear; `landmark` must
            equal `horizon`; `n.grid` dropped; P(win)+P(loss)≤1 enforced by rescaling.
  63e7027 — robustness: `min.cens.surv` exposed on `clinicalWinRatio()`/`clinicalPSNB()`/
            `clinicalRMTIF()` (default 0.05); `.tvCensLaggedSurv()` now uses the analyst's
            SL library instead of fixed defaults. Matters with heavy crossover (EVOQUE 49%).
  9f330b81 — audit round 2: log-scale RR CIs fixed in `addWaldInference()`; `clinicalRMTIF()`
             gains `crossover=` and `min.cens.surv=`; `clinicalPSNB()` reach guard; etc.
             CI parsing unchanged for WinRatio (was already log-scale). Transparent to bridge.

Python-side wiring (crossover_col, min_cens_surv accepted and passed to R) is live.
R-side calls are scaffolded but inactive — TODO comments at each clinicalWinRatio()/
clinicalPSNB()/clinicalRMTIF() call site; uncomment when #36 merges.

Each PRO spec is a dict with keys:
  marker     (required) column name in the DataFrame
  landmark   float, must equal horizon (post-#36 constraint)
  margin     float win margin δ (default 0 = any difference is a win)
  direction  "higher.better" (default) or "lower.better"
  type       "continuous" (default) or "ordinal"

Gracefully returns [] if rpy2 or concrete is unavailable.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_bench.estimators.base import BaseEstimator
from causal_bench.estimators.concrete_rmst import _concrete_available, prepare_for_r
from causal_bench.metrics import EstimatorResult

_R_BRIDGE = Path(__file__).parent.parent.parent / "r_scripts" / "concrete_bridge.R"


class ConcretePROWinRatioEstimator(BaseEstimator):
    """Win ratio estimator with PRO tiers via concrete clinicalWinRatio(pro=...).

    Wraps the post-PR-#35 clinicalWinRatio() signature. Hard-event tiers
    (death, hospitalisations) are specified via the existing illness_time /
    terminal_time columns; PRO tiers are appended at the bottom via pro_specs.

    Parameters
    ----------
    pro_specs : list of dicts, each with:
        marker      — DataFrame column name for the PRO value at landmark
        landmark    — float, defaults to horizon
        margin      — float win margin (default 0)
        direction   — "higher.better" | "lower.better" (default "higher.better")
        type        — "continuous" | "ordinal" (default "continuous")
    horizon : float, evaluation horizon (years). None → max observed time.
    illness_time_cols : list of non-terminal event time columns (e.g. HF hosp).
        If None, uses ["T_illness"] when present, else death-only hierarchy.
    terminal_time_col : terminal event time column (default "T_obs").
    terminal_status_col : 0/1 death indicator column (default "Delta").
    covariate_cols : covariate columns for nuisance models (default W1–W4).
    """

    def __init__(
        self,
        pro_specs: list[dict[str, Any]],
        horizon: float | None = 1.0,
        illness_time_cols: list[str] | None = None,
        terminal_time_col: str = "T_obs",
        terminal_status_col: str = "Delta",
        covariate_cols: list[str] | None = None,
        crossover_col: str | None = None,
        min_cens_surv: float = 0.05,
    ):
        if not pro_specs:
            raise ValueError("pro_specs must be a non-empty list of PRO tier dicts")
        self._pro_specs = pro_specs
        self._horizon = horizon
        self._illness_time_cols = illness_time_cols
        self._terminal_time_col = terminal_time_col
        self._terminal_status_col = terminal_status_col
        self._covariate_cols = covariate_cols or ["W1", "W2", "W3", "W4"]
        self._crossover_col = crossover_col
        self._min_cens_surv = min_cens_surv

    @property
    def name(self) -> str:
        markers = "+".join(s["marker"] for s in self._pro_specs)
        suffix = f",xover={self._crossover_col}" if self._crossover_col else ""
        return f"concrete_PRO_WR[{markers}{suffix}]"

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
        run_fn = ro.globalenv["run_concrete_pro_win_ratio"]

        use_horizon = self._horizon if self._horizon is not None else horizon

        # Post-#36: landmark must equal horizon (final-visit design).
        for s in self._pro_specs:
            lm = s.get("landmark", use_horizon)
            if abs(float(lm) - use_horizon) > 1e-9:
                warnings.warn(
                    f"{self.name}: PRO spec landmark={lm} != horizon={use_horizon}. "
                    "concrete PR #36 requires landmark == horizon (final-visit design). "
                    "This will error when #36 is installed.",
                    UserWarning,
                    stacklevel=2,
                )

        # Determine illness-time columns
        illness_cols = self._illness_time_cols
        if illness_cols is None:
            illness_cols = ["T_illness"] if "T_illness" in df.columns else []

        df_r = prepare_for_r(df.copy())
        df_r[self._terminal_status_col] = df_r[self._terminal_status_col].astype(int)

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r)

        # Build R list of PRO spec lists
        r_pro_specs = ro.ListVector([
            ro.ListVector({
                "marker":    ro.StrVector([s["marker"]]),
                "landmark":  ro.FloatVector([float(s.get("landmark", use_horizon))]),
                "margin":    ro.FloatVector([float(s.get("margin", 0.0))]),
                "direction": ro.StrVector([s.get("direction", "higher.better")]),
                "type":      ro.StrVector([s.get("type", "continuous")]),
            })
            for s in self._pro_specs
        ])

        r_illness      = ro.StrVector(illness_cols) if illness_cols else ro.rinterface.NULL
        r_covars       = ro.StrVector(self._covariate_cols)
        r_crossover    = (ro.StrVector([self._crossover_col])
                          if self._crossover_col else ro.rinterface.NULL)
        r_min_cens_surv = ro.FloatVector([self._min_cens_surv])

        try:
            result = run_fn(
                r_df,
                ro.FloatVector([use_horizon]),
                r_illness,
                ro.StrVector([self._terminal_time_col]),
                ro.StrVector([self._terminal_status_col]),
                r_covars,
                r_pro_specs,
                r_crossover,
                r_min_cens_surv,
            )
        except Exception as exc:
            warnings.warn(f"{self.name}: R call failed — {exc}", stacklevel=2)
            return []

        try:
            wr    = float(result.rx2("WR")[0])
            se    = float(result.rx2("SE")[0])
            lo    = float(result.rx2("CI_lower")[0])
            hi    = float(result.rx2("CI_upper")[0])
            n_tiers = int(result.rx2("n_tiers")[0])
        except Exception as exc:
            warnings.warn(f"{self.name}: result parsing failed — {exc}", stacklevel=2)
            return []

        return [
            EstimatorResult(
                estimator=self.name,
                estimand="WR",
                estimate=wr,
                se=se,
                ci_lower=lo,
                ci_upper=hi,
                extra={"n_tiers": n_tiers, "pro_markers": [s["marker"] for s in self._pro_specs]},
            )
        ]
