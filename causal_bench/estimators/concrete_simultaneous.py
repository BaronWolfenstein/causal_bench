"""Simultaneous confidence bands across the estimand family via concrete PR #31.

Calls run_concrete_simultaneous() in the R bridge, which runs doConcrete
once with all requested horizons, then calls getOutput(Simultaneous=TRUE).
getOutput internally calls getSimultaneous(), which:
  1. Stacks per-subject IC vectors from all arms and horizons into an n×q matrix
  2. Estimates the joint correlation R = cor(IC_matrix)
  3. Draws the simultaneous critical value q = quantile(max_j|Z_j|, 1-alpha)
     via 1000 Gaussian-multiplier samples from N(0, R)
  4. Returns SimCI Low/Hi = Pt Est ± se × q for each RD estimand

Returns one EstimatorResult per estimand (RD_t<t>, RMST_t<t>).
RD results carry simultaneous CIs in convergence_info["sim_ci_lo/hi"].
The n×q IC matrix is stored in convergence_info["ic_matrix"] of the first
result so Experiment 12 can run its own Gaussian-multiplier bootstrap for
non-concrete estimators.
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


class ConcreteSimultaneousEstimator(BaseEstimator):
    """Multi-horizon TMLE with simultaneous confidence bands.

    Parameters
    ----------
    horizons : tuple[float, ...]
        Target times for the RD and RMST estimands.  Passed as TargetTime to
        concrete's formatArguments so all horizons are fit in one doConcrete
        call — required for getSimultaneous to see the full stacked IC matrix.
    signif : float
        Significance level for simultaneous bands (default 0.05 → 95% bands).
    """

    def __init__(self, horizons: tuple[float, ...] = (1.0, 2.0), signif: float = 0.05):
        self.horizons = tuple(horizons)
        self.signif = signif

    @property
    def name(self) -> str:
        h_str = "_".join(f"{h:g}" for h in self.horizons)
        return f"concrete_simult_t{h_str}"

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
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
        run_fn = ro.globalenv["run_concrete_simultaneous"]

        df_r = df.copy()
        df_r["event_type"] = df_r["Delta"].astype(int)
        df_r = prepare_for_r(df_r)

        r_horizons = ro.FloatVector(list(self.horizons))

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r)

        try:
            r_result = run_fn(r_df, r_horizons, signif=ro.FloatVector([self.signif]))
        except Exception as exc:
            warnings.warn(f"{self.name}: R bridge failed — {exc}", stacklevel=2)
            return []

        return self._parse_r_result(r_result)

    def _parse_r_result(self, r_result) -> list[EstimatorResult]:
        import rpy2.robjects as ro
        import rpy2.robjects.pandas2ri as pandas2ri
        from rpy2.robjects.conversion import localconverter

        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                results_df = ro.conversion.rpy2py(r_result.rx2("results"))
        except Exception:
            return []

        if results_df is None or len(results_df) == 0:
            return []

        # Extract IC matrix (n × q numpy array) and keep column names for indexing
        ic_matrix: np.ndarray | None = None
        ic_col_names: list[str] = []
        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                ic_df = ro.conversion.rpy2py(r_result.rx2("ic_matrix"))
            if ic_df is not None and len(ic_df.columns) > 0:
                ic_matrix = ic_df.to_numpy(dtype=float)
                ic_col_names = list(ic_df.columns)
        except Exception:
            pass

        # Simultaneous critical value (shared across RD horizons)
        sim_q: float | None = None
        try:
            sim_q = float(r_result.rx2("sim_q")[0])
            if not np.isfinite(sim_q):
                sim_q = None
        except Exception:
            pass

        # TMLE diagnostics
        tmle_diag: dict | None = None
        try:
            with localconverter(ro.default_converter + pandas2ri.converter):
                diag_df = ro.conversion.rpy2py(r_result.rx2("tmle_diag"))
            if diag_df is not None and len(diag_df) > 0:
                tmle_diag = diag_df.to_dict(orient="list")
        except Exception:
            pass

        out: list[EstimatorResult] = []
        first = True
        for _, row in results_df.iterrows():
            pt      = float(row.get("point", np.nan))
            se      = float(row.get("se",    np.nan))
            est_lbl = str(row.get("estimand", f"est_{_}"))

            if not (np.isfinite(pt) and np.isfinite(se) and se > 0):
                continue

            conv_info: dict = {
                "estimand_label": est_lbl,
                "horizon": float(row.get("horizon", np.nan)),
            }

            # Simultaneous CI (populated for RD estimands; NaN for RMST)
            s_lo = float(row.get("sim_ci_lo", np.nan))
            s_hi = float(row.get("sim_ci_hi", np.nan))
            if np.isfinite(s_lo) and np.isfinite(s_hi):
                conv_info["sim_ci_lo"] = s_lo
                conv_info["sim_ci_hi"] = s_hi
            if sim_q is not None:
                conv_info["sim_q"] = sim_q

            # Attach IC matrix to first valid result for Exp 12 bootstrap
            if first and ic_matrix is not None:
                conv_info["ic_matrix"] = ic_matrix
                first = False
            if tmle_diag is not None:
                conv_info["tmle_diag"] = tmle_diag

            # Per-subject IC column for RD estimands only (ic_matrix contains IC_RD).
            # RMST estimands have a different IC structure not captured here.
            ic_vec: np.ndarray | None = None
            if ic_matrix is not None and ic_col_names and est_lbl.startswith("RD_t"):
                horizon_tag = est_lbl.split("_t")[-1]  # e.g. "0.4" from "RD_t0.4"
                matches = [j for j, c in enumerate(ic_col_names) if horizon_tag in c]
                if matches:
                    ic_vec = ic_matrix[:, matches[0]]

            out.append(EstimatorResult(
                name             = self.name,
                estimand         = est_lbl,
                point_estimate   = pt,
                standard_error   = se,
                ci_lower         = float(row.get("ci_lo", pt - 1.96 * se)),
                ci_upper         = float(row.get("ci_hi", pt + 1.96 * se)),
                convergence_info = conv_info,
                ic               = ic_vec,
            ))

        return out
