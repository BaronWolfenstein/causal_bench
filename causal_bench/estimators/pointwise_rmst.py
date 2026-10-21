"""Pointwise RMST estimator: integrate AIPW risk differences over K time points.

Demonstrates discretization bias vs direct RMST targeting (concrete).
Bias is O(1/K): large at K=2, small at K=20 (right-endpoint Riemann sum).
Audit fix 2026-10-21: the pointwise risk differences are now IPCW-AIPW (see _ipcw_rd). Before, patients censored
before t_k were counted as event-free, a censoring bias that does NOT shrink with K and so contaminated the
discretization-bias demonstration.
"""
import numpy as np
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner
from causal_bench.estimators._ipcw_rd import censoring_survival_fn, ipcw_aipw_ic


class PointwiseRMSTEstimator(BaseEstimator):
    """RMST via right-endpoint Riemann integration of pointwise IPCW-AIPW estimates.

    Parameters
    ----------
    n_grid : int
        Number of evenly-spaced time points in (0, horizon]. Bias ~ O(1/n_grid).
    n_folds : int
        K-fold cross-fitting for SuperLearner.
    random_state : int
    """

    def __init__(self, n_grid=10, n_folds=5, random_state=42):
        self.n_grid = n_grid
        self.n_folds = n_folds
        self.random_state = random_state

    @property
    def name(self):
        return f"RMST_K{self.n_grid}"

    def estimate(self, df, horizon=1.0, estimand="ATE"):
        W_cols = ["W1", "W2", "W3", "W4"]
        n = len(df)
        A = df["A"].values

        # Propensity model — fit once, reuse across all time points
        g_sl = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        g_sl.fit(df[W_cols].values, A)
        g = g_sl.predict_proba(df[W_cols].values)   # P(A=1|W)

        # Time grid: K points evenly spaced in (0, horizon]
        # Use linspace(0, horizon, n_grid+1)[1:] so first point != 0
        t_grid = np.linspace(0, horizon, self.n_grid + 1)[1:]
        delta_t = t_grid[0]  # uniform spacing

        # Accumulate integrated EIF across time points
        IC_integrated = np.zeros(n)
        rmst_diff = 0.0

        Wm = df[W_cols].values.astype(float)
        Gfn = censoring_survival_fn(df, Wm, horizon)
        for t_k in t_grid:
            # IPCW-AIPW IC for RD(t_k) = F_1(t_k) - F_0(t_k) (same full-data SuperLearner Q style as before)
            IC_k = ipcw_aipw_ic(df, Wm, A.astype(float), g, t_k, Gfn, n_folds=self.n_folds,
                                random_state=self.random_state, crossfit=False, horizon=horizon)
            # RMST_diff = integral (F_0 - F_1) dt  ⇒  += -RD(t_k) * delta_t
            IC_integrated += -IC_k * delta_t
            rmst_diff    += -np.mean(IC_k) * delta_t

        se = float(np.std(IC_integrated, ddof=1) / np.sqrt(n))
        ci_lower = rmst_diff - 1.96 * se
        ci_upper = rmst_diff + 1.96 * se

        return [EstimatorResult(
            name=self.name,
            estimand=estimand,
            point_estimate=float(rmst_diff),
            standard_error=se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        )]
