"""Pointwise RMST estimator: integrate AIPW risk differences over K time points.

Demonstrates discretization bias vs direct RMST targeting (concrete).
Bias is O(1/K): large at K=2, small at K=20.
"""
import numpy as np
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner


class PointwiseRMSTEstimator(BaseEstimator):
    """RMST via trapezoid integration of pointwise AIPW estimates.

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

        for t_k in t_grid:
            # Binary outcome at t_k: event occurred by t_k
            Y_k = ((df["T_obs"] <= t_k) & (df["Delta"] == 1)).astype(float).values

            # Outcome model at t_k
            X_AW = np.column_stack([A, df[W_cols].values])
            Q_sl = SuperLearner(task="regression", n_folds=self.n_folds,
                                random_state=self.random_state)
            Q_sl.fit(X_AW, Y_k)

            X_1W = np.column_stack([np.ones(n),  df[W_cols].values])
            X_0W = np.column_stack([np.zeros(n), df[W_cols].values])
            Q1 = np.clip(Q_sl.predict(X_1W), 0, 1)
            Q0 = np.clip(Q_sl.predict(X_0W), 0, 1)
            Q_AW = np.clip(Q_sl.predict(X_AW), 0, 1)

            # AIPW EIF for RD at t_k: F_1(t_k) - F_0(t_k)
            augment = (A / g) * (Y_k - Q1) - ((1 - A) / (1 - g)) * (Y_k - Q0)
            IC_k = (Q1 - Q0) + augment

            # RMST contribution: -RD(t_k) * delta_t  [since RMST_diff = integral (F_0-F_1) dt]
            # Note: RD = F_1 - F_0, so RMST_diff += (F_0 - F_1)*delta_t = -RD*delta_t
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
