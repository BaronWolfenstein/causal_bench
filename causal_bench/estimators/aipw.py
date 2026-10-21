"""AIPW: Augmented IPW / doubly-robust EIF estimator."""
import numpy as np
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner
from causal_bench.estimators._ipcw_rd import censoring_survival_fn, ipcw_aipw_ic


class AIPWEstimator(BaseEstimator):
    """Augmented IPW (doubly-robust EIF estimator).

    No targeting step — uses plug-in EIF directly.
    Doubly robust: consistent if Q or g is correctly specified.
    """

    name = "AIPW"

    def __init__(self, n_folds=5, random_state=42, ipcw=False):
        # ipcw=False keeps the documented "no censoring model" contract (exp8/exp13 use AIPW as the censoring-naive
        # contrast to TMLE+IPCW); ipcw=True adds a Cox censoring model (see _ipcw_rd).
        self.n_folds = n_folds
        self.random_state = random_state
        self.ipcw = ipcw

    def estimate(self, df, horizon=1.0, estimand="ATE"):
        # Audit fix 2026-10-21 (see _ipcw_rd): ONE cross-fitted SuperLearner Q in both the plug-in and residual terms
        # (was SL plug-in + RidgeCV residuals, so not doubly robust); propensity truncated at [0.01, 0.99].
        W_cols = ["W1", "W2", "W3", "W4"]
        n = len(df)
        A = df["A"].values.astype(float)
        W = df[W_cols].values.astype(float)

        g_sl = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        g_sl.fit(W, A)
        g_oof = g_sl.oof_predictions_            # genuine out-of-fold propensity

        Gfn = censoring_survival_fn(df, W, horizon) if self.ipcw else None
        IC = ipcw_aipw_ic(df, W, A, g_oof, horizon, Gfn, n_folds=self.n_folds,
                          random_state=self.random_state, crossfit=True, horizon=horizon)
        point = float(np.mean(IC))
        IC = IC - point
        se = float(np.std(IC, ddof=1) / np.sqrt(n))

        return [EstimatorResult(
            name="AIPW",
            estimand=estimand,
            point_estimate=point,
            standard_error=se,
            ci_lower=float(point - 1.96 * se),
            ci_upper=float(point + 1.96 * se),
            ic=IC,
        )]
