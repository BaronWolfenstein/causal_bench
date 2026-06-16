"""IPW estimator: Horvitz-Thompson with weight truncation."""
import numpy as np
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner


class IPWEstimator(BaseEstimator):
    """Inverse probability weighting (Horvitz-Thompson)."""

    name = "IPW"

    def __init__(self, n_folds=5, random_state=42, trim_quantile=0.01):
        self.n_folds = n_folds
        self.random_state = random_state
        self.trim_quantile = trim_quantile  # trim top/bottom trim_quantile of weights

    def estimate(self, df, horizon=1.0, estimand="ATE"):
        W_cols = ["W1", "W2", "W3", "W4"]
        n = len(df)
        A = df["A"].values
        Y = ((df["T_obs"] <= horizon) & (df["Delta"] == 1)).astype(float).values

        # Propensity score via SuperLearner
        g_sl = SuperLearner(
            task="classification",
            n_folds=self.n_folds,
            random_state=self.random_state,
        )
        g_sl.fit(df[W_cols].values, A)
        g = g_sl.predict_proba(df[W_cols].values)  # P(A=1|W)

        # Horvitz-Thompson weights
        w1 = A / g           # treated arm weight
        w0 = (1 - A) / (1 - g)  # control arm weight

        # Stabilize: multiply by marginal P(A)
        p_A = A.mean()
        w1 *= p_A
        w0 *= (1 - p_A)

        # Truncate at trim_quantile / 1-trim_quantile of combined weights
        all_w = np.concatenate([w1[A == 1], w0[A == 0]])
        lo = np.quantile(all_w, self.trim_quantile)
        hi = np.quantile(all_w, 1 - self.trim_quantile)
        w1 = np.clip(w1, lo, hi)
        w0 = np.clip(w0, lo, hi)

        # IPW point estimate
        mu1 = np.sum(w1 * Y) / np.sum(w1)
        mu0 = np.sum(w0 * Y) / np.sum(w0)
        point = mu1 - mu0

        # Sandwich SE via influence function
        # IC_i = w1_i*Y_i/sum(w1) - w0_i*Y_i/sum(w0) - point/n  ... simplified
        # Use direct Hajek estimator IC:
        #   psi1_i = w1_i*(Y_i - mu1) / mean(w1)
        #   psi0_i = w0_i*(Y_i - mu0) / mean(w0)
        psi1 = w1 * (Y - mu1) / w1.mean()
        psi0 = w0 * (Y - mu0) / w0.mean()
        IC = psi1 - psi0
        se = float(np.std(IC, ddof=1) / np.sqrt(n))

        ci_lower = point - 1.96 * se
        ci_upper = point + 1.96 * se

        return [EstimatorResult(
            name="IPW",
            estimand=estimand,
            point_estimate=float(point),
            standard_error=se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            ic=IC,
        )]
