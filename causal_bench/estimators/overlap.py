"""Overlap weighting estimator (Li, Morgan & Zaslavsky 2018)."""
import numpy as np
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner


class OverlapEstimator(BaseEstimator):
    """Overlap weighting (ATO estimand).

    Weights: h(x) = g(x)*(1-g(x)) — maximally upweights overlap region.
    Targets Average Treatment effect in the Overlap population (ATO), not ATE.
    """

    name = "Overlap"

    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state

    def estimate(self, df, horizon=1.0, estimand="ATE"):
        W_cols = ["W1", "W2", "W3", "W4"]
        n = len(df)
        A = df["A"].values
        Y = ((df["T_obs"] <= horizon) & (df["Delta"] == 1)).astype(float).values

        # Propensity score
        g_sl = SuperLearner(
            task="classification",
            n_folds=self.n_folds,
            random_state=self.random_state,
        )
        g_sl.fit(df[W_cols].values, A)
        g = g_sl.predict_proba(df[W_cols].values)

        # Overlap weights h(x) = g(x)*(1-g(x))
        h = g * (1 - g)

        # Weighted means
        w1 = h * A
        w0 = h * (1 - A)
        mu1 = np.sum(w1 * Y) / np.sum(w1)
        mu0 = np.sum(w0 * Y) / np.sum(w0)
        point = mu1 - mu0

        # Influence function SE (Mao, Li & Greene 2019)
        # tau_i = h_i*(A_i*(Y_i - mu1)/sum(w1) - (1-A_i)*(Y_i - mu0)/sum(w0))
        psi1 = w1 * (Y - mu1) / w1.mean()
        psi0 = w0 * (Y - mu0) / w0.mean()
        IC = psi1 - psi0
        se = float(np.std(IC, ddof=1) / np.sqrt(n))

        ci_lower = point - 1.96 * se
        ci_upper = point + 1.96 * se

        # Note: ATO != ATE under heterogeneous effects; report estimand as passed
        return [EstimatorResult(
            name="Overlap",
            estimand=estimand,
            point_estimate=float(point),
            standard_error=se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        )]
