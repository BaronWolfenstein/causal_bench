"""AIPW: Augmented IPW / doubly-robust EIF estimator."""
import numpy as np
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner


class AIPWEstimator(BaseEstimator):
    """Augmented IPW (doubly-robust EIF estimator).

    No targeting step — uses plug-in EIF directly.
    Doubly robust: consistent if Q or g is correctly specified.
    """

    name = "AIPW"

    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state

    def estimate(self, df, horizon=1.0, estimand="ATE"):
        W_cols = ["W1", "W2", "W3", "W4"]
        n = len(df)
        A = df["A"].values
        Y = ((df["T_obs"] <= horizon) & (df["Delta"] == 1)).astype(float).values

        # --- Outcome model Q: E[Y | A, W] via SuperLearner ---
        X_AW = np.column_stack([A, df[W_cols].values])
        Q_sl = SuperLearner(task="regression", n_folds=self.n_folds,
                            random_state=self.random_state)
        Q_sl.fit(X_AW, Y)

        X_1W = np.column_stack([np.ones(n), df[W_cols].values])
        X_0W = np.column_stack([np.zeros(n), df[W_cols].values])
        Q_A1 = np.clip(Q_sl.predict(X_1W), 0, 1)
        Q_A0 = np.clip(Q_sl.predict(X_0W), 0, 1)
        Q_AW = np.clip(Q_sl.predict(X_AW), 0, 1)  # noqa: F841

        # --- Propensity model g: P(A=1|W) via SuperLearner ---
        g_sl = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        g_sl.fit(df[W_cols].values, A)
        g = g_sl.predict_proba(df[W_cols].values)

        # --- AIPW EIF (plug-in, no targeting) ---
        # psi_i = (Q_A1 - Q_A0) + A/g*(Y - Q_A1) - (1-A)/(1-g)*(Y - Q_A0)
        augment = (A / g) * (Y - Q_A1) - ((1 - A) / (1 - g)) * (Y - Q_A0)
        IC = (Q_A1 - Q_A0) + augment
        point = float(np.mean(IC))
        se = float(np.std(IC, ddof=1) / np.sqrt(n))

        ci_lower = point - 1.96 * se
        ci_upper = point + 1.96 * se

        return [EstimatorResult(
            name="AIPW",
            estimand=estimand,
            point_estimate=point,
            standard_error=se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        )]
