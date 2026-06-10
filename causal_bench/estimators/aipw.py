"""AIPW: Augmented IPW / doubly-robust EIF estimator."""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
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
        W = df[W_cols].values.astype(float)

        # --- Outcome model Q: E[Y | A, W] via SuperLearner (full data for point est.) ---
        X_AW = np.column_stack([A, W])
        X_1W = np.column_stack([np.ones(n), W])
        X_0W = np.column_stack([np.zeros(n), W])
        Q_sl = SuperLearner(task="regression", n_folds=self.n_folds,
                            random_state=self.random_state)
        Q_sl.fit(X_AW, Y)

        Q_A1 = np.clip(Q_sl.predict(X_1W), 0, 1)
        Q_A0 = np.clip(Q_sl.predict(X_0W), 0, 1)

        # --- Propensity model g: P(A=1|W) via SuperLearner ---
        g_sl = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        g_sl.fit(W, A)
        g = g_sl.predict_proba(W)
        # OOF g from SuperLearner's genuine out-of-fold predictions
        g_oof = g_sl.oof_predictions_

        # OOF Q: K-fold cross-fitted counterfactual predictions for unbiased IC variance.
        Q_A1_oof = np.zeros(n)
        Q_A0_oof = np.zeros(n)
        for tr, val in KFold(self.n_folds, shuffle=True,
                             random_state=self.random_state).split(X_AW):
            qc = RidgeCV().fit(X_AW[tr], Y[tr])
            Q_A1_oof[val] = np.clip(qc.predict(X_1W[val]), 0, 1)
            Q_A0_oof[val] = np.clip(qc.predict(X_0W[val]), 0, 1)

        # --- AIPW EIF: point estimate from full-data Q, SE from OOF IC ---
        point = float(np.mean(Q_A1 - Q_A0))
        IC = ((Q_A1 - Q_A0 - point)
              + (A / g_oof) * (Y - Q_A1_oof)
              - ((1 - A) / (1 - g_oof)) * (Y - Q_A0_oof))
        # Recenter to remove residual bias from OOF mismatch
        point = point + float(np.mean(IC))
        IC = IC - float(np.mean(IC))
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
            ic=IC,
        )]
