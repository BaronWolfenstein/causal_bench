# causal_bench/estimators/tmle_ipcw.py
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy import stats
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner


class TMLEIPCWEstimator(BaseEstimator):

    def __init__(self, use_compliance: bool = False, n_folds: int = 5,
                 random_state: int = 42):
        self.use_compliance = use_compliance
        self.n_folds = n_folds
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "TMLE+IPCW+Comply" if self.use_compliance else "TMLE+IPCW"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        W_cols = ["W1", "W2", "W3", "W4"]
        A = df["A"].values.astype(float)
        T_obs = df["T_obs"].values
        Delta = df["Delta"].values.astype(float)
        W = df[W_cols].values.astype(float)
        n = len(A)

        Y = ((T_obs <= horizon) & (Delta == 1)).astype(float)

        # ── Step 1: Censoring model ──
        censor_feature_cols = W_cols + ["A"]
        if self.use_compliance and "compliance" in df.columns:
            censor_feature_cols = censor_feature_cols + ["compliance"]

        censor_df = df[censor_feature_cols + ["T_obs", "Delta"]].copy()
        censor_df = censor_df.rename(columns={"Delta": "event_obs"})
        censor_df["C_indicator"] = 1.0 - censor_df["event_obs"]

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(censor_df[censor_feature_cols + ["T_obs", "C_indicator"]],
                    duration_col="T_obs", event_col="C_indicator",
                    fit_options={"max_steps": 50})

            unique_times = np.sort(np.unique(T_obs))
            sf = cph.predict_survival_function(
                censor_df[censor_feature_cols], times=unique_times
            )
            # For each patient i, look up G(C > T_obs_i | covariates_i)
            G = np.ones(n)
            for i, t in enumerate(T_obs):
                col = sf.iloc[:, i]
                idx_before = sf.index <= t
                if idx_before.any():
                    G[i] = float(col[idx_before].iloc[-1])
                else:
                    G[i] = 1.0
        except Exception:
            G = np.ones(n)

        G = np.clip(G, 0.05, 1.0)
        ipcw = Delta / G

        # ── Step 2: Propensity model ──
        sl_g = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        sl_g.fit(W, A)
        g = sl_g.predict_proba(W)

        # ── Step 3: Outcome model (IPCW-weighted logistic) ──
        AW = np.column_stack([A, W])
        AW1 = np.column_stack([np.ones(n), W])
        AW0 = np.column_stack([np.zeros(n), W])

        q_model = LogisticRegression(max_iter=1000, C=1.0)
        sample_weights = ipcw.copy()
        sample_weights = sample_weights / sample_weights.mean()  # normalize
        q_model.fit(AW, Y, sample_weight=sample_weights)

        Q_AW = np.clip(expit(q_model.decision_function(AW)), 1e-5, 1 - 1e-5)
        Q_1W = np.clip(expit(q_model.decision_function(AW1)), 1e-5, 1 - 1e-5)
        Q_0W = np.clip(expit(q_model.decision_function(AW0)), 1e-5, 1 - 1e-5)

        # ── Steps 4-5: Targeting + EIF SE ──
        results = []
        estimands_to_run = ["ATE", "ATT"] if estimand == "ATT" else ["ATE"]
        for est in estimands_to_run:
            point, se = self._target_and_se(Y, A, g, Q_AW, Q_1W, Q_0W, ipcw, est, n)
            z = stats.norm.ppf(0.975)
            results.append(EstimatorResult(
                name=self.name, estimand=est,
                point_estimate=float(point), standard_error=float(se),
                ci_lower=float(point - z * se), ci_upper=float(point + z * se),
            ))
        return results

    def _target_and_se(self, Y, A, g, Q_AW, Q_1W, Q_0W, ipcw, estimand, n):
        if estimand == "ATE":
            H = ipcw * (A / g - (1 - A) / (1 - g))
            H1 = 1.0 / g
            H0 = -1.0 / (1 - g)
        else:  # ATT
            H = ipcw * (A - (1 - A) * g / (1 - g))
            H1 = np.ones(n)
            H0 = -g / (1 - g)

        # One-step targeting: epsilon via Newton update
        denom = np.mean(H ** 2)
        eps = np.mean(H * (Y - Q_AW)) / denom if denom > 1e-10 else 0.0
        # Clip eps to prevent runaway updates
        eps = np.clip(eps, -2.0, 2.0)

        Q1_star = expit(logit(Q_1W) + eps * H1)
        Q0_star = expit(logit(Q_0W) + eps * H0)

        if estimand == "ATE":
            point = np.mean(Q1_star - Q0_star)
            IC = ((Q1_star - Q0_star - point)
                  + ipcw * (A / g) * (Y - Q1_star)
                  - ipcw * ((1 - A) / (1 - g)) * (Y - Q0_star))
        else:
            pi = np.mean(A)
            if pi < 1e-8:
                return float("nan"), float("nan")
            point = np.mean(A * (Q1_star - Q0_star)) / pi
            IC = (A * (Q1_star - Q0_star) / pi - point
                  + ipcw * A / pi * (Y - Q1_star)
                  - ipcw * (1 - A) * g / (1 - g) / pi * (Y - Q0_star))

        se = np.sqrt(np.var(IC, ddof=1) / n)
        return point, se
