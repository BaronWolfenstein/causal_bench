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
        # Only pre-horizon dropouts are informative censoring events.
        # Admin-censored (T_obs >= horizon) ran out of follow-up, not informatively censored.
        censor_df["C_indicator"] = (
            (censor_df["event_obs"] == 0) & (censor_df["T_obs"] < horizon - 1e-9)
        ).astype(float)

        G = self._fit_G(censor_df, censor_feature_cols, T_obs, n)
        # Events: upweight by 1/G to represent censored patients with similar patterns.
        # Admin-censored (T_obs >= horizon): Y=0 is known; ipcw=1 (no upweighting needed).
        # Pre-horizon dropouts: ipcw=0 (outcome unknown; excluded from outcome model).
        admin_censored = (Delta == 0) & (T_obs >= horizon - 1e-9)
        ipcw = np.where(Delta == 1, 1.0 / G, np.where(admin_censored, 1.0, 0.0))

        # ── Step 2: Propensity model ──
        sl_g = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        sl_g.fit(W, A)
        g = sl_g.predict_proba(W)
        # OOF g: SuperLearner stores genuine out-of-fold predictions during fit.
        g_oof = sl_g.oof_predictions_

        # ── Step 3: Outcome model (IPCW-weighted logistic) ──
        AW = np.column_stack([A, W])
        AW1 = np.column_stack([np.ones(n), W])
        AW0 = np.column_stack([np.zeros(n), W])

        sample_weights = ipcw / max(ipcw.mean(), 1e-10)
        q_model = LogisticRegression(max_iter=1000, C=1.0)
        q_model.fit(AW, Y, sample_weight=sample_weights)

        Q_AW = np.clip(expit(q_model.decision_function(AW)), 1e-5, 1 - 1e-5)
        Q_1W = np.clip(expit(q_model.decision_function(AW1)), 1e-5, 1 - 1e-5)
        Q_0W = np.clip(expit(q_model.decision_function(AW0)), 1e-5, 1 - 1e-5)

        # OOF Q: K-fold cross-fitted predictions for unbiased IC variance.
        from sklearn.model_selection import KFold
        Q_1W_oof = np.zeros(n)
        Q_0W_oof = np.zeros(n)
        for tr, val in KFold(self.n_folds, shuffle=True,
                             random_state=self.random_state).split(AW):
            sw_tr = sample_weights[tr] / max(sample_weights[tr].mean(), 1e-10)
            qc = LogisticRegression(max_iter=1000, C=1.0)
            qc.fit(AW[tr], Y[tr], sample_weight=sw_tr)
            Q_1W_oof[val] = np.clip(expit(qc.decision_function(AW1[val])), 1e-5, 1-1e-5)
            Q_0W_oof[val] = np.clip(expit(qc.decision_function(AW0[val])), 1e-5, 1-1e-5)

        # ── Steps 4-5: Targeting + EIF SE ──
        results = []
        estimands_to_run = ["ATE", "ATT"] if estimand == "ATT" else ["ATE"]
        for est in estimands_to_run:
            point, se, IC = self._target_and_se(
                Y, A, g, Q_AW, Q_1W, Q_0W, ipcw, est, n,
                g_oof=g_oof, Q_1W_oof=Q_1W_oof, Q_0W_oof=Q_0W_oof,
            )
            z = stats.norm.ppf(0.975)
            results.append(EstimatorResult(
                name=self.name, estimand=est,
                point_estimate=float(point), standard_error=float(se),
                ci_lower=float(point - z * se), ci_upper=float(point + z * se),
                ic=IC,
            ))
        return results

    def _fit_G(self, censor_df, censor_feature_cols, T_obs, n):
        """Fit censoring survival model on the full dataset; return clipped G array."""
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(censor_df[censor_feature_cols + ["T_obs", "C_indicator"]],
                    duration_col="T_obs", event_col="C_indicator",
                    fit_options={"max_steps": 50})
            G = self._predict_G_sf(cph, censor_df[censor_feature_cols], T_obs, n)
        except Exception:
            G = np.ones(n)
        return np.clip(G, 0.05, 1.0)

    @staticmethod
    def _predict_G_sf(cph, cov_df, T_obs, n):
        """Read G(C > T_obs_i | covariates_i) from a fitted CoxPHFitter."""
        unique_times = np.sort(np.unique(T_obs))
        sf = cph.predict_survival_function(cov_df, times=unique_times)
        G = np.ones(n)
        for i, t in enumerate(T_obs):
            col = sf.iloc[:, i]
            idx_before = sf.index <= t
            if idx_before.any():
                G[i] = float(col[idx_before].iloc[-1])
        return G

    def _target_and_se(self, Y, A, g, Q_AW, Q_1W, Q_0W, ipcw, estimand, n,
                       g_oof=None, Q_1W_oof=None, Q_0W_oof=None):
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
        eps = np.clip(eps, -2.0, 2.0)

        Q1_star = expit(logit(Q_1W) + eps * H1)
        Q0_star = expit(logit(Q_0W) + eps * H0)

        # When OOF predictions are available, use a fully cross-fitted (DML) IC for SE.
        # All terms use OOF quantities so no in-sample bias inflates the variance.
        # The full-data Q*/g are still used for the point estimate (targeting improves bias).
        use_oof = (g_oof is not None) and (Q_1W_oof is not None)
        g_ic  = g_oof    if use_oof else g
        Q1_ic = Q_1W_oof if use_oof else Q1_star
        Q0_ic = Q_0W_oof if use_oof else Q0_star

        if estimand == "ATE":
            point = np.mean(Q1_star - Q0_star)
            IC = ((Q1_ic - Q0_ic - point)
                  + ipcw * (A / g_ic) * (Y - Q1_ic)
                  - ipcw * ((1 - A) / (1 - g_ic)) * (Y - Q0_ic))
        else:
            pi = np.mean(A)
            if pi < 1e-8:
                return float("nan"), float("nan"), None
            point = np.mean(A * (Q1_star - Q0_star)) / pi
            IC = (A * (Q1_ic - Q0_ic) / pi - point
                  + ipcw * A / pi * (Y - Q1_ic)
                  - ipcw * (1 - A) * g_ic / (1 - g_ic) / pi * (Y - Q0_ic))

        se = np.sqrt(np.var(IC, ddof=1) / n)
        return point, se, IC
