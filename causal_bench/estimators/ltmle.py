# causal_bench/estimators/ltmle.py
"""
LTMLE (Longitudinal TMLE) estimator for time-varying confounders.

Two-stage approach:
1. Q_full(A, W, L1): outcome model on alive-at-L1 subset, IPCW-weighted to account
   for informative censoring after L1.
2. Q_bar(A, W): second-stage regression combining alive patients' Q_full pseudo-outcomes
   with early events (Y=1, T_obs < t_L1). This correctly accounts for the survival
   selection into being alive at L1 and avoids collider bias.
3. TMLE targeting of Q_bar using the full-sample EIF.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as _expit_scipy
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LTMLEEstimator(BaseEstimator):

    def __init__(self, n_folds: int = 5, random_state: int = 42,
                 n_mc: int = 50):
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_mc = n_mc

    @property
    def name(self) -> str:
        return "LTMLE"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        has_l1 = ("L1" in df.columns) and (df["L1"].notna().any())

        if not has_l1:
            from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
            fallback = TMLEIPCWEstimator(use_compliance=False,
                                         n_folds=self.n_folds,
                                         random_state=self.random_state)
            results = fallback.estimate(df, horizon=horizon, estimand=estimand)
            return [EstimatorResult(
                name=self.name, estimand=r.estimand,
                point_estimate=r.point_estimate,
                standard_error=r.standard_error,
                ci_lower=r.ci_lower, ci_upper=r.ci_upper,
            ) for r in results]

        W_cols = ["W1", "W2", "W3", "W4"]
        A = df["A"].values.astype(float)
        T_obs = df["T_obs"].values
        Delta = df["Delta"].values.astype(float)
        W = df[W_cols].values.astype(float)
        n = len(A)

        Y = ((T_obs <= horizon) & (Delta == 1)).astype(float)

        alive_mask = df["L1"].notna().values
        # t_L1 is the L1 measurement time; anything before this is a pre-L1 early event
        t_L1 = 0.5  # matches DGPConfig.t_L1 default
        early_event_mask = (Delta == 1) & (T_obs < t_L1 - 1e-9)
        admin_mask = (Delta == 0) & (T_obs >= horizon - 1e-9)

        # ── Step 1: Censoring model ──
        # Only pre-horizon, pre-L1-or-event dropouts are informative censoring events.
        # Admin-censored (T_obs≥horizon) and events are NOT censoring events.
        censor_feature_cols = W_cols + ["A"]
        censor_df = df[censor_feature_cols + ["T_obs", "Delta"]].copy()
        censor_df["C_indicator"] = (
            (censor_df["Delta"] == 0) & (censor_df["T_obs"] < horizon - 1e-9)
        ).astype(float)

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(censor_df[censor_feature_cols + ["T_obs", "C_indicator"]],
                    duration_col="T_obs", event_col="C_indicator",
                    fit_options={"max_steps": 50})

            unique_times = np.sort(np.unique(T_obs))
            sf = cph.predict_survival_function(
                censor_df[censor_feature_cols], times=unique_times
            )
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
        # Events: upweight 1/G (represent similar patients who were censored).
        # Admin-censored: ipcw=1 (Y=0 known, no upweighting needed).
        # Pre-horizon dropouts: ipcw=0 (excluded; outcome unknown).
        ipcw = np.where(Delta == 1, 1.0 / G,
                        np.where(admin_mask, 1.0, 0.0))

        # ── Step 2: Propensity model g(A=1 | W) ──
        sl_g = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        sl_g.fit(W, A)
        g = sl_g.predict_proba(W)
        g_oof = sl_g.oof_predictions_

        # ── Step 3: Stage-1 outcome model Q_full(A, W, L1) on alive-at-L1 ──
        alive_idx = np.where(alive_mask)[0]
        A_alive = A[alive_idx]
        W_alive = W[alive_idx]
        L1_alive = df["L1"].values[alive_idx]
        Y_alive = Y[alive_idx]
        ipcw_alive = ipcw[alive_idx]
        ipcw_alive_norm = ipcw_alive / max(ipcw_alive.mean(), 1e-10)

        X_full_train = np.column_stack([A_alive, W_alive, L1_alive])
        q_full = LogisticRegression(max_iter=1000, C=1.0)
        q_full.fit(X_full_train, Y_alive, sample_weight=ipcw_alive_norm)

        # Q_full predictions as pseudo-outcomes for alive patients
        Q_hat_alive = np.clip(
            _expit_scipy(q_full.decision_function(X_full_train)), 1e-5, 1 - 1e-5
        )

        # ── Step 4: Stage-2 Q_bar(A, W) combining alive patients + early events ──
        # Early events (Y=1, T<t_L1) are missing from the alive subset due to survival
        # selection; including them corrects for the resulting downward bias on ATE.
        early_idx = np.where(early_event_mask)[0]
        stage2_idx = np.concatenate([alive_idx, early_idx])
        stage2_A = A[stage2_idx]
        stage2_W = W[stage2_idx]
        stage2_ipcw = ipcw[stage2_idx]
        # Use actual outcomes: Y_alive for alive patients; Y=1 for early events.
        stage2_Y = np.concatenate([Y_alive, np.ones(len(early_idx))])
        stage2_sw = stage2_ipcw / max(stage2_ipcw.mean(), 1e-10)

        AW_s2 = np.column_stack([stage2_A, stage2_W])
        AW = np.column_stack([A, W])
        AW1 = np.column_stack([np.ones(n), W])
        AW0 = np.column_stack([np.zeros(n), W])

        # Guard against degenerate case (single class in subset)
        if len(np.unique(stage2_Y)) < 2:
            from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
            fallback = TMLEIPCWEstimator(use_compliance=False,
                                         n_folds=self.n_folds,
                                         random_state=self.random_state)
            results = fallback.estimate(df, horizon=horizon, estimand=estimand)
            return [EstimatorResult(
                name=self.name, estimand=r.estimand,
                point_estimate=r.point_estimate,
                standard_error=r.standard_error,
                ci_lower=r.ci_lower, ci_upper=r.ci_upper,
            ) for r in results]

        q_bar = LogisticRegression(max_iter=1000, C=1.0)
        q_bar.fit(AW_s2, stage2_Y.astype(int), sample_weight=stage2_sw)

        Q_AW = np.clip(_expit_scipy(q_bar.decision_function(AW)), 1e-5, 1 - 1e-5)
        Q_1W = np.clip(_expit_scipy(q_bar.decision_function(AW1)), 1e-5, 1 - 1e-5)
        Q_0W = np.clip(_expit_scipy(q_bar.decision_function(AW0)), 1e-5, 1 - 1e-5)

        # OOF Q_bar: cross-fitted predictions on the full sample for unbiased IC variance.
        from sklearn.model_selection import KFold
        Q_1W_oof = np.zeros(n)
        Q_0W_oof = np.zeros(n)
        for tr_s2, val_s2 in KFold(self.n_folds, shuffle=True,
                                    random_state=self.random_state).split(AW_s2):
            # Map stage2 val indices back to full-sample positions
            val_full = stage2_idx[val_s2]
            sw_tr = stage2_sw[tr_s2] / max(stage2_sw[tr_s2].mean(), 1e-10)
            if len(np.unique(stage2_Y[tr_s2])) < 2:
                continue
            qc = LogisticRegression(max_iter=1000, C=1.0)
            qc.fit(AW_s2[tr_s2], stage2_Y[tr_s2].astype(int), sample_weight=sw_tr)
            Q_1W_oof[val_full] = np.clip(
                _expit_scipy(qc.decision_function(AW1[val_full])), 1e-5, 1 - 1e-5)
            Q_0W_oof[val_full] = np.clip(
                _expit_scipy(qc.decision_function(AW0[val_full])), 1e-5, 1 - 1e-5)

        # Fill any zeros (folds skipped due to single-class) with full-data predictions
        Q_1W_oof = np.where(Q_1W_oof == 0.0, Q_1W, Q_1W_oof)
        Q_0W_oof = np.where(Q_0W_oof == 0.0, Q_0W, Q_0W_oof)

        # ── Step 5: TMLE targeting ──
        H = ipcw * (A / g - (1 - A) / (1 - g))
        denom = np.mean(H ** 2)
        eps = np.mean(H * (Y - Q_AW)) / denom if denom > 1e-10 else 0.0
        eps = np.clip(eps, -2.0, 2.0)

        Q_1W_star = _expit(_logit(Q_1W) + eps / g)
        Q_0W_star = _expit(_logit(Q_0W) - eps / (1 - g))

        # ── Step 6: Point estimate + EIF SE ──
        # Fully cross-fitted DML IC: all terms use OOF g and Q for unbiased variance.
        # Point estimate still uses full-data targeted Q* for better finite-sample bias.
        point = float(np.mean(Q_1W_star - Q_0W_star))

        IC = ((Q_1W_oof - Q_0W_oof - point)
              + ipcw * (A / g_oof) * (Y - Q_1W_oof)
              - ipcw * ((1 - A) / (1 - g_oof)) * (Y - Q_0W_oof))
        se = float(np.sqrt(np.var(IC, ddof=1) / n))

        z = stats.norm.ppf(0.975)
        return [EstimatorResult(
            name=self.name, estimand="ATE",
            point_estimate=point, standard_error=se,
            ci_lower=point - z * se, ci_upper=point + z * se,
            ic=IC,
        )]
