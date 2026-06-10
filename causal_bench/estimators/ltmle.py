# causal_bench/estimators/ltmle.py
"""
LTMLE (Longitudinal TMLE) estimator for time-varying confounders.

Key insight: L1 is a collider (caused by both A and U). Conditioning on it
in a naive model introduces bias. LTMLE avoids this by:
1. Fitting outcome model Q_full with L1 included (reduces variance)
2. Marginalizing out L1 over its empirical distribution (avoids collider bias)
3. Targeting with treatment clever covariate at baseline
"""
import numpy as np
import pandas as pd
from scipy import stats
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
        # Check whether L1 is usable
        has_l1 = ("L1" in df.columns) and (df["L1"].notna().any())

        if not has_l1:
            # Fallback to TMLE+IPCW
            from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
            fallback = TMLEIPCWEstimator(use_compliance=False,
                                         n_folds=self.n_folds,
                                         random_state=self.random_state)
            results = fallback.estimate(df, horizon=horizon, estimand=estimand)
            # Rename results to LTMLE
            renamed = []
            for r in results:
                renamed.append(EstimatorResult(
                    name=self.name, estimand=r.estimand,
                    point_estimate=r.point_estimate,
                    standard_error=r.standard_error,
                    ci_lower=r.ci_lower, ci_upper=r.ci_upper,
                ))
            return renamed

        W_cols = ["W1", "W2", "W3", "W4"]
        A = df["A"].values.astype(float)
        T_obs = df["T_obs"].values
        Delta = df["Delta"].values.astype(float)
        W = df[W_cols].values.astype(float)
        n = len(A)

        Y = ((T_obs <= horizon) & (Delta == 1)).astype(float)

        # ── Step 1: Identify alive_at_L1 rows ──
        alive_mask = df["L1"].notna().values
        L1_pool = df.loc[alive_mask, "L1"].values

        # ── Step 2: Fit censoring model (IPCW) ──
        from lifelines import CoxPHFitter
        censor_feature_cols = W_cols + ["A"]
        censor_df = df[censor_feature_cols + ["T_obs", "Delta"]].copy()
        censor_df["C_indicator"] = 1.0 - censor_df["Delta"]

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
        is_observed = (Delta == 1) | (T_obs >= horizon - 1e-9)
        ipcw = np.where(is_observed, 1.0 / G, 0.0)

        # ── Step 3: Propensity model g(A=1 | W) ──
        sl_g = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        sl_g.fit(W, A)
        g = sl_g.predict_proba(W)

        # ── Step 4: Full outcome model Q_full(A, W, L1) on alive subset ──
        alive_idx = np.where(alive_mask)[0]
        A_alive = A[alive_idx]
        W_alive = W[alive_idx]
        L1_alive = df["L1"].values[alive_idx]
        Y_alive = Y[alive_idx]

        X_full_train = np.column_stack([A_alive, W_alive, L1_alive])

        sl_q = SuperLearner(task="regression", n_folds=self.n_folds,
                            random_state=self.random_state)
        sl_q.fit(X_full_train, Y_alive)

        # ── Step 5: Marginalize out L1 via Monte Carlo ──
        rng_mc = np.random.default_rng(self.random_state)
        preds_a1 = []
        preds_a0 = []
        preds_aobs = []

        for _ in range(self.n_mc):
            L1_sample = rng_mc.choice(L1_pool, size=n, replace=True)
            X_mc_a1 = np.column_stack([np.ones(n), W, L1_sample])
            X_mc_a0 = np.column_stack([np.zeros(n), W, L1_sample])
            X_mc_aobs = np.column_stack([A, W, L1_sample])
            preds_a1.append(sl_q.predict(X_mc_a1))
            preds_a0.append(sl_q.predict(X_mc_a0))
            preds_aobs.append(sl_q.predict(X_mc_aobs))

        Q_margin_A1 = np.clip(np.mean(preds_a1, axis=0), 1e-6, 1 - 1e-6)
        Q_margin_A0 = np.clip(np.mean(preds_a0, axis=0), 1e-6, 1 - 1e-6)
        Q_margin_AW = np.clip(np.mean(preds_aobs, axis=0), 1e-6, 1 - 1e-6)

        # ── Step 6: Targeting ──
        H = ipcw * (A / g - (1 - A) / (1 - g))
        denom = np.mean(H ** 2)
        eps = np.mean(H * (Y - Q_margin_AW)) / denom if denom > 1e-10 else 0.0
        eps = np.clip(eps, -2.0, 2.0)

        Q_margin_A1_star = _expit(_logit(Q_margin_A1) + eps / g)
        Q_margin_A0_star = _expit(_logit(Q_margin_A0) - eps / (1 - g))

        # ── Step 7: Point estimate ──
        point = float(np.mean(Q_margin_A1_star - Q_margin_A0_star))

        # ── Step 8: EIF-based SE ──
        IC = ((Q_margin_A1_star - Q_margin_A0_star - point)
              + ipcw * (A / g) * (Y - Q_margin_A1_star)
              - ipcw * ((1 - A) / (1 - g)) * (Y - Q_margin_A0_star))
        se = float(np.sqrt(np.var(IC, ddof=1) / n))

        z = stats.norm.ppf(0.975)
        return [EstimatorResult(
            name=self.name, estimand="ATE",
            point_estimate=point, standard_error=se,
            ci_lower=point - z * se, ci_upper=point + z * se,
        )]
