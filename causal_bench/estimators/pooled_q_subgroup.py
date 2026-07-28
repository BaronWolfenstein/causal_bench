# causal_bench/estimators/pooled_q_subgroup.py
"""Pooled-Q subgroup estimator (issues #77 / #189).

Single-arm-safe pooled-Q borrowing (Qiu et al., arXiv:2605.15483 — the
pooled-outcome-regression piece ONLY; NOT the A-TMLE treatment-effect borrowing,
whose S×A interaction bias term does not exist single-arm).

Estimand (#77, event-rate functional implemented here): for each pre-specified
subgroup s,

    psi_s = E[Y | S=s] = E[Q(W) | S=s],   Y = 1{composite event by `horizon`}

the one-sample targeted subgroup event rate (tested elsewhere against a
performance goal). The RMST functional (#189) reuses the SAME pooled Q via
`concrete_RMST` and is a separate target — build the pooled-Q core once, emit
both estimands.

Mechanism. A pooled outcome regression `Q(W)=P(Y=1|W)` is fit on ALL patients
(borrows the covariate->outcome map from everyone), IPCW-weighted for censoring,
then TMLE-targeted *per subgroup* via the subgroup-membership clever covariate
`H = 1{S=s}/pi_s * ipcw` for a valid influence-function SE. `pooled=False` fits
`Q` within each subgroup instead (borrows nothing) — the high-variance baseline
whose small-subgroup SE the pooled version should beat.

Efficient IC for psi_s (one-sample, IPCW):
    D_i = (1{S_i=s}/pi_s) * [ ipcw_i*(Y_i - Q*_i) + Q*_i - psi_s ].
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy import stats
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter

from causal_bench.estimators.base import BaseEstimator
from causal_bench.estimators.tmle_ipcw import _fit_q, _q_predict, TMLEIPCWEstimator
from causal_bench.metrics import EstimatorResult


class PooledQSubgroupEstimator(BaseEstimator):

    def __init__(self, pooled: bool = True, subgroup_col: str = "subgroup_label",
                 q_learner=None, use_compliance: bool = False,
                 random_state: int = 42):
        self.pooled = pooled
        self.subgroup_col = subgroup_col
        self.q_learner = q_learner
        self.use_compliance = use_compliance
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "pooled-Q subgroup rate" if self.pooled else "subgroup-only rate"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "subgroup_event_rate") -> list[EstimatorResult]:
        W_cols = ["W1", "W2", "W3", "W4"]
        W = df[W_cols].values.astype(float)
        T_obs = df["T_obs"].values
        Delta = df["Delta"].values.astype(float)
        n = len(df)
        Y = ((T_obs <= horizon) & (Delta == 1)).astype(float)

        if self.subgroup_col not in df.columns:
            raise ValueError(
                f"subgroup column '{self.subgroup_col}' not in df; "
                f"pooled-Q subgroup estimand needs pre-specified subgroups")
        S = df[self.subgroup_col].values

        # IPCW: one censoring model on the full data, shared across subgroups.
        ipcw = self._ipcw(df, W_cols, T_obs, Delta, horizon, n)
        sw = ipcw / max(ipcw.mean(), 1e-10)

        q_proto = (self.q_learner if self.q_learner is not None
                   else LogisticRegression(max_iter=1000, C=1.0))

        # Pooled outcome regression on ALL patients WITH the subgroup as a covariate
        # (Qiu's pooled-Q): the W coefficients are shared/borrowed across subgroups
        # while each subgroup gets its own level, so marginalizing E[Q(W,S=s)|S=s] over
        # subgroup s's own W distribution recovers E[Y|S=s]. Omitting S here would leave
        # a subgroup shift the (one-step) targeting cannot fully absorb. Subgroup
        # indicators are one-hot (drop_first; the logistic intercept carries the base
        # level), so it generalizes past binary subgroups.
        Q_pooled = None
        if self.pooled:
            S_dum = pd.get_dummies(pd.Series(np.asarray(S)), drop_first=True).values.astype(float)
            WS = np.column_stack([W, S_dum]) if S_dum.shape[1] else W
            q_all = _fit_q(q_proto, WS, Y, sw)
            Q_pooled = np.clip(_q_predict(q_all, WS), 1e-5, 1 - 1e-5)

        results = []
        z = stats.norm.ppf(0.975)
        for s in np.unique(S):
            in_s = (S == s).astype(float)
            pi_s = in_s.mean()
            if pi_s < 1e-8:
                continue

            if self.pooled:
                Q = Q_pooled
            else:
                m = (S == s)
                if m.sum() < 2:                 # cannot fit within a singleton subgroup
                    continue
                q_s = _fit_q(q_proto, W[m], Y[m], sw[m])
                Q = np.clip(_q_predict(q_s, W), 1e-5, 1 - 1e-5)

            # One-sample IPCW-TMLE targeting of E[Y|S=s].
            H = (in_s / pi_s) * ipcw                       # clever covariate (0 outside s)
            denom = float(np.mean(H ** 2))
            eps = float(np.mean(H * (Y - Q)) / denom) if denom > 1e-12 else 0.0
            eps = float(np.clip(eps, -10.0, 10.0))
            Q_star = expit(logit(Q) + eps * H)             # fluctuated only inside s

            psi = float(np.sum(in_s * Q_star) / (pi_s * n))  # = mean_{i in s} Q*_i
            IC = (in_s / pi_s) * (ipcw * (Y - Q_star) + Q_star - psi)
            se = float(np.sqrt(np.var(IC, ddof=1) / n))

            results.append(EstimatorResult(
                name=self.name, estimand=f"rate|S={s}",
                point_estimate=psi, standard_error=se,
                ci_lower=psi - z * se, ci_upper=psi + z * se,
                ess=float(pi_s * n), ic=IC,
            ))
        return results

    def _ipcw(self, df, W_cols, T_obs, Delta, horizon, n):
        """IPCW weights from a Cox censoring model (informative pre-horizon dropout
        only; admin-censored get weight 1). Reuses TMLEIPCWEstimator._predict_G_sf."""
        feats = list(W_cols)
        if "A" in df.columns:
            feats = feats + ["A"]
        if self.use_compliance and "compliance" in df.columns:
            feats = feats + ["compliance"]
        # Drop (near-)constant columns: single-arm A has zero variance and makes the
        # Cox fit emit NaN params (which it does NOT raise on — it only warns), so the
        # try/except below cannot catch it; excluding them up front is the real fix.
        feats = [c for c in feats if float(np.std(df[c].values)) > 1e-9]
        cdf = df[feats + ["T_obs"]].copy()
        cdf["C_indicator"] = ((Delta == 0) & (T_obs < horizon - 1e-9)).astype(float)
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cdf[feats + ["T_obs", "C_indicator"]],
                    duration_col="T_obs", event_col="C_indicator",
                    fit_options={"max_steps": 50})
            G = TMLEIPCWEstimator._predict_G_sf(cph, cdf[feats], T_obs, n)
        except Exception:
            G = np.ones(n)
        # Belt-and-suspenders: a non-raising bad fit can still leave NaN in G.
        G = np.nan_to_num(np.asarray(G, float), nan=1.0)
        G = np.clip(G, 0.05, 1.0)
        admin_censored = (Delta == 0) & (T_obs >= horizon - 1e-9)
        return np.where(Delta == 1, 1.0 / G, np.where(admin_censored, 1.0, 0.0))
