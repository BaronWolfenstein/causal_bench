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

Audit 2026-10-21 rewrite (the docstring above is now what the code does):
- Stage 2 regresses the stage-1 PSEUDO-OUTCOMES (Ỹ = 1 for early events, Q2*(A,W,L1) if alive at t_L1). The old
  code fit Q_full but regressed the observed Y, so L1 never entered the estimate.
- Each stage is targeted by a LOGISTIC fluctuation with clever covariate H = A/g − (1−A)/(1−g) and IPCW weights.
  The old one-step was computed on the probability scale, applied on the logit scale, and dropped the IPCW factor.
- Censoring weights are 1/G(min(T, t)-) for every patient with known status, incl. event-free at the horizon
  (the old version gave those weight 1).
- The IC is the two-time-point ICE-TMLE influence function, with the same fitted Q* as the point estimate.
- CAVEAT: in dgp.survival, L1 is drawn after T and C and affects neither (L1 = f(A, W3, U)). No DGP in the repo
  therefore exercises time-varying confounding; on it LTMLE and TMLE+IPCW target the same total effect.
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
                 n_mc: int = 50, t_L1: float = 0.5):
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_mc = n_mc
        self.t_L1 = t_L1          # L1 measurement time (DGPConfig.t_L1 default)

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

        alive = df["L1"].notna().values                          # alive AND uncensored at t_L1 (L1 observed)
        t_L1 = float(self.t_L1)
        L1 = np.where(alive, df["L1"].values, 0.0)
        ev1 = Delta == 1

        # ── Step 1: censoring survival G(s- | A, W) (Cox; competing events count as censoring, repo convention).
        # Audit fix 2026-10-21: EVERY patient whose status is known is weighted 1/G(min(T, t)-), including those
        # event-free at the horizon. The old ipcw = 1 for them ("Y=0 is known") under-represented the uncensored
        # non-events relative to the 1/G-weighted events.
        # Two-interval censoring model (the point of LTMLE): [0, t_L1] given (A, W); (t_L1, τ] given (A, W, L1) among
        # those alive at t_L1 (left-truncated at t_L1). Dropout that depends on L1 is then ignorable.
        cens = (Delta == 0) & (T_obs < horizon - 1e-9)                  # competing events count as censoring
        G1 = _cox_G(np.minimum(T_obs, t_L1), cens & (T_obs < t_L1), np.column_stack([A, W]))
        G2 = _cox_G(T_obs[alive], cens[alive], np.column_stack([A, W, L1])[alive], entry=t_L1)
        G1_tL1 = G1(np.full(n, t_L1))
        G2_full = np.ones(n); G2_full[alive] = G2(np.minimum(T_obs[alive], horizon))
        known1 = (T_obs >= t_L1) | (ev1 & (T_obs < t_L1))           # status at t_L1 known
        w1 = known1 / G1(np.minimum(T_obs, t_L1))
        known2 = alive & ((T_obs >= horizon - 1e-9) | ev1)          # alive at t_L1 and status at horizon known
        w2 = known2 / np.maximum(G1_tL1 * G2_full, 0.05)
        early = ev1 & (T_obs < t_L1)

        # ── Step 2: propensity g(A=1 | W) ──
        sl_g = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        sl_g.fit(W, A)
        g = np.clip(sl_g.predict_proba(W), 0.01, 0.99)
        HA = A / g - (1 - A) / (1 - g)

        # ── Step 3: stage-2-in-time regression Q2(A, W, L1) = P(Y=1 | A, W, L1, alive at t_L1), targeted ──
        X2 = np.column_stack([A, W, L1])
        r2 = known2
        Q2 = _wlogit_fit_predict(X2[r2], Y[r2], w2[r2], X2)
        eps2 = _fluctuate(Y[r2], Q2[r2], HA[r2], w2[r2])
        Q2s = _expit(_logit(Q2) + eps2 * HA)

        # ── Step 4: stage-1-in-time regression on the PSEUDO-OUTCOME (sequential regression / ICE), targeted ──
        # Ỹ = 1 for an event before t_L1, Q2*(A, W, L1) for patients alive at t_L1. Regress Ỹ on (A, W) among
        # those whose status at t_L1 is known. (The old code fit Q2 but then regressed the OBSERVED Y here, so L1
        # never entered the estimate, contradicting this module's docstring.)
        Yt = np.where(early, 1.0, np.where(alive, Q2s, 0.0))
        X1 = np.column_stack([A, W]); X1a = np.column_stack([np.ones(n), W]); X0a = np.column_stack([np.zeros(n), W])
        r1 = known1
        if r1.sum() < 10:
            raise RuntimeError("LTMLE: too few patients with known status at t_L1")
        Q1_A, Q1_1, Q1_0 = _wlogit_fit_predict(X1[r1], Yt[r1], w1[r1], X1, X1a, X0a)
        eps1 = _fluctuate(Yt[r1], Q1_A[r1], HA[r1], w1[r1])
        Q1_1s = _expit(_logit(Q1_1) + eps1 / g)
        Q1_0s = _expit(_logit(Q1_0) - eps1 / (1 - g))
        Q1_As = np.where(A == 1, Q1_1s, Q1_0s)

        # ── Step 5: estimate + EIF (ICE-TMLE influence function, sum over the two time points) ──
        point = float(np.mean(Q1_1s - Q1_0s))
        IC = ((Q1_1s - Q1_0s - point)
              + w1 * HA * (Yt - Q1_As)
              + w2 * HA * (Y - Q2s))
        se = float(np.sqrt(np.var(IC, ddof=1) / n))
        z = stats.norm.ppf(0.975)
        return [EstimatorResult(
            name=self.name, estimand="ATE",
            point_estimate=point, standard_error=se,
            ci_lower=point - z * se, ci_upper=point + z * se,
            ic=IC,
        )]


def _cox_G(dur, event, X, entry=None, floor=0.05):
    """Censoring survival from a Cox model: s -> P(no censoring in (entry, s) | X), left limit, floored."""
    n = len(dur)
    if event.sum() < 5:
        return lambda s: np.ones(n)
    d = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])]); d["T"] = dur; d["C"] = event.astype(int)
    kw = {}
    if entry is not None:
        d["E"] = entry; kw["entry_col"] = "E"
    cph = CoxPHFitter(penalizer=1e-3).fit(d, duration_col="T", event_col="C", **kw)
    H0 = cph.baseline_cumulative_hazard_; tt = H0.index.values.astype(float); h0 = H0.values[:, 0]
    risk = np.asarray(cph.predict_partial_hazard(d[[c for c in d.columns if c.startswith("x")]])).ravel()
    H_at = lambda s: np.where(np.searchsorted(tt, s, side="left") - 1 >= 0, h0[np.clip(np.searchsorted(tt, s, side="left") - 1, 0, None)], 0.0)
    H_entry = H_at(np.full(n, entry)) if entry is not None else 0.0
    return lambda s: np.maximum(np.exp(-(H_at(np.asarray(s, float)) - H_entry) * risk), floor)


def _wlogit_fit_predict(X, y, w, *Xp):
    """Weighted logistic regression allowing FRACTIONAL y in [0, 1] (quasi-binomial via the duplicate-row trick)."""
    y = np.clip(np.asarray(y, float), 0.0, 1.0); w = np.asarray(w, float)
    Xd = np.vstack([X, X]); yd = np.r_[np.ones(len(y)), np.zeros(len(y))]; wd = np.r_[w * y, w * (1 - y)]
    keep = wd > 0
    wd = wd / max(wd[keep].mean(), 1e-12)
    m = LogisticRegression(max_iter=2000, C=1.0).fit(Xd[keep], yd[keep], sample_weight=wd[keep])
    out = [np.clip(_expit_scipy(m.decision_function(x)), 1e-5, 1 - 1e-5) for x in Xp]
    return out[0] if len(out) == 1 else out


def _fluctuate(y, q, h, w, iters=50):
    """Logistic TMLE fluctuation: solve Σ w h (y − expit(logit q + ε h)) = 0 for ε (1-D Newton, offset logit q)."""
    eps = 0.0; off = _logit(q)
    for _ in range(iters):
        p = _expit(off + eps * h)
        score = np.sum(w * h * (y - p)); info = np.sum(w * h * h * p * (1 - p))
        if info <= 1e-12:
            break
        step = score / info; eps += step
        if abs(step) < 1e-10:
            break
    return float(eps)
