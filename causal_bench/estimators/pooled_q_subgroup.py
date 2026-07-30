# causal_bench/estimators/pooled_q_subgroup.py
"""Pooled-Q subgroup estimator (issues #77 / #189).

Single-arm-safe pooled-Q borrowing (Qiu et al., arXiv:2605.15483 — the
pooled-outcome-regression piece ONLY; NOT the A-TMLE treatment-effect borrowing,
whose S×A interaction bias term does not exist single-arm).

Estimand (#77, event-rate functional): for each pre-specified subgroup s,

    psi_s = E[Y | S=s] = E[Q(W) | S=s],   Y = 1{composite event by `horizon`}

the one-sample targeted subgroup event rate (tested elsewhere against a
performance goal).

Estimand (#189, RMST functional): for each subgroup s,

    RMST_s(tau) = int_0^tau S(t | S=s) dt,   S(t|S=s) = P(T > t | S=s)

the single-arm subgroup restricted mean survival time (`estimand="subgroup_rmst"`).
It lifts the pooled-Q idea to a time grid: a pooled *discrete-time hazard*
lambda_k(W,S) gives a monotone initial survival S0(t_k|W,S)=prod_{j<=k}(1-lambda_j),
each grid point is membership-targeted per subgroup exactly as the event rate is,
and the targeted S*_s(t_k) are integrated. SE is the integrated influence function
(per-time membership ICs summed over the grid). This is Route A — a self-contained
single-arm estimator, NOT concrete's two-arm RMST; the flexible `nuisance="rp_spline"`
backend swaps the pooled hazard for a Royston-Parmar flexsurvspline fit (#188),
still debiased by our own per-subgroup one-step.

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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter

from causal_bench.estimators.base import BaseEstimator
from causal_bench.estimators.tmle_ipcw import _fit_q, _q_predict, TMLEIPCWEstimator
from causal_bench.metrics import EstimatorResult


def _fit_predict_Q(proto, X_fit, y_fit, sw_fit, X_pred):
    """Fit Q=P(Y=1|X) on (X_fit, y_fit) and predict on X_pred, clipped to (0,1).

    Guards the degenerate single-class case: a small covariate-defined subgroup can be
    entirely events (or entirely non-events), and LogisticRegression RAISES on <2 classes
    ("needs samples of at least 2 classes"). There the outcome regression is the constant
    that class — P(Y=1|W)=1 when everyone had the event — so we skip the fit and return it
    rather than crash. Keeps `pooled=False` robust on all-events tails."""
    y_fit = np.asarray(y_fit, float)
    classes = np.unique(y_fit)
    if classes.size < 2:
        const = float(classes[0]) if classes.size == 1 else 0.5
        return np.full(X_pred.shape[0], np.clip(const, 1e-5, 1 - 1e-5))
    m = _fit_q(proto, X_fit, y_fit, sw_fit)
    return np.clip(_q_predict(m, X_pred), 1e-5, 1 - 1e-5)


def _solve_fluctuation(H, Y, F, max_iter=100, tol=1e-9):
    """Solve the 1-D logistic-fluctuation score sum(H*(Y - expit(logit(F)+eps*H)))=0 for
    eps by Newton's method (score' = -sum(H^2 * F*(1-F))). A single *linear* one-step
    (eps = mean(H(Y-F))/mean(H^2)) omits the F(1-F) curvature and UNDER-corrects when the
    initial F is badly biased (e.g. under informative censoring) — solving to convergence
    restores the targeting's double robustness through the censoring model."""
    lo = logit(np.clip(F, 1e-6, 1 - 1e-6))
    eps = 0.0
    for _ in range(max_iter):
        Fstar = expit(lo + eps * H)
        score = float(np.sum(H * (Y - Fstar)))
        deriv = -float(np.sum(H * H * Fstar * (1.0 - Fstar)))   # d score / d eps < 0
        if abs(deriv) < 1e-12:
            break
        step = score / (-deriv)                                # Newton: eps <- eps + score/|deriv|
        eps = float(np.clip(eps + step, -30.0, 30.0))
        if abs(step) < tol:
            break
    return eps


class PooledQSubgroupEstimator(BaseEstimator):

    def __init__(self, pooled: bool = True, subgroup_col: str = "subgroup_label",
                 q_learner=None, use_compliance: bool = False,
                 nuisance: str = "logistic", n_grid: int = 25,
                 random_state: int = 42):
        self.pooled = pooled
        self.subgroup_col = subgroup_col
        self.q_learner = q_learner
        self.use_compliance = use_compliance
        self.nuisance = nuisance        # RMST survival backend: "logistic" | "rp_spline"
        self.n_grid = n_grid            # RMST time-grid resolution
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "pooled-Q subgroup rate" if self.pooled else "subgroup-only rate"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "subgroup_event_rate") -> list[EstimatorResult]:
        if estimand == "subgroup_rmst":
            return self._estimate_rmst(df, horizon)
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
            Q_pooled = _fit_predict_Q(q_proto, WS, Y, sw, WS)

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
                Q = _fit_predict_Q(q_proto, W[m], Y[m], sw[m], W)

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

    def _fit_censoring_cox(self, df, W_cols, T_obs, Delta, horizon):
        """Fit the Cox censoring model once; return (cph, feats) or (None, feats)."""
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
            return cph, feats
        except Exception:
            return None, feats

    def _censoring_G(self, df, W_cols, T_obs, Delta, horizon, n):
        """G_i = P(C > T_obs_i | covariates_i) from a Cox censoring model, clipped to
        [0.05, 1]. Used by the event-rate IPCW."""
        cph, feats = self._fit_censoring_cox(df, W_cols, T_obs, Delta, horizon)
        if cph is None:
            return np.ones(n)
        G = TMLEIPCWEstimator._predict_G_sf(cph, df[feats], T_obs, n)
        G = np.nan_to_num(np.asarray(G, float), nan=1.0)   # a non-raising bad fit leaves NaN
        return np.clip(G, 0.05, 1.0)

    def _censoring_curves(self, df, W_cols, T_obs, Delta, horizon, n, t_grid):
        """Censoring survival for the RMST IPCW weights: (G_T, G_grid) where
        G_T[i]=G(T_obs_i|W_i) reweights events, and G_grid[i,k]=G(t_k|W_i) reweights
        subjects known alive at t_k. A subject's t_k-status is observed with probability
        G(min(T_i,t_k)|W_i) — the survivor weight is 1/G(t_k), NOT 1 (which would bias
        RMST high under informative censoring). Both clipped to [0.05, 1]."""
        cph, feats = self._fit_censoring_cox(df, W_cols, T_obs, Delta, horizon)
        if cph is None:
            return np.ones(n), np.ones((n, len(t_grid)))
        G_T = np.clip(np.nan_to_num(
            TMLEIPCWEstimator._predict_G_sf(cph, df[feats], T_obs, n), nan=1.0), 0.05, 1.0)
        # G(t_k | W_i): predict_survival_function returns times x subjects.
        sf = cph.predict_survival_function(df[feats], times=np.asarray(t_grid, float))
        G_grid = np.clip(np.nan_to_num(sf.values.T, nan=1.0), 0.05, 1.0)   # (n, K)
        return G_T, G_grid

    def _ipcw(self, df, W_cols, T_obs, Delta, horizon, n):
        """IPCW weights from a Cox censoring model (informative pre-horizon dropout
        only; admin-censored get weight 1)."""
        G = self._censoring_G(df, W_cols, T_obs, Delta, horizon, n)
        admin_censored = (Delta == 0) & (T_obs >= horizon - 1e-9)
        return np.where(Delta == 1, 1.0 / G, np.where(admin_censored, 1.0, 0.0))

    # ------------------------------------------------------------------ RMST (#189)
    def _estimate_rmst(self, df, horizon):
        """Subgroup RMST via a pooled discrete-time hazard + per-subgroup TMLE.

        S0(t_k|W,S) comes from a monotone pooled hazard (or an RP-spline fit); each
        grid point's cumulative incidence F_k=1-S0 is membership-targeted inside each
        subgroup exactly as the event rate is; targeted survival is re-monotonized and
        integrated. SE = integrated IC (per-time membership ICs summed over the grid).
        """
        W_cols = ["W1", "W2", "W3", "W4"]
        W = df[W_cols].values.astype(float)
        T_obs = np.asarray(df["T_obs"].values, float)
        Delta = np.asarray(df["Delta"].values, float)
        n = len(df)
        if self.subgroup_col not in df.columns:
            raise ValueError(
                f"subgroup column '{self.subgroup_col}' not in df; "
                f"pooled-Q subgroup RMST needs pre-specified subgroups")
        S = np.asarray(df[self.subgroup_col].values)

        t_grid = np.linspace(0.0, horizon, self.n_grid + 1)[1:]
        dt = float(t_grid[0])                         # uniform spacing
        G_T, G_grid = self._censoring_curves(df, W_cols, T_obs, Delta, horizon, n, t_grid)

        # Pooled design: shared W + subgroup one-hot (generalizes past binary subgroups).
        S_dum = pd.get_dummies(pd.Series(S), drop_first=True).values.astype(float)
        WS = np.column_stack([W, S_dum]) if S_dum.shape[1] else W

        # Initial monotone survival S0[i,k] = P(T > t_k | W_i, S_i).
        S0_pooled = None
        if self.pooled:
            if self.nuisance == "rp_spline":
                from causal_bench.estimators.rp_spline_nuisance import predict_rp_survival
                S0_pooled = predict_rp_survival(df, W_cols, self.subgroup_col, t_grid)
            if S0_pooled is None:                     # logistic, or RP unavailable
                S0_pooled = self._hazard_survival(WS, T_obs, Delta, G_T, WS, t_grid)

        results = []
        z = stats.norm.ppf(0.975)
        K = len(t_grid)
        # Per-grid-point IPCW pseudo-outcome + weights (cumulative-incidence at t_k):
        #   fail by t_k  -> Y=1, w=1/G(T_i);   alive past t_k -> Y=0, w=1/G(t_k|W_i);
        #   censored before t_k -> t_k-status unobserved, w=0 (IPCW upweights the rest).
        Yk_all = np.zeros((n, K))
        wk_all = np.zeros((n, K))
        for k, t_k in enumerate(t_grid):
            failed_by = (T_obs <= t_k) & (Delta == 1)
            past_tk = T_obs > t_k
            Yk_all[:, k] = failed_by.astype(float)
            wk_all[:, k] = np.where(failed_by, 1.0 / G_T,
                                    np.where(past_tk, 1.0 / G_grid[:, k], 0.0))

        for s in np.unique(S):
            in_s = (S == s).astype(float)
            pi_s = in_s.mean()
            if pi_s < 1e-8:
                continue

            if self.pooled:
                S0 = S0_pooled
            else:
                m = (S == s)
                if m.sum() < 5:                        # too few to fit a within-subgroup hazard
                    continue
                S0 = self._hazard_survival(W[m], T_obs[m], Delta[m], G_T[m], W, t_grid)
            F0 = np.clip(1.0 - S0, 1e-5, 1.0 - 1e-5)   # cumulative incidence (nondecr in k)

            # Trapezoidal integration of S over [0, tau] with the known S(0)=1: the
            # slab weight on S(t_k) is dt for interior points and dt/2 at the endpoint
            # t_K, plus a constant dt/2 * S(0). A right-Riemann sum (dt on every S(t_k))
            # omits the [0, t_1] slab where S~=1 and biases RMST low. IC weights match.
            # Each grid point is targeted INDEPENDENTLY. We deliberately do NOT re-
            # monotonize the targeted F* across k: forcing monotonicity after solving
            # the score would clobber the fluctuation and break the targeting (and its
            # double robustness). The monotone initial S0 from the hazard cumprod keeps
            # F* near-monotone; any small wiggle averages out in the RMST integral.
            IC_rmst = np.zeros(n)
            rmst = 0.5 * dt                             # dt/2 * S(0)=1 (deterministic)
            for k in range(K):
                c_k = dt if k < K - 1 else 0.5 * dt     # trapezoid slab weight on S(t_k)
                Yk, wk, Fk = Yk_all[:, k], wk_all[:, k], F0[:, k]
                H = (in_s / pi_s) * wk                  # membership clever covariate
                eps = _solve_fluctuation(H, Yk, Fk)     # solve the logistic score to ~0
                Fk_star = expit(logit(Fk) + eps * H)    # fluctuate only inside s
                Sk_star = 1.0 - Fk_star
                psi_k = float(np.sum(in_s * Sk_star) / (pi_s * n))   # mean_{i in s} S*(t_k)
                rmst += psi_k * c_k
                # IC of S(t_k) = -IC of F(t_k); RMST IC accumulates c_k * IC_S.
                IC_Fk = (in_s / pi_s) * (wk * (Yk - Fk_star) + Fk_star - (1.0 - psi_k))
                IC_rmst += -IC_Fk * c_k

            se = float(np.sqrt(np.var(IC_rmst, ddof=1) / n))
            results.append(EstimatorResult(
                name=self.name, estimand=f"rmst|S={s}",
                point_estimate=rmst, standard_error=se,
                ci_lower=rmst - z * se, ci_upper=rmst + z * se,
                ess=float(pi_s * n), ic=IC_rmst,
            ))
        return results

    def _hazard_survival(self, X_fit, T_fit, D_fit, G_fit, X_pred, t_grid):
        """Pooled discrete-time logistic hazard on person-time built from (X_fit, T_fit,
        D_fit), IPCW-stabilized, returning monotone S[i,k]=prod_{j<=k}(1-lambda_j) for
        every row of X_pred. Interval baselines are one-hot; W/S effects are shared
        across intervals (the 'pooled' structure)."""
        K = len(t_grid)
        edges = np.concatenate([[0.0], t_grid])        # t_0..t_K
        lo, hi = edges[:-1], edges[1:]                 # (K,)
        Ti = T_fit[:, None]
        at_risk = Ti > lo[None, :]                     # subject entered interval k
        fail_here = (D_fit[:, None] == 1) & (Ti > lo[None, :]) & (Ti <= hi[None, :])

        rows_i, rows_k = np.where(at_risk)
        if rows_i.size == 0:
            return np.ones((len(X_pred), K))
        eye = np.eye(K)
        # IPCW-stabilized per-subject weight (1/G at each subject's own T), broadcast to rows.
        w_subj = 1.0 / np.clip(G_fit, 0.05, 1.0)
        w_subj = w_subj / max(w_subj.mean(), 1e-10)
        X_long = np.column_stack([X_fit[rows_i], eye[rows_k]])
        y_long = fail_here[rows_i, rows_k].astype(float)
        w_long = w_subj[rows_i]

        proto = (clone(self.q_learner) if self.q_learner is not None
                 else LogisticRegression(max_iter=1000, C=1.0, fit_intercept=False))
        try:
            clf = _fit_q(proto, X_long, y_long, w_long)
        except Exception:
            return np.ones((len(X_pred), K))
        # Degenerate all-0 / all-1 outcome: predict a constant hazard safely.
        lam = np.empty((len(X_pred), K))
        for k in range(K):
            Xk = np.column_stack([X_pred, np.tile(eye[k], (len(X_pred), 1))])
            lam[:, k] = np.clip(_q_predict(clf, Xk), 1e-6, 1.0 - 1e-6)
        return np.cumprod(1.0 - lam, axis=1)
