"""exp54 — SURVIVAL variant of exp45: time-varying treatment + time-to-event outcome.

Fuses exp45 (sequential treatment with treatment-confounder feedback, scalar outcome) with exp51
(survival CATE, point treatment). The Figma story: a *sequential* lifecycle program on new teams with
CHURN (time-to-event) as the outcome, informative censoring, and a treatment-affected confounder.

    W → A0 → L1 → A1 → T(churn),   with  A0→L1 (the week-1 nudge raises engagement),
    L1→A1 (feedback: low-engagement teams get the re-nudge), L1→hazard, plus informative censoring.

    A0  week-1 intervention (tutorial / starter-template push)
    L1  week-1 engagement (files, collaborators, DAU) — RAISED by A0, DRIVES who gets A1, PREDICTS churn
    A1  week-4 re-engagement nudge, TARGETED by (low) L1
    T   time until the team churns; C informative censoring (admin end + covariate-dependent dropout)

The trap (same as exp45, now on a survival outcome): L1 is simultaneously a *mediator* of A0 (the
week-1 nudge works partly by raising engagement) and a *confounder* of A1. Adjusting for L1 in the
outcome blocks A0's benefit-through-engagement; not adjusting biases A1. → needs g-methods.

Estimands (per dynamic regime ā=(a0,a1)): the regime survival S(τ|ā), the RMST ("expected active-days
over τ"), the regime contrast S(τ|1,1)−S(τ|0,0), and the A0 survival blip S(τ|1,·)−S(τ|0,·) (the TOTAL
A0 effect, including the L1-mediated path). Methods (all self-validating vs interventional MC truth):

  • **ICE-survival g-computation** with IPCW censoring — the g-formula backbone (survival analogue of
    exp45's `ice_regime_mean`): weighted sequential regression Q2(W,A0,L1,A1)→Q1(W,A0)→mean, on the
    τ-survival indicator, IPCW-weighted for informative censoring. Recovers all regimes + RMST.
  • **last-blip g-estimation** ψ1 — the structural log-HR of A1 is the coefficient of A1 in a Cox model
    adjusting for the FULL pre-A1 history (W,A0,L1); valid because L1 precedes A1 (survival analogue of
    exp45's final `g_estimation` blip). NOTE: the marginal A0 hazard blip is NOT cleanly recoverable on
    the hazard scale (Cox non-collapsibility) — which is exactly why the A0 total effect is reported on
    the collapsible survival/RMST scale via ICE, not as a hazard ratio.
  • **naive baselines (biased)** — per-(A0,A1) Kaplan–Meier (ignores confounding + feedback) and a Cox
    that adjusts the mediator L1 in the outcome (blocks A0's engagement-mediated benefit).

Beyond the repo's two-timepoint `LTMLEEstimator`; the DR survival-LTMLE / continuous-time CONCRETE
column (survival-SNMM targeting, optimal dynamic regime) stays phase-2 / gated on the concrete
default-learner attenuation fix (#223) — see docs/plans/2026-09-23-*. All estimators here are
OLS/Cox/KM (numpy/sklearn/lifelines) — no MCMC, no concrete.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _expit(x): return 1.0 / (1.0 + np.exp(-x))


# structural coefficients (fixed; the truth the estimators must recover)
_GAMMA = 0.5                     # A0 ~ expit(γ·W)
_B_L0, _B_LW = 0.8, 0.5          # L1 = _B_L0·A0 + _B_LW·W + noise (engagement, raised by A0)
_A0C, _A_L = -0.2, -0.6          # A1 ~ expit(_A0C + _A_L·L1): low engagement → more re-nudge (feedback)
_H0, _HW, _HL = -0.9, 0.4, -0.5  # log-hazard of churn: base, W, L1 (engagement protects)
_PSI0, _PSI1 = -0.4, -0.5        # structural log-HR blips of A0, A1 (both reduce churn)
_D0, _DW = -1.0, 0.3             # log-hazard of censoring (informative in W)


def sim_survival_timevarying(n, *, tau=3.0, seed=0):
    """Draw the sequential-treatment survival cohort. Columns: W, A0, L1, A1, T_obs, event (0/1), tau."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=n)
    A0 = (rng.random(n) < _expit(_GAMMA * W)).astype(float)
    L1 = _B_L0 * A0 + _B_LW * W + rng.normal(size=n)              # confounder affected by A0
    A1 = (rng.random(n) < _expit(_A0C + _A_L * L1)).astype(float)  # feedback: A1 targeted by (low) L1
    lamT = np.exp(_H0 + _HW * W + _HL * L1 + _PSI0 * A0 + _PSI1 * A1)
    T = rng.exponential(1.0 / lamT)
    lamC = np.exp(_D0 + _DW * W)
    C = rng.exponential(1.0 / lamC)
    T_obs = np.minimum(np.minimum(T, C), tau)
    event = ((T <= C) & (T <= tau)).astype(int)                  # 1=churn observed, 0=censored (informative or admin)
    return pd.DataFrame({"W": W, "A0": A0, "L1": L1, "A1": A1, "T_obs": T_obs, "event": event, "tau": tau})


def true_regime_survival(*, tau=3.0, n=2_000_000, seed=99):
    """Interventional MC truth per regime ā=(a0,a1): S(τ|ā)=E[e^{−λτ}], RMST(τ|ā)=E[(1−e^{−λτ})/λ],
    over do(A0=a0) (which sets L1's distribution) and do(A1=a1). Plus the (1,1)-(0,0) contrast and the
    A0 survival blip S(τ|1,a1)−S(τ|0,a1)."""
    rng = np.random.default_rng(seed)
    S, R = {}, {}
    for a0 in (0.0, 1.0):
        for a1 in (0.0, 1.0):
            W = rng.normal(size=n)
            L1 = _B_L0 * a0 + _B_LW * W + rng.normal(size=n)     # do(A0=a0): L1 responds
            lam = np.exp(_H0 + _HW * W + _HL * L1 + _PSI0 * a0 + _PSI1 * a1)
            S[(a0, a1)] = float(np.mean(np.exp(-lam * tau)))
            R[(a0, a1)] = float(np.mean((1.0 - np.exp(-lam * tau)) / lam))
    return {"surv": S, "rmst": R,
            "contrast_surv": S[(1.0, 1.0)] - S[(0.0, 0.0)],
            "contrast_rmst": R[(1.0, 1.0)] - R[(0.0, 0.0)],
            "blip_A0_surv": {a1: S[(1.0, a1)] - S[(0.0, a1)] for a1 in (0.0, 1.0)},
            "psi_logHR": (_PSI0, _PSI1)}


# ---------------------------------------------------------------- IPCW (informative censoring)
def _ipcw(df, horizon):
    """Inverse-probability-of-censoring weights + τ-survival indicator at `horizon`.

    Censored-before-`horizon` units (event==0 & T_obs<horizon) are uninformative for 1{T>horizon}; the
    rest are reweighted by 1/K_C(min(T_obs,horizon)|history), K_C fit by Cox on (W,A0,L1,A1). Returns
    (w, Y) with Y=1{survived past horizon}; w=0 on the censored (they drop out of the weighted fit)."""
    from lifelines import CoxPHFitter
    t = df["T_obs"].values
    censored_before = ((df["event"].values == 0) & (t < horizon - 1e-9))     # uninformative for 1{T>horizon}
    fit = pd.DataFrame({"dur": t, "cev": censored_before.astype(int),
                        "W": df["W"], "A0": df["A0"], "L1": df["L1"], "A1": df["A1"]})
    cph = CoxPHFitter(penalizer=1e-4).fit(fit, duration_col="dur", event_col="cev")
    H0 = cph.baseline_cumulative_hazard_.iloc[:, 0]              # cumulative baseline hazard of censoring
    ph = cph.predict_partial_hazard(fit).values                 # exp(β'x) per unit
    et = np.minimum(t, horizon)                                 # evaluate K_C at min(T_obs, horizon)
    idx = np.searchsorted(H0.index.values, et, side="right") - 1
    H0et = np.where(idx >= 0, H0.values[np.clip(idx, 0, len(H0) - 1)], 0.0)
    Kc = np.exp(-H0et * ph)                                     # P(uncensored past et | history)
    w = np.where(censored_before, 0.0, 1.0 / np.clip(Kc, 1e-3, None))
    Y = (t >= horizon - 1e-9).astype(float)                     # survived past horizon
    return w, Y


# ---------------------------------------------------------------- estimators
def ice_survival_regime(df, a0, a1, horizon):
    """ICE-survival g-computation of S(horizon|ā) — the LTMLE g-computation backbone with IPCW.
    Q2 = E[1{T>h} | A0,L1,A1,W] (IPCW-weighted), set A1=a1; Q1 = E[Q2 | A0,W], set A0=a0; mean."""
    w, Y = _ipcw(df, horizon)
    A0v, L1, A1v, W = (df[k].values for k in ("A0", "L1", "A1", "W"))
    X2 = np.column_stack([A0v, L1, A1v, W])
    # numpy 2.x emits spurious "divide/overflow/invalid in matmul" FPE flags from the SIMD BLAS path
    # even on finite data (weights + outputs verified finite); silence them around the linear predicts.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        q2m = LinearRegression().fit(X2, Y, sample_weight=w)                   # weighted: censored (w=0) drop
        q2 = q2m.predict(np.column_stack([A0v, L1, np.full_like(A1v, a1), W]))  # do(A1=a1), all units
        q1m = LinearRegression().fit(np.column_stack([A0v, W]), q2)           # pseudo-outcome, unweighted
        q1 = q1m.predict(np.column_stack([np.full_like(A0v, a0), W]))         # do(A0=a0)
    return float(np.clip(q1, 0.0, 1.0).mean())


def ice_regime_table(df, tau):
    """S(τ|ā) for all four regimes + the (1,1)-(0,0) contrast + the A0 survival blip, via ICE."""
    S = {(a0, a1): ice_survival_regime(df, a0, a1, tau) for a0 in (0.0, 1.0) for a1 in (0.0, 1.0)}
    return {"surv": S, "contrast_surv": S[(1.0, 1.0)] - S[(0.0, 0.0)],
            "blip_A0_surv": {a1: S[(1.0, a1)] - S[(0.0, a1)] for a1 in (0.0, 1.0)}}


def ice_rmst_regime(df, a0, a1, tau, grid=None):
    """RMST(τ|ā) = ∫₀^τ S(t|ā) dt via ICE-survival at a horizon grid, trapezoid (S(0)=1)."""
    grid = np.asarray(grid) if grid is not None else np.linspace(tau / 6, tau, 6)
    ts = np.concatenate([[0.0], grid])
    s = [1.0] + [ice_survival_regime(df, a0, a1, float(t)) for t in grid]
    s = np.array(s)
    return float(np.sum((s[:-1] + s[1:]) / 2 * np.diff(ts)))


def g_estimation_psi1(df):
    """Last-blip g-estimation: the structural log-HR of A1 = Cox coefficient of A1 adjusting the FULL
    pre-A1 history (W, A0, L1). Valid because L1 precedes A1 (the survival analogue of exp45's final
    peel-off blip). Returns the estimated ψ1 (log hazard ratio)."""
    from lifelines import CoxPHFitter
    d = df[["T_obs", "event", "A1", "W", "A0", "L1"]].copy()
    cph = CoxPHFitter(penalizer=1e-4).fit(d, duration_col="T_obs", event_col="event")
    return float(cph.params_["A1"])


def naive_km_regime(df, tau):
    """Biased comparator 1: per-(A0,A1) Kaplan–Meier survival at τ, contrast (1,1)-(0,0). Ignores W
    confounding of A0 and L1 confounding of A1 → observational, not interventional."""
    from lifelines import KaplanMeierFitter
    def s(a0, a1):
        d = df[(df["A0"] == a0) & (df["A1"] == a1)]
        return float(KaplanMeierFitter().fit(d["T_obs"], d["event"])
                     .survival_function_at_times([tau]).iloc[0])
    S = {(a0, a1): s(a0, a1) for a0 in (0.0, 1.0) for a1 in (0.0, 1.0)}
    return {"surv": S, "contrast_surv": S[(1.0, 1.0)] - S[(0.0, 0.0)]}


def naive_cox_A0(df):
    """Biased comparator 2: Cox of churn on (A0, A1, L1, W) — adjusting the mediator L1 blocks A0's
    engagement-mediated benefit, so the A0 log-HR is attenuated toward 0. Returns coef[A0]."""
    from lifelines import CoxPHFitter
    d = df[["T_obs", "event", "A0", "A1", "L1", "W"]].copy()
    cph = CoxPHFitter(penalizer=1e-4).fit(d, duration_col="T_obs", event_col="event")
    return float(cph.params_["A0"])


# ---------------------------------------------------------------- constraint-based structure discovery
def discover_timevarying_structure(df, seed=0):
    """Recover L1's DUAL ROLE — mediator of A0 *and* confounder of A1 — from observed (W,A0,L1,A1,event) with the
    ZFCI conditional-independence detector (exp39/exp46). This is the STRUCTURE layer's *power*-use: unlike the
    confounded case (blind to a hidden U by construction), here the ambiguous role of an OBSERVED variable is
    exactly what CI structure resolves — so the g-methods requirement is *validated empirically*, not asserted.

    Key tests (temporal order W < A0 < L1 < A1 < T fixes the arrow directions):
      • A0 ⫫̸ L1 | W          ⇒ A0→L1  (L1 is a CHILD of A0 = a mediator of A0's effect)
      • L1 ⫫̸ A1 | (W,A0)     ⇒ L1→A1  (feedback: L1 is a PARENT of A1)
      • L1 ⫫̸ Y  | (W,A0,A1)  ⇒ L1→Y   (L1 is a PARENT of churn) ⟹ with L1→A1, L1 CONFOUNDS the A1→Y relation
      • A1 ⫫̸ Y  | (W,A0,L1)  ⇒ A1→Y   (the direct last-blip ψ1)
      • A1 ⫫  A0 | L1         ⇒ negative control: A1 is driven ONLY by L1, so discovery discriminates (SUPPORTS)

    L1 ∈ descendants(A0) ∧ L1 ∈ parents(A1)∩parents(Y) ⟹ treatment-affected confounder: adjusting L1 blocks
    A0's L1-mediated path (biases the A0 total effect), NOT adjusting it leaves A1 confounded ⟹ no single static
    adjustment set works ⟹ g-methods. Returns each verdict + the derived reading. Self-checks against the planted DGP."""
    from causal_bench.detectors.zero_flow_ci import zero_flow_ci_test
    W, A0, L1, A1 = (df[k].values.astype(float) for k in ("W", "A0", "L1", "A1"))
    Y = df["event"].values.astype(float)

    def ci(X, Yv, *Zs):
        Z = np.column_stack(Zs) if Zs else np.zeros((len(X), 1))
        r = zero_flow_ci_test(X, Yv, Z, rng=np.random.default_rng(seed))
        return {"verdict": r.verdict, "p": round(float(r.p_value), 3)}

    tests = {
        "A0 ⫫ L1 | W":         ci(A0, L1, W),
        "L1 ⫫ A1 | (W,A0)":    ci(L1, A1, W, A0),
        "L1 ⫫ Y  | (W,A0,A1)": ci(L1, Y, W, A0, A1),
        "A1 ⫫ Y  | (W,A0,L1)": ci(A1, Y, W, A0, L1),
        "A1 ⫫ A0 | L1  (neg control)": ci(A1, A0, L1),
    }
    v = {k: t["verdict"] for k, t in tests.items()}
    l1_child_of_A0 = v["A0 ⫫ L1 | W"] == "refutes"
    l1_confounds_A1 = (v["L1 ⫫ A1 | (W,A0)"] == "refutes") and (v["L1 ⫫ Y  | (W,A0,A1)"] == "refutes")
    discriminates = v["A1 ⫫ A0 | L1  (neg control)"] == "supports"
    return {"tests": tests,
            "L1_is_mediator_of_A0": l1_child_of_A0,
            "L1_is_confounder_of_A1": l1_confounds_A1,
            "treatment_affected_confounder": l1_child_of_A0 and l1_confounds_A1,
            "discriminates": discriminates}
