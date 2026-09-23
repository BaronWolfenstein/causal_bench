"""exp51 — SURVIVAL UPLIFT: heterogeneous treatment effect on a time-to-event outcome (CATE by an
effect modifier V) under confounding + informative censoring, using McCoy's CONCRETE fork (continuous-
time TMLE) as the doubly-robust engine.

Motivation: product "uplift" for retention/churn is a *time-to-event* CATE — which users (by a modifier
V) get the most survival benefit from the intervention. Standard uplift assumes a point outcome and
ignorable censoring; here treatment A is confounded by W1 and censoring is *informative* (depends on W1),
so a naive per-arm Kaplan–Meier RMST is biased. Per-stratum CONCRETE RMST (doubly robust, models the
censoring) recovers the true per-stratum uplift.

DGP (concrete-schema compatible):
    W1~N(0,1) confounder; V:=W2~Bern(1/2) effect modifier; W3,W4~N(0,1);
    A ~ Bern(expit(γ·W1))                                  (confounding)
    T ~ Exp(λ),  log λ = β0 + βW·W1 + βA·A + βAV·A·V       (treatment effect MODIFIED by V)
    C ~ Exp(λ_c), log λ_c = δ0 + δW·W1                     (INFORMATIVE censoring)
    T_obs = min(T, C, τ);  Delta = 1{T ≤ min(C, τ)}        (event_type ∈ {0,1})
Estimand: per-stratum SURVIVAL-PROBABILITY uplift CATE(v) = S(τ|A=1,V=v) − S(τ|A=0,V=v) (retention/
churn benefit), and its heterogeneity CATE(1)−CATE(0). Self-validating against interventional MC truth.
Note: causal_bench's DR survival estimators (TMLE-IPCW, concrete) report a RISK difference (event-
probability, = −CATE), so the estimators negate it to the survival-benefit scale.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_G_A = 0.7                       # A ~ expit(γ·W1) confounding (moderate)
_B0, _BW, _BA, _BAV = -0.7, 0.5, -0.6, -0.7     # log-hazard of T (treatment effect modified by V)
_D0, _DW = -0.8, 0.3            # log-hazard of censoring (mildly informative in W1)


def _expit(x): return 1.0 / (1.0 + np.exp(-x))


def sim_survival_uplift(n, *, tau=3.0, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=n); V = rng.binomial(1, 0.5, n).astype(float)
    W3 = rng.normal(size=n); W4 = rng.normal(size=n)
    A = (rng.random(n) < _expit(_G_A * W1)).astype(int)
    lamT = np.exp(_B0 + _BW * W1 + _BA * A + _BAV * A * V)
    T = rng.exponential(1.0 / lamT)
    lamC = np.exp(_D0 + _DW * W1)
    C = rng.exponential(1.0 / lamC)
    T_obs = np.minimum(np.minimum(T, C), tau)
    Delta = ((T <= C) & (T <= tau)).astype(int)             # 1=event, 0=censored (admin or informative)
    return pd.DataFrame({"T_obs": T_obs, "Delta": Delta, "A": A,
                         "W1": W1, "W2": V, "W3": W3, "W4": W4})


def true_cate(*, tau=3.0, n=2_000_000, seed=99):
    """Interventional MC truth: per-stratum SURVIVAL-PROBABILITY uplift CATE(v) = S(τ|A=1,V=v) −
    S(τ|A=0,V=v) (the retention/churn estimand), and its heterogeneity. S(τ|a,v) = E_{W1}[e^{−λτ}]."""
    rng = np.random.default_rng(seed); W1 = rng.normal(size=n)
    def surv(a, v):
        lam = np.exp(_B0 + _BW * W1 + _BA * a + _BAV * a * v)
        return float(np.mean(np.exp(-lam * tau)))
    cate = {v: surv(1, v) - surv(0, v) for v in (0.0, 1.0)}
    return {"cate": cate, "heterogeneity": cate[1.0] - cate[0.0]}


# ------------------------------------------------------------------ estimators (per stratum V=v)
# ESTIMAND: survival-probability difference at τ, CATE(v)=S(τ|1,v)−S(τ|0,v) (positive = treatment
# improves retention). The DR estimators (TMLE-IPCW, concrete) report a RISK difference
# (P(event by τ|1)−P(event|0) = −CATE), so we NEGATE their point estimate.
def km_cate(df, tau):
    """Naive per-arm Kaplan–Meier survival-difference per stratum — no confounding/censoring
    adjustment (biased low: A⊥̸W1 confounding makes treated look sicker + informative censoring)."""
    from lifelines import KaplanMeierFitter
    def surv(d_arm):
        return float(KaplanMeierFitter().fit(d_arm["T_obs"], d_arm["Delta"])
                     .survival_function_at_times([tau]).iloc[0])
    out = {}
    for v in (0.0, 1.0):
        d = df[df["W2"] == v]
        out[v] = surv(d[d["A"] == 1]) - surv(d[d["A"] == 0])
    return out


def tmle_cate(df, tau):
    """Per-stratum TMLE-IPCW survival-difference (doubly robust: adjusts W1–W4 + models censoring).
    Reports a risk difference → negated to the survival-benefit CATE."""
    from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
    out = {}
    for v in (0.0, 1.0):
        res = TMLEIPCWEstimator().estimate(df[df["W2"] == v].reset_index(drop=True), horizon=tau)
        out[v] = -float(res[0].point_estimate) if res else float("nan")
    return out


def concrete_cate(df, tau):
    """Per-stratum CONCRETE (McCoy fork) survival-difference — continuous-time TMLE, DR. The bridge
    returns the 'Life Years Lost'/CIF risk-difference contrast, so we NEGATE to the survival benefit.
    Returns {} if the R package is unavailable."""
    from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator, _concrete_available
    if not _concrete_available():
        return {}
    est = ConcreteRMSTEstimator(horizon=tau)
    out = {}
    for v in (0.0, 1.0):
        res = est.estimate(df[df["W2"] == v].reset_index(drop=True), horizon=tau)
        out[v] = -float(res[0].point_estimate) if res else float("nan")
    return out
