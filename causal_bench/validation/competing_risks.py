"""exp53 — COMPETING RISKS: 'treating a competing event as censoring is biased'.

The classic competing-risks error, product-framed. Two competing exits: cause 1 = CHURN (the event of
interest), cause 2 = UPGRADE (a competing exit — an upgraded team can no longer churn from the plan they
left). Plus administrative/random censoring. The estimand is the cause-1 cumulative incidence CIF₁(τ|A) =
P(churn by τ under treatment A), and the treatment effect on it.

The trap: the naive 1−KM (Kaplan–Meier for churn, treating UPGRADE as if it were censoring) OVER-estimates
CIF₁ — it pretends upgraded teams remain at risk of churn. The correct estimator is the Aalen–Johansen CIF,
which accounts for the competing exit removing them. Treatment A is RANDOMIZED here so the ONLY bias is the
competing-risks one (no confounding to disentangle). Self-validating vs interventional MC truth.

This is a Python-only teaching demo (no CONCRETE). The doubly-robust *continuous-time RMTIF* version — restricted
mean time in the ACTIVE state under confounding — is the one load-bearing case for concrete (grid-TMLE can't
represent time-in-state with competing exits). It inherits #223's residual-confounding contrast-compression (the
default-learner-attenuation gate was FALSIFIED), so it needs the adjustment fixes + the Kish-ESS overlap guardrail
(exp52/exp55). See docs/plans/2026-09-23-concrete-rmst-fixes.md and the linked issue.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_B10, _BA1 = -0.9, -0.6      # churn (cause 1) log-hazard: base, treatment effect (A reduces churn)
_B20 = -1.1                  # upgrade (cause 2) log-hazard (competing)
_LAMC = 0.25                 # random censoring rate


def sim_competing(n, *, tau=3.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.binomial(1, 0.5, n).astype(int)                       # RANDOMIZED — isolates the competing-risks bias
    lam1 = np.exp(_B10 + _BA1 * A); lam2 = np.exp(np.full(n, _B20))
    T1 = rng.exponential(1.0 / lam1); T2 = rng.exponential(1.0 / lam2)
    C = rng.exponential(1.0 / _LAMC)
    T = np.minimum(T1, T2); cause = np.where(T1 < T2, 1, 2)
    T_obs = np.minimum(np.minimum(T, C), tau)
    event = np.where((T <= C) & (T <= tau), cause, 0)             # 0=censored, 1=churn, 2=upgrade
    return pd.DataFrame({"T_obs": T_obs, "event": event, "A": A})


def true_cif1(*, tau=3.0, n=2_000_000, seed=99):
    """Interventional MC truth: CIF₁(τ|a) = P(T1≤τ, T1<T2 | do(A=a)) per arm, + the treatment effect."""
    rng = np.random.default_rng(seed)
    def cif(a):
        lam1 = np.exp(_B10 + _BA1 * a); lam2 = np.exp(_B20)
        T1 = rng.exponential(1.0 / lam1, n); T2 = rng.exponential(1.0 / lam2, n)
        return float(np.mean((T1 <= tau) & (T1 < T2)))
    c0, c1 = cif(0.0), cif(1.0)
    return {"cif1": {0.0: c0, 1.0: c1}, "effect": c1 - c0}


# ---- estimators (per arm) ----
def naive_1km_cif(df, tau):
    """WRONG: 1 − Kaplan–Meier for churn, treating UPGRADE (event==2) as censoring. Over-estimates CIF₁."""
    from lifelines import KaplanMeierFitter
    out = {}
    for a in (0.0, 1.0):
        d = df[df["A"] == a]
        churn = (d["event"] == 1).astype(int)                    # upgrade & admin-censor both counted as censored
        km = KaplanMeierFitter().fit(d["T_obs"], churn)
        out[a] = 1.0 - float(km.survival_function_at_times([tau]).iloc[0])
    return out


def aalen_johansen_cif(df, tau):
    """CORRECT: Aalen–Johansen cumulative incidence for cause 1 (churn), accounting for the competing upgrade."""
    from lifelines import AalenJohansenFitter
    out = {}
    for a in (0.0, 1.0):
        d = df[df["A"] == a]
        ajf = AalenJohansenFitter(calculate_variance=False)
        ajf.fit(d["T_obs"].values, d["event"].values, event_of_interest=1)
        cif = ajf.cumulative_density_                            # index = times, one column
        col = cif.columns[0]
        out[a] = float(cif.loc[cif.index <= tau, col].iloc[-1]) if (cif.index <= tau).any() else 0.0
    return out
