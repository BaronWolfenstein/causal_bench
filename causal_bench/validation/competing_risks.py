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
    C = rng.exponential(1.0 / _LAMC, n)                           # per-subject iid censoring (was a shared scalar)
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


# ---------------------------------------------------------------- continuous-time RMTL + efficient-influence SE
# Restricted Mean Time Lost to churn: RMTL₁(τ) = ∫₀^τ F₁(u) du (area under the cause-1 CIF); RMST_active = τ − RMTL₁
# (time-in-state). The efficient influence function is a martingale integral ∫ W_c(t) dM_c(t); the integrand
# weights W₁,W₂ (Andersen; RMTL directional derivative through the competing CIFs) are computed in O(1) per
# event time using the running-scalar identity ∫ₜ^τ F₁ = RMTL₁(τ) − RMTL₁(t), so no O(N²) look-ahead integral.
def _calculate_integrand_weights(times, dL1, dL2, tau):
    """Port of the O(1) forward-pass integrand weights (paste-2 math): given cause-specific hazard increments
    dΛ₁,dΛ₂ at ascending `times` (≤τ handled by the caller), return (W1, W2, F1_at, F2_at, rmtl1) where
    W₁(t)=(1−F₂(t))(τ−t) − ∫ₜ^τF₁,  W₂(t)=F₁(t)(τ−t) − ∫ₜ^τF₁, and ∫ₜ^τF₁ = rmtl1 − running_rmtl(t)."""
    m = len(times)
    F1 = np.zeros(m); F2 = np.zeros(m); S = 1.0; f1 = 0.0; f2 = 0.0
    # first pass: CIFs at each event time, and total RMTL₁(τ) (step-integral of F1)
    last = 0.0; rmtl1 = 0.0
    for k in range(m):
        t = times[k]
        rmtl1 += f1 * (t - last)                                 # accrue area under F1 up to t (F1 just before)
        f1 += S * dL1[k]; f2 += S * dL2[k]; S *= (1.0 - dL1[k] - dL2[k])
        F1[k] = f1; F2[k] = f2; last = t
    rmtl1 += f1 * (tau - last)                                   # tail area t_last→τ
    # second pass: O(1) weights via the running-scalar remaining-area
    W1 = np.zeros(m); W2 = np.zeros(m); running = 0.0; last = 0.0; f1p = 0.0
    for k in range(m):
        t = times[k]
        running += f1p * (t - last)                              # area under F1 accrued up to t
        remaining = rmtl1 - running                              # = ∫ₜ^τ F1(u) du, in O(1)
        W1[k] = (1.0 - F2[k]) * (tau - t) - remaining
        W2[k] = F1[k] * (tau - t) - remaining
        f1p = F1[k]; last = t
    return W1, W2, F1, F2, rmtl1


def rmtl1_arm(d, tau):
    """One-sample continuous-time RMTL₁(τ) for an arm `d` (cols T_obs,event) with its efficient-influence SE.
    Independent censoring is absorbed by the risk set Y(t) (classic Aalen–Johansen martingale variance), so the
    IC is φ_i = Σ_c ∫₀^τ W_c(t)[dN_{c,i}−Y_i dΛ_c]/π(t), π(t)=Y(t)/n. Returns (rmtl1, se, per-subject IC)."""
    T = d["T_obs"].values.astype(float); E = d["event"].values.astype(int); n = len(T)
    order = np.argsort(T); Ts = T[order]; Es = E[order]
    times = np.unique(Ts[Es >= 1])                               # distinct EVENT times (cause 1 or 2)
    times = times[times <= tau + 1e-12]
    if len(times) == 0:
        return 0.0, 0.0, np.zeros(n)
    # risk set + cause counts at each event time (Y = #{T_obs >= t})
    Y = np.array([np.sum(T >= t - 1e-12) for t in times], float)
    n1 = np.array([np.sum((T == t) & (E == 1)) for t in times], float)  # exact ties on the simulated grid
    n2 = np.array([np.sum((T == t) & (E == 2)) for t in times], float)
    dL1 = n1 / Y; dL2 = n2 / Y
    W1, W2, F1, F2, rmtl1 = _calculate_integrand_weights(times, dL1, dL2, tau)
    pi = Y / n                                                   # empirical P(at risk at t)
    g = (W1 * dL1 + W2 * dL2) / pi                               # compensator term per event time
    G = np.cumsum(g)                                             # G(t_k) = Σ_{j≤k} g_j
    # per-subject IC: own-event jump minus cumulative compensator up to T_obs_i (at-risk while t_k ≤ T_obs_i)
    tk_index = {t: k for k, t in enumerate(times)}
    IC = np.zeros(n)
    Gtot_at = np.searchsorted(times, T, side="right") - 1        # last event-time index ≤ T_obs_i
    IC = np.where(Gtot_at >= 0, -G[np.clip(Gtot_at, 0, len(G) - 1)], 0.0)
    for i in range(n):
        if E[i] in (1, 2) and T[i] <= tau + 1e-12:
            k = tk_index.get(T[i])
            if k is not None:
                w = W1[k] if E[i] == 1 else W2[k]
                IC[i] += w / pi[k]
    se = float(np.sqrt(np.var(IC, ddof=1) / n))
    return float(rmtl1), se, IC


def rmtl1_effect(df, tau):
    """Continuous-time RMTL₁ churn-time-lost per arm + the treatment effect, each with its efficient-influence SE.
    (Randomized A ⇒ the two arms are independent one-sample functionals; effect SE combines the arm ICs.)"""
    out = {}; se = {}
    for a in (0.0, 1.0):
        r, s, _ = rmtl1_arm(df[df["A"] == a].reset_index(drop=True), tau)
        out[a] = r; se[a] = s
    eff = out[1.0] - out[0.0]
    eff_se = float(np.sqrt(se[0.0] ** 2 + se[1.0] ** 2))
    return {"rmtl1": out, "se": se, "effect": eff, "effect_se": eff_se}


def true_rmtl1(*, tau=3.0, n=2_000_000, seed=98):
    """Interventional MC truth: RMTL₁(τ|a) = ∫₀^τ CIF₁(u|a) du = E[max(τ−T1,0)·1{T1<T2} | do(A=a)]."""
    rng = np.random.default_rng(seed)
    def rmtl(a):
        lam1 = np.exp(_B10 + _BA1 * a); lam2 = np.exp(_B20)
        T1 = rng.exponential(1.0 / lam1, n); T2 = rng.exponential(1.0 / lam2, n)
        return float(np.mean(np.maximum(tau - T1, 0.0) * (T1 < T2)))
    r0, r1 = rmtl(0.0), rmtl(1.0)
    return {"rmtl1": {0.0: r0, 1.0: r1}, "effect": r1 - r0}
