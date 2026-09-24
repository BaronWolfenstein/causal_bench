"""exp52 — UPLIFT TARGETING POLICY VALUE (doubly robust) + Qini curve.

What uplift modeling is *for*: not just estimating CATE, but turning it into a targeting policy ("treat the
top-k% by predicted uplift") and evaluating that policy's VALUE. The catch a product-DS must get right:
treatment is confounded, so the naive top-k outcome mean (the usual Qini) is BIASED. The doubly-robust
(AIPW) off-policy value estimator fixes it — the causal-correct version of Qini.

DGP (point treatment, heterogeneous, confounded):
    W1,W2 ~ N(0,1);  e(W)=expit(γ·W1);  A ~ Bern(e)                 (confounding)
    τ(W) = τ0 + τ1·W1 + τ2·W2   (HETEROGENEOUS CATE; some units harmed)
    Y = μ0(W) + τ(W)·A + noise
Policy value V(π)=E[Y(π(W))]=E[μ0 + τ·π]. Truth (interventional): treat-none, treat-all, and the OPTIMAL
policy π*=1{τ>cost}. Estimated from observational data via cross-fit nuisances → DR-learner τ̂ (ranking) →
AIPW policy value V̂_DR(π). Self-validating against the MC truth.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

_G = 0.8                                    # confounding strength
_T0, _T1, _T2 = 0.5, 1.0, -0.8              # heterogeneous CATE τ(W)=τ0+τ1 W1+τ2 W2
_M1, _M2 = 1.0, 0.5                         # baseline μ0(W)=μ1 W1+μ2 W2


def _expit(x): return 1.0 / (1.0 + np.exp(-x))
def _tau(W1, W2): return _T0 + _T1 * W1 + _T2 * W2
def _mu0(W1, W2): return _M1 * W1 + _M2 * W2


def sim_uplift(n, *, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=n); W2 = rng.normal(size=n)
    A = (rng.random(n) < _expit(_G * W1)).astype(int)
    Y = _mu0(W1, W2) + _tau(W1, W2) * A + rng.normal(size=n)
    return {"W1": W1, "W2": W2, "A": A, "Y": Y}


def true_values(*, cost=0.0, n=2_000_000, seed=99):
    rng = np.random.default_rng(seed); W1 = rng.normal(size=n); W2 = rng.normal(size=n)
    mu0 = _mu0(W1, W2); tau = _tau(W1, W2)
    v_none = float(mu0.mean()); v_all = float((mu0 + tau).mean())
    v_opt = float((mu0 + tau * (tau > cost)).mean())          # optimal targeting
    return {"v_none": v_none, "v_all": v_all, "v_opt": v_opt,
            "achievable": v_opt - v_none, "frac_treated_opt": float((tau > cost).mean())}


# ---- cross-fit nuisances: e(W)=P(A=1|W), μ_a(W)=E[Y|A=a,W] ----
def _crossfit(d, folds=2, seed=0):
    W = np.column_stack([d["W1"], d["W2"]]); A = np.asarray(d["A"]); Y = np.asarray(d["Y"], float); n = len(Y)
    rng = np.random.default_rng(seed); idx = rng.permutation(n); fold = np.array_split(idx, folds)
    ehat = np.zeros(n); mu0 = np.zeros(n); mu1 = np.zeros(n)
    for f in range(folds):
        te = fold[f]; tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        ehat[te] = LogisticRegression(max_iter=1000).fit(W[tr], A[tr]).predict_proba(W[te])[:, 1]
        m1 = LinearRegression().fit(W[tr][A[tr] == 1], Y[tr][A[tr] == 1])
        m0 = LinearRegression().fit(W[tr][A[tr] == 0], Y[tr][A[tr] == 0])
        mu1[te] = m1.predict(W[te]); mu0[te] = m0.predict(W[te])
    ehat = np.clip(ehat, 0.02, 0.98)
    return W, A, Y, ehat, mu0, mu1


def dr_learner_cate(d, folds=2, seed=0):
    """DR-learner (Kennedy): AIPW pseudo-outcome regressed on W → τ̂(W)."""
    W, A, Y, e, mu0, mu1 = _crossfit(d, folds, seed)
    phi = (mu1 - mu0) + A / e * (Y - mu1) - (1 - A) / (1 - e) * (Y - mu0)
    tau_hat = LinearRegression().fit(W, phi).predict(W)
    return tau_hat, (W, A, Y, e, mu0, mu1)


def aipw_policy_value(nuis, pi):
    """Doubly-robust off-policy value V̂(π)=mean[ μ_{π} + 1{A=π}/P(A=π|W)·(Y−μ_A) ] for deterministic π∈{0,1}^n."""
    W, A, Y, e, mu0, mu1 = nuis
    mu_pi = np.where(pi == 1, mu1, mu0)
    p_pi = np.where(pi == 1, e, 1 - e)
    corr = np.where(A == pi, (Y - np.where(A == 1, mu1, mu0)) / p_pi, 0.0)
    return float(np.mean(mu_pi + corr))


def policy_value_ess(nuis, pi):
    """Kish ESS of the AIPW off-policy correction weights w = 1{A=π}/P(A=π|W) — the overlap diagnostic for
    V̂(π). As the TARGET policy π diverges from the logging propensity e(W), 1/p_pi explodes on the matched
    units, the ESS collapses, and V̂(π) is dominated by a few high-weight units (its variance blows up). This
    is the off-policy-EVALUATION analogue of the PPO reuse gate ESS(π_θ/π_old): both measure how usable a batch
    drawn under one policy is for a *different* target policy. Returns (ess, ess_frac = ess/n).

    NB: this is our own diagnostic (Kish/Smola), not stated in Martin's RL-as-nonequilibrium-MC note — that note
    supplies the staleness *mechanism* (the refresh rate ν and the ratio r=π_θ/π_old), never an ESS gate."""
    from causal_bench.sampling import kish_ess
    _, A, _, e, _, _ = nuis
    A = np.asarray(A)
    p_pi = np.where(pi == 1, e, 1 - e)
    w = np.where(A == pi, 1.0 / p_pi, 0.0)                      # AIPW correction weight (0 on unmatched units)
    ess = kish_ess(np.log(np.clip(w, 1e-300, None)))
    return float(ess), float(ess / len(A))


# ---------------------------------------------------------------- unmeasured-confounder variant
# ESS is NECESSARY but NOT SUFFICIENT. Here a hidden binary U ("enterprise mandate") confounds A and Y,
# while the TRUE treatment effect is exactly 0. A depends on U (hidden) and W1 (observed), so the observed
# propensity e(W1) is smoothly distributed → the Kish ESS looks healthy (green light), yet V̂_DR is biased:
# AIPW targets the W-adjusted functional E[E[Y|A=1,W]]−E[E[Y|A=0,W]], which ≠ the interventional effect
# because W does not block the A←U→Y back-door. The overlap diagnostic cannot see a DAG violation — the same
# lesson as the collider/estimand-discipline thread: a robust estimator can't rescue a mis-specified estimand.
_GU, _GW_A, _BU, _PU = 1.5, 0.5, 2.0, 0.2   # U→A, W1→A, U→Y strengths; P(U=1)


def sim_uplift_confounded(n, *, seed=0):
    """Uplift cohort with an UNMEASURED confounder U and a TRUE treatment effect of exactly 0. U is withheld
    from the returned data (only W1,W2,A,Y observed). A ~ Bern(expit(_GW_A·W1 + _GU·U − 0.5))."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=n); W2 = rng.normal(size=n)
    U = (rng.random(n) < _PU).astype(float)                        # unmeasured
    A = (rng.random(n) < _expit(_GW_A * W1 + _GU * U - 0.5)).astype(int)
    Y = _mu0(W1, W2) + _BU * U + 0.0 * A + rng.normal(size=n)      # τ≡0; U drives Y (the hidden confounding)
    return {"W1": W1, "W2": W2, "A": A, "Y": Y}                    # U NOT returned


def ate_overlap_ess(A, e):
    """Kish ESS of the ATE inverse-probability weights w = A/ê + (1−A)/(1−ê) across ALL units — the standard
    positivity/overlap diagnostic. (Distinct from `policy_value_ess`, whose treat-all/none correction zeroes
    one arm and so caps near 50% by construction, regardless of overlap.) Returns (ess, ess_frac=ess/n)."""
    from causal_bench.sampling import kish_ess
    A = np.asarray(A); e = np.clip(np.asarray(e), 1e-6, 1 - 1e-6)
    w = A / e + (1 - A) / (1 - e)
    ess = kish_ess(np.log(np.clip(w, 1e-300, None)))
    return float(ess), float(ess / len(A))


def confounded_trap(d, folds=2, seed=0):
    """Fit the SAME cross-fit AIPW machinery on the observed (W1,W2,A,Y) — U hidden — and report the pair that
    makes the point: the DR treat-all vs treat-none contrast (biased away from the true 0) alongside a HEALTHY
    ATE overlap ESS and a moderate propensity spread. Returns the trap: green overlap, wrong answer."""
    n = len(d["A"])
    _, nuis = dr_learner_cate(d, folds=folds, seed=seed)           # nuisances on OBSERVED W only
    A, e = nuis[1], nuis[3]
    contrast = aipw_policy_value(nuis, np.ones(n, int)) - aipw_policy_value(nuis, np.zeros(n, int))
    ess, essf = ate_overlap_ess(A, e)                              # the genuine positivity diagnostic (all units)
    return {"contrast_dr": float(contrast), "contrast_true": 0.0,
            "ess": ess, "essf": float(essf),
            "e_min": float(e.min()), "e_max": float(e.max())}


def naive_policy_value(d, pi):
    """Naive (confounded) value: mean observed Y among those whose observed A matches π — the biased Qini path."""
    A = np.asarray(d["A"]); Y = np.asarray(d["Y"], float)
    match = A == pi
    return float(Y[match].mean()) if match.any() else float("nan")


def qini(tau_hat, nuis, d, ks=None):
    """Uplift curve V(top-k)−V(none) vs k for both DR and naive value; Qini coeff = area above random line."""
    n = len(tau_hat); order = np.argsort(-tau_hat)                 # rank by predicted uplift, descending
    ks = ks if ks is not None else np.linspace(0.0, 1.0, 21)
    v_none_dr = aipw_policy_value(nuis, np.zeros(n, int))
    v_all_dr = aipw_policy_value(nuis, np.ones(n, int))
    v_none_nv = naive_policy_value(d, np.zeros(n, int))
    up_dr, up_nv, ess_dr = [], [], []
    for k in ks:
        pi = np.zeros(n, int); pi[order[:int(round(k * n))]] = 1
        up_dr.append(aipw_policy_value(nuis, pi) - v_none_dr)
        ess_dr.append(policy_value_ess(nuis, pi)[1])               # per-bin overlap of the top-k targeting policy
        up_nv.append((naive_policy_value(d, pi) - v_none_nv) if 0 < k < 1 else (up_nv[-1] if up_nv else 0.0))
    up_dr = np.array(up_dr); up_nv = np.array(up_nv); ess_dr = np.array(ess_dr)
    rand = ks * (v_all_dr - v_none_dr)                             # random targeting = linear
    _diff = up_dr - rand
    qini_dr = float(np.sum((_diff[:-1] + _diff[1:]) / 2 * np.diff(ks)))   # trapezoid (numpy-version-safe)
    # ess_frac_dr flags where the uplift curve stops being trustworthy: as overlap thins along the ranking the
    # AIPW value's effective n collapses, so a low-ESS bin's uplift point is variance-dominated, not signal.
    return {"ks": ks, "uplift_dr": up_dr, "uplift_naive": up_nv, "rand": rand, "ess_frac_dr": ess_dr,
            "min_ess_frac_dr": float(ess_dr.min()),
            "qini_dr": qini_dr, "v_all_minus_none_dr": v_all_dr - v_none_dr}
