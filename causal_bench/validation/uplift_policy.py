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
    up_dr, up_nv = [], []
    for k in ks:
        pi = np.zeros(n, int); pi[order[:int(round(k * n))]] = 1
        up_dr.append(aipw_policy_value(nuis, pi) - v_none_dr)
        up_nv.append((naive_policy_value(d, pi) - v_none_nv) if 0 < k < 1 else (up_nv[-1] if up_nv else 0.0))
    up_dr = np.array(up_dr); up_nv = np.array(up_nv)
    rand = ks * (v_all_dr - v_none_dr)                             # random targeting = linear
    _diff = up_dr - rand
    qini_dr = float(np.sum((_diff[:-1] + _diff[1:]) / 2 * np.diff(ks)))   # trapezoid (numpy-version-safe)
    return {"ks": ks, "uplift_dr": up_dr, "uplift_naive": up_nv, "rand": rand,
            "qini_dr": qini_dr, "v_all_minus_none_dr": v_all_dr - v_none_dr}
