"""exp45 — estimators for the time-varying app co-intervention (spec 2026-07-29).

A T=2 longitudinal structure with treatment-confounder feedback — the multi-period case the
repo's two-timepoint `LTMLEEstimator` cannot represent:

    W → A0 → L1 → A1 → Y,   with  A0→L1,  L1→A1 (feedback),  L1→Y.

Linear-additive structural nested mean model: the blip of A1 is ψ1 (+ effect modification by
L1), the blip of A0 is ψ0. Two estimands, two routes (both target different objects; naive is
biased by the feedback):
  • **regime mean** E[Y_ā] and contrast — via **sequential regression (ICE)**, the g-computation
    backbone of multi-period LTMLE (targeting is the DR refinement, noted for phase 2);
  • **structural blip ψ** — via **g-estimation** (blip-down / peel-off), which also recovers
    effect modification.
Self-validating: closed-form regime means + ψ (verified by a large interventional MC draw).

All estimators are OLS/logistic (numpy/sklearn) — no MCMC.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# structural coefficients (fixed; the truth the estimators must recover)
_B_LW, _B_L0 = 0.5, 0.8          # L1 = _B_L0*A0 + _B_LW*W + noise
_B_YW, _B_YL = 0.5, 0.7          # Y  = _B_YW*W + _B_YL*L1 + ψ0*A0 + (ψ1+em*L1)*A1 + noise


def sim_longitudinal(n, *, psi0=0.3, psi1=0.5, effect_mod=0.0, seed=0):
    """Draw the T=2 cohort. Returns a dict of arrays {W, A0, L1, A1, Y}."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=n)
    A0 = (rng.random(n) < _expit(0.3 * W)).astype(float)
    L1 = _B_L0 * A0 + _B_LW * W + rng.normal(size=n)          # confounder affected by A0
    A1 = (rng.random(n) < _expit(0.6 * L1)).astype(float)     # feedback: A1 depends on L1
    Y = _B_YW * W + _B_YL * L1 + psi0 * A0 + (psi1 + effect_mod * L1) * A1 + rng.normal(size=n)
    return {"W": W, "A0": A0, "L1": L1, "A1": A1, "Y": Y}


def true_values(*, psi0=0.3, psi1=0.5, effect_mod=0.0, n=400_000, seed=99):
    """Interventional MC truth: regime means E[Y_{a0,a1}] and the blip ψ=(ψ0, ψ1)."""
    rng = np.random.default_rng(seed)
    means = {}
    for a0 in (0.0, 1.0):
        for a1 in (0.0, 1.0):
            W = rng.normal(size=n)
            L1 = _B_L0 * a0 + _B_LW * W + rng.normal(size=n)          # do(A0=a0)
            Y = _B_YW * W + _B_YL * L1 + psi0 * a0 + (psi1 + effect_mod * L1) * a1 + rng.normal(size=n)
            means[(a0, a1)] = float(Y.mean())
    contrast = means[(1.0, 1.0)] - means[(0.0, 0.0)]
    # Structural blips (what g-estimation targets): the A0 blip is the TOTAL effect of A0 not
    # through A1 — direct + the L1-mediated path — so it is ψ0 + _B_YL·_B_L0, NOT ψ0 alone.
    blip_A0 = means[(1.0, 0.0)] - means[(0.0, 0.0)]
    blip_A1 = means[(0.0, 1.0)] - means[(0.0, 0.0)]
    return {"regime_means": means, "contrast": contrast,
            "blip_A0": blip_A0, "blip_A1": blip_A1, "psi_params": (psi0, psi1)}


# --------------------------------------------------------------- estimators

def g_estimation(data):
    """g-estimation of the linear SNMM (blip-down). Returns (ψ0_hat, ψ1_hat).

    ψ1 (last blip) = OLS coefficient of A1 adjusting for ALL pre-A1 history (W, A0, L1) — valid
    because L1 precedes A1. Peel it off: H = Y − ψ1·A1 (the outcome absent the A1 blip). ψ0 = OLS
    coefficient of A0 in H adjusting for ONLY pre-A0 covariates (W) — L1 is post-A0 and must be
    excluded."""
    W, A0, L1, A1, Y = (data[k] for k in ("W", "A0", "L1", "A1", "Y"))
    X1 = np.column_stack([A1, W, A0, L1])
    psi1 = LinearRegression().fit(X1, Y).coef_[0]
    H = Y - psi1 * A1
    X0 = np.column_stack([A0, W])
    psi0 = LinearRegression().fit(X0, H).coef_[0]
    return float(psi0), float(psi1)


def g_estimation_effect_mod(data):
    """Recover the A1 blip's modification by L1: ψ1 + em·L1, i.e. the coefficient of the A1×L1
    interaction — the object g-estimation surfaces that a regime-mean does not."""
    W, A0, L1, A1, Y = (data[k] for k in ("W", "A0", "L1", "A1", "Y"))
    X = np.column_stack([A1, A1 * L1, W, A0, L1])
    coef = LinearRegression().fit(X, Y).coef_
    return {"psi1": float(coef[0]), "effect_mod": float(coef[1])}


def msm_iptw(data, *, stabilized=True, trunc=0.01):
    """Time-varying IPTW marginal structural model (Robins) — the entry-rung g-method, added as a
    BASELINE against g-estimation (ψ) and ICE/LTMLE (regime contrast). Fits the saturated MSM
    E[Y_{a0,a1}] = β0 + β1 a0 + β2 a1 + β3 a0 a1 by weighted OLS with stabilized inverse-
    treatment-probability weights over the two decision points:
        sw = [P(A0)·P(A1|A0)] / [P(A0|W)·P(A1|W,A0,L1)],
    and returns the regime contrast E[Y_{11}]−E[Y_{00}] = β1+β2+β3. Unlike `naive_effects` it does
    NOT put the feedback confounder L1 in the OUTCOME model — L1 enters only the A1 treatment model —
    so it is unbiased under **sequential exchangeability** (no unmeasured time-varying confounding).
    SINGLY robust (needs BOTH treatment models correct); LTMLE / g-estimation are the DR upgrades.
    """
    W, A0, L1, A1, Y = (np.asarray(data[k], float) for k in ("W", "A0", "L1", "A1", "Y"))
    n = len(Y)
    # treatment model at t=0: P(A0=1 | W); at t=1: P(A1=1 | W, A0, L1)  (L1 = pre-A1 history)
    g0 = LogisticRegression().fit(W.reshape(-1, 1), A0).predict_proba(W.reshape(-1, 1))[:, 1]
    p0 = np.where(A0 == 1, g0, 1 - g0)
    X1 = np.column_stack([W, A0, L1])
    g1 = LogisticRegression().fit(X1, A1).predict_proba(X1)[:, 1]
    p1 = np.where(A1 == 1, g1, 1 - g1)
    denom = p0 * p1
    if stabilized:
        pA0 = A0.mean(); num0 = np.where(A0 == 1, pA0, 1 - pA0)
        gA1 = LogisticRegression().fit(A0.reshape(-1, 1), A1).predict_proba(A0.reshape(-1, 1))[:, 1]
        num1 = np.where(A1 == 1, gA1, 1 - gA1)
        w = (num0 * num1) / denom
    else:
        w = 1.0 / denom
    # weight-health / positivity diagnostic BEFORE truncation: Kish ESS = (Σw)²/Σw² (the same helper the
    # twisted SMC-IPCW uses as its resample trigger — low ESS = weights collapsed onto few units = poor overlap).
    from causal_bench.sampling import kish_ess
    ess = kish_ess(np.log(np.clip(w, 1e-300, None)))
    lo, hi = np.quantile(w, [trunc, 1 - trunc]); w = np.clip(w, lo, hi)   # positivity truncation
    sw = np.sqrt(w)
    Xm = np.column_stack([np.ones(n), A0, A1, A0 * A1]) * sw[:, None]     # saturated MSM, weighted OLS
    beta = np.linalg.lstsq(Xm, Y * sw, rcond=None)[0]
    return {"contrast": float(beta[1] + beta[2] + beta[3]), "beta": [float(b) for b in beta],
            "ess": float(ess), "ess_frac": float(ess / n), "max_weight": float(w.max())}


def ice_regime_mean(data, a0, a1):
    """Sequential regression (ICE) estimate of E[Y_{a0,a1}] — the LTMLE g-computation backbone.
    Q2 = E[Y|A0,L1,A1,W] set to A1=a1; Q1 = E[Q2|A0,W] set to A0=a0; mean of Q1."""
    W, A0, L1, A1, Y = (data[k] for k in ("W", "A0", "L1", "A1", "Y"))
    q2_model = LinearRegression().fit(np.column_stack([A0, L1, A1, W]), Y)
    q2 = q2_model.predict(np.column_stack([A0, L1, np.full_like(A1, a1), W]))   # do(A1=a1)
    q1_model = LinearRegression().fit(np.column_stack([A0, W]), q2)
    q1 = q1_model.predict(np.column_stack([np.full_like(A0, a0), W]))           # do(A0=a0)
    return float(q1.mean())


def ice_contrast(data):
    """The regime contrast E[Y_{1,1}] − E[Y_{0,0}] via sequential regression."""
    return ice_regime_mean(data, 1.0, 1.0) - ice_regime_mean(data, 0.0, 0.0)


def naive_effects(data):
    """The biased comparator: OLS of Y on (A0, A1) adjusting for L1 and W. Adjusting for the
    feedback confounder L1 blocks A0's effect through L1 (and L1 is a collider on A0→L1←...), so
    the A0 coefficient is biased for the causal blip — the whole point of needing g-methods."""
    W, A0, L1, A1, Y = (data[k] for k in ("W", "A0", "L1", "A1", "Y"))
    coef = LinearRegression().fit(np.column_stack([A0, A1, L1, W]), Y).coef_
    return {"A0": float(coef[0]), "A1": float(coef[1])}
