"""exp57 — CALIBRATION vs CAUSAL COUNT: a calibrated probability you can *count* on observationally is NOT a
count you can *act* on. The counting/calibration-axis analogue of exp52's ESS "necessary but not sufficient."

Motivation (R.D. Montgomery, "A Probability You Can Count On"): calibrated event probabilities can be SUMMED to
an expected count — "among items called 0.8, ~80% come true," so Σᵢ p̂ᵢ estimates the count, with Poisson-binomial
variance Σᵢ p̂ᵢ(1−p̂ᵢ). True and useful — for the OBSERVED world. But a product question like "how many teams
would we RETAIN if we shipped the nudge to everyone?" is INTERVENTIONAL, and a calibrated *predictor* answers it
wrong under confounding: calibration is necessary for an honest observational count, not sufficient for a causal one.

DGP (confounded binary churn; U hidden):
    W ~ N(0,1) observed;  U ~ Bern(_PU) UNMEASURED ("enterprise mandate");
    A ~ Bern(expit(_A0 + _AW·W + _AU·U))         (U confounds who gets the nudge)
    churn Y ~ Bern(expit(_B0 + _BW·W + _BU·U + _BA·A))   (A protective; U raises churn AND uptake)

Three claims, each self-validated against interventional MC truth:
  1. **Calibration holds** — a logistic p̂(W,A) on observed data has small ECE (Montgomery's necessary condition).
  2. **The observational count is right** — Σ p̂(Wᵢ,Aᵢ) ≈ true # churns (Montgomery's counting works observationally).
  3. **The interventional count is wrong** — Σ p̂(Wᵢ, A=1) ≠ true count under do(A=1) (calibration ≠ causal validity),
     while the ORACLE Σ p̂(Wᵢ,Uᵢ,A=1) recovers it → the gap is the *unmeasured confounder*, not the method.

The SUFFICIENCY side (the point of the exercise): the thing that *addresses* the gap ESS/calibration are blind to
is a **sensitivity analysis** — the VanderWeele–Ding **E-value**: the minimum strength of unmeasured confounding
(on the risk-ratio scale, with BOTH treatment and outcome) that could explain away the naive interventional
effect. It does NOT prove no confounding (that is irreducibly impossible from data); it BOUNDS the vulnerability.
So "sufficient" is never a single green light — it is the conjunction (positivity=ESS) ∧ (functional form=DR/EIC)
∧ (estimand=DAG discipline) ∧ (a sensitivity analysis bounding the unmeasured-confounding gap).

How this composes with the structure layer (ZFCI / Markov blanket, exp39/exp46): those tools sit UPSTREAM — they
pick variables / recover the graph — while calibration here is DOWNSTREAM output-honesty; they are complementary,
not substitutes. And a ZFCI test over the OBSERVED variables is necessary-but-not-sufficient in the *same* way this
demo's calibration is: it cannot see the hidden U (no observed-variable CI test can), so it too would pass while
the causal count stays biased. The Markov blanket is the optimal *predictive* feature set (good for the calibrated
p̂ above), but is NOT the causal adjustment set (it contains colliders) — the estimand-discipline distinction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_PU = 0.35
_A0, _AW, _AU = -0.3, 0.7, 1.3        # A ~ expit(_A0 + _AW·W + _AU·U): U drives uptake
_B0, _BW, _BU, _BA = -0.4, 0.5, 1.4, -0.8   # churn: base, W, U (raises churn), A (protective)


def _expit(x): return 1.0 / (1.0 + np.exp(-x))


def sim_churn_confounded(n, *, seed=0):
    """Confounded binary-churn cohort. Returns observed (W, A, Y); U is withheld."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=n)
    U = (rng.random(n) < _PU).astype(float)                       # UNMEASURED
    A = (rng.random(n) < _expit(_A0 + _AW * W + _AU * U)).astype(int)
    Y = (rng.random(n) < _expit(_B0 + _BW * W + _BU * U + _BA * A)).astype(int)  # churn
    return pd.DataFrame({"W": W, "A": A, "Y": Y, "_U": U})        # _U kept for the oracle; drop before "observed" use


def true_rates(*, n=4_000_000, seed=99):
    """Interventional MC truth (per-capita churn rates → counts scale by n): observational E[Y], and do(A=a)."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=n); U = (rng.random(n) < _PU).astype(float)
    A = (rng.random(n) < _expit(_A0 + _AW * W + _AU * U)).astype(int)
    obs = float(np.mean(_expit(_B0 + _BW * W + _BU * U + _BA * A)))
    do1 = float(np.mean(_expit(_B0 + _BW * W + _BU * U + _BA * 1)))
    do0 = float(np.mean(_expit(_B0 + _BW * W + _BU * U + _BA * 0)))
    return {"rate_obs": obs, "rate_do1": do1, "rate_do0": do0,
            "rr_true": do1 / do0}                                  # true interventional risk ratio (churn)


def _ece(p, y, bins=10):
    """Expected calibration error: Σ_bin (n_bin/n)·|mean(y)−mean(p)| over equal-width probability bins."""
    p = np.asarray(p); y = np.asarray(y); edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def calibration_count_demo(df):
    """Fit a calibrated predictor on OBSERVED (W,A), and contrast observational vs interventional counts, plus the
    oracle-with-U. Returns per-capita rates + counts (n = len(df)) + calibration ECE + Poisson-binomial variance."""
    n = len(df)
    W = df["W"].values.reshape(-1, 1); A = df["A"].values.reshape(-1, 1); Y = df["Y"].values
    X = np.column_stack([W, A])
    clf = LogisticRegression(max_iter=1000).fit(X, Y)             # p̂(W,A) — calibrated predictor (U unseen)
    p_obs = clf.predict_proba(X)[:, 1]                            # p̂ at observed A
    p_do1 = clf.predict_proba(np.column_stack([W, np.ones((n, 1))]))[:, 1]   # p̂ with A:=1 (naive g-comp)
    p_do0 = clf.predict_proba(np.column_stack([W, np.zeros((n, 1))]))[:, 1]

    # oracle: same fit but WITH the hidden U (proves the gap is U, not the method/estimator class)
    Xo = np.column_stack([W, df["_U"].values.reshape(-1, 1), A])
    clfo = LogisticRegression(max_iter=1000).fit(Xo, Y)
    U = df["_U"].values.reshape(-1, 1)
    o_do1 = clfo.predict_proba(np.column_stack([W, U, np.ones((n, 1))]))[:, 1]
    o_do0 = clfo.predict_proba(np.column_stack([W, U, np.zeros((n, 1))]))[:, 1]

    return {
        "n": n,
        "ece": _ece(p_obs, Y),                                    # calibration (should be small)
        "rate_obs_hat": float(p_obs.mean()),                      # observational count / n  (should match truth)
        "rate_do1_naive": float(p_do1.mean()),                    # naive interventional (biased)
        "rate_do0_naive": float(p_do0.mean()),
        "rr_naive": float(p_do1.mean() / p_do0.mean()),           # naive interventional RR (the vulnerable estimate)
        "rate_do1_oracle": float(o_do1.mean()),                   # with U → recovers truth
        "rate_do0_oracle": float(o_do0.mean()),
        "pb_var_obs": float(np.sum(p_obs * (1 - p_obs))),         # Montgomery Poisson-binomial variance of the count
    }


def evalue_rr(rr):
    """VanderWeele–Ding E-value for a risk ratio: the minimum joint strength of association (RR scale) that an
    unmeasured confounder would need with BOTH treatment and outcome to fully explain away the observed rr. For a
    protective rr<1, apply to 1/rr (VanderWeele–Ding 2017, Ann Intern Med 167:268)."""
    r = rr if rr >= 1 else 1.0 / rr
    return float(r + np.sqrt(r * (r - 1.0)))
