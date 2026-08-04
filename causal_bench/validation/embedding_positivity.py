"""High-dim learned-embedding covariates as a fraught causal adjustment set (exp50).

Controlled synthetic reproduction of the SMB-demo finding (payoff_v7): a low-dim
true confounder ``Ustar`` drives treatment ``A`` and outcome ``Y``; the observed
covariate ``W`` is a high-fidelity high-dim encoding of ``Ustar``. As the
confounding strength rises, ``A`` becomes near-deterministic given the confounder
(severe positivity), and a FLEXIBLE outcome model (HistGradientBoosting) then
**attenuates the treatment effect toward null** -- a doubly-robust AIPW does not
rescue it, because the failure is the adjustment SET / support, not the nuisances.

Key mechanism (refined from the linear scaffold, see issue #206):
  * A *linear* outcome model does NOT show this -- it recovers the effect.
  * A *flexible* outcome model + severe positivity attenuates even the ORACLE
    (adjusting for the true low-dim Ustar), so it is the flexible-Q x positivity
    interaction, not high-dimensionality per se. The high-dim embedding's role in
    the real case is to CREATE the severe positivity (near-perfect propensity).

First-cut positivity-robust responses (constant effect => every subpopulation ATE
== tau, so these all target tau):
  * trimmed  -- restrict to the propensity-overlap region [lo, hi], AIPW there.
  * ato      -- overlap-weighted estimand (ATO): downweight extreme-propensity units.

Self-validating: (i) at conf=0 (no confounding) every method is unbiased; (ii) the
attenuation grows monotonically with confounding strength.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

_CLIP = (0.02, 0.98)


def simulate(n, d=50, fidelity=6.0, conf=3.0, tau=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Ustar = rng.normal(size=(n, 2))
    e = 1.0 / (1.0 + np.exp(-conf * (Ustar @ np.array([1.3, -1.0]))))
    A = (rng.random(n) < e).astype(float)
    Y = tau * A + Ustar @ np.array([1.5, 1.0]) + rng.normal(size=n)
    B = rng.normal(size=(2, d)) / np.sqrt(2)
    W = Ustar @ B + (1.0 / fidelity) * rng.normal(size=(n, d))
    return dict(W=W, Ustar=Ustar, A=A, Y=Y, e_true=e, tau=float(tau))


def _propensity(W, A):
    return np.clip(LogisticRegression(max_iter=2000, C=1.0).fit(W, A).predict_proba(W)[:, 1], *_CLIP)


def _q_predict(W, A, Y, flex):
    n = len(A)
    XA = np.column_stack([A, W])
    if flex:
        m = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                          learning_rate=0.1, random_state=0).fit(XA, Y)
    else:
        m = Ridge(alpha=1.0).fit(XA, Y)
    Q1 = m.predict(np.column_stack([np.ones(n), W]))
    Q0 = m.predict(np.column_stack([np.zeros(n), W]))
    return Q1, Q0


def _aipw(W, A, Y, flex=True, e=None):
    e = _propensity(W, A) if e is None else e
    Q1, Q0 = _q_predict(W, A, Y, flex)
    QA = A * Q1 + (1 - A) * Q0
    H = A / e - (1 - A) / (1 - e)
    return float(np.mean(Q1 - Q0 + H * (Y - QA)))


def _trimmed_aipw(W, A, Y, flex=True, lo=0.1, hi=0.9):
    e = _propensity(W, A)
    m = (e >= lo) & (e <= hi)
    if m.sum() < 50 or A[m].sum() < 10 or (1 - A[m]).sum() < 10:
        return float("nan")
    return _aipw(W[m], A[m], Y[m], flex=flex)


def _ato(W, A, Y):
    """Overlap-weighted ATO (Hajek): treated w=1-e, control w=e."""
    e = _propensity(W, A)
    wt, wc = (1 - e) * A, e * (1 - A)
    return float((wt * Y).sum() / wt.sum() - (wc * Y).sum() / wc.sum())


def one_rep(n, d, fidelity, conf, tau, seed, flex=True):
    s = simulate(n, d=d, fidelity=fidelity, conf=conf, tau=tau, seed=seed)
    W, A, Y, U = s["W"], s["A"], s["Y"], s["Ustar"]
    return {
        "oracle_Ustar": _aipw(U, A, Y, flex=flex),
        "naive_fullW": _aipw(W, A, Y, flex=flex),
        "trimmed_W": _trimmed_aipw(W, A, Y, flex=flex),
        "ato_W": _ato(W, A, Y),
        "frac_extreme": float(np.mean((s["e_true"] < 0.05) | (s["e_true"] > 0.95))),
    }


def report_rows(n=2500, d=50, fidelity=6.0, tau=1.0, n_reps=12,
                confs=(0.0, 1.0, 2.0, 3.0, 5.0), flex=True, seed=0):
    methods = ["oracle_Ustar", "naive_fullW", "trimmed_W", "ato_W"]
    rows = []
    for conf in confs:
        acc = {m: [] for m in methods}; fe = []
        for r in range(n_reps):
            res = one_rep(n, d, fidelity, conf, tau, seed + r, flex=flex)
            for m in methods:
                acc[m].append(res[m] - tau)          # bias vs true ATE (== every-subpop ATE)
            fe.append(res["frac_extreme"])
        rows.append({"conf": conf, "frac_extreme": float(np.mean(fe)),
                     **{m: float(np.nanmean(acc[m])) for m in methods}})
    return rows
