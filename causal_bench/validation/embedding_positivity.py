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

Positivity-robust / reduction responses (constant effect => every subpopulation ATE
== tau, so these all target tau):
  * dr_ato     -- tier-1: augmented (doubly-robust) overlap weighting; ~halves the bias
    but does not fully recover (ATO != ATE).
  * prognostic -- tier-2: adjust for the 1-D PROGNOSTIC score alone (control-outcome
    surface, Hansen 2008). Unlike the propensity score it is not treatment-degenerate,
    so it escapes the positivity trap and LARGELY RECOVERS the effect.

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


def _dr_ato(W, A, Y, flex=True):
    """Augmented (doubly-robust) overlap-weighted ATO -- the tier-1 positivity-robust
    response. Bounded overlap weights h=e(1-e); ~halves the attenuation but does not
    fully recover (ATO != ATE)."""
    e = _propensity(W, A)
    Q1, Q0 = _q_predict(W, A, Y, flex)
    h = e * (1 - e)
    plug = np.sum(h * (Q1 - Q0)) / np.sum(h)
    c1 = np.sum(A * (1 - e) * (Y - Q1)) / np.sum(h)
    c0 = np.sum((1 - A) * e * (Y - Q0)) / np.sum(h)
    return float(plug + c1 - c0)


def _prognostic_score(W, A, Y, flex):
    """Prognostic score E[Y | A=0, W], fit on CONTROLS only (Hansen 2008)."""
    Wc, Yc = W[A == 0], Y[A == 0]
    if flex:
        m = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                          learning_rate=0.1, random_state=0).fit(Wc, Yc)
    else:
        m = Ridge(alpha=1.0).fit(Wc, Yc)
    return m.predict(W)


def _prognostic_aipw(W, A, Y, flex=True):
    """Tier-2 candidate: adjust for the 1-D PROGNOSTIC score alone. Unlike the
    propensity score (which IS the positivity direction), the prognostic score is
    not treatment-degenerate, so it controls confounding while escaping the
    positivity trap -- and largely RECOVERS the effect where full-W AIPW fails."""
    prog = _prognostic_score(W, A, Y, flex)
    return _aipw(prog[:, None], A, Y, flex=flex)


def one_rep(n, d, fidelity, conf, tau, seed, flex=True):
    s = simulate(n, d=d, fidelity=fidelity, conf=conf, tau=tau, seed=seed)
    W, A, Y, U = s["W"], s["A"], s["Y"], s["Ustar"]
    return {
        "oracle_Ustar": _aipw(U, A, Y, flex=flex),        # adjust for the true confounder
        "naive_fullW": _aipw(W, A, Y, flex=flex),         # the pathology
        "dr_ato_W": _dr_ato(W, A, Y, flex=flex),          # tier-1: positivity-robust, ~halves bias
        "prog_score_W": _prognostic_aipw(W, A, Y, flex=flex),  # tier-2: largely recovers
        "frac_extreme": float(np.mean((s["e_true"] < 0.05) | (s["e_true"] > 0.95))),
    }


def report_rows(n=2500, d=50, fidelity=6.0, tau=1.0, n_reps=12,
                confs=(0.0, 1.0, 2.0, 3.0, 5.0), flex=True, seed=0):
    methods = ["oracle_Ustar", "naive_fullW", "dr_ato_W", "prog_score_W"]
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
