"""Point-treatment AIPW/TMLE with a crossfit toggle, for exp33 attribution.

The production estimators (aipw.py, tmle_ipcw.py) hardwire SuperLearner
and always use cross-fitted nuisances in the IC, so they cannot express
the crossfit-OFF condition the Donsker theory licenses. These thin
estimators take explicit learners and a crossfit flag; they are built
for attribution in exp33 (and reuse in phase 2), not as production
replacements. No censoring, binary Y, ATE only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit
from sklearn.base import clone

from causal_bench.crossfit import make_folds

_P_CLIP = (0.01, 0.99)
_Q_CLIP = (1e-5, 1 - 1e-5)


@dataclass
class PointResult:
    point: float
    se: float
    ci_lower: float
    ci_upper: float
    ic: np.ndarray


def _predict_binary(model, X) -> np.ndarray:
    """P(y=1|X) from a classifier, or clipped predictions from a regressor."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return np.clip(model.predict(X), 0.0, 1.0)


class NuisanceFits:
    """Fitted nuisances evaluated on the analysis sample.

    g/Q1/Q0 are in-sample fits (crossfit=False) or out-of-fold
    (crossfit=True). predict() evaluates on new data, averaging the
    fold models under crossfit — used for the Monte Carlo (population)
    side of the empirical-process measurement in exp33.
    """

    def __init__(self, g, Q1, Q0, models):
        self.g = g
        self.Q1 = Q1
        self.Q0 = Q0
        self._models = models          # list of (g_model, q_model)

    def predict(self, W_new: np.ndarray):
        W_new = np.asarray(W_new, dtype=float)
        n = len(W_new)
        X1 = np.column_stack([np.ones(n), W_new])
        X0 = np.column_stack([np.zeros(n), W_new])
        gs, q1s, q0s = [], [], []
        for g_m, q_m in self._models:
            gs.append(_predict_binary(g_m, W_new))
            q1s.append(_predict_binary(q_m, X1))
            q0s.append(_predict_binary(q_m, X0))
        return (np.mean(gs, axis=0), np.mean(q1s, axis=0), np.mean(q0s, axis=0))


class _OracleNuisanceFits(NuisanceFits):
    def __init__(self, W, surface):
        from causal_bench.dgp.point_treatment import true_Q, true_g
        self._surface = surface
        super().__init__(true_g(W, surface), true_Q(1, W, surface),
                         true_Q(0, W, surface), models=[])

    def predict(self, W_new):
        from causal_bench.dgp.point_treatment import true_Q, true_g
        W_new = np.asarray(W_new, dtype=float)
        return (true_g(W_new, self._surface),
                true_Q(1, W_new, self._surface),
                true_Q(0, W_new, self._surface))


def oracle_nuisances(W: np.ndarray, surface: str) -> NuisanceFits:
    return _OracleNuisanceFits(np.asarray(W, dtype=float), surface)


def fit_nuisances(W, A, Y, g_learner, q_learner, crossfit: bool,
                  n_folds: int = 5, random_state: int = 0) -> NuisanceFits:
    W = np.asarray(W, dtype=float)
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(A)
    Xq = np.column_stack([A, W])
    X1 = np.column_stack([np.ones(n), W])
    X0 = np.column_stack([np.zeros(n), W])

    if not crossfit:
        g_m = clone(g_learner).fit(W, A)
        q_m = clone(q_learner).fit(Xq, Y)
        return NuisanceFits(_predict_binary(g_m, W),
                            _predict_binary(q_m, X1),
                            _predict_binary(q_m, X0),
                            models=[(g_m, q_m)])

    g = np.zeros(n)
    Q1 = np.zeros(n)
    Q0 = np.zeros(n)
    models = []
    for tr, val in make_folds(W, A, n_folds=n_folds, mode="iid",
                              stratify=True, random_state=random_state):
        g_m = clone(g_learner).fit(W[tr], A[tr])
        q_m = clone(q_learner).fit(Xq[tr], Y[tr])
        g[val] = _predict_binary(g_m, W[val])
        Q1[val] = _predict_binary(q_m, X1[val])
        Q0[val] = _predict_binary(q_m, X0[val])
        models.append((g_m, q_m))
    return NuisanceFits(g, Q1, Q0, models)


def _prep(A, Y, nf):
    g = np.clip(nf.g, *_P_CLIP)
    Q1 = np.clip(nf.Q1, *_Q_CLIP)
    Q0 = np.clip(nf.Q0, *_Q_CLIP)
    QA = A * Q1 + (1 - A) * Q0
    H = A / g - (1 - A) / (1 - g)
    return g, Q1, Q0, QA, H


def _result(point, ic, n):
    ic = ic - float(np.mean(ic))
    se = float(np.sqrt(np.var(ic, ddof=1) / n))
    return PointResult(point=float(point), se=se,
                       ci_lower=float(point - 1.96 * se),
                       ci_upper=float(point + 1.96 * se), ic=ic)


def point_aipw(A, Y, nf: NuisanceFits) -> PointResult:
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    g, Q1, Q0, QA, H = _prep(A, Y, nf)
    eif = Q1 - Q0 + H * (Y - QA)
    point = float(np.mean(eif))
    return _result(point, eif - point, len(A))


def point_tmle(A, Y, nf: NuisanceFits) -> PointResult:
    A = np.asarray(A, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(A)
    g, Q1, Q0, QA, H = _prep(A, Y, nf)

    # One-step Newton targeting (as in tmle_ipcw._target_and_se, sans IPCW)
    denom = float(np.mean(H ** 2))
    eps = float(np.mean(H * (Y - QA))) / denom if denom > 1e-10 else 0.0
    eps = float(np.clip(eps, -2.0, 2.0))

    Q1s = expit(logit(Q1) + eps / g)
    Q0s = expit(logit(Q0) - eps / (1 - g))
    QAs = A * Q1s + (1 - A) * Q0s

    point = float(np.mean(Q1s - Q0s))
    ic = (Q1s - Q0s - point) + H * (Y - QAs)
    return _result(point, ic, n)
