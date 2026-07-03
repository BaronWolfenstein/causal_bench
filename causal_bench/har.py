"""Highly Adaptive Ridge (HAR).

Schuler (arXiv:2410.02680): ridge regression over HAL's zero-order
tensor-product indicator basis, computed implicitly through the
dominance-counting kernel

    K(x, x') = sum_i 2^{|{j : X_ij <= min(x_j, x'_j)}|}

so the n·2^p basis is never materialized. Dimension-free rate matching
HAL's, but the guarantee requires square-integrable sectional
derivatives — strictly stronger than bounded sectional variation, and
jump discontinuities fall outside it (exp33's jumpy arm probes this).

Squared-error only: HARClassifier is least-squares on the binary label
with clipped probabilities. Tail calibration is a known caveat the
benchmark measures, not a bug.

Lambda is selected by exact leave-one-out CV from a single
eigendecomposition of the training kernel.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

_CLIP = 1e-6


def har_kernel(A: np.ndarray, B: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """K[k, l] = sum_i 2^{#coords j where X_train[i,j] <= min(A[k,j], B[l,j])}."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    X = np.asarray(X_train, dtype=float)
    dom_a = (X[None, :, :] <= A[:, None, :])   # (m, n, p)
    dom_b = (X[None, :, :] <= B[:, None, :])   # (r, n, p)
    K = np.zeros((A.shape[0], B.shape[0]))
    for i in range(X.shape[0]):
        counts = dom_a[:, i, :].astype(np.float64) @ dom_b[:, i, :].T.astype(np.float64)
        K += np.exp2(counts)
    return K


class _HARBase(BaseEstimator):
    _is_classifier = False

    def __init__(self, lambdas=None, jitter=1e-10, random_state=0):
        self.lambdas = lambdas
        self.jitter = jitter
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        self.X_ = X
        K = har_kernel(X, X, X)
        w, V = np.linalg.eigh(K + self.jitter * np.eye(n))
        w = np.clip(w, 0.0, None)

        self.y_mean_ = float(y.mean())
        yc = y - self.y_mean_
        Vty = V.T @ yc

        scale = max(np.trace(K) / n, 1e-12)
        lambdas = (np.asarray(self.lambdas, dtype=float) if self.lambdas is not None
                   else np.logspace(-4, 4, 30) * scale)

        best_loo, best_lam = np.inf, lambdas[0]
        for lam in lambdas:
            shrink = w / (w + lam)
            yhat_c = V @ (shrink * Vty)
            diag_h = np.einsum("ij,j,ij->i", V, shrink, V)
            denom = np.clip(1.0 - diag_h, 1e-8, None)
            loo = float(np.mean(((yc - yhat_c) / denom) ** 2))
            if loo < best_loo:
                best_loo, best_lam = loo, lam

        self.lambda_ = float(best_lam)
        self.alpha_ = V @ (Vty / (w + best_lam))
        if self._is_classifier:
            self.classes_ = np.array([0, 1])
        return self

    def _raw_predict(self, X):
        k = har_kernel(np.asarray(X, dtype=float), self.X_, self.X_)
        return k @ self.alpha_ + self.y_mean_


class HARRegressor(_HARBase, RegressorMixin):
    def predict(self, X):
        return self._raw_predict(X)


class HARClassifier(_HARBase, ClassifierMixin):
    _is_classifier = True

    def predict_proba(self, X):
        p1 = np.clip(self._raw_predict(X), _CLIP, 1 - _CLIP)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
