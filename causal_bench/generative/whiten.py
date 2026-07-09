"""ZCA whitening: Z = (X - mu) U Λ^{-1/2} Uᵀ. Cheap, invertible, gives identity
covariance while staying in the original orientation (unlike PCA whitening), so
separation AUC is preserved — the invariant the localization diagnostic checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ZCA:
    mean: np.ndarray
    W: np.ndarray        # whitening matrix
    W_inv: np.ndarray    # de-whitening matrix

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) @ self.W

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.W_inv + self.mean


def zca_fit(X: np.ndarray, eps: float = 1e-6) -> ZCA:
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    W_inv = U @ np.diag(np.sqrt(S + eps)) @ U.T
    return ZCA(mean=mu, W=W, W_inv=W_inv)
