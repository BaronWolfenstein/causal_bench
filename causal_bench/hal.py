"""Sklearn-compatible wrappers around hal9001::fit_hal (R).

HAL (Highly Adaptive LASSO) satisfies the regularity conditions required for
the doubly-robust remainder in TMLE to be o(n^{-1/2}), making the IC-based SE
asymptotically honest.  See van der Laan & Bibaut (2017).

Both classes follow the sklearn estimator protocol so they drop into
SuperLearner as ordinary base learners.  rpy2 is required; if hal9001 is not
installed the classes raise ImportError at fit() time rather than at import.

**max_degree WARNING**: Always use max_degree=1 (the default here).  Higher
values expand the basis combinatorially — max_degree=2 on p=5 covariates at
n=500 can exceed glmnet's integer limit and raise an overflow error.
Pairwise interactions are already covered by the RKS learners in the default
SuperLearner library.

**Runtime WARNING**: HAL runs glmnet cross-validation on a large step-function
basis, taking ~3 min/simulation at n=500 inside a 5-fold SuperLearner.  It is
NOT included in the default learner library for this reason.  Use
super_learner.hal_classifiers() / hal_regressors() to opt in explicitly.
"""
from __future__ import annotations

import numpy as np


def _hal9001_available() -> bool:
    try:
        import rpy2.robjects.packages as rpacks
        rpacks.importr("hal9001")
        return True
    except Exception:
        return False


def _fit_hal_r(X: np.ndarray, y: np.ndarray, family: str):
    """Call hal9001::fit_hal via inline R, returning the fit object by name.

    Routes through ro.r() to avoid rpy2 matrix-conversion quirks that trigger
    glmnet integer-overflow errors when passing numpy arrays directly.
    max_degree=1: univariate step-function basis gives the convergence rate
    guarantee; pairwise interactions are covered by RKS learners.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    importr("hal9001")
    n, p = X.shape

    # Push data as plain R numeric/FloatVector, then reshape to matrix in R
    ro.globalenv["hal_X_flat"] = ro.FloatVector(
        X.astype(np.float64).ravel(order="F").tolist())
    ro.globalenv["hal_y"] = ro.FloatVector(y.astype(np.float64).tolist())

    ro.r(f"""
        .hal_X <- matrix(hal_X_flat, nrow={n}L, ncol={p}L)
        .hal_fit <- fit_hal(
            X                 = .hal_X,
            Y                 = hal_y,
            family            = "{family}",
            max_degree        = 1L,
            smoothness_orders = 0L,
            reduce_basis      = 1 / sqrt({n})
        )
    """)
    return ro.globalenv[".hal_fit"]


def _predict_hal_r(fit, X: np.ndarray) -> np.ndarray:
    """Predict from a hal9001 fit object stored in R global env."""
    import rpy2.robjects as ro

    n, p = X.shape
    ro.globalenv["hal_Xnew_flat"] = ro.FloatVector(
        X.astype(np.float64).ravel(order="F").tolist())

    ro.r(f"""
        .hal_Xnew <- matrix(hal_Xnew_flat, nrow={n}L, ncol={p}L)
        .hal_pred <- predict(.hal_fit, new_data = .hal_Xnew, type = "response")
    """)
    pred = ro.globalenv[".hal_pred"]
    return np.asarray(list(pred), dtype=float).ravel()


class HALClassifier:
    """Logistic HAL via hal9001::fit_hal(family='binomial').

    Drop-in sklearn classifier; exposes predict_proba() clipped to [1e-6, 1-1e-6].
    """

    def __init__(self):
        self._fit = None

    def fit(self, X, y):
        if not _hal9001_available():
            raise ImportError("hal9001 R package is required for HALClassifier")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._fit = _fit_hal_r(X, y, family="binomial")
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = _predict_hal_r(self._fit, X)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class HALRegressor:
    """Gaussian HAL via hal9001::fit_hal(family='gaussian').

    Drop-in sklearn regressor; exposes predict().
    """

    def __init__(self):
        self._fit = None

    def fit(self, X, y):
        if not _hal9001_available():
            raise ImportError("hal9001 R package is required for HALRegressor")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._fit = _fit_hal_r(X, y, family="gaussian")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return _predict_hal_r(self._fit, X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
