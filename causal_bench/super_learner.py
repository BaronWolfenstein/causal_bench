import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from causal_bench.crossfit import make_folds


def _rks_classifier(gamma: float, n_components: int = 200,
                    random_state=None) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rbf",    RBFSampler(gamma=gamma, n_components=n_components,
                              random_state=random_state)),
        ("lr",     LogisticRegression(max_iter=1000, C=1.0)),
    ])


def _rks_regressor(gamma: float, n_components: int = 200,
                   random_state=None) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rbf",    RBFSampler(gamma=gamma, n_components=n_components,
                              random_state=random_state)),
        ("ridge",  RidgeCV()),
    ])


def _default_classifiers(random_state=None):
    return [
        LogisticRegression(max_iter=1000, C=1.0),
        RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                               random_state=random_state),
        GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                   random_state=random_state),
        _rks_classifier(gamma=0.1,  random_state=random_state),
        _rks_classifier(gamma=1.0,  random_state=random_state),
        _rks_classifier(gamma=10.0, random_state=random_state),
    ]


def _default_regressors(random_state=None):
    return [
        RidgeCV(),
        RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                              random_state=random_state),
        GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                  random_state=random_state),
        _rks_regressor(gamma=0.1,  random_state=random_state),
        _rks_regressor(gamma=1.0,  random_state=random_state),
        _rks_regressor(gamma=10.0, random_state=random_state),
    ]


def hal_classifiers(random_state=None):
    """Return default classifiers augmented with HALClassifier.

    HAL satisfies the regularity conditions for doubly-robust remainder
    o(n^{-1/2}), but glmnet CV inside each SuperLearner fold makes it
    ~3 min/sim at n=500.  Use only when runtime is not a concern.

    Important: HALClassifier is fixed at max_degree=1 (univariate step
    functions).  Do NOT increase max_degree — the basis explodes
    combinatorially and triggers glmnet integer-overflow errors.
    """
    from causal_bench.hal import HALClassifier
    return _default_classifiers(random_state) + [HALClassifier()]


def hal_regressors(random_state=None):
    """Return default regressors augmented with HALRegressor.

    See hal_classifiers() for runtime and max_degree warnings.
    """
    from causal_bench.hal import HALRegressor
    return _default_regressors(random_state) + [HALRegressor()]


class SuperLearner:
    def __init__(self, candidates=None, n_folds=5, task="classification",
                 random_state=None, fold_mode="iid"):
        self.candidates = candidates
        self.n_folds = n_folds
        self.task = task
        self.random_state = random_state
        self.fold_mode = fold_mode
        self.weights_: np.ndarray | None = None
        self._fitted_candidates: list | None = None

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        candidates = self.candidates or (
            _default_classifiers(self.random_state) if self.task == "classification"
            else _default_regressors(self.random_state)
        )
        n, k = len(y), len(candidates)
        oof = np.zeros((n, k))

        folds = make_folds(
            X, y, n_folds=self.n_folds, mode=self.fold_mode, groups=groups,
            random_state=self.random_state, stratify=(self.task == "classification"),
        )

        for train_idx, val_idx in folds:
            for j, est in enumerate(candidates):
                m = clone(est)
                m.fit(X[train_idx], y[train_idx])
                if self.task == "classification":
                    oof[val_idx, j] = m.predict_proba(X[val_idx])[:, 1]
                else:
                    oof[val_idx, j] = m.predict(X[val_idx])

        coefs, _ = nnls(oof, y)
        total = coefs.sum()
        self.weights_ = coefs / total if total > 1e-10 else np.ones(k) / k

        # Store OOF predictions for unbiased IC variance computation in TMLE/AIPW.
        # These are genuine out-of-fold predictions with ensemble weights fixed above.
        self.oof_predictions_ = np.clip(oof @ self.weights_, 1e-6, 1 - 1e-6)

        # Refit all candidates on full data
        self._fitted_candidates = []
        for est in candidates:
            m = clone(est)
            m.fit(X, y)
            self._fitted_candidates.append(m)

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Returns P(y=1|X), clipped to [1e-6, 1-1e-6]."""
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self._fitted_candidates
        ])
        return np.clip(preds @ self.weights_, 1e-6, 1 - 1e-6)

    def predict(self, X) -> np.ndarray:
        """Returns regression predictions."""
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([m.predict(X) for m in self._fitted_candidates])
        return preds @ self.weights_
