import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.base import clone


def _default_classifiers():
    return [
        LogisticRegression(max_iter=1000, C=1.0),
        RandomForestClassifier(n_estimators=100, min_samples_leaf=5),
        GradientBoostingClassifier(n_estimators=100, max_depth=3),
    ]


def _default_regressors():
    return [
        RidgeCV(),
        RandomForestRegressor(n_estimators=100, min_samples_leaf=5),
        GradientBoostingRegressor(n_estimators=100, max_depth=3),
    ]


class SuperLearner:
    def __init__(self, candidates=None, n_folds=5, task="classification",
                 random_state=None):
        self.candidates = candidates
        self.n_folds = n_folds
        self.task = task
        self.random_state = random_state
        self.weights_: np.ndarray | None = None
        self._fitted_candidates: list | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        candidates = self.candidates or (
            _default_classifiers() if self.task == "classification"
            else _default_regressors()
        )
        n, k = len(y), len(candidates)
        oof = np.zeros((n, k))

        splitter = (
            StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                            random_state=self.random_state)
            if self.task == "classification"
            else KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=self.random_state)
        )

        for _, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
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
