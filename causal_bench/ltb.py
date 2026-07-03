"""Lassoed Tree Boosting (LTB), a.k.a. the Selectively Adaptive Lasso.

Schuler, Li & van der Laan (arXiv:2205.10697v6): gradient-boosted trees
generate a basis; an L1 regression over per-tree margin contributions
selects and reweights it, with block-wise early stopping on validation
error. The fit lies in the cadlag bounded-sectional-variation (Donsker)
class with a dimension-free O_P(n^{-1/3} log-factor) L2 rate — the pair
of conditions that licenses IC-based SEs for AIPW/TMLE *without*
cross-fitting (benchmarked in exp33).

Classifier note: the paper's theory is stated for squared-error loss;
LTBClassifier uses the natural logistic-lasso analogue over the same
tree basis (L1-penalized logistic regression), so probabilities come
through the link and need no clipping.

Follows hal.py's convention: sklearn estimator protocol; the xgboost
import happens at fit() time so a missing backend raises then, not at
module import.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


class _LTBBase(BaseEstimator):
    _is_classifier = False

    def __init__(self, max_depth=3, learning_rate=0.1, block_size=10,
                 max_blocks=30, patience=3, val_fraction=0.2, cv=5,
                 random_state=0):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.patience = patience
        self.val_fraction = val_fraction
        self.cv = cv
        self.random_state = random_state

    # -- tree basis ---------------------------------------------------------
    def _tree_basis(self, X: np.ndarray, n_trees: int) -> np.ndarray:
        """Column k = margin contribution of tree k alone (cumulative diffs)."""
        import xgboost as xgb
        d = xgb.DMatrix(np.asarray(X, dtype=float))
        cum = np.column_stack([
            self.booster_.predict(d, iteration_range=(0, k), output_margin=True)
            for k in range(1, n_trees + 1)
        ])
        H = np.empty_like(cum)
        H[:, 0] = cum[:, 0]        # includes base_score; constant, absorbed
        H[:, 1:] = np.diff(cum, axis=1)   # by the lasso intercept
        return H

    def _fit_l1(self, H, y):
        if self._is_classifier:
            return LogisticRegressionCV(
                penalty="l1", solver="saga", Cs=10, cv=self.cv,
                max_iter=5000, scoring="neg_log_loss",
                random_state=self.random_state).fit(H, y)
        return LassoCV(cv=self.cv, random_state=self.random_state).fit(H, y)

    def _val_error(self, model, H, y):
        if self._is_classifier:
            p = np.clip(model.predict_proba(H)[:, 1], 1e-12, 1 - 1e-12)
            return log_loss(y, p, labels=[0, 1])
        return float(np.mean((model.predict(H) - y) ** 2))

    # -- fit ----------------------------------------------------------------
    def fit(self, X, y):
        import xgboost as xgb  # fit-time import per hal.py convention
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        strat = y if self._is_classifier else None
        Xtr, Xval, ytr, yval = train_test_split(
            X, y, test_size=self.val_fraction,
            random_state=self.random_state, stratify=strat)

        params = {
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "objective": ("binary:logistic" if self._is_classifier
                          else "reg:squarederror"),
            "seed": self.random_state,
            "nthread": 1,
        }
        dtrain = xgb.DMatrix(Xtr, label=ytr)

        self.booster_ = None
        best_err, best_k, stale = np.inf, None, 0
        for block in range(1, self.max_blocks + 1):
            self.booster_ = xgb.train(
                params, dtrain, num_boost_round=self.block_size,
                xgb_model=self.booster_)
            k = block * self.block_size
            model = self._fit_l1(self._tree_basis(Xtr, k), ytr)
            err = self._val_error(model, self._tree_basis(Xval, k), yval)
            if err < best_err:
                best_err, best_k, stale = err, k, 0
            else:
                stale += 1
                if stale >= self.patience:   # paper: 3 validation increases
                    break

        self.n_trees_ = best_k
        # Final L1 fit on the full data over the selected basis size.
        self.l1_model_ = self._fit_l1(self._tree_basis(X, self.n_trees_), y)
        if self._is_classifier:
            self.classes_ = self.l1_model_.classes_.astype(int)
        return self


class LTBRegressor(_LTBBase, RegressorMixin):
    def predict(self, X):
        return self.l1_model_.predict(self._tree_basis(np.asarray(X, float),
                                                        self.n_trees_))


class LTBClassifier(_LTBBase, ClassifierMixin):
    _is_classifier = True

    def predict_proba(self, X):
        H = self._tree_basis(np.asarray(X, float), self.n_trees_)
        return self.l1_model_.predict_proba(H)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
