# Exp33 Donsker-Class Nuisance Learners (LTB/HAR) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Lassoed Tree Boosting (LTB) and Highly Adaptive Ridge (HAR) as sklearn-protocol learners, a point-treatment DGP with jumpy/smooth nuisance surfaces, a crossfit-toggleable point AIPW/TMLE harness, and the exp33 benchmark measuring the empirical-process and remainder terms directly (spec: `docs/superpowers/specs/2026-07-02-ltb-har-benchmark-design.md`, phase 1 only).

**Architecture:** Two new learner modules follow `hal.py`'s sklearn-estimator pattern. A new point-treatment DGP module exposes true nuisance functions so exp33 can compute (P_n−P)(D(f̂)−D(f₀)) per simulation against a fixed Monte Carlo evaluation sample. A thin `estimators/point.py` provides AIPW/TMLE parametrized by `(g_learner, q_learner, crossfit)`; production estimators are untouched.

**Tech Stack:** Python, numpy/scipy/sklearn, xgboost (new core dep), pytest; optional rpy2/hal9001 for the HAL reference arm.

## Global Constraints

- New core dependency: `xgboost>=2.0` in `pyproject.toml` `dependencies` (spec §10).
- `ltb.py` imports xgboost **inside fit()** so ImportError surfaces at fit time (matches `hal.py` convention).
- HAL arm must skip gracefully (warning, not error) when rpy2/hal9001 unavailable.
- No changes to any existing estimator, `super_learner.py`, or the concrete bridge (spec §5, §9, phase 2 is out of scope).
- All learners: sklearn protocol (`get_params`/`set_params` via `sklearn.base.BaseEstimator` subclassing, clone-safe: `__init__` only stores params).
- Classifier probability outputs clipped/lying in `[1e-6, 1−1e-6]` where noted.
- DGP columns follow repo convention `W1..W4`, `A`, `Y`; g₀ ∈ [0.1, 0.9] by construction.
- Commit after every task; commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- PR body at the end must say "part of #N" — **never** any closing keyword, even negated.
- Run commands from repo root `/Users/noahrahman/git/causal_bench` on branch `exp33-donsker-learners`.

---

### Task 0: Tracking issue + dependency

**Files:**
- Modify: `pyproject.toml` (dependencies list, line ~10-22)

**Interfaces:**
- Produces: issue number `#N` used in the final PR body; xgboost importable for Task 2.

- [ ] **Step 1: Create the tracking issue (both phases, per spec §9)**

```bash
gh issue create \
  --title "Donsker-class nuisance learners (LTB/HAR): exp33 benchmark, then TMLE-IPCW wiring" \
  --body "$(cat <<'EOF'
Spec: docs/superpowers/specs/2026-07-02-ltb-har-benchmark-design.md

Two phases, one PR each (each PR body will say "part of" this issue; close manually after phase 2):

- Phase 1: LTB (arXiv:2205.10697v6) + HAR (arXiv:2410.02680) sklearn learners, point-treatment DGP (jumpy/smooth), crossfit-toggleable point AIPW/TMLE, exp33 grid with directly-measured empirical-process and remainder terms. HAL reference arm via existing hal.py at reduced n_sims.
- Phase 2: ltb_/har_ opt-in SuperLearner lists; TMLEIPCWEstimator g_learner/q_learner override with Cox _fit_G fixed; rerun edwards_realistic + exp16.

Future work parked in the spec: phase 3 (LTB discrete-time hazard for G), candidate exp34 (pooled-Q subgroup event rates, Qiu et al. arXiv:2605.15483).
EOF
)"
```

Expected: prints the new issue URL. Record the number as `#N` for the final PR.

- [ ] **Step 2: Add xgboost to core dependencies**

In `pyproject.toml`, in the `dependencies = [` list, add after the `"scikit-learn>=1.3",` line:

```toml
    "xgboost>=2.0",
```

- [ ] **Step 3: Install and verify**

Run: `pip install -e ".[dev]" && python -c "import xgboost; print(xgboost.__version__)"`
Expected: prints a version ≥ 2.0.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add xgboost core dependency for LTB (part of #N)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 1: Point-treatment DGP

**Files:**
- Create: `causal_bench/dgp/point_treatment.py`
- Test: `tests/test_point_treatment_dgp.py`

**Interfaces:**
- Produces:
  - `draw_point_treatment(n: int, surface: str, seed: int) -> pd.DataFrame` with columns `W1..W4` (float), `A` (0/1 int), `Y` (0/1 int).
  - `true_g(W: np.ndarray, surface: str) -> np.ndarray` — P(A=1|W), W shape (n,4), values in [0.1, 0.9].
  - `true_Q(a: int, W: np.ndarray, surface: str) -> np.ndarray` — E[Y|A=a,W].
  - `true_tau(surface: str) -> float` — cached MC integral (N=2×10⁶, fixed seed).
  - `SURFACES = ("jumpy", "smooth")`, `GATE = 0.6` (the W1 threshold).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_point_treatment_dgp.py`:

```python
import numpy as np
import pytest

from causal_bench.dgp.point_treatment import (
    GATE, SURFACES, draw_point_treatment, true_Q, true_g, true_tau)


@pytest.mark.parametrize("surface", SURFACES)
def test_draw_columns_and_types(surface):
    df = draw_point_treatment(n=500, surface=surface, seed=1)
    assert list(df.columns) == ["W1", "W2", "W3", "W4", "A", "Y"]
    assert set(df["A"].unique()) <= {0, 1}
    assert set(df["Y"].unique()) <= {0, 1}
    assert len(df) == 500


@pytest.mark.parametrize("surface", SURFACES)
def test_g_bounds(surface):
    rng = np.random.default_rng(2)
    W = rng.normal(size=(2000, 4)) * 3  # deliberately wide tails
    g = true_g(W, surface)
    assert g.min() >= 0.1 - 1e-12 and g.max() <= 0.9 + 1e-12


def test_true_tau_cached_and_stable():
    t1 = true_tau("jumpy")
    t2 = true_tau("jumpy")
    assert t1 == t2  # cached
    assert -1.0 < t1 < 1.0 and t1 != 0.0


def test_jumpy_surface_is_discontinuous_smooth_is_not():
    base = np.zeros((1, 4))
    lo, hi = base.copy(), base.copy()
    lo[0, 0], hi[0, 0] = GATE - 1e-6, GATE + 1e-6
    jump_q = abs(true_Q(1, hi, "jumpy") - true_Q(1, lo, "jumpy"))[0]
    jump_g = abs(true_g(hi, "jumpy") - true_g(lo, "jumpy"))[0]
    smooth_q = abs(true_Q(1, hi, "smooth") - true_Q(1, lo, "smooth"))[0]
    assert jump_q > 0.05 and jump_g > 0.05
    assert smooth_q < 1e-4


def test_empirical_tau_matches_true_tau():
    # Oracle G-computation on a big draw should land near the cached truth.
    df = draw_point_treatment(n=200_000, surface="smooth", seed=7)
    W = df[["W1", "W2", "W3", "W4"]].values
    emp = float(np.mean(true_Q(1, W, "smooth") - true_Q(0, W, "smooth")))
    assert abs(emp - true_tau("smooth")) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_point_treatment_dgp.py -v`
Expected: FAIL (ModuleNotFoundError: `causal_bench.dgp.point_treatment`).

- [ ] **Step 3: Write the DGP module**

Create `causal_bench/dgp/point_treatment.py`:

```python
"""Point-treatment binary-outcome DGP for exp33 (Donsker learner benchmark).

No censoring machinery: the estimand is the plain ATE, so the empirical-
process and remainder terms of AIPW/TMLE are directly computable against
the exposed truth (`true_g`, `true_Q`, `true_tau`).

Two nuisance-surface variants:
- "jumpy": threshold gate on W1 (LVEDD-style) in BOTH g0 and Q0 — cadlag
  with genuine jumps: inside LTB/HAL's function class, outside HAR's
  square-integrable-derivative condition.
- "smooth": the same structural strength via tanh, inside every class.

Positivity is healthy by construction (g0 in [0.1, 0.9]); positivity
stress is exp2's job.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit

SURFACES = ("jumpy", "smooth")
GATE = 0.6          # threshold on W1
_TAU_MC_N = 2_000_000
_TAU_MC_SEED = 20260702

# Correlated covariates: W ~ N(0, S), unit variances, mild correlation.
_CHOL = np.linalg.cholesky(
    np.array([[1.0, 0.3, 0.2, 0.0],
              [0.3, 1.0, 0.3, 0.1],
              [0.2, 0.3, 1.0, 0.2],
              [0.0, 0.1, 0.2, 1.0]]))


def _draw_W(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((n, 4)) @ _CHOL.T


def _gate_term(W: np.ndarray, surface: str) -> np.ndarray:
    """The W1 feature: hard indicator (jumpy) or tanh ramp (smooth)."""
    if surface == "jumpy":
        return (W[:, 0] >= GATE).astype(float)
    # tanh(2x) spans ~[-1,1]; rescale to [0,1] so both variants share range
    return 0.5 * (1.0 + np.tanh(2.0 * (W[:, 0] - GATE)))


def true_g(W: np.ndarray, surface: str) -> np.ndarray:
    """P(A=1 | W), bounded in [0.1, 0.9] by construction."""
    s = _gate_term(W, surface)
    lin = -0.3 + 0.7 * W[:, 1] - 0.5 * W[:, 2] + 1.4 * s
    return 0.1 + 0.8 * expit(lin)


def true_Q(a: int, W: np.ndarray, surface: str) -> np.ndarray:
    """E[Y | A=a, W]."""
    s = _gate_term(W, surface)
    lin = (-0.8 + 0.6 * W[:, 1] + 0.4 * W[:, 2] * W[:, 3]
           + 1.1 * s + a * (-0.9 + 0.8 * s - 0.3 * W[:, 3]))
    return expit(lin)


@lru_cache(maxsize=None)
def true_tau(surface: str) -> float:
    """ATE by Monte Carlo integration over the W distribution (cached)."""
    rng = np.random.default_rng(_TAU_MC_SEED)
    W = _draw_W(_TAU_MC_N, rng)
    return float(np.mean(true_Q(1, W, surface) - true_Q(0, W, surface)))


def draw_point_treatment(n: int, surface: str, seed: int) -> pd.DataFrame:
    """One simulated trial: columns W1..W4, A, Y."""
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
    rng = np.random.default_rng(seed)
    W = _draw_W(n, rng)
    A = rng.binomial(1, true_g(W, surface))
    pY = np.where(A == 1, true_Q(1, W, surface), true_Q(0, W, surface))
    Y = rng.binomial(1, pY)
    return pd.DataFrame({
        "W1": W[:, 0], "W2": W[:, 1], "W3": W[:, 2], "W4": W[:, 3],
        "A": A.astype(int), "Y": Y.astype(int),
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_point_treatment_dgp.py -v`
Expected: 7 PASS (two are parametrized ×2). The `true_tau` test takes a few seconds (2M-row MC, cached after first call).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/dgp/point_treatment.py tests/test_point_treatment_dgp.py
git commit -m "feat: point-treatment DGP with jumpy/smooth nuisance surfaces (part of #N)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: LTB learner

**Files:**
- Create: `causal_bench/ltb.py`
- Test: `tests/test_ltb.py`

**Interfaces:**
- Produces:
  - `LTBRegressor(max_depth=3, learning_rate=0.1, block_size=10, max_blocks=30, patience=3, val_fraction=0.2, cv=5, random_state=0)` with `fit(X, y)`, `predict(X)`.
  - `LTBClassifier(...same params...)` with `fit(X, y)`, `predict_proba(X)` (shape (n,2)), `predict(X)`, `classes_`.
  - Both sklearn-clone-safe; xgboost imported inside `fit()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ltb.py`:

```python
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_bench.ltb import LTBClassifier, LTBRegressor


def _step_data(n, seed, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 3))
    y = (X[:, 0] > 0).astype(float) + 0.3 * (X[:, 1] > 0.5) + rng.normal(0, noise, n)
    return X, y


def test_regressor_clone_and_shapes():
    m = LTBRegressor(max_blocks=5, random_state=0)
    m2 = clone(m)  # raises if __init__ mutates params
    X, y = _step_data(300, 0)
    m2.fit(X, y)
    assert m2.predict(X[:10]).shape == (10,)


def test_regressor_beats_linear_on_step_function():
    Xtr, ytr = _step_data(800, 1)
    Xte, yte = _step_data(800, 2)
    ltb = LTBRegressor(random_state=0).fit(Xtr, ytr)
    lin = LinearRegression().fit(Xtr, ytr)
    mse_ltb = np.mean((ltb.predict(Xte) - yte) ** 2)
    mse_lin = np.mean((lin.predict(Xte) - yte) ** 2)
    assert mse_ltb < mse_lin


def test_classifier_proba_valid_and_beats_logistic_on_step():
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(1000, 3))
    p = 0.15 + 0.7 * (X[:, 0] > 0)
    y = rng.binomial(1, p)
    Xte = rng.uniform(-2, 2, size=(1000, 3))
    pte = 0.15 + 0.7 * (Xte[:, 0] > 0)
    yte = rng.binomial(1, pte)

    ltb = LTBClassifier(random_state=0).fit(X, y)
    proba = ltb.predict_proba(Xte)
    assert proba.shape == (1000, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(ltb.classes_) == [0, 1]

    lr = LogisticRegression(max_iter=1000).fit(X, y)
    rmse_ltb = np.sqrt(np.mean((proba[:, 1] - pte) ** 2))
    rmse_lr = np.sqrt(np.mean((lr.predict_proba(Xte)[:, 1] - pte) ** 2))
    assert rmse_ltb < rmse_lr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ltb.py -v`
Expected: FAIL (ModuleNotFoundError: `causal_bench.ltb`).

- [ ] **Step 3: Write the LTB module**

Create `causal_bench/ltb.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ltb.py -v`
Expected: 3 PASS (the classifier test takes ~30-60s due to saga; if it exceeds ~2 min, reduce `max_blocks` in the test via `LTBClassifier(max_blocks=10, random_state=0)` — accuracy assertions still hold).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/ltb.py tests/test_ltb.py
git commit -m "feat: Lassoed Tree Boosting learners (arXiv:2205.10697) (part of #N)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: HAR learner

**Files:**
- Create: `causal_bench/har.py`
- Test: `tests/test_har.py`

**Interfaces:**
- Produces:
  - `HARRegressor(lambdas=None, jitter=1e-10, random_state=0)` with `fit(X, y)`, `predict(X)`.
  - `HARClassifier(...)` with `predict_proba(X)` (shape (n,2), clipped to [1e-6, 1−1e-6]), `predict(X)`, `classes_`.
  - Module-level `har_kernel(A, B, X_train) -> np.ndarray` (shape (len(A), len(B))) for direct testing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_har.py`:

```python
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge

from causal_bench.har import HARClassifier, HARRegressor, har_kernel


def test_kernel_hand_computed():
    # p=1, training points {0, 1}. K(a,b) = sum_i 2^{#coords where X_i <= min(a,b)}
    X = np.array([[0.0], [1.0]])
    K = har_kernel(np.array([[2.0]]), np.array([[3.0]]), X)
    # both training points dominated: 2^1 + 2^1 = 4
    assert K.shape == (1, 1) and K[0, 0] == 4.0
    K2 = har_kernel(np.array([[0.5]]), np.array([[3.0]]), X)
    # min(0.5,3)=0.5: only X=0 dominated -> 2^1 + 2^0 = 3
    assert K2[0, 0] == 3.0


def test_regressor_clone_and_beats_ridge_on_smooth_nonlinear():
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(-2, 2, size=(400, 3))
    Xte = rng.uniform(-2, 2, size=(400, 3))
    f = lambda X: np.tanh(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    ytr = f(Xtr) + rng.normal(0, 0.3, 400)
    yte = f(Xte)

    har = clone(HARRegressor(random_state=0)).fit(Xtr, ytr)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    mse_har = np.mean((har.predict(Xte) - yte) ** 2)
    mse_ridge = np.mean((ridge.predict(Xte) - yte) ** 2)
    assert mse_har < mse_ridge


def test_classifier_proba_clipped_and_shaped():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(300, 3))
    y = rng.binomial(1, 0.2 + 0.6 * (np.tanh(X[:, 0]) > 0))
    m = HARClassifier(random_state=0).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (300, 2)
    assert proba[:, 1].min() >= 1e-6 and proba[:, 1].max() <= 1 - 1e-6
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(m.classes_) == [0, 1]
    assert set(np.unique(m.predict(X))) <= {0, 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_har.py -v`
Expected: FAIL (ModuleNotFoundError: `causal_bench.har`).

- [ ] **Step 3: Write the HAR module**

Create `causal_bench/har.py`:

```python
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
```

Note the `har_kernel(Xnew, self.X_, self.X_)` call in `_raw_predict`: the second argument is the point set the fitted `alpha_` lives on and the third is the dominance set; at fit time all three coincide.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_har.py -v`
Expected: 3 PASS in well under a minute (n ≤ 400).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/har.py tests/test_har.py
git commit -m "feat: Highly Adaptive Ridge learners (arXiv:2410.02680) (part of #N)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Point AIPW/TMLE attribution harness

**Files:**
- Create: `causal_bench/estimators/point.py`
- Test: `tests/test_point_estimators.py`

**Interfaces:**
- Consumes: `causal_bench.crossfit.make_folds`; DGP truth functions from Task 1 (tests only).
- Produces:
  - `PointResult` dataclass: `point, se, ci_lower, ci_upper, ic` (floats + np.ndarray).
  - `NuisanceFits` with attributes `g, Q1, Q0` (np.ndarray, in-sample or OOF per crossfit) and method `predict(W_new) -> (g, Q1, Q0)` (fold-model averages under crossfit).
  - `fit_nuisances(W, A, Y, g_learner, q_learner, crossfit, n_folds=5, random_state=0) -> NuisanceFits`.
  - `oracle_nuisances(W, surface) -> NuisanceFits` (wraps `true_g`/`true_Q`).
  - `point_aipw(A, Y, nf) -> PointResult` and `point_tmle(A, Y, nf) -> PointResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_point_estimators.py`:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

from causal_bench.dgp.point_treatment import draw_point_treatment, true_tau
from causal_bench.estimators.point import (
    fit_nuisances, oracle_nuisances, point_aipw, point_tmle)

W_COLS = ["W1", "W2", "W3", "W4"]


def _sim(seed, n=700, surface="smooth"):
    df = draw_point_treatment(n=n, surface=surface, seed=seed)
    return df[W_COLS].values, df["A"].values.astype(float), df["Y"].values.astype(float)


def test_oracle_aipw_and_tmle_recover_truth():
    tau0 = true_tau("smooth")
    pts_a, pts_t = [], []
    for seed in range(40):
        W, A, Y = _sim(seed)
        nf = oracle_nuisances(W, "smooth")
        pts_a.append(point_aipw(A, Y, nf).point)
        pts_t.append(point_tmle(A, Y, nf).point)
    assert abs(np.mean(pts_a) - tau0) < 0.02
    assert abs(np.mean(pts_t) - tau0) < 0.02


def test_oracle_ci_covers():
    tau0 = true_tau("smooth")
    cover = 0
    for seed in range(60):
        W, A, Y = _sim(seed)
        r = point_aipw(A, Y, oracle_nuisances(W, "smooth"))
        cover += (r.ci_lower <= tau0 <= r.ci_upper)
    assert cover / 60 > 0.85   # ~95% nominal, MC slack


def test_crossfit_toggle_changes_nuisances_but_not_shape():
    W, A, Y = _sim(0)
    g_l = LogisticRegression(max_iter=1000)
    q_l = LogisticRegression(max_iter=1000)
    nf_off = fit_nuisances(W, A, Y, g_l, q_l, crossfit=False, random_state=0)
    nf_on = fit_nuisances(W, A, Y, g_l, q_l, crossfit=True, random_state=0)
    assert nf_off.g.shape == nf_on.g.shape == (700,)
    assert not np.allclose(nf_off.g, nf_on.g)      # OOF differs from in-sample
    g_new, Q1_new, Q0_new = nf_on.predict(W[:5])
    assert g_new.shape == Q1_new.shape == Q0_new.shape == (5,)


def test_tmle_point_within_bounds():
    W, A, Y = _sim(1)
    g_l = LogisticRegression(max_iter=1000)
    q_l = LogisticRegression(max_iter=1000)
    nf = fit_nuisances(W, A, Y, g_l, q_l, crossfit=False, random_state=0)
    r = point_tmle(A, Y, nf)
    assert -1.0 <= r.point <= 1.0
    assert r.se > 0 and r.ci_lower < r.point < r.ci_upper
    assert abs(float(np.mean(r.ic))) < 1e-8    # IC centered
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_point_estimators.py -v`
Expected: FAIL (ModuleNotFoundError: `causal_bench.estimators.point`).

- [ ] **Step 3: Write the module**

Create `causal_bench/estimators/point.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_point_estimators.py -v`
Expected: 4 PASS (the two oracle tests run 40-60 sims at n=700 each; ~30s total).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/estimators/point.py tests/test_point_estimators.py
git commit -m "feat: point AIPW/TMLE with crossfit toggle for exp33 attribution (part of #N)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Exp33 core — EP/remainder measurement and cell runner

**Files:**
- Create: `experiments/exp33_donsker_learners.py`
- Test: `tests/test_exp33_donsker_learners.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2, 3, 4 (exact signatures listed there).
- Produces (importable from the experiment module, exp29/exp30 precedent):
  - `LEARNERS = ("logistic", "xgboost", "ltb", "har", "hal", "oracle")`.
  - `make_learners(name: str, seed: int) -> tuple[g_learner, q_learner] | None` (None for "oracle"; raises `ImportError` for "hal" when rpy2/hal9001 missing).
  - `mc_eval_sample(surface: str) -> pd.DataFrame` — fixed N=100_000 draw per surface, module-level cached, seed 424242.
  - `eif0_values(g, Q1, Q0, A, Y) -> np.ndarray` — uncentered EIF `Q1−Q0+H(Y−QA)`.
  - `ep_and_remainder(nf, df_sim, surface) -> tuple[float, float]` — (√n·EP, remainder).
  - `run_cell(learner, crossfit, surface, n, n_sims, base_seed) -> pd.DataFrame` — one row per (sim, estimator).
  - `summarize(df) -> pd.DataFrame` — per-cell aggregates.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exp33_donsker_learners.py`:

```python
import numpy as np

from causal_bench.dgp.point_treatment import draw_point_treatment, true_tau
from causal_bench.estimators.point import oracle_nuisances
from experiments.exp33_donsker_learners import (
    ep_and_remainder, run_cell, summarize)

W_COLS = ["W1", "W2", "W3", "W4"]


def test_oracle_ep_and_remainder_are_zero():
    # With f_hat == f_0, eif0(f_hat) - eif0(f_0) == 0 pointwise, so both
    # the EP term and the remainder must vanish (remainder up to MC error).
    df = draw_point_treatment(n=700, surface="smooth", seed=0)
    nf = oracle_nuisances(df[W_COLS].values, "smooth")
    ep, rem = ep_and_remainder(nf, df, "smooth")
    assert ep == 0.0
    assert abs(rem) < 0.01   # MC error of the fixed 1e5 evaluation sample


def test_run_cell_logistic_shape_and_columns():
    out = run_cell("logistic", crossfit=False, surface="smooth",
                   n=300, n_sims=3, base_seed=0)
    assert len(out) == 6        # 3 sims x {aipw, tmle}
    for col in ["learner", "crossfit", "surface", "estimator", "sim", "point",
                "se", "covered", "g_rmse", "q_rmse", "sqrtn_ep", "remainder"]:
        assert col in out.columns
    assert set(out["estimator"]) == {"aipw", "tmle"}
    assert out["point"].between(-1, 1).all()


def test_summarize_aggregates():
    out = run_cell("logistic", crossfit=True, surface="jumpy",
                   n=300, n_sims=3, base_seed=1)
    summ = summarize(out)
    assert len(summ) == 2       # one row per estimator within the cell
    for col in ["bias", "rmse", "coverage", "se_ratio", "g_rmse",
                "sqrtn_ep_mean", "sqrtn_ep_sd"]:
        assert col in summ.columns
    tau0 = true_tau("jumpy")
    assert np.isfinite(summ["bias"]).all()
    assert (summ["rmse"] >= abs(summ["bias"]) - 1e-12).all()
    assert summ["coverage"].between(0, 1).all()
    assert np.isfinite(tau0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_exp33_donsker_learners.py -v`
Expected: FAIL (ModuleNotFoundError: `experiments.exp33_donsker_learners`).

- [ ] **Step 3: Write the experiment core**

Create `experiments/exp33_donsker_learners.py`:

```python
"""Exp 33: do Donsker-class learners license AIPW/TMLE without cross-fitting?

Spec: docs/superpowers/specs/2026-07-02-ltb-har-benchmark-design.md.

Grid: learner in {logistic, xgboost, ltb, har, hal, oracle}
      x crossfit in {off, on} x surface in {jumpy, smooth}, n=700.

Beyond the usual bias/RMSE/coverage/se_ratio, each simulation measures
the two terms of the estimator expansion directly against the DGP truth:

  EP        = (P_n - P)[eif0(f_hat) - eif0(f_0)]   (reported as sqrt(n)*EP)
  remainder = P[eif0(f_hat)] - tau_0

where the population part P[.] is evaluated on a fixed independent
Monte Carlo sample (N=1e5 per surface). The Donsker theory's claim is
precisely that sqrt(n)*EP -> 0 without cross-fitting for LTB/HAL (and
HAR on the smooth surface only); xgboost is the non-Donsker control.
Under crossfit, the P_n part uses out-of-fold nuisances and the P part
averages the fold models (see NuisanceFits.predict).

The oracle arm (f_hat = f_0) pins both terms at zero by construction.
HAL runs via the existing rpy2 wrappers at reduced n_sims and is
skipped with a warning when rpy2/hal9001 is unavailable.
"""
from __future__ import annotations

import argparse
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.dgp.point_treatment import (
    SURFACES, draw_point_treatment, true_Q, true_g, true_tau)
from causal_bench.estimators.point import (
    fit_nuisances, oracle_nuisances, point_aipw, point_tmle)

OUT_DIR = Path("results/exp33_donsker_learners")
W_COLS = ["W1", "W2", "W3", "W4"]
LEARNERS = ("logistic", "xgboost", "ltb", "har", "hal", "oracle")
_MC_N = 100_000
_MC_SEED = 424242


def make_learners(name: str, seed: int):
    """(g_learner, q_learner) for a grid arm; None for the oracle arm."""
    if name == "oracle":
        return None
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return (LogisticRegression(max_iter=1000, C=1.0),
                LogisticRegression(max_iter=1000, C=1.0))
    if name == "xgboost":
        from xgboost import XGBClassifier
        mk = lambda: XGBClassifier(n_estimators=300, max_depth=3,
                                   learning_rate=0.05, random_state=seed,
                                   n_jobs=1, verbosity=0)
        return (mk(), mk())
    if name == "ltb":
        from causal_bench.ltb import LTBClassifier
        return (LTBClassifier(random_state=seed), LTBClassifier(random_state=seed))
    if name == "har":
        from causal_bench.har import HARClassifier
        return (HARClassifier(random_state=seed), HARClassifier(random_state=seed))
    if name == "hal":
        from causal_bench.hal import HALClassifier  # raises if rpy2 missing
        return (HALClassifier(), HALClassifier())
    raise ValueError(f"unknown learner {name!r}")


@lru_cache(maxsize=None)
def mc_eval_sample(surface: str) -> pd.DataFrame:
    """Fixed independent draw used as the population P[.] in EP/remainder."""
    return draw_point_treatment(n=_MC_N, surface=surface, seed=_MC_SEED)


def eif0_values(g, Q1, Q0, A, Y) -> np.ndarray:
    """Uncentered efficient influence function value eif0 = Q1-Q0+H(Y-QA)."""
    g = np.clip(g, 0.01, 0.99)
    QA = A * Q1 + (1 - A) * Q0
    H = A / g - (1 - A) / (1 - g)
    return Q1 - Q0 + H * (Y - QA)


def ep_and_remainder(nf, df_sim: pd.DataFrame, surface: str):
    """(sqrt(n)*EP, remainder) for the fitted nuisances nf on this sim."""
    W = df_sim[W_COLS].values
    A = df_sim["A"].values.astype(float)
    Y = df_sim["Y"].values.astype(float)
    n = len(A)

    diff_sim = (eif0_values(nf.g, nf.Q1, nf.Q0, A, Y)
                - eif0_values(true_g(W, surface), true_Q(1, W, surface),
                              true_Q(0, W, surface), A, Y))

    mc = mc_eval_sample(surface)
    W_mc = mc[W_COLS].values
    A_mc = mc["A"].values.astype(float)
    Y_mc = mc["Y"].values.astype(float)
    g_mc, Q1_mc, Q0_mc = nf.predict(W_mc)
    eif_hat_mc = eif0_values(g_mc, Q1_mc, Q0_mc, A_mc, Y_mc)
    diff_mc = eif_hat_mc - eif0_values(
        true_g(W_mc, surface), true_Q(1, W_mc, surface),
        true_Q(0, W_mc, surface), A_mc, Y_mc)

    ep = float(np.mean(diff_sim)) - float(np.mean(diff_mc))
    remainder = float(np.mean(eif_hat_mc)) - true_tau(surface)
    return float(np.sqrt(n)) * ep, remainder


def run_cell(learner: str, crossfit: bool, surface: str, n: int,
             n_sims: int, base_seed: int) -> pd.DataFrame:
    """All simulations for one (learner, crossfit, surface) cell."""
    tau0 = true_tau(surface)
    rows = []
    for sim in range(n_sims):
        seed = base_seed + 1000 * sim
        df = draw_point_treatment(n=n, surface=surface, seed=seed)
        W = df[W_COLS].values
        A = df["A"].values.astype(float)
        Y = df["Y"].values.astype(float)

        if learner == "oracle":
            nf = oracle_nuisances(W, surface)
        else:
            g_l, q_l = make_learners(learner, seed)
            nf = fit_nuisances(W, A, Y, g_l, q_l, crossfit=crossfit,
                               random_state=seed)

        g_rmse = float(np.sqrt(np.mean((nf.g - true_g(W, surface)) ** 2)))
        q_rmse = float(np.sqrt(np.mean(
            (nf.Q1 - true_Q(1, W, surface)) ** 2
            + (nf.Q0 - true_Q(0, W, surface)) ** 2) / np.sqrt(2)))
        sqrtn_ep, remainder = ep_and_remainder(nf, df, surface)

        for est_name, est in (("aipw", point_aipw), ("tmle", point_tmle)):
            r = est(A, Y, nf)
            rows.append({
                "learner": learner, "crossfit": crossfit, "surface": surface,
                "estimator": est_name, "sim": sim, "n": n,
                "point": r.point, "se": r.se,
                "covered": bool(r.ci_lower <= tau0 <= r.ci_upper),
                "g_rmse": g_rmse, "q_rmse": q_rmse,
                "sqrtn_ep": sqrtn_ep, "remainder": remainder,
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (learner, crossfit, surface, estimator) cell aggregates."""
    out = []
    keys = ["learner", "crossfit", "surface", "estimator"]
    for vals, grp in df.groupby(keys):
        tau0 = true_tau(vals[keys.index("surface")])
        emp_sd = float(grp["point"].std(ddof=1)) if len(grp) > 1 else float("nan")
        out.append(dict(
            zip(keys, vals),
            n_sims=len(grp),
            bias=float(grp["point"].mean() - tau0),
            rmse=float(np.sqrt(np.mean((grp["point"] - tau0) ** 2))),
            coverage=float(grp["covered"].mean()),
            se_ratio=float(grp["se"].mean() / emp_sd) if emp_sd else float("nan"),
            g_rmse=float(grp["g_rmse"].mean()),
            q_rmse=float(grp["q_rmse"].mean()),
            sqrtn_ep_mean=float(grp["sqrtn_ep"].mean()),
            sqrtn_ep_sd=float(grp["sqrtn_ep"].std(ddof=1)) if len(grp) > 1
                        else float("nan"),
            remainder_mean=float(grp["remainder"].mean()),
        ))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--n-sims-hal", type=int, default=50)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--skip-hal", action="store_true")
    ap.add_argument("--learners", nargs="+", default=list(LEARNERS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for surface in SURFACES:
        for learner in args.learners:
            if learner == "hal":
                if args.skip_hal:
                    continue
                try:
                    make_learners("hal", 0)
                except Exception as e:  # rpy2/hal9001 absent
                    warnings.warn(f"skipping HAL arm: {e}")
                    continue
            n_sims = args.n_sims_hal if learner == "hal" else args.n_sims
            crossfits = (False,) if learner == "oracle" else (False, True)
            for crossfit in crossfits:
                print(f"[exp33] {surface} / {learner} / crossfit={crossfit} "
                      f"({n_sims} sims)")
                frames.append(run_cell(learner, crossfit, surface,
                                       n=args.n, n_sims=n_sims,
                                       base_seed=args.seed))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(OUT_DIR / "raw.csv", index=False)
    summ = summarize(raw)
    summ.to_csv(OUT_DIR / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.sort_values(["surface", "estimator", "learner", "crossfit"])
                  .to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_exp33_donsker_learners.py -v`
Expected: 3 PASS (~30s; logistic-only cells at n=300).

- [ ] **Step 5: Commit**

```bash
git add experiments/exp33_donsker_learners.py tests/test_exp33_donsker_learners.py
git commit -m "feat: exp33 Donsker-learner benchmark with direct EP/remainder measurement (part of #N)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: End-to-end smoke run, full suite, PR

**Files:**
- No new files; verification and delivery only.

**Interfaces:**
- Consumes: everything above; issue `#N` from Task 0.

- [ ] **Step 1: Smoke-run the experiment end-to-end (fast arms)**

Run:
```bash
python experiments/exp33_donsker_learners.py \
  --n-sims 3 --n 300 --skip-hal --learners logistic ltb har oracle
```
Expected: prints per-cell progress lines then a summary table; `results/exp33_donsker_learners/raw.csv` and `summary.csv` exist. Sanity-read the table: oracle rows have `sqrtn_ep_mean == 0` exactly and `|bias|` small; every row has finite metrics. (3 sims says nothing about coverage — this is a plumbing check only.)

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: all tests pass, including the pre-existing suite (no production estimator was touched, so any pre-existing failure must also exist on `main` — verify with `git stash && pytest <failing test> && git stash pop` before concluding it's ours).

- [ ] **Step 3: Confirm results artifacts are not committed**

Run: `git status --short`
Expected: `results/` output is untracked/ignored; only source, tests, docs staged-or-clean. Do not commit `results/`.

- [ ] **Step 4: Push and open the phase-1 PR**

```bash
git push -u origin exp33-donsker-learners
gh pr create \
  --title "exp33: Donsker-class nuisance learners (LTB/HAR) benchmark" \
  --body "$(cat <<'EOF'
Part of #N (phase 1 of 2; phase 2 wires the learners into TMLE-IPCW).

Spec: docs/superpowers/specs/2026-07-02-ltb-har-benchmark-design.md

- causal_bench/ltb.py: Lassoed Tree Boosting (arXiv:2205.10697v6) — xgboost tree basis + L1 selection, block early stopping; sklearn protocol per hal.py.
- causal_bench/har.py: Highly Adaptive Ridge (arXiv:2410.02680) — dominance kernel + exact-LOO ridge; squared-error caveat documented.
- causal_bench/dgp/point_treatment.py: jumpy/smooth point-treatment DGP exposing true_g/true_Q/true_tau.
- causal_bench/estimators/point.py: AIPW/TMLE with (g_learner, q_learner, crossfit) — attribution harness, production estimators untouched.
- experiments/exp33_donsker_learners.py: learner x crossfit x surface grid; measures sqrt(n)*(P_n-P)(D(f_hat)-D(f_0)) and the remainder directly against DGP truth; HAL reference arm optional via rpy2.
- xgboost>=2.0 added to core dependencies.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed. Body says "part of #N" only — no closing keywords.

---

## Self-Review (completed at plan-writing time)

- **Spec coverage:** §2 LTB → Task 2; §3 HAR → Task 3; §4 DGP → Task 1; §5 harness → Task 4; §6 grid/metrics → Task 5; §7 tests → Tasks 1-5 test steps + Task 6 full-suite; §9 tracking → Tasks 0/6; §10 xgboost → Task 0. Phase 2/3/exp34 are spec-only by design.
- **Placeholder scan:** none; every step carries runnable code/commands.
- **Type consistency:** `NuisanceFits.predict(W_new) -> (g, Q1, Q0)` consumed identically in Task 5's `ep_and_remainder`; `make_learners` returns the `(g_learner, q_learner)` tuple `fit_nuisances` expects; `PointResult` field names match usage in `run_cell`.
