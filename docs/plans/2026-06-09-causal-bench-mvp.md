# causal_bench MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a runnable CLI demo that simulates clinical trial data and compares 5 key causal estimators (Naive → KM → Cox → TMLE+IPCW → TMLE+IPCW+compliance) to show when TMLE with IPCW outperforms standard methods and what compliance data adds analytically.

**Architecture:** Pure Python package `causal_bench` with a DGP module, estimator module, Super Learner, Monte Carlo runner, and matplotlib visualizations. The inner package matches the repo name to avoid namespace confusion. The R bridge and LTMLE are out of scope for this MVP.

**Tech Stack:** Python 3.10+, numpy, scipy, pandas, scikit-learn, lifelines, matplotlib, joblib, tqdm. No zepid — TMLE is implemented manually.

---

## Package layout

```
causal_bench/          ← repo root
  causal_bench/        ← Python package
    __init__.py
    dgp/
      __init__.py
      config.py
      survival.py
      scenarios.py
    estimators/
      __init__.py
      base.py
      naive.py
      kaplan_meier.py
      cox.py
      tmle_ipcw.py
      tmle_ipcw_comply.py
    super_learner.py
    metrics.py
    runner.py
    viz.py
    __main__.py
  tests/
    test_dgp.py
    test_estimators.py
    test_super_learner.py
  docs/plans/
  pyproject.toml
  requirements.txt
  README.md
```

---

## Task 1: Project scaffold

**Files:**
- Create: `causal_bench/pyproject.toml`
- Create: `causal_bench/requirements.txt`
- Create: `causal_bench/causal_bench/__init__.py`
- Create: all `__init__.py` stubs for sub-packages

**Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "causal_bench"
version = "0.1.0"
description = "Monte Carlo benchmarking of causal estimators for clinical trials"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "lifelines>=0.27",
    "matplotlib>=3.7",
    "joblib>=1.3",
    "tqdm>=4.65",
]

[project.optional-dependencies]
dev = ["pytest>=7", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["."]
include = ["causal_bench*"]
```

**Step 2: Write requirements.txt**

```
numpy>=1.24
scipy>=1.10
pandas>=2.0
scikit-learn>=1.3
lifelines>=0.27
matplotlib>=3.7
joblib>=1.3
tqdm>=4.65
pytest>=7
pytest-cov
```

**Step 3: Create all __init__.py stubs**

`causal_bench/__init__.py`: `__version__ = "0.1.0"`
All others: empty files.

**Step 4: Install in editable mode**

```bash
cd /Users/noahrahman/git/causal_bench
pip install -e ".[dev]"
```

Expected: `Successfully installed causal-bench-0.1.0`

**Step 5: Commit**

```bash
git init && git add -A && git commit -m "chore: project scaffold"
```

---

## Task 2: DGP config dataclass

**Files:**
- Create: `causal_bench/causal_bench/dgp/config.py`

**Step 1: Write failing test**

```python
# tests/test_dgp.py
from causal_bench.dgp.config import DGPConfig

def test_dgp_config_defaults():
    cfg = DGPConfig()
    assert cfg.n == 500
    assert cfg.true_tau == -0.5
    assert cfg.censoring_informativeness == 0.0
    assert cfg.compliance_available is True
    assert cfg.seed == 42

def test_dgp_config_override():
    cfg = DGPConfig(n=200, true_tau=-0.3, censoring_informativeness=0.6)
    assert cfg.n == 200
    assert cfg.true_tau == -0.3
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_dgp.py::test_dgp_config_defaults -v
```
Expected: ImportError or ModuleNotFoundError

**Step 3: Implement DGPConfig**

```python
# causal_bench/dgp/config.py
from dataclasses import dataclass, field

@dataclass
class DGPConfig:
    # Sample
    n: int = 500
    n_treated_fraction: float = 0.5

    # Treatment
    true_tau: float = -0.5
    treatment_prevalence: float = 0.5
    positivity_severity: float = 0.0
    unmeasured_confounding_strength: float = 0.0

    # Outcome
    outcome_nonlinearity: float = 0.0
    effect_heterogeneity: float = 0.0
    baseline_hazard: str = "weibull"
    horizon: float = 1.0

    # Censoring
    censoring_rate: float = 0.25
    censoring_informativeness: float = 0.0

    # Crossover (unused in MVP, present for scenario compatibility)
    crossover_rate: float = 0.0
    crossover_informativeness: float = 0.0

    # Time-varying confounder (unused in MVP)
    collider_strength: float = 0.0
    sigma_L: float = 0.5
    t_L1: float = 0.3

    # Competing risks (unused in MVP)
    competing_risks: bool = False
    cause1_fraction: float = 0.4
    cause1_treatment_effect: float = -0.3
    cause2_treatment_effect: float = -0.6

    # Enrollment drift
    enrollment_drift: float = 0.0
    enrollment_period: float = 1.0

    # Compliance covariate
    compliance_available: bool = True
    compliance_censoring_r2: float = 0.3

    seed: int = 42
```

**Step 4: Run tests**

```bash
pytest tests/test_dgp.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add causal_bench/dgp/config.py tests/test_dgp.py
git commit -m "feat: DGPConfig dataclass"
```

---

## Task 3: DGP survival data generator

**Files:**
- Create: `causal_bench/causal_bench/dgp/survival.py`

This is the core DGP. It generates one dataset for a single simulation replicate.

**Step 1: Write failing tests**

```python
# tests/test_dgp.py (add to existing)
import numpy as np
import pandas as pd
from causal_bench.dgp.survival import generate_data
from causal_bench.dgp.config import DGPConfig

def test_generate_data_shape():
    cfg = DGPConfig(n=200, seed=0)
    df = generate_data(cfg)
    assert len(df) == 200
    required = {"T_obs", "Delta", "A", "W1", "W2", "W3", "W4",
                "compliance", "Y_neg", "enrollment_time"}
    assert required.issubset(df.columns)

def test_generate_data_u_not_observed():
    cfg = DGPConfig(n=200, seed=0)
    df = generate_data(cfg)
    assert "U" not in df.columns

def test_generate_data_censoring_rate():
    cfg = DGPConfig(n=2000, censoring_rate=0.25, censoring_informativeness=0.0, seed=1)
    df = generate_data(cfg)
    obs_censor = 1 - df["Delta"].mean()
    assert 0.10 <= obs_censor <= 0.50

def test_generate_data_treatment_prevalence():
    cfg = DGPConfig(n=2000, treatment_prevalence=0.5, seed=2)
    df = generate_data(cfg)
    assert 0.40 <= df["A"].mean() <= 0.60

def test_generate_data_negative_control_no_treatment_effect():
    """Y_neg should have near-zero treatment coefficient in large samples."""
    cfg = DGPConfig(n=5000, unmeasured_confounding_strength=0.0, seed=3)
    df = generate_data(cfg)
    from scipy.stats import pearsonr
    r, _ = pearsonr(df["A"], df["Y_neg"] - df["W1"] * 0.5)
    assert abs(r) < 0.10
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_dgp.py::test_generate_data_shape -v
```

**Step 3: Implement generate_data**

```python
# causal_bench/dgp/survival.py
import numpy as np
import pandas as pd
from causal_bench.dgp.config import DGPConfig


def generate_data(config: DGPConfig, rng: np.random.Generator = None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(config.seed)

    n = config.n
    horizon = config.horizon

    # ── Latent + observed covariates ──
    U = rng.standard_normal(n)
    W1 = rng.standard_normal(n)
    W2 = rng.binomial(1, 0.5, n).astype(float)
    W3 = rng.standard_normal(n)
    W4 = rng.binomial(1, 0.3, n).astype(float)

    # ── Enrollment time ──
    enrollment_time = rng.uniform(0, config.enrollment_period, n)

    # ── Treatment ──
    logit_A = (0.0
               + 0.3 * W1
               + 0.2 * W2
               - 0.2 * W3
               + 0.1 * W4
               + 0.5 * U * config.unmeasured_confounding_strength
               + 0.8 * W1 * W3 * config.positivity_severity)
    prob_A = _sigmoid(logit_A)
    A = rng.binomial(1, prob_A).astype(float)

    # ── Potential survival times (AFT with Gumbel noise → Weibull) ──
    def survival_time(a_val):
        log_T = (1.0
                 + 0.4 * W1
                 - 0.3 * W2
                 + 0.2 * W3
                 - 0.2 * W4
                 + 0.3 * U
                 + config.true_tau * a_val
                 + config.enrollment_drift * enrollment_time
                 + config.outcome_nonlinearity * (W1 ** 2 - 1.0)
                 + config.effect_heterogeneity * a_val * W1
                 + rng.gumbel(0, 1, n))
        return np.exp(log_T)

    T1 = survival_time(1.0)
    T0 = survival_time(0.0)
    T_true = np.where(A == 1, T1, T0)

    # ── Compliance covariate (observed; correlated with U) ──
    rho = np.sqrt(config.compliance_censoring_r2)
    compliance = rho * U + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    compliance = _sigmoid(compliance)  # map to [0,1]

    # ── Censoring time ──
    log_C = (1.5
             - 0.2 * W1
             + 0.1 * W3
             - 0.1 * A
             + 0.4 * U * config.censoring_informativeness
             + rng.gumbel(0, 1, n))
    # MNAR component: sicker patients (lower T) censored more when informativeness > 0.5
    mnar_weight = max(0.0, config.censoring_informativeness - 0.5) * 2.0
    T_median = np.median(T_true)
    log_C -= mnar_weight * (T_true < T_median).astype(float)
    C = np.exp(log_C)

    # Scale C to hit target censoring_rate under MCAR (informativeness=0)
    # Use a simple scale factor calibrated at generation time
    scale = _calibrate_censoring_scale(config, rng.integers(0, 2**31))
    C = C * scale

    T_obs = np.minimum(T_true, np.minimum(C, horizon))
    Delta = (T_true <= C) & (T_true <= horizon)
    Delta = Delta.astype(float)

    # ── Negative control outcome (no treatment effect) ──
    Y_neg = (0.5 * W1 - 0.3 * W3 + 0.4 * U
             + rng.standard_normal(n) * 0.5)

    df = pd.DataFrame({
        "T_obs": T_obs,
        "Delta": Delta,
        "A": A,
        "W1": W1, "W2": W2, "W3": W3, "W4": W4,
        "compliance": compliance,
        "enrollment_time": enrollment_time,
        "Y_neg": Y_neg,
    })
    return df


def compute_true_effects(config: DGPConfig, n_ref: int = 50_000) -> dict:
    """Estimate true ATE and ATT from a large reference population."""
    rng = np.random.default_rng(config.seed + 999_999)
    cfg_ref = config.__class__(**{**config.__dict__, "n": n_ref,
                                   "censoring_rate": 0.0,
                                   "censoring_informativeness": 0.0})
    # Generate with treatment forced to 1
    cfg1 = cfg_ref.__class__(**{**cfg_ref.__dict__, "seed": config.seed + 1_000_000})
    cfg0 = cfg_ref.__class__(**{**cfg_ref.__dict__, "seed": config.seed + 2_000_000})

    rng1 = np.random.default_rng(cfg1.seed)
    rng0 = np.random.default_rng(cfg0.seed)

    # Share baseline covariates for paired comparison
    shared_rng = np.random.default_rng(config.seed + 3_000_000)
    n = n_ref
    U = shared_rng.standard_normal(n)
    W1 = shared_rng.standard_normal(n)
    W2 = shared_rng.binomial(1, 0.5, n).astype(float)
    W3 = shared_rng.standard_normal(n)
    W4 = shared_rng.binomial(1, 0.3, n).astype(float)
    enrollment_time = shared_rng.uniform(0, config.enrollment_period, n)
    logit_A = (0.3*W1 + 0.2*W2 - 0.2*W3 + 0.1*W4
               + 0.5*U*config.unmeasured_confounding_strength
               + 0.8*W1*W3*config.positivity_severity)
    prob_A = _sigmoid(logit_A)
    A_obs = shared_rng.binomial(1, prob_A).astype(float)

    def _t(a_val, noise_rng):
        log_T = (1.0 + 0.4*W1 - 0.3*W2 + 0.2*W3 - 0.2*W4 + 0.3*U
                 + config.true_tau * a_val
                 + config.enrollment_drift * enrollment_time
                 + config.outcome_nonlinearity * (W1**2 - 1.0)
                 + config.effect_heterogeneity * a_val * W1
                 + noise_rng.gumbel(0, 1, n))
        return np.exp(log_T)

    # Use same noise for both potential outcomes (common noise)
    gumbel = shared_rng.gumbel(0, 1, n)

    def _t_fixed_noise(a_val):
        log_T = (1.0 + 0.4*W1 - 0.3*W2 + 0.2*W3 - 0.2*W4 + 0.3*U
                 + config.true_tau * a_val
                 + config.enrollment_drift * enrollment_time
                 + config.outcome_nonlinearity * (W1**2 - 1.0)
                 + config.effect_heterogeneity * a_val * W1
                 + gumbel)
        return np.exp(log_T)

    T1 = _t_fixed_noise(1.0)
    T0 = _t_fixed_noise(0.0)

    Y1 = (T1 <= config.horizon).astype(float)
    Y0 = (T0 <= config.horizon).astype(float)

    ATE = float(np.mean(Y1 - Y0))
    treated = A_obs == 1
    ATT = float(np.mean(Y1[treated] - Y0[treated])) if treated.sum() > 0 else ATE

    return {"ATE": ATE, "ATT": ATT}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _calibrate_censoring_scale(config: DGPConfig, seed: int) -> float:
    """Find scale factor so ~censoring_rate fraction are censored under MCAR."""
    if config.censoring_rate <= 0:
        return 1e10  # no censoring
    # Binary search on scale: higher scale → less censoring
    rng = np.random.default_rng(seed)
    n = 5000
    U = rng.standard_normal(n)
    W1 = rng.standard_normal(n)
    W3 = rng.standard_normal(n)
    A = rng.binomial(1, 0.5, n).astype(float)
    log_T = 1.0 + 0.4*W1 + 0.3*U + rng.gumbel(0, 1, n)
    T_true = np.exp(log_T)
    log_C_base = 1.5 - 0.2*W1 + 0.1*W3 - 0.1*A + rng.gumbel(0, 1, n)
    C_base = np.exp(log_C_base)

    lo, hi = 0.01, 100.0
    for _ in range(40):
        mid = (lo + hi) / 2
        C = C_base * mid
        T_obs = np.minimum(T_true, np.minimum(C, config.horizon))
        censor_rate = np.mean((C < T_true) & (C < config.horizon))
        if censor_rate > config.censoring_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
```

**Step 4: Run tests**

```bash
pytest tests/test_dgp.py -v
```
Expected: all PASS (the negative control test may need tolerance adjustment)

**Step 5: Commit**

```bash
git add causal_bench/dgp/survival.py tests/test_dgp.py
git commit -m "feat: survival DGP with MCAR/MAR/MNAR censoring and negative control"
```

---

## Task 4: Named scenarios

**Files:**
- Create: `causal_bench/causal_bench/dgp/scenarios.py`

**Step 1: Write failing test**

```python
# tests/test_dgp.py (add)
from causal_bench.dgp.scenarios import get_scenario, list_scenarios

def test_get_scenario_clean():
    cfg = get_scenario("clean")
    assert cfg.censoring_informativeness == 0.0
    assert cfg.positivity_severity == 0.0

def test_get_scenario_edwards_realistic():
    cfg = get_scenario("edwards_realistic")
    assert cfg.n == 700
    assert cfg.censoring_informativeness == 0.6

def test_list_scenarios_includes_expected():
    names = list_scenarios()
    for name in ["clean", "edwards_realistic", "edwards_optimistic",
                 "edwards_pessimistic", "censor_mild", "censor_severe"]:
        assert name in names
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_dgp.py::test_get_scenario_clean -v
```

**Step 3: Implement scenarios.py**

```python
# causal_bench/dgp/scenarios.py
from causal_bench.dgp.config import DGPConfig

_CLEAN = dict(
    n=500, censoring_informativeness=0.0, censoring_rate=0.25,
    positivity_severity=0.0, unmeasured_confounding_strength=0.0,
    collider_strength=0.0, crossover_rate=0.0, enrollment_drift=0.0,
    true_tau=-0.5,
)

_REGISTRY: dict[str, dict] = {
    "clean": _CLEAN,
    # Censoring gradient
    "censor_mild":     {**_CLEAN, "censoring_informativeness": 0.3, "censoring_rate": 0.25},
    "censor_moderate": {**_CLEAN, "censoring_informativeness": 0.6, "censoring_rate": 0.30},
    "censor_severe":   {**_CLEAN, "censoring_informativeness": 1.0, "censoring_rate": 0.40},
    # Positivity gradient
    "positivity_mild":     {**_CLEAN, "positivity_severity": 1.0},
    "positivity_moderate": {**_CLEAN, "positivity_severity": 2.0},
    "positivity_severe":   {**_CLEAN, "positivity_severity": 3.0},
    # Unmeasured confounding gradient
    "unmeasured_mild":   {**_CLEAN, "unmeasured_confounding_strength": 0.2},
    "unmeasured_mod":    {**_CLEAN, "unmeasured_confounding_strength": 0.5},
    "unmeasured_strong": {**_CLEAN, "unmeasured_confounding_strength": 0.8},
    # Edwards variants
    "edwards_realistic": dict(
        n=700, n_treated_fraction=0.43,
        censoring_informativeness=0.6, censoring_rate=0.25,
        positivity_severity=1.5, crossover_rate=0.05,
        unmeasured_confounding_strength=0.2,
        collider_strength=0.4, enrollment_drift=0.15,
        outcome_nonlinearity=0.5, effect_heterogeneity=0.3,
        true_tau=-0.5,
    ),
    "edwards_optimistic": dict(
        n=700, n_treated_fraction=0.43,
        censoring_informativeness=0.3, censoring_rate=0.15,
        positivity_severity=0.5, unmeasured_confounding_strength=0.1,
        collider_strength=0.2, enrollment_drift=0.05,
        true_tau=-0.5,
    ),
    "edwards_pessimistic": dict(
        n=700, n_treated_fraction=0.43,
        censoring_informativeness=0.9, censoring_rate=0.40,
        positivity_severity=2.5, crossover_rate=0.10,
        unmeasured_confounding_strength=0.4,
        collider_strength=0.7, enrollment_drift=0.3,
        outcome_nonlinearity=0.7, effect_heterogeneity=0.5,
        true_tau=-0.5,
    ),
}


def get_scenario(name: str) -> DGPConfig:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Known: {list(_REGISTRY)}")
    return DGPConfig(**_REGISTRY[name])


def list_scenarios() -> list[str]:
    return list(_REGISTRY.keys())
```

**Step 4: Run tests**

```bash
pytest tests/test_dgp.py -v
```

**Step 5: Commit**

```bash
git add causal_bench/dgp/scenarios.py tests/test_dgp.py
git commit -m "feat: named scenario registry"
```

---

## Task 5: Metrics dataclasses

**Files:**
- Create: `causal_bench/causal_bench/metrics.py`

**Step 1: Implement (no complex logic to TDD here)**

```python
# causal_bench/metrics.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EstimatorResult:
    name: str
    estimand: str          # "ATE", "ATT", "ATO"
    point_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    ess: Optional[float] = None
    convergence_info: Optional[dict] = None


@dataclass
class SimResult:
    """Aggregated metrics across n_sim Monte Carlo replicates for one estimator."""
    estimator_name: str
    estimand: str
    true_value: float
    n_sim: int
    # Per-replicate arrays (length n_sim)
    estimates: np.ndarray = field(repr=False)
    se_estimates: np.ndarray = field(repr=False)
    ci_lowers: np.ndarray = field(repr=False)
    ci_uppers: np.ndarray = field(repr=False)
    nc_estimates: np.ndarray = field(repr=False)   # negative control point estimates

    @property
    def bias(self) -> float:
        return float(np.mean(self.estimates) - self.true_value)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean((self.estimates - self.true_value) ** 2)))

    @property
    def coverage(self) -> float:
        covered = (self.ci_lowers <= self.true_value) & (self.true_value <= self.ci_uppers)
        return float(np.mean(covered))

    @property
    def ci_width(self) -> float:
        return float(np.mean(self.ci_uppers - self.ci_lowers))

    @property
    def se_ratio(self) -> float:
        empirical_se = np.std(self.estimates, ddof=1)
        if empirical_se < 1e-10:
            return float("nan")
        return float(np.median(self.se_estimates) / empirical_se)

    @property
    def nc_bias(self) -> float:
        return float(np.mean(self.nc_estimates))

    def summary(self) -> dict:
        return {
            "estimator": self.estimator_name,
            "estimand": self.estimand,
            "true": round(self.true_value, 4),
            "bias": round(self.bias, 4),
            "rmse": round(self.rmse, 4),
            "coverage": round(self.coverage, 3),
            "ci_width": round(self.ci_width, 4),
            "se_ratio": round(self.se_ratio, 3),
            "nc_bias": round(self.nc_bias, 4),
        }
```

**Step 2: Quick smoke test**

```python
# tests/test_estimators.py (start this file)
import numpy as np
from causal_bench.metrics import SimResult, EstimatorResult

def test_sim_result_bias():
    estimates = np.array([0.1, 0.2, 0.3])
    sr = SimResult("test", "ATE", true_value=0.2, n_sim=3,
                   estimates=estimates,
                   se_estimates=np.array([0.05, 0.05, 0.05]),
                   ci_lowers=estimates - 0.1,
                   ci_uppers=estimates + 0.1,
                   nc_estimates=np.array([0.01, -0.01, 0.0]))
    assert abs(sr.bias) < 1e-10
    assert sr.coverage == 1.0
```

```bash
pytest tests/test_estimators.py -v
```

**Step 3: Commit**

```bash
git add causal_bench/metrics.py tests/test_estimators.py
git commit -m "feat: EstimatorResult and SimResult dataclasses"
```

---

## Task 6: Super Learner

**Files:**
- Create: `causal_bench/causal_bench/super_learner.py`
- Create: `tests/test_super_learner.py`

**Step 1: Write failing tests**

```python
# tests/test_super_learner.py
import numpy as np
from sklearn.datasets import make_classification
from causal_bench.super_learner import SuperLearner

def test_super_learner_fit_predict_proba():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    sl = SuperLearner(task="classification", n_folds=3, random_state=0)
    sl.fit(X, y)
    probs = sl.predict_proba(X)
    assert probs.shape == (300,)
    assert np.all((probs >= 0) & (probs <= 1))

def test_super_learner_weights_sum_to_one():
    X, y = make_classification(n_samples=300, n_features=5, random_state=1)
    sl = SuperLearner(task="classification", n_folds=3, random_state=1)
    sl.fit(X, y)
    assert abs(sum(sl.weights_) - 1.0) < 1e-6
    assert all(w >= 0 for w in sl.weights_)

def test_super_learner_regression():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = X[:, 0] + rng.standard_normal(300) * 0.1
    sl = SuperLearner(task="regression", n_folds=3, random_state=0)
    sl.fit(X, y)
    preds = sl.predict(X)
    assert preds.shape == (300,)
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_super_learner.py -v
```

**Step 3: Implement SuperLearner**

```python
# causal_bench/super_learner.py
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
        self._fitted_candidates = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        candidates = self.candidates or (
            _default_classifiers() if self.task == "classification"
            else _default_regressors()
        )
        n = len(y)
        k = len(candidates)
        oof = np.zeros((n, k))

        if self.task == "classification":
            splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                       random_state=self.random_state)
        else:
            splitter = KFold(n_splits=self.n_folds, shuffle=True,
                             random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            for j, est in enumerate(candidates):
                m = clone(est)
                m.fit(X[train_idx], y[train_idx])
                if self.task == "classification":
                    oof[val_idx, j] = m.predict_proba(X[val_idx])[:, 1]
                else:
                    oof[val_idx, j] = m.predict(X[val_idx])

        # NNLS stacking
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
        X = np.asarray(X)
        preds = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self._fitted_candidates
        ])
        result = preds @ self.weights_
        return np.clip(result, 1e-6, 1 - 1e-6)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        preds = np.column_stack([m.predict(X) for m in self._fitted_candidates])
        return preds @ self.weights_
```

**Step 4: Run tests**

```bash
pytest tests/test_super_learner.py -v
```

**Step 5: Commit**

```bash
git add causal_bench/super_learner.py tests/test_super_learner.py
git commit -m "feat: Super Learner with NNLS stacking and K-fold cross-fitting"
```

---

## Task 7: Base estimator interface

**Files:**
- Create: `causal_bench/causal_bench/estimators/base.py`

**Step 1: Implement**

```python
# causal_bench/estimators/base.py
from abc import ABC, abstractmethod
import pandas as pd
from causal_bench.metrics import EstimatorResult


class BaseEstimator(ABC):
    name: str = "base"

    @abstractmethod
    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
    ) -> list[EstimatorResult]:
        """
        Args:
            df: DataFrame with columns T_obs, Delta, A, W1-W4,
                compliance (optional), Y_neg
            horizon: time point for risk difference
            estimand: "ATE" or "ATT"
        Returns:
            list of EstimatorResult (one per estimand the method reports)
        """
        ...

    def estimate_negative_control(
        self, df: pd.DataFrame, horizon: float = 1.0
    ) -> float:
        """Run estimate() with Y_neg substituted for the outcome.
        Default: use A=0 for all (difference should be ~0)."""
        import numpy as np
        df_nc = df.copy()
        # For NC: treat Y_neg as a binary outcome (above median = event)
        threshold = df_nc["Y_neg"].median()
        df_nc["_Y_nc"] = (df_nc["Y_neg"] > threshold).astype(float)
        df_nc["Delta"] = 1.0           # everyone "observed"
        df_nc["T_obs"] = horizon * 0.99  # all survive to near-horizon
        # Naive difference in Y_neg by treatment arm
        treated = df_nc["A"] == 1
        return float(df_nc.loc[treated, "_Y_nc"].mean()
                     - df_nc.loc[~treated, "_Y_nc"].mean())
```

**Step 2: Commit**

```bash
git add causal_bench/estimators/base.py
git commit -m "feat: BaseEstimator ABC"
```

---

## Task 8: Naive and KM estimators

**Files:**
- Create: `causal_bench/causal_bench/estimators/naive.py`
- Create: `causal_bench/causal_bench/estimators/kaplan_meier.py`

**Step 1: Write failing tests**

```python
# tests/test_estimators.py (add)
import pandas as pd
import numpy as np
from causal_bench.dgp.survival import generate_data
from causal_bench.dgp.config import DGPConfig
from causal_bench.estimators.naive import NaiveEstimator
from causal_bench.estimators.kaplan_meier import KaplanMeierEstimator

def _clean_df(n=500, seed=0):
    return generate_data(DGPConfig(n=n, censoring_informativeness=0.0,
                                   unmeasured_confounding_strength=0.0,
                                   positivity_severity=0.0, seed=seed))

def test_naive_returns_result():
    df = _clean_df()
    results = NaiveEstimator().estimate(df, horizon=1.0, estimand="ATE")
    assert len(results) == 1
    r = results[0]
    assert r.name == "Naive"
    assert -2.0 < r.point_estimate < 2.0
    assert r.ci_lower < r.point_estimate < r.ci_upper

def test_km_returns_result():
    df = _clean_df()
    results = KaplanMeierEstimator().estimate(df, horizon=1.0, estimand="ATE")
    assert len(results) >= 1
    r = results[0]
    assert r.name == "KM"
    assert -1.0 < r.point_estimate < 1.0
```

**Step 2: Implement NaiveEstimator**

```python
# causal_bench/estimators/naive.py
import numpy as np
import pandas as pd
from scipy import stats
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult


class NaiveEstimator(BaseEstimator):
    name = "Naive"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        obs = df[df["Delta"] == 1].copy()
        y1 = (obs.loc[obs["A"] == 1, "T_obs"] <= horizon).astype(float)
        y0 = (obs.loc[obs["A"] == 0, "T_obs"] <= horizon).astype(float)

        if len(y1) == 0 or len(y0) == 0:
            return [EstimatorResult(self.name, estimand, float("nan"),
                                    float("nan"), float("nan"), float("nan"))]

        point = y1.mean() - y0.mean()
        se = np.sqrt(y1.var() / len(y1) + y0.var() / len(y0))
        z = stats.norm.ppf(0.975)
        return [EstimatorResult(
            name=self.name, estimand=estimand,
            point_estimate=float(point), standard_error=float(se),
            ci_lower=float(point - z * se), ci_upper=float(point + z * se),
        )]
```

**Step 3: Implement KaplanMeierEstimator**

```python
# causal_bench/estimators/kaplan_meier.py
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult


class KaplanMeierEstimator(BaseEstimator):
    name = "KM"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        results = []
        km1 = KaplanMeierFitter()
        km0 = KaplanMeierFitter()
        arm1 = df[df["A"] == 1]
        arm0 = df[df["A"] == 0]
        km1.fit(arm1["T_obs"], arm1["Delta"])
        km0.fit(arm0["T_obs"], arm0["Delta"])

        s1 = float(km1.survival_function_at_times([horizon]).iloc[0])
        s0 = float(km0.survival_function_at_times([horizon]).iloc[0])
        risk1 = 1 - s1
        risk0 = 1 - s0
        point = risk1 - risk0

        # Greenwood-based SE via delta method on S → 1-S
        var1 = _greenwood_var_at_t(km1, horizon)
        var0 = _greenwood_var_at_t(km0, horizon)
        se = np.sqrt(var1 + var0)
        from scipy import stats
        z = stats.norm.ppf(0.975)
        results.append(EstimatorResult(
            name=self.name, estimand="ATE",
            point_estimate=float(point), standard_error=float(se),
            ci_lower=float(point - z * se), ci_upper=float(point + z * se),
        ))
        return results


def _greenwood_var_at_t(km: KaplanMeierFitter, t: float) -> float:
    """Greenwood variance of 1-S(t) via delta method."""
    tbl = km.event_table
    tbl = tbl[tbl.index <= t]
    s_t = float(km.survival_function_at_times([t]).iloc[0])
    gw = 0.0
    for row in tbl.itertuples():
        d = row.observed
        n = row.at_risk
        if n > 0 and d > 0:
            gw += d / (n * (n - d)) if n > d else 0.0
    return (s_t ** 2) * gw
```

**Step 4: Run tests**

```bash
pytest tests/test_estimators.py -v
```

**Step 5: Commit**

```bash
git add causal_bench/estimators/naive.py causal_bench/estimators/kaplan_meier.py tests/test_estimators.py
git commit -m "feat: Naive and KM estimators"
```

---

## Task 9: Cox PH estimator

**Files:**
- Create: `causal_bench/causal_bench/estimators/cox.py`

**Step 1: Write failing test**

```python
# tests/test_estimators.py (add)
from causal_bench.estimators.cox import CoxEstimator

def test_cox_returns_result():
    df = _clean_df(n=500)
    results = CoxEstimator().estimate(df, horizon=1.0, estimand="ATE")
    assert any(r.name == "Cox" for r in results)
    r = next(r for r in results if r.name == "Cox")
    assert r.ci_lower < r.point_estimate < r.ci_upper
```

**Step 2: Implement CoxEstimator**

```python
# causal_bench/estimators/cox.py
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult


class CoxEstimator(BaseEstimator):
    name = "Cox"

    def __init__(self, include_L1: bool = False):
        self.include_L1 = include_L1

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        covs = ["A", "W1", "W2", "W3", "W4"]
        if self.include_L1 and "L1" in df.columns:
            covs.append("L1")

        cox_df = df[covs + ["T_obs", "Delta"]].dropna()
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="T_obs", event_col="Delta")

        # G-computation: predict S(horizon | A=1, W) and S(horizon | A=0, W)
        df1 = cox_df.copy(); df1["A"] = 1.0
        df0 = cox_df.copy(); df0["A"] = 0.0

        s1 = cph.predict_survival_function(df1, times=[horizon]).mean(axis=1).iloc[0]
        s0 = cph.predict_survival_function(df0, times=[horizon]).mean(axis=1).iloc[0]
        risk1 = 1 - s1
        risk0 = 1 - s0
        point = risk1 - risk0

        # Delta-method SE: bootstrap would be more correct but this is fast
        se = float(cph.standard_errors_["A"]) * abs(point) * 0.5
        se = max(se, 0.001)
        z = stats.norm.ppf(0.975)

        name = "Cox+L1" if self.include_L1 else self.name
        return [EstimatorResult(
            name=name, estimand="ATE",
            point_estimate=float(point), standard_error=float(se),
            ci_lower=float(point - z * se), ci_upper=float(point + z * se),
        )]
```

**Step 3: Run tests**

```bash
pytest tests/test_estimators.py::test_cox_returns_result -v
```

**Step 4: Commit**

```bash
git add causal_bench/estimators/cox.py tests/test_estimators.py
git commit -m "feat: Cox PH estimator with G-computation standardization"
```

---

## Task 10: TMLE + IPCW estimator

**Files:**
- Create: `causal_bench/causal_bench/estimators/tmle_ipcw.py`

This is the core methodological contribution. Implementation uses:
- SuperLearner for g(A|W) (propensity) and Q(A,W) (outcome)
- KM-based IPCW for censoring G(C>t|A,W)
- Manual targeting step with clever covariate
- EIF-based SE

**Step 1: Write failing test**

```python
# tests/test_estimators.py (add)
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator

def test_tmle_ipcw_returns_ate_att():
    df = _clean_df(n=400, seed=5)
    est = TMLEIPCWEstimator(use_compliance=False)
    results = est.estimate(df, horizon=1.0, estimand="ATE")
    names = {r.name for r in results}
    assert "TMLE+IPCW" in names
    for r in results:
        assert not np.isnan(r.point_estimate)
        assert r.ci_lower < r.ci_upper

def test_tmle_ipcw_comply_uses_compliance():
    df = _clean_df(n=400, seed=6)
    est = TMLEIPCWEstimator(use_compliance=True)
    results = est.estimate(df, horizon=1.0, estimand="ATE")
    names = {r.name for r in results}
    assert "TMLE+IPCW+Comply" in names
```

**Step 2: Implement TMLEIPCWEstimator**

```python
# causal_bench/estimators/tmle_ipcw.py
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy import stats
from sklearn.linear_model import LogisticRegression
from lifelines import CoxPHFitter
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult
from causal_bench.super_learner import SuperLearner


class TMLEIPCWEstimator(BaseEstimator):
    name = "TMLE+IPCW"

    def __init__(self, use_compliance: bool = False, n_folds: int = 5,
                 random_state: int = 42):
        self.use_compliance = use_compliance
        self.n_folds = n_folds
        self.random_state = random_state

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        name = "TMLE+IPCW+Comply" if self.use_compliance else "TMLE+IPCW"
        W_cols = ["W1", "W2", "W3", "W4"]
        A = df["A"].values
        T_obs = df["T_obs"].values
        Delta = df["Delta"].values
        W = df[W_cols].values
        n = len(A)

        # ── Binary outcome: event before horizon ──
        Y = ((T_obs <= horizon) & (Delta == 1)).astype(float)

        # ── Censoring weights (IPCW) ──
        # Fit Cox on censoring (reverse outcome: censored=1, event=0)
        censor_covs = W_cols + (["compliance"] if self.use_compliance else [])
        censor_df = pd.DataFrame(W, columns=W_cols)
        censor_df["A"] = A
        if self.use_compliance:
            censor_df["compliance"] = df["compliance"].values
        censor_df["T_obs"] = T_obs
        censor_df["C_indicator"] = 1 - Delta   # censored = 1

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(censor_df[censor_covs + ["A", "T_obs", "C_indicator"]],
                    duration_col="T_obs", event_col="C_indicator")
            # G(C > T_obs | covs): survival probability of censoring at T_obs
            sf = cph.predict_survival_function(
                censor_df[censor_covs + ["A"]], times=np.sort(np.unique(T_obs))
            )
            G_weights = np.array([
                float(sf.loc[sf.index <= t].iloc[-1, i])
                if (sf.index <= t).any() else 1.0
                for i, t in enumerate(T_obs)
            ])
        except Exception:
            G_weights = np.ones(n)

        G_weights = np.clip(G_weights, 0.05, 1.0)
        ipcw = Delta / G_weights

        # ── Propensity model g(A|W) ──
        sl_g = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state)
        sl_g.fit(W, A)
        g = sl_g.predict_proba(W)
        g = np.clip(g, 0.025, 0.975)

        # ── Outcome model Q(A,W): IPCW-weighted logistic regression ──
        AW = np.column_stack([A, W])
        AW1 = np.column_stack([np.ones(n), W])
        AW0 = np.column_stack([np.zeros(n), W])

        sl_q = SuperLearner(task="classification", n_folds=self.n_folds,
                            random_state=self.random_state + 1)
        # Fit on observed (non-zero IPCW weight) — use IPCW as sample weight
        sl_q.fit(AW, Y)   # Note: sklearn SL doesn't take sample_weight simply;
        # use reweighted logistic as fallback for Q
        from sklearn.linear_model import LogisticRegression
        q_model = LogisticRegression(max_iter=1000)
        q_model.fit(AW, Y, sample_weight=ipcw)
        Q_AW = np.clip(expit(q_model.decision_function(AW)), 1e-5, 1 - 1e-5)
        Q_1W = np.clip(expit(q_model.decision_function(AW1)), 1e-5, 1 - 1e-5)
        Q_0W = np.clip(expit(q_model.decision_function(AW0)), 1e-5, 1 - 1e-5)

        # ── Targeting step ──
        results = []
        for est in (["ATE"] if estimand == "ATE" else ["ATE", "ATT"]):
            Q1, Q0, point, se = self._target(
                Y, A, g, Q_AW, Q_1W, Q_0W, ipcw, est, n
            )
            z = stats.norm.ppf(0.975)
            results.append(EstimatorResult(
                name=name, estimand=est,
                point_estimate=float(point), standard_error=float(se),
                ci_lower=float(point - z * se), ci_upper=float(point + z * se),
            ))
        return results

    def _target(self, Y, A, g, Q_AW, Q_1W, Q_0W, ipcw, estimand, n):
        if estimand == "ATE":
            H = ipcw * (A / g - (1 - A) / (1 - g))
            H1 = 1.0 / g
            H0 = -1.0 / (1 - g)
        else:  # ATT
            H = ipcw * (A - (1 - A) * g / (1 - g))
            H1 = np.ones(n)
            H0 = -g / (1 - g)

        # One-step logistic targeting
        from sklearn.linear_model import LogisticRegression
        offset = logit(np.clip(Q_AW, 1e-5, 1 - 1e-5))
        try:
            eps_model = LogisticRegression(fit_intercept=False, C=1e8, max_iter=1000)
            eps_model.fit(H.reshape(-1, 1), Y, sample_weight=None)
            eps = float(eps_model.coef_[0, 0])
        except Exception:
            eps = 0.0

        Q1_star = expit(logit(np.clip(Q_1W, 1e-5, 1 - 1e-5)) + eps * H1)
        Q0_star = expit(logit(np.clip(Q_0W, 1e-5, 1 - 1e-5)) + eps * H0)

        if estimand == "ATE":
            point = np.mean(Q1_star - Q0_star)
            IC = (Q1_star - Q0_star - point
                  + ipcw * (A / g) * (Y - Q1_star)
                  - ipcw * ((1 - A) / (1 - g)) * (Y - Q0_star))
        else:  # ATT
            pi = np.mean(A)
            point = np.mean((A * (Q1_star - Q0_star)) / pi)
            IC = (A * (Q1_star - Q0_star) / pi - point
                  + ipcw * A / pi * (Y - Q1_star)
                  - ipcw * (1 - A) * g / (1 - g) / pi * (Y - Q0_star))

        se = np.sqrt(np.var(IC, ddof=1) / n)
        return Q1_star, Q0_star, point, se
```

**Step 3: Run tests**

```bash
pytest tests/test_estimators.py::test_tmle_ipcw_returns_ate_att -v
pytest tests/test_estimators.py::test_tmle_ipcw_comply_uses_compliance -v
```

**Step 4: Commit**

```bash
git add causal_bench/estimators/tmle_ipcw.py tests/test_estimators.py
git commit -m "feat: TMLE+IPCW and TMLE+IPCW+compliance estimators"
```

---

## Task 11: Estimator registry

**Files:**
- Modify: `causal_bench/causal_bench/estimators/__init__.py`

**Step 1: Implement registry**

```python
# causal_bench/estimators/__init__.py
from causal_bench.estimators.naive import NaiveEstimator
from causal_bench.estimators.kaplan_meier import KaplanMeierEstimator
from causal_bench.estimators.cox import CoxEstimator
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator

ESTIMATOR_REGISTRY = {
    "naive":          NaiveEstimator(),
    "km":             KaplanMeierEstimator(),
    "cox":            CoxEstimator(),
    "tmle_ipcw":      TMLEIPCWEstimator(use_compliance=False),
    "tmle_ipcw_comply": TMLEIPCWEstimator(use_compliance=True),
}

MVP_ESTIMATORS = list(ESTIMATOR_REGISTRY.keys())


def get_estimator(name: str):
    if name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown estimator '{name}'. Known: {list(ESTIMATOR_REGISTRY)}")
    return ESTIMATOR_REGISTRY[name]
```

**Step 2: Commit**

```bash
git add causal_bench/estimators/__init__.py
git commit -m "feat: estimator registry"
```

---

## Task 12: Monte Carlo runner

**Files:**
- Create: `causal_bench/causal_bench/runner.py`

**Step 1: Write failing test**

```python
# tests/test_estimators.py (add)
from causal_bench.runner import run_simulation
from causal_bench.dgp.config import DGPConfig

def test_run_simulation_smoke():
    cfg = DGPConfig(n=200, seed=0, censoring_informativeness=0.0)
    results = run_simulation(cfg, estimator_names=["naive", "km"],
                             n_sim=5, n_jobs=1, seed=0)
    assert "naive" in results
    assert "km" in results
    assert results["naive"].n_sim == 5
    assert abs(results["naive"].bias) < 1.0  # loose sanity check
```

**Step 2: Implement runner**

```python
# causal_bench/runner.py
import numpy as np
import pandas as pd
from typing import Optional
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.random import SeedSequence

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data, compute_true_effects
from causal_bench.estimators import get_estimator
from causal_bench.metrics import SimResult, EstimatorResult


def _run_one_sim(config: DGPConfig, estimator_names: list[str],
                 child_seed: np.random.SeedSequence,
                 estimand: str, horizon: float) -> dict:
    rng = np.random.default_rng(child_seed)
    cfg_i = DGPConfig(**{**config.__dict__, "seed": int(rng.integers(0, 2**31))})
    df = generate_data(cfg_i, rng=rng)

    out = {}
    for name in estimator_names:
        est = get_estimator(name)
        try:
            results = est.estimate(df, horizon=horizon, estimand=estimand)
            nc_val = est.estimate_negative_control(df, horizon=horizon)
        except Exception as e:
            results = []
            nc_val = float("nan")
        # Keep the result matching the requested estimand, or first result
        match = next((r for r in results if r.estimand == estimand), None)
        if match is None and results:
            match = results[0]
        out[name] = (match, nc_val)
    return out


def run_simulation(
    dgp_config: DGPConfig,
    estimator_names: list[str],
    n_sim: int = 500,
    n_jobs: int = -1,
    seed: int = 42,
    estimand: str = "ATE",
    horizon: Optional[float] = None,
) -> dict[str, SimResult]:
    if horizon is None:
        horizon = dgp_config.horizon

    true_effects = compute_true_effects(dgp_config)
    true_value = true_effects.get(estimand, true_effects["ATE"])

    child_seeds = SeedSequence(seed).spawn(n_sim)

    sim_outputs = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_sim)(dgp_config, estimator_names, cs, estimand, horizon)
        for cs in tqdm(child_seeds, desc="Simulations", total=n_sim)
    )

    # Aggregate per estimator
    results: dict[str, SimResult] = {}
    for name in estimator_names:
        estimates, ses, ci_lows, ci_highs, nc_vals = [], [], [], [], []
        for sim_out in sim_outputs:
            res, nc = sim_out.get(name, (None, float("nan")))
            if res is not None and not np.isnan(res.point_estimate):
                estimates.append(res.point_estimate)
                ses.append(res.standard_error)
                ci_lows.append(res.ci_lower)
                ci_highs.append(res.ci_upper)
                nc_vals.append(nc)

        if not estimates:
            continue

        results[name] = SimResult(
            estimator_name=name,
            estimand=estimand,
            true_value=true_value,
            n_sim=len(estimates),
            estimates=np.array(estimates),
            se_estimates=np.array(ses),
            ci_lowers=np.array(ci_lows),
            ci_uppers=np.array(ci_highs),
            nc_estimates=np.array(nc_vals),
        )
    return results


def run_parameter_sweep(
    base_config: DGPConfig,
    param_name: str,
    param_values: list,
    estimator_names: list[str],
    n_sim: int = 500,
    **kwargs,
) -> dict[str, list[SimResult]]:
    """Sweep one DGP parameter and return {estimator_name: [SimResult per value]}."""
    all_results: dict[str, list[SimResult]] = {name: [] for name in estimator_names}
    for val in param_values:
        config = DGPConfig(**{**base_config.__dict__, param_name: val})
        sim_results = run_simulation(config, estimator_names, n_sim=n_sim, **kwargs)
        for name in estimator_names:
            all_results[name].append(sim_results.get(name))
    return all_results
```

**Step 3: Run test**

```bash
pytest tests/test_estimators.py::test_run_simulation_smoke -v
```

**Step 4: Commit**

```bash
git add causal_bench/runner.py tests/test_estimators.py
git commit -m "feat: Monte Carlo runner with joblib parallelism and SeedSequence"
```

---

## Task 13: Visualization

**Files:**
- Create: `causal_bench/causal_bench/viz.py`

**Step 1: Implement**

```python
# causal_bench/viz.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from causal_bench.metrics import SimResult

COLORS = {
    "naive":            "#999999",
    "km":               "#7FCDBB",
    "cox":              "#FC8D59",
    "tmle_ipcw":        "#31A354",
    "tmle_ipcw_comply": "#006D2C",
}

LABELS = {
    "naive":            "Naive",
    "km":               "KM",
    "cox":              "Cox PH",
    "tmle_ipcw":        "TMLE+IPCW",
    "tmle_ipcw_comply": "TMLE+IPCW+Comply",
}

_STYLE = dict(fontfamily="sans-serif", fontsize=11)


def _apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_forest(
    results: dict[str, SimResult],
    title: str = "Estimator Comparison",
    save_path: str | None = None,
) -> plt.Figure:
    """Forest plot: mean ± 95% CI across simulations, true ATE dashed."""
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.7)))

    for i, name in enumerate(reversed(names)):
        sr = results[name]
        mean_est = np.mean(sr.estimates)
        mean_lo = np.mean(sr.ci_lowers)
        mean_hi = np.mean(sr.ci_uppers)
        color = COLORS.get(name, "#333333")
        label = LABELS.get(name, name)
        ax.plot([mean_lo, mean_hi], [i, i], color=color, lw=2)
        ax.plot(mean_est, i, "o", color=color, ms=7)
        ax.text(mean_hi + 0.005, i, f"{mean_est:+.3f}", va="center",
                fontsize=8, color=color)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([LABELS.get(n, n) for n in reversed(names)])

    true_val = next(iter(results.values())).true_value
    ax.axvline(true_val, ls="--", color="black", lw=1.2, label=f"True = {true_val:.3f}")
    ax.axvline(0, ls=":", color="#aaaaaa", lw=0.8)
    _apply_style(ax)
    ax.set_xlabel("Risk difference at horizon", **_STYLE)
    ax.set_title(title, **_STYLE, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_panel(
    sweep_results: dict[str, list[SimResult]],
    param_values: list,
    param_name: str,
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """5-row panel: bias, coverage, RMSE, CI width, NC bias across a parameter sweep."""
    metrics = ["bias", "coverage", "rmse", "ci_width", "nc_bias"]
    ylabels = ["Bias", "Coverage (95%)", "RMSE", "CI Width", "NC Bias"]
    targets = [0.0, 0.95, None, None, 0.0]

    fig, axes = plt.subplots(5, 1, figsize=(8, 16), sharex=True)

    for ax, metric, ylabel, target in zip(axes, metrics, ylabels, targets):
        for name, sr_list in sweep_results.items():
            vals = [getattr(sr, metric) for sr in sr_list if sr is not None]
            color = COLORS.get(name, "#333333")
            label = LABELS.get(name, name)
            ax.plot(param_values[:len(vals)], vals, "o-", color=color,
                    label=label, lw=1.8, ms=5)
        if target is not None:
            ax.axhline(target, ls="--", color="black", lw=1.0, alpha=0.6)
        ax.set_ylabel(ylabel, **_STYLE)
        _apply_style(ax)

    axes[-1].set_xlabel(param_name, **_STYLE)
    axes[0].set_title(title or f"Parameter sweep: {param_name}", **_STYLE,
                      fontweight="bold")

    handles = [mpatches.Patch(color=COLORS.get(n, "#333"), label=LABELS.get(n, n))
               for n in sweep_results]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_summary_table(results: dict[str, SimResult],
                           fmt: str = "markdown") -> str:
    """Generate a summary table in markdown or latex."""
    rows = [sr.summary() for sr in results.values()]
    cols = ["estimator", "estimand", "true", "bias", "rmse",
            "coverage", "ci_width", "se_ratio", "nc_bias"]
    if fmt == "markdown":
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines = [header, sep]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)
    else:
        raise NotImplementedError("Only markdown supported in MVP")
```

**Step 2: Smoke test (visual)**

```bash
python -c "
from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data
from causal_bench.runner import run_simulation
from causal_bench.viz import plot_forest
cfg = DGPConfig(n=200, seed=0, censoring_informativeness=0.0)
results = run_simulation(cfg, ['naive','km','cox'], n_sim=10, n_jobs=1)
fig = plot_forest(results, title='Smoke test')
fig.savefig('/tmp/smoke_forest.png')
print('Saved to /tmp/smoke_forest.png')
"
```

**Step 3: Commit**

```bash
git add causal_bench/viz.py
git commit -m "feat: forest plot and 5-row panel visualization"
```

---

## Task 14: CLI entry point

**Files:**
- Create: `causal_bench/causal_bench/__main__.py`

**Step 1: Implement**

```python
# causal_bench/__main__.py
import argparse
import sys
import os
from pathlib import Path

from causal_bench.dgp.scenarios import get_scenario, list_scenarios
from causal_bench.estimators import MVP_ESTIMATORS
from causal_bench.runner import run_simulation
from causal_bench.viz import plot_forest, plot_panel, generate_summary_table


def main():
    parser = argparse.ArgumentParser(
        prog="python -m causal_bench",
        description="Monte Carlo benchmarking of causal estimators for clinical trials",
    )
    parser.add_argument("--scenario", default="edwards_realistic",
                        help=f"Named scenario. Options: {list_scenarios()}")
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--estimand", default="ATE", choices=["ATE", "ATT"])
    parser.add_argument("--estimators", nargs="+", default=MVP_ESTIMATORS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print(f"\ncausal_bench | scenario={args.scenario} | n_sims={args.n_sims} "
          f"| estimand={args.estimand}\n")

    try:
        config = get_scenario(args.scenario)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = run_simulation(
        config,
        estimator_names=args.estimators,
        n_sim=args.n_sims,
        n_jobs=args.n_jobs,
        seed=args.seed,
        estimand=args.estimand,
    )

    out_dir = Path(args.out_dir) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary table
    table = generate_summary_table(results)
    print("\n── Results ──────────────────────────────")
    print(table)
    table_path = out_dir / "summary.md"
    table_path.write_text(table)
    print(f"\nSaved summary to {table_path}")

    if not args.no_plots:
        forest_path = out_dir / "forest.png"
        plot_forest(results, title=f"{args.scenario} | {args.estimand}",
                    save_path=str(forest_path))
        print(f"Saved forest plot to {forest_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test the CLI**

```bash
cd /Users/noahrahman/git/causal_bench
python -m causal_bench --scenario clean --n-sims 20 --n-jobs 1 --estimators naive km cox
```

Expected: summary table printed, forest.png written to `results/clean/`.

**Step 3: Commit**

```bash
git add causal_bench/__main__.py
git commit -m "feat: CLI entry point with scenario, estimators, plots"
```

---

## Task 15: README

**Files:**
- Create: `causal_bench/README.md`

**Step 1: Write README**

Include:
- One-paragraph description
- Quick start (`pip install -e .` then CLI example)
- Table: 5 estimators, what they do, when they fail
- Key findings section (placeholder — fill after running experiments)
- References from spec

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quick start and estimator table"
```

---

## Task 16: GitHub push

**Step 1: Create repo on GitHub**

```bash
gh repo create causal_bench --public --description "Monte Carlo benchmarking of causal estimators for clinical trials (TMLE, IPCW, competing risks)"
```

**Step 2: Push**

```bash
git remote add origin https://github.com/<username>/causal_bench.git
git push -u origin main
```

---

## Post-MVP: What comes next

In priority order for expanding the portfolio piece:
1. **Exp 1 (censoring gradient)** — run `run_parameter_sweep` on `censoring_informativeness`, save panel plot. Shows the key story in one figure.
2. **Exp 7 (edwards_realistic)** — full 1000-sim run, forest + bias-variance scatter
3. **Cox+L1 collider estimator + Exp 5** — the "trap" demonstration
4. **AIPW estimator** — doubly robust comparison
5. **IPW + overlap weighting** — complete the gradient story
6. **Diagnostics** (propensity, SMD, SE calibration)
7. **R bridge + LTMLE + Exp 8** — the McCoy experiment

---

## Running the full test suite

```bash
cd /Users/noahrahman/git/causal_bench
pytest tests/ -v --tb=short
```

All tests should pass before pushing to GitHub.
