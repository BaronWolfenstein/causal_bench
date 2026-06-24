# Censoring Discriminated Union Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three flat censoring fields on `DGPConfig` (`censoring_mechanism`, `censoring_informativeness`, `censoring_beta_T`) with a typed discriminated union (`IndependentCensoringConfig | CovariateDependentCensoringConfig | InformativeCensoringConfig`) so variant-specific parameters are structurally scoped, per-variant field leakage is impossible, and `survival.py` dispatch is exhaustive.

**Architecture:** Add three frozen Pydantic variant models + a `CensoringConfig` type alias to `config.py`, swap the three flat fields on `DGPConfig` for a single `censoring: CensoringConfig` field, update dispatch in `survival.py` to `isinstance`, and migrate all construction call sites (scenarios, experiments, tests) from flat kwargs to typed objects. The `@lru_cache` in `_calibrate_censoring_scale` continues to work because frozen Pydantic models are hashable.

**Tech Stack:** Python 3.10+, Pydantic v2, pytest.

## Global Constraints

- No behavior change to any censoring mechanism — only structural refactor.
- Do NOT rewrite any censoring DGP math.
- `extra="forbid"` and `frozen=True` on all three variant models.
- `frozen=True` on the variants is load-bearing: it makes them hashable for `@lru_cache`.
- Do not add `mode="before"` coercion. Callers pass typed objects.
- Reopen GitHub issue #12 before starting. Close it after the final push + green tests.

---

## File Map

| File | Change |
|------|--------|
| `causal_bench/dgp/config.py` | Add 3 variant classes + `CensoringConfig` alias; remove 3 flat fields; remove `censoring_beta_T` validator check |
| `causal_bench/dgp/survival.py` | Update `_calibrate_censoring_scale` signature + dispatch; update `generate_data` dispatch |
| `causal_bench/dgp/scenarios.py` | Replace flat `censoring_informativeness` entries with `CensoringConfig` objects |
| `tests/test_dgp.py` | Update construction sites + assertions to use new API |
| `tests/test_estimators.py` | Drop redundant `censoring_informativeness=0.0` kwargs (3 sites) |
| `experiments/exp2_positivity.py` | Drop `censoring_informativeness=0.0` (covered by default) |
| `experiments/exp3_unmeasured.py` | Drop `censoring_informativeness=0.0` (covered by default) |
| `experiments/exp4_crossover.py` | `censoring=CovariateDependentCensoringConfig(informativeness=0.3)` |
| `experiments/exp5_collider.py` | `censoring=CovariateDependentCensoringConfig(informativeness=0.3)` |
| `experiments/exp6_drift.py` | Drop `censoring_informativeness=0.0` (covered by default) |
| `experiments/exp13_censoring_sweep.py` | Typed GRID entries + 2 read-site fixes (`cfg.censoring_mechanism` → `cfg.censoring.kind`) |
| `experiments/exp15_sequential_monitoring.py` | `censoring=CovariateDependentCensoringConfig(informativeness=0.6)` in `_EDWARDS_BASE` |
| `experiments/diagnose_se_coverage.py` | Drop `censoring_informativeness=0.0` (2 sites, covered by default) |

---

### Task 1: Reopen issue #12

**Files:** none

- [ ] **Step 1: Reopen the issue**

```bash
gh issue reopen 12 --repo BaronWolfenstein/causal_bench --comment "Reopening to track discriminated union follow-on per spec docs/superpowers/specs/2026-06-24-censoring-discriminated-union-design.md"
```

Expected: issue #12 state changes to open.

---

### Task 2: Add variant types to `config.py` + update `DGPConfig`

**Files:**
- Modify: `causal_bench/dgp/config.py`
- Test: `tests/test_dgp.py`

**Interfaces:**
- Produces: `IndependentCensoringConfig`, `CovariateDependentCensoringConfig`, `InformativeCensoringConfig`, `CensoringConfig` — all importable from `causal_bench.dgp.config`
- `DGPConfig.censoring: CensoringConfig` (replaces `censoring_mechanism`, `censoring_informativeness`, `censoring_beta_T`)
- Default: `DGPConfig().censoring` is `CovariateDependentCensoringConfig(informativeness=0.0)`

- [ ] **Step 1: Write failing tests**

In `tests/test_dgp.py`, replace the three existing tests that reference flat censoring fields:

```python
# Replace test_dgp_config_defaults
def test_dgp_config_defaults():
    from causal_bench.dgp.config import CovariateDependentCensoringConfig
    cfg = DGPConfig()
    assert cfg.n == 500
    assert cfg.true_tau == -0.5
    assert isinstance(cfg.censoring, CovariateDependentCensoringConfig)
    assert cfg.censoring.informativeness == 0.0
    assert cfg.seed == 42


# Replace test_dgp_config_override
def test_dgp_config_override():
    from causal_bench.dgp.config import CovariateDependentCensoringConfig
    cfg = DGPConfig(n=200, true_tau=-0.3, censoring=CovariateDependentCensoringConfig(informativeness=0.6))
    assert cfg.n == 200
    assert cfg.true_tau == -0.3
    assert cfg.censoring.informativeness == 0.6
```

Also add two new tests after `test_dgp_config_override`:

```python
def test_dgp_config_informative_censoring():
    from causal_bench.dgp.config import InformativeCensoringConfig
    cfg = DGPConfig(censoring=InformativeCensoringConfig(beta_T=-0.8))
    assert cfg.censoring.beta_T == -0.8


def test_dgp_config_independent_censoring():
    from causal_bench.dgp.config import IndependentCensoringConfig
    cfg = DGPConfig(censoring=IndependentCensoringConfig())
    assert cfg.censoring.kind == "independent"
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/test_dgp.py::test_dgp_config_defaults tests/test_dgp.py::test_dgp_config_override tests/test_dgp.py::test_dgp_config_informative_censoring tests/test_dgp.py::test_dgp_config_independent_censoring -v 2>&1 | tail -15
```

Expected: 4 FAILs.

- [ ] **Step 3: Implement — update `config.py`**

Replace the import line at the top of `causal_bench/dgp/config.py`:

```python
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator
```

Add the following variant classes and alias **before** the `DGPConfig` class (after the `_STRATA_ELIGIBLE_COLS` line):

```python
class IndependentCensoringConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["independent"] = "independent"


class CovariateDependentCensoringConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["covariate_dependent"] = "covariate_dependent"
    informativeness: float = Field(0.0, ge=0.0, le=1.0)


class InformativeCensoringConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["informative"] = "informative"
    beta_T: float = 0.0


CensoringConfig = Annotated[
    Union[IndependentCensoringConfig, CovariateDependentCensoringConfig, InformativeCensoringConfig],
    Field(discriminator="kind"),
]
```

In `DGPConfig`, replace the three flat censoring fields (the `censoring_mechanism`, `censoring_informativeness`, `censoring_beta_T` lines) with:

```python
    censoring_rate: float = Field(0.25, ge=0.0, lt=1.0)
    censoring: CensoringConfig = Field(default_factory=CovariateDependentCensoringConfig)
```

Remove the `censoring_beta_T` cross-field check from `_check_couplings` (the block starting with `# censoring_beta_T only has any effect when censoring_mechanism ==` through the closing `raise ValueError`).

- [ ] **Step 4: Run the new tests to verify they pass**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/test_dgp.py::test_dgp_config_defaults tests/test_dgp.py::test_dgp_config_override tests/test_dgp.py::test_dgp_config_informative_censoring tests/test_dgp.py::test_dgp_config_independent_censoring -v 2>&1 | tail -10
```

Expected: 4 PASSes.

- [ ] **Step 5: Check which other tests now fail (survival + scenario call sites not yet updated)**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/test_dgp.py -q 2>&1 | tail -10
```

Expected: failures only in `test_generate_data_*` and `test_get_scenario_*`. Config-layer tests all pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/noahrahman/git/causal_bench && git add causal_bench/dgp/config.py tests/test_dgp.py && git commit -m "$(cat <<'EOF'
refactor: add CensoringConfig discriminated union variants to config.py

Add IndependentCensoringConfig, CovariateDependentCensoringConfig,
InformativeCensoringConfig, and CensoringConfig alias. Replace three flat
censoring fields on DGPConfig with censoring: CensoringConfig. Remove the
now-redundant censoring_beta_T cross-field validator.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Update `survival.py` dispatch

**Files:**
- Modify: `causal_bench/dgp/survival.py`

**Interfaces:**
- Consumes: `IndependentCensoringConfig`, `CovariateDependentCensoringConfig`, `InformativeCensoringConfig` from `causal_bench.dgp.config`
- `_calibrate_censoring_scale(censoring_rate, horizon, censoring)` — `censoring` is a `CensoringConfig` object

- [ ] **Step 1: Update imports in `survival.py`**

Replace the existing `from causal_bench.dgp.config import DGPConfig` line with:

```python
from causal_bench.dgp.config import (
    DGPConfig,
    CensoringConfig,
    IndependentCensoringConfig,
    CovariateDependentCensoringConfig,
    InformativeCensoringConfig,
)
```

- [ ] **Step 2: Replace `_calibrate_censoring_scale` (lines 22–66)**

```python
@lru_cache(maxsize=256)
def _calibrate_censoring_scale(
    censoring_rate: float,
    horizon: float,
    censoring: CensoringConfig,
) -> float:
    """Scale factor so achieved censoring_rate matches target under given mechanism."""
    if censoring_rate <= 0:
        return 1e10
    rng = np.random.default_rng(0)
    n = 5000
    U  = rng.standard_normal(n)
    W1 = rng.standard_normal(n)
    W3 = rng.standard_normal(n)
    A  = rng.binomial(1, 0.5, n).astype(float)
    log_T  = 0.0 + 0.4 * W1 + 0.3 * U + rng.gumbel(0, 1, n)
    T_true = np.exp(log_T)
    gumbel_c = rng.gumbel(0, 1, n)

    if isinstance(censoring, IndependentCensoringConfig):
        log_C_base = 1.5 + gumbel_c
    elif isinstance(censoring, InformativeCensoringConfig):
        log_C_base = 1.5 + censoring.beta_T * T_true + gumbel_c
    else:  # CovariateDependentCensoringConfig
        log_C_base = (1.5 - 0.2 * W1 + 0.1 * W3 - 0.1 * A
                      + 0.4 * U * censoring.informativeness
                      + gumbel_c)
        mnar_weight = max(0.0, censoring.informativeness - 0.5) * 2.0
        if mnar_weight > 0:
            log_C_base -= mnar_weight * (T_true < np.median(T_true)).astype(float)

    C_base = np.exp(np.clip(log_C_base, -700, 700))
    lo, hi = 0.01, 100.0
    for _ in range(40):
        mid = (lo + hi) / 2
        C = C_base * mid
        censor_rate = np.mean((C < T_true) & (C < horizon))
        if censor_rate > censoring_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
```

- [ ] **Step 3: Replace censoring block in `generate_data` (lines 205–237)**

```python
    # --- Censoring ---
    scale_factor = _calibrate_censoring_scale(
        config.censoring_rate, config.horizon, config.censoring
    )

    gumbel_c = rng.gumbel(0, 1, n)
    if isinstance(config.censoring, IndependentCensoringConfig):
        # Pure random dropout: C doesn't depend on covariates, treatment, or T_true.
        log_C_base = 1.5 + gumbel_c
    elif isinstance(config.censoring, InformativeCensoringConfig):
        # MNAR: censoring time directly depends on the (unobservable) event time.
        # IPCW conditional only on W, A cannot correct this — it requires T_true.
        log_C_base = 1.5 + config.censoring.beta_T * T_true + gumbel_c
    else:
        # CovariateDependentCensoringConfig — MAR conditional on W, A; optional MNAR-via-U component
        log_C_base = (
            1.5
            - 0.2 * W1
            + 0.1 * W3
            - 0.1 * A
            + 0.4 * U * config.censoring.informativeness
            + gumbel_c
        )
        # MNAR component: early events are more likely to be censored
        mnar_weight = max(0.0, config.censoring.informativeness - 0.5) * 2
        if mnar_weight > 0:
            median_T = np.median(T_true)
            log_C_base -= mnar_weight * (T_true < median_T).astype(float)

    C = np.exp(np.clip(log_C_base, -700, 700)) * scale_factor  # avoid 0/inf overflow at extreme beta_T * T_true
```

- [ ] **Step 4: Run `test_dgp.py` — survival tests now pass, scenarios still failing**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/test_dgp.py -q 2>&1 | tail -10
```

Expected: `test_generate_data_*` tests pass. Only scenario tests still failing.

- [ ] **Step 5: Commit**

```bash
cd /Users/noahrahman/git/causal_bench && git add causal_bench/dgp/survival.py && git commit -m "$(cat <<'EOF'
refactor: update survival.py dispatch to CensoringConfig isinstance checks

Replace flat-field access (config.censoring_mechanism, .censoring_informativeness,
.censoring_beta_T) with isinstance dispatch on config.censoring. @lru_cache
in _calibrate_censoring_scale continues to work because frozen Pydantic models
are hashable.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Migrate `scenarios.py` and update scenario tests

**Files:**
- Modify: `causal_bench/dgp/scenarios.py`
- Modify: `tests/test_dgp.py`

**Interfaces:**
- Consumes: `CovariateDependentCensoringConfig` from `causal_bench.dgp.config`
- All scenarios use `CovariateDependentCensoringConfig` (none use `informative` or `independent`)

- [ ] **Step 1: Write failing scenario tests**

In `tests/test_dgp.py`, replace the two assertions that check `cfg.censoring_informativeness`:

```python
# Replace test_get_scenario_clean
def test_get_scenario_clean():
    from causal_bench.dgp.config import CovariateDependentCensoringConfig
    cfg = get_scenario("clean")
    assert isinstance(cfg.censoring, CovariateDependentCensoringConfig)
    assert cfg.censoring.informativeness == 0.0
    assert cfg.positivity_severity == 0.0
    assert cfg.true_tau == -0.5


# Replace test_get_scenario_edwards_realistic
def test_get_scenario_edwards_realistic():
    from causal_bench.dgp.config import CovariateDependentCensoringConfig
    cfg = get_scenario("edwards_realistic")
    assert cfg.n == 700
    assert isinstance(cfg.censoring, CovariateDependentCensoringConfig)
    assert cfg.censoring.informativeness == 0.6
    assert cfg.positivity_severity == 1.5
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/test_dgp.py::test_get_scenario_clean tests/test_dgp.py::test_get_scenario_edwards_realistic -v 2>&1 | tail -10
```

Expected: 2 FAILs.

- [ ] **Step 3: Rewrite `scenarios.py`**

```python
from causal_bench.dgp.config import DGPConfig, CovariateDependentCensoringConfig

_CENS = CovariateDependentCensoringConfig  # local alias

_CLEAN = dict(
    n=500, censoring=_CENS(informativeness=0.0), censoring_rate=0.25,
    positivity_severity=0.0, unmeasured_confounding_strength=0.0,
    collider_strength=0.0, crossover_rate=0.0, enrollment_drift=0.0,
    true_tau=-0.5,
)

_REGISTRY: dict[str, dict] = {
    "clean": _CLEAN,
    # Censoring gradient
    "censor_mild":     {**_CLEAN, "censoring": _CENS(informativeness=0.3), "censoring_rate": 0.25},
    "censor_moderate": {**_CLEAN, "censoring": _CENS(informativeness=0.6), "censoring_rate": 0.30},
    "censor_severe":   {**_CLEAN, "censoring": _CENS(informativeness=1.0), "censoring_rate": 0.40},
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
        n=700,
        censoring=_CENS(informativeness=0.6), censoring_rate=0.25,
        positivity_severity=1.5, crossover_rate=0.05,
        unmeasured_confounding_strength=0.2,
        collider_strength=0.4, enrollment_drift=0.15,
        outcome_nonlinearity=0.5, effect_heterogeneity=0.3,
        true_tau=-0.5,
    ),
    "edwards_optimistic": dict(
        n=700,
        censoring=_CENS(informativeness=0.3), censoring_rate=0.15,
        positivity_severity=0.5, unmeasured_confounding_strength=0.1,
        collider_strength=0.2, enrollment_drift=0.05,
        true_tau=-0.5,
    ),
    "edwards_pessimistic": dict(
        n=700,
        censoring=_CENS(informativeness=0.9), censoring_rate=0.40,
        positivity_severity=2.5, crossover_rate=0.10,
        unmeasured_confounding_strength=0.4,
        collider_strength=0.7, enrollment_drift=0.3,
        outcome_nonlinearity=0.7, effect_heterogeneity=0.5,
        true_tau=-0.5,
    ),
    # Stratified block randomization — for Exp 11 / SE correction benchmark
    # W2 (Bern 0.5) × W4 (Bern 0.3) → 4 strata; block size 4.
    # Strata account for ~20% of outcome variance via their W2/W4 prognostic effects.
    "stratified_base": {
        **_CLEAN,
        "strata_cols": ("W2", "W4"),
        "strata_block_size": 4,
        "censoring": _CENS(informativeness=0.0),
        "censoring_rate": 0.20,
    },
    # Competing risks — for Exp 8 / McCoy experiment
    # cause-1 (primary event): treatment effect is true_tau, same as single-event case.
    # cause-2 (competing event): cause2_treatment_effect controls treatment's effect on
    # the competing cause's hazard.
    "competing_risks_base": {
        **_CLEAN,
        "n": 600, "competing_risks": True,
        "censoring": _CENS(informativeness=0.3), "censoring_rate": 0.20,
        "true_tau": -0.3,
    },
    # ENCIRCLE-calibrated — for Exp 16 / calibrated replication
    # Calibrated to published 1-year ENCIRCLE marginals (device arm, n=299):
    #   composite 25.2% (device) / ~45% (historical performance goal)
    #   mortality 13.9%, HF hospitalization 16.7%, ~5.4% overlap
    #   ~19% missing at 1-year visit
    # DGP validation (n=100k): device comp≈0.257, HFH≈0.166, death≈0.090
    #                           control comp≈0.465, ATE≈−0.144
    # NOTE: true_tau > 0 means device EXTENDS survival (fewer bad events).
    # The reverse sign convention from other scenarios where true_tau < 0 means
    # treatment causes the event sooner (those scenarios use a beneficial-event framing).
    "encircle_calibrated": {
        "n": 700,
        "treatment_prevalence": 0.43,       # ~299 treated in n=700
        "true_tau": 0.48,                   # device reduces HFH risk (extends T1)
        "competing_risks": True,
        "cause2_treatment_effect": 0.32,    # device also reduces mortality (extends T2)
        "hfh_death_escalation": 0.55,       # HFH-prone patients die sooner (shared frailty)
        "censoring_rate": 0.19,             # ~19% missing at 1-year visit
        "censoring": _CENS(informativeness=0.25),  # mild informative: sicker patients miss more
        "horizon": 0.77,
        "positivity_severity": 0.5,         # mild enrollment heterogeneity
        "unmeasured_confounding_strength": 0.1,
    },
}


def get_scenario(name: str) -> DGPConfig:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Known: {list(_REGISTRY)}")
    return DGPConfig(**_REGISTRY[name])


def list_scenarios() -> list[str]:
    return list(_REGISTRY.keys())
```

- [ ] **Step 4: Run full `test_dgp.py` — all tests should pass**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/test_dgp.py -q 2>&1 | tail -5
```

Expected: all pass, 0 failures.

- [ ] **Step 5: Commit**

```bash
cd /Users/noahrahman/git/causal_bench && git add causal_bench/dgp/scenarios.py tests/test_dgp.py && git commit -m "$(cat <<'EOF'
refactor: migrate scenarios.py to CensoringConfig objects + update tests

Replace flat censoring_informativeness entries in the scenario registry with
CovariateDependentCensoringConfig(informativeness=X). Update test_get_scenario_*
assertions to check cfg.censoring.informativeness.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Update `test_estimators.py`

**Files:**
- Modify: `tests/test_estimators.py`

- [ ] **Step 1: Drop `censoring_informativeness=0.0` kwargs at 3 sites**

Line ~85 — remove `censoring_informativeness=0.0,` from the `DGPConfig` call:
```python
# before
return generate_data(DGPConfig(n=n, censoring_informativeness=0.0,
# after
return generate_data(DGPConfig(n=n,
```

Line ~177 — remove `censoring_informativeness=0.0`:
```python
# before
cfg = _DGPConfig(n=150, seed=0, censoring_informativeness=0.0)
# after
cfg = _DGPConfig(n=150, seed=0)
```

Line ~334 — remove `censoring_informativeness=0.0`:
```python
# before
cfg = DGPConfig(n=500, censoring_informativeness=0.0, seed=1)
# after
cfg = DGPConfig(n=500, seed=1)
```

- [ ] **Step 2: Run `test_estimators.py`**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/test_estimators.py -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
cd /Users/noahrahman/git/causal_bench && git add tests/test_estimators.py && git commit -m "$(cat <<'EOF'
refactor: drop redundant censoring_informativeness=0.0 from test_estimators.py

Default CensoringConfig is CovariateDependentCensoringConfig(informativeness=0.0).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Update experiment files (exp2, exp3, exp4, exp5, exp6, exp15, diagnose)

**Files:**
- Modify: `experiments/exp2_positivity.py`
- Modify: `experiments/exp3_unmeasured.py`
- Modify: `experiments/exp4_crossover.py`
- Modify: `experiments/exp5_collider.py`
- Modify: `experiments/exp6_drift.py`
- Modify: `experiments/exp15_sequential_monitoring.py`
- Modify: `experiments/diagnose_se_coverage.py`

- [ ] **Step 1: Update imports in exp4, exp5, exp15**

These three files need `CovariateDependentCensoringConfig` imported. Add it to each file's existing `from causal_bench.dgp.config import DGPConfig` line:

```python
from causal_bench.dgp.config import DGPConfig, CovariateDependentCensoringConfig
```

- [ ] **Step 2: exp2_positivity.py — drop `censoring_informativeness=0.0`**

```python
# before  (line ~25)
base = DGPConfig(n=500, censoring_informativeness=0.0, positivity_severity=0.0,
# after
base = DGPConfig(n=500, positivity_severity=0.0,
```

- [ ] **Step 3: exp3_unmeasured.py — drop `censoring_informativeness=0.0`**

```python
# before  (line ~27)
base = DGPConfig(n=500, censoring_informativeness=0.0,
# after
base = DGPConfig(n=500,
```

- [ ] **Step 4: exp4_crossover.py — typed censoring**

```python
# before  (line ~31)
base = DGPConfig(n=500, censoring_informativeness=0.3, censoring_rate=0.25,
# after
base = DGPConfig(n=500, censoring=CovariateDependentCensoringConfig(informativeness=0.3), censoring_rate=0.25,
```

- [ ] **Step 5: exp5_collider.py — typed censoring**

```python
# before  (line ~21)
base = DGPConfig(n=500, censoring_informativeness=0.3, true_tau=-0.5,
# after
base = DGPConfig(n=500, censoring=CovariateDependentCensoringConfig(informativeness=0.3), true_tau=-0.5,
```

- [ ] **Step 6: exp6_drift.py — drop `censoring_informativeness=0.0`**

```python
# before  (line ~27)
base = DGPConfig(n=500, censoring_informativeness=0.0, enrollment_drift=0.0,
# after
base = DGPConfig(n=500, enrollment_drift=0.0,
```

- [ ] **Step 7: exp15_sequential_monitoring.py — typed censoring in `_EDWARDS_BASE`**

```python
# before  (line ~44-45)
_EDWARDS_BASE = dict(
    censoring_informativeness=0.6, censoring_rate=0.25,
# after
_EDWARDS_BASE = dict(
    censoring=CovariateDependentCensoringConfig(informativeness=0.6), censoring_rate=0.25,
```

- [ ] **Step 8: diagnose_se_coverage.py — drop `censoring_informativeness=0.0` (2 sites)**

Remove `censoring_informativeness=0.0,` from both `DGPConfig(n=500, censoring_informativeness=0.0, ...)` calls (lines ~184 and ~217).

- [ ] **Step 9: Verify all experiment files import cleanly**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -c "
import experiments.exp2_positivity
import experiments.exp3_unmeasured
import experiments.exp4_crossover
import experiments.exp5_collider
import experiments.exp6_drift
import experiments.exp15_sequential_monitoring
" 2>&1
```

Expected: no output.

- [ ] **Step 10: Commit**

```bash
cd /Users/noahrahman/git/causal_bench && git add experiments/exp2_positivity.py experiments/exp3_unmeasured.py experiments/exp4_crossover.py experiments/exp5_collider.py experiments/exp6_drift.py experiments/exp15_sequential_monitoring.py experiments/diagnose_se_coverage.py && git commit -m "$(cat <<'EOF'
refactor: update exp2–6, exp15, diagnose to use CensoringConfig objects

Replace censoring_informativeness flat kwargs with CovariateDependentCensoringConfig
objects. Drop redundant informativeness=0.0 args (covered by default).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update `exp13_censoring_sweep.py`

**Files:**
- Modify: `experiments/exp13_censoring_sweep.py`

- [ ] **Step 1: Update imports**

Replace `from causal_bench.dgp.config import DGPConfig` with:

```python
from causal_bench.dgp.config import (
    DGPConfig,
    IndependentCensoringConfig,
    CovariateDependentCensoringConfig,
    InformativeCensoringConfig,
)
```

- [ ] **Step 2: Replace GRID list (lines 59–69)**

```python
GRID = [
    {"label": "independent",           "censoring": IndependentCensoringConfig()},
    {"label": "covdep_info0.0",        "censoring": CovariateDependentCensoringConfig(informativeness=0.0)},
    {"label": "covdep_info0.3",        "censoring": CovariateDependentCensoringConfig(informativeness=0.3)},
    {"label": "covdep_info0.6",        "censoring": CovariateDependentCensoringConfig(informativeness=0.6)},
    {"label": "covdep_info0.9",        "censoring": CovariateDependentCensoringConfig(informativeness=0.9)},
    {"label": "informative_betaT-0.8", "censoring": InformativeCensoringConfig(beta_T=-0.8)},
    {"label": "informative_betaT-0.4", "censoring": InformativeCensoringConfig(beta_T=-0.4)},
    {"label": "informative_betaT+0.4", "censoring": InformativeCensoringConfig(beta_T=0.4)},
    {"label": "informative_betaT+0.8", "censoring": InformativeCensoringConfig(beta_T=0.8)},
]
```

- [ ] **Step 3: Fix read sites**

Line ~179:
```python
# before
print(f"\nExp 13: cell={label} | mechanism={cfg.censoring_mechanism} | n_sims={n_sims}", flush=True)
# after
print(f"\nExp 13: cell={label} | mechanism={cfg.censoring.kind} | n_sims={n_sims}", flush=True)
```

Line ~223:
```python
# before
row["mechanism"] = cfg.censoring_mechanism
# after
row["mechanism"] = cfg.censoring.kind
```

- [ ] **Step 4: Verify import**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -c "import experiments.exp13_censoring_sweep" 2>&1
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
cd /Users/noahrahman/git/causal_bench && git add experiments/exp13_censoring_sweep.py && git commit -m "$(cat <<'EOF'
refactor: migrate exp13 GRID to typed CensoringConfig objects

Replace flat censoring_mechanism/censoring_informativeness/censoring_beta_T
dict entries with typed config objects. Fix cfg.censoring_mechanism read
sites to cfg.censoring.kind.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Final verification + push + close issue #12

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/noahrahman/git/causal_bench && python3 -m pytest tests/ -q 2>&1 | tail -10
```

Expected: all tests pass, 0 failures.

- [ ] **Step 2: Verify no flat field references remain**

```bash
cd /Users/noahrahman/git/causal_bench && grep -rn "censoring_mechanism\|censoring_informativeness\|censoring_beta_T" causal_bench/ experiments/ tests/ --include="*.py" 2>&1
```

Expected: no output (zero remaining references).

- [ ] **Step 3: Push**

```bash
cd /Users/noahrahman/git/causal_bench && git push
```

- [ ] **Step 4: Close issue #12**

```bash
gh issue close 12 --repo BaronWolfenstein/causal_bench --comment "Discriminated union delivered. Three flat censoring fields replaced with IndependentCensoringConfig | CovariateDependentCensoringConfig | InformativeCensoringConfig across config.py, survival.py, scenarios.py, all experiments, and tests. All tests pass."
```
