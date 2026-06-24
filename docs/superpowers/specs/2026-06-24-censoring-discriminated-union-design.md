# Design: DGPConfig censoring_mechanism → discriminated union

**Date:** 2026-06-24
**Issue:** [#12](https://github.com/BaronWolfenstein/causal_bench/issues/12) (reopen before starting; close after commit + push + tests pass)
**Scope:** Pure structural refactor — no behavior changes, no new DGP logic.

## Context

Issue #12's pydantic refactor used `Literal["independent", "covariate_dependent", "informative"]` for `censoring_mechanism` as the smaller change, deferring the discriminated union as a follow-on. This spec delivers that follow-on.

The three flat censoring fields (`censoring_mechanism`, `censoring_informativeness`, `censoring_beta_T`) are per-variant parameters that belong together. The `Literal` approach requires a cross-field validator to catch `censoring_beta_T != 0` on the wrong mechanism. The discriminated union makes misconfiguration structurally impossible and gives `survival.py` exhaustive dispatch that the type checker can verify.

## Design

### 1. New types in `config.py`

Three frozen/forbid Pydantic models with a `kind` Literal discriminator, plus a type alias:

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
    IndependentCensoringConfig | CovariateDependentCensoringConfig | InformativeCensoringConfig,
    Field(discriminator="kind")
]
```

`frozen=True` on variants is load-bearing: it makes them hashable, which `_calibrate_censoring_scale`'s `@lru_cache` requires.

### 2. `DGPConfig` changes

- **Remove:** `censoring_mechanism`, `censoring_informativeness`, `censoring_beta_T`
- **Add:** `censoring: CensoringConfig = Field(default_factory=CovariateDependentCensoringConfig)`
- **Remove** the `censoring_beta_T` cross-field check from `_check_couplings` — structural enforcement replaces it (`extra="forbid"` on each variant prevents setting params on the wrong type)
- `with_overrides` is unchanged in signature; callers pass `censoring=InformativeCensoringConfig(beta_T=-0.8)` as a single argument

### 3. `survival.py` dispatch

`_calibrate_censoring_scale` signature: replace `censoring_informativeness`, `censoring_mechanism`, `censoring_beta_T` params with a single `censoring: CensoringConfig`. `@lru_cache` continues to work because frozen Pydantic models are hashable. Dispatch via `isinstance` or `match` — exhaustive.

Same change in the inline dispatch inside `generate_data`.

### 4. Call-site migration

**`scenarios.py`** — flat dict entries:
```python
# before
{"censoring_informativeness": 0.3, ...}
# after
{"censoring": CovariateDependentCensoringConfig(informativeness=0.3), ...}
```

**`exp13_censoring_sweep.py`** — cell list:
```python
# before
{"censoring_mechanism": "informative", "censoring_beta_T": -0.8}
# after
{"censoring": InformativeCensoringConfig(beta_T=-0.8)}
```
Read sites `cfg.censoring_mechanism` → `cfg.censoring.kind`.

**`exp12`, `exp14`** — use `DGPConfig.model_construct(**cfg_dict)` to reconstruct from serialized dicts. `model_construct` bypasses validation and would leave `censoring` as a raw dict. Switch to `DGPConfig.model_validate(cfg_dict)` so the discriminated union routes the nested `{"kind": ..., ...}` dict to the right variant.

**Tests** — construction sites update flat censoring kwargs to `censoring=` objects. No test logic changes.

## Non-goals

- No behavior change to any censoring mechanism
- No changes to `EstimatorResult`, `SimResult`, or the estimator layer
- No new DGP parameters
- `cause2_treatment_effect` and `hfh_death_escalation` cross-field validators are untouched

## Process

1. Reopen issue #12 before starting implementation
2. Implement, run tests, push
3. Close issue #12 once CI passes
