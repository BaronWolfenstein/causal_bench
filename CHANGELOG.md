# Changelog

## v0.2.1 (2026-06-19)

### New estimator
- `ClinicalPSNBEstimator` (`"clinical_PSNB"`) — rpy2 bridge to `concrete::clinicalPSNB()` (McCoy PR #34, now merged). Wraps the priority-standardized net benefit (PSNB) and win ratio (PSWR) with a configurable `charter` vector; returns two `EstimatorResult` objects per call. PSWR CIs use concrete's log-scale construction.

### Fixes
- R bridge `run_clinical_psnb()`: corrected extraction to match actual `clinicalPSNB()` output — a `data.table` keyed by `Estimand` column, not a named list. PSWR normal-theory CI fallback removed (log-scale CIs from concrete are already correct).
- `exp10_win_ratio.py`: corrected two stale "Exp 9" labels in print/warn strings.

### Docs
- README install comment updated to mention PR #33 (RMT-IF) and PR #34 (PSNB), both now merged.
- `ClinicalRMTIFEstimator` docstring: removed stale "(PR #33)" pending marker.

## v0.2.0 (2026-06-17)

First versioned release. Scope substantially exceeds the original MVP plan.

### DGP and scenarios
- `DGPConfig` (Pydantic `BaseModel`): `frozen=True`, `extra="forbid"`, `Field` bounds on all physically-constrained parameters, and cross-field coupling validators (even strata block size, valid strata column names, `censoring_beta_T`/mechanism pairing, `cause2_treatment_effect`/`competing_risks` pairing).
- 18 named scenarios across censoring, positivity, unmeasured confounding, Edwards trial variants, stratified randomization, and competing risks.
- Provenance-linked synthetic augmentation (`AugmentationConfig`, `generate_augmented_data`) with a controllable leakage knob and pandera provenance-integrity check on output.
- Competing-risks DGP with cause-specific first-event times and `event_type` encoding.

### Estimators
- **Python**: Naive, Kaplan-Meier, Cox, IPW, AIPW, Overlap weighting, Pointwise RMST, TMLE+IPCW (with bootstrap and CV variants), LTMLE.
- **R bridge** (requires `concrete` R package via `rpy2`): `ConcreteRMSTEstimator`, `ConcreteSimultaneousEstimator`, `ConcreteClinicalRMTIFEstimator`, `ConcreteWinRatioEstimator`.
- Super Learner ensemble with cross-fitting; `fold_mode="group"` respects provenance groups for augmented datasets.
- pandera schema at the `prepare_for_r` boundary validates `T_obs`, `event_type`, `A`, and `W1–W4` before crossing into R.

### Validation hardening (all five trust boundaries)
- `DGPConfig`: Field bounds + model validators (§1).
- R-bridge DataFrame: pandera `DataFrameSchema` inside `prepare_for_r` (§2).
- `EstimatorResult`: `se > 0` or NaN; `ci_lower ≤ point_estimate ≤ ci_upper` (§3).
- `ComparisonSpec`: estimand-coherence gate; `allow_known_mismatch=True` required for known mismatches (e.g. Exp 8 cause-specific CIF vs all-cause RD) (§4).
- `AugmentationConfig`: typed config replacing loose kwargs; pandera provenance check `synth_groups ⊆ real_groups` (§5).

### Experiments (14 total)
Exp 1–11 cover the core assumption-violation scenarios (censoring, positivity, unmeasured confounding, crossover, collider trap, enrollment drift, Edwards composite, McCoy RMST, sample size, win ratio, stratified randomization). Exp 12–14 add simultaneous confidence bands, censoring-mechanism sweep, and provenance-linked augmentation cross-fitting.

### Metrics and runner
- `SimResult`: bias, RMSE, coverage, CI width, SE ratio, negative-control bias; `to_parquet`/`from_parquet` persistence.
- `SimResultFamily`: joint pointwise and simultaneous coverage across an estimand family.
- `run_simulation`: per-replicate seed sequencing, `model_construct` bypass in workers, per-estimator `ComparisonSpec` annotation, error classification.

### Breaking changes from v0.1.0
- `DGPConfig` fields removed: `cause1_fraction`, `cause1_treatment_effect`, `compliance_available`, `n_treated_fraction` (all were accepted but never read by `generate_data()`).
- `DGPConfig.cause2_treatment_effect` default changed from `-0.6` to `0.0`.
- `generate_augmented_data` signature changed: loose `n_real`/`n_synth_per_real`/`leakage_strength` kwargs replaced by `AugmentationConfig`.
- `extra="forbid"` on `DGPConfig`: misspelled constructor kwargs now raise `ValidationError` instead of being silently ignored.
