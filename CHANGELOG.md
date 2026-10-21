# Changelog

## Unreleased (2026-10-21 estimator audit)

### Fixes
- `AIPWEstimator`: the point estimate used a SuperLearner Q in the plug-in term but a separate RidgeCV Q in the residual
  terms, so it was not doubly robust (bias E[(Q_SL − Q_ridge)₁ − (Q_SL − Q_ridge)₀] even with a correct propensity).
  It now uses ONE cross-fitted SuperLearner Q in both terms, with the propensity truncated to [0.01, 0.99].
  - Still no censoring model by default (documented contrast to TMLE+IPCW in exp8/exp13). New opt-in `ipcw=True`.
  - Known-truth check (n=3000, 3 seeds): no-censoring bias −0.0001 → −0.0000 (small on this DGP);
    `ipcw=True` at 40% censoring gives −0.0019 vs −0.017 without.
- `PointwiseRMSTEstimator` (`rmst_k2…k20`): it counted patients censored before t_k as event-free (no censoring
  weights, although index.qmd lists its nuisances as Q + g + G). That censoring bias does not shrink with K, so it
  contaminated the O(1/K) discretization demonstration. The pointwise risk differences are now IPCW-AIPW with a Cox
  censoring model (competing events treated as censoring, the repo convention).
  - Known-truth check, K=20 at 40% censoring: bias +0.0127 → +0.0035, equal to the no-censoring discretization
    residual +0.0033.
  - The docstring said "trapezoid"; the code is a right-endpoint Riemann sum.
- Shared helper `estimators/_ipcw_rd.py`; tests `tests/test_ipcw_rd.py`.
- `TMLEIPCWEstimator` (MVP) and `LTMLEEstimator`: patients event-free at the horizon got IPCW weight 1 ("Y=0 is
  known"), but they are observed only because they were not censored, so they need 1/G like events. Weight 1
  under-represented them.
  - Known truth, n=3000, 6 seeds: the censoring-induced shift in bias at 40% censoring goes from −0.016 to −0.003,
    with the absolute bias moving −0.0080 → +0.0047. The 0% case is unchanged.
  - The reported SE grows with censoring as it should (0.0154 → 0.0225 at 40%).
  - `test_default_path_numerically_unchanged` re-pinned (intentional).
- `LTMLEEstimator` rewritten as a two-time-point ICE-TMLE, which is what its docstring always described:
  - stage 2 regresses the stage-1 pseudo-outcomes Q2*(A,W,L1) (the old code fit them, then regressed the observed
    Y, so L1 never entered);
  - logistic fluctuations with IPCW-weighted clever covariates (the old one-step mixed probability and logit
    scales and dropped the IPCW factor);
  - an influence function built from the same Q* as the point estimate.
  - CAVEAT: in `dgp.survival` L1 is drawn after T and C and affects neither (L1 = f(A, W3, U)), so no DGP here
    exercises time-varying confounding, and LTMLE ≈ TMLE+IPCW on it (validated: both +0.005 at 40% censoring).
    README's "corrects for time-varying confounders" is untested until such a DGP exists.
- NOTED, NOT FIXED: on this DGP at n=3000 the EIF SEs of TMLE+IPCW and LTMLE are ~2× the empirical SD across seeds
  (old and new alike; 6 seeds, so rough). This is conservative, not anti-conservative, but se_ratio should be
  rechecked with exp13's harness.
- **Affected results to regenerate:** every experiment using `tmle_ipcw` (MVP), `ltmle`, `aipw` or `rmst_k*`. That
  includes exp2, exp3, exp7, exp8, exp13, exp16 and exp37 at least; grep `experiments/` for `tmle_ipcw`.

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
