# Exp33: Donsker-Class Nuisance Learners (LTB / HAR) — Design

**Date:** 2026-07-02
**Status:** Approved design, pre-implementation. Phase 1 implemented under this spec; Phase 2 specified here but implemented in a follow-on PR.
**Context:** HAL is the repo's only nuisance learner whose theory licenses IC-based SEs *without* cross-fitting (Donsker class + faster-than-n^{-1/4} rate), but at ~3 min/simulation via rpy2 (`hal.py`) it is excluded from the default SuperLearner library and unusable at Monte Carlo scale. Two recent alternatives claim HAL's dimension-free O_P(n^{-1/3}·(log n)^{2(p-1)/3}) rate at ML cost: **Lassoed Tree Boosting** (LTB; Schuler, Li & van der Laan, arXiv:2205.10697v6 — v1 was titled "The Selectively Adaptive Lasso") and **Highly Adaptive Ridge** (HAR; Schuler, arXiv:2410.02680). Neither paper ships code, and neither runs the causal downstream. This experiment implements both and measures whether they actually deliver AIPW/TMLE validity without cross-fitting at ENCIRCLE scale (n≈700).

## 1. What the theory claims and what we test

Cross-fitting exists solely to bypass the Donsker condition on nuisance estimators. If the fitted nuisances (a) lie in a fixed Donsker class and (b) converge faster than n^{-1/4} in L2, the empirical-process term (P_n − P)(D(f̂) − D(f₀)) is o_p(n^{-1/2}) on the full sample and AIPW/TMLE are asymptotically efficient without sample splitting.

- **LTB** delivers both ingredients in the càdlàg bounded-sectional-variation class; the paper proves the rate but defers the plug-in/efficiency argument ("future work"). The load-bearing caveat: the Donsker class is "sectional variation norm ≤ M" for *fixed* M, and data-adaptive tuning only bounds the fitted norm in probability — an asymptotic argument, not a finite-sample guarantee.
- **HAR** requires square-integrable sectional derivatives — strictly stronger than bounded sectional variation, and it excludes jump discontinuities. ENCIRCLE-style nuisances plausibly *have* jumps (LVEDD ≥ 75 mm gate, threshold-driven Heart-Team behavior), so the jumpy DGP arm below is the case where HAR's guarantee formally fails. HAR's kernel trick is also squared-error-only.

The primary question is therefore empirical and finite-sample: **does crossfit-off match crossfit-on for LTB/HAR (as theory predicts) while breaking for xgboost (as theory also predicts), and does the failure show up in the specific term the theory governs?**

### Why a point-treatment DGP (and not the existing survival scenarios)

In `tmle_ipcw.py` the EIF involves three nuisances (g, Q, and the censoring model G through the IPCW weights), and the no-crossfit condition must hold jointly for all three. Worse, `tmle_ipcw_cv.py` documents that the observed se_ratio ≈ 0.81–0.85 undercoverage in exp6/exp9 is dominated by the doubly-robust remainder and SuperLearner model-selection variance — neither of which is the empirical-process term. Testing LTB/HAR there first would measure noise. A point-treatment DGP with known truth lets us **compute the empirical-process term directly per simulation** rather than inferring it from coverage, and an oracle-nuisance arm pins the remainder at zero as a control. Phase 2 then carries the winners into the survival machinery one nuisance at a time, following the repo's established swap-one-thing pattern (`tmle_ipcw_cv` overrides only `_fit_G`; exp30 swaps only the propensity to HAL).

## 2. Component: LTB learner (`causal_bench/ltb.py`)

`LTBRegressor` and `LTBClassifier`, following `hal.py`'s sklearn-protocol pattern (fit/predict/predict_proba, clone-safe `get_params`/`set_params`) so they drop into SuperLearner unchanged.

Algorithm (per the paper, §Algorithm 1):

1. Fit xgboost with early stopping on a validation split (default 20%), squared-error loss for the regressor, logistic loss for the classifier.
2. Extract the per-tree basis: column k of the design matrix H is the margin contribution of tree k alone, computed as the difference between `predict(..., iteration_range=(0, k+1), output_margin=True)` and `iteration_range=(0, k)`.
3. Lasso the outcome on H: `LassoCV` for the regressor; `LogisticRegressionCV(penalty="l1", solver="saga")` for the classifier (the natural logistic-lasso analogue; the paper's theory is stated for squared error — documented in the docstring).
4. Iterate: add trees in blocks of 10, re-lasso, and stop when validation error exceeds that of every prior solution with smaller L1 coefficient norm (the paper's early-stopping rule).

Final predictor: f(x) = H(x)β̂. Classifier probabilities come through the logistic link and need no clipping. Hyperparameter defaults: `max_depth=3`, block size 10, patience mirroring the paper's "3 increases in validation error".

## 3. Component: HAR learner (`causal_bench/har.py`)

`HARRegressor` and `HARClassifier`, same sklearn-protocol pattern.

- Kernel: K(x, x′) = Σ_i 2^{|s_i(x, x′)|} where s_i(x, x′) = {j : X_ij ≤ min(x_j, x′_j)} counts the coordinates in which training point i is dominated by both arguments. Built by numpy broadcasting over training points — O(n²·p) memory-lite passes, trivial at n ≈ 700 (the O(n³) solve is a 700×700 eigendecomposition).
- Ridge solve: eigendecompose K once, then sweep a log-spaced λ grid with K-fold CV using the cached eigenbasis; refit at the CV-selected λ.
- `HARClassifier` is least-squares on the binary label with predictions clipped to [1e-6, 1−1e-6]. This is a documented caveat, not a bug: the HAR kernel trick applies only to squared-error loss, so probability estimates can be poorly calibrated in the tails — one of the things the benchmark measures.

## 4. Component: point-treatment DGP (`causal_bench/dgp/point_treatment.py`)

No censoring machinery. Columns follow the repo convention: `W1..W4`, `A`, `Y` (binary). Default n = 700 (ENCIRCLE scale), continuous W ~ correlated Gaussians rescaled to clinically-plausible ranges.

Two surface variants, selected by `surface="jumpy" | "smooth"`:

- **jumpy**: g₀(W) and Q₀(A, W) both contain threshold terms — an indicator gate on W1 (LVEDD-style, e.g. 1{W1 ≥ c} shifting both treatment propensity and outcome risk) plus a threshold×covariate interaction. Càdlàg with genuine jumps: inside LTB/HAL's function class, outside HAR's smoothness condition.
- **smooth**: the same structural strength expressed through smooth nonlinearities (tanh/quadratic), inside every learner's class. Control arm where HAR's guarantee applies.

The module exposes `true_g(W)`, `true_Q(a, W)`, and `true_tau(surface)` (computed once per surface by Monte Carlo integration at N = 2×10⁶ with a fixed seed, cached at module level). Positivity is kept healthy by construction (g₀ ∈ [0.1, 0.9]): positivity stress is exp2's job, not this experiment's.

## 5. Component: attribution harness (point AIPW + TMLE)

The existing `aipw.py` hardwires SuperLearner and always uses cross-fitted (OOF) nuisances in the IC, so it cannot express the crossfit-off condition. Exp33 gets its own thin estimator pair in `causal_bench/estimators/point.py` (importable by tests and by phase 2, unlike experiment-local code):

- `point_aipw(df, q_learner, g_learner, crossfit: bool)` — plug-in EIF, no targeting.
- `point_tmle(df, q_learner, g_learner, crossfit: bool)` — one-step logistic targeting as in `tmle_ipcw._target_and_se`, minus the IPCW terms.

With `crossfit=False`, nuisances are fit and evaluated on the full sample and the IC uses those same fits — the condition the Donsker theory licenses. With `crossfit=True`, both g and Q are 5-fold cross-fitted via `crossfit.make_folds` (iid mode; there is no provenance grouping in this DGP) and the IC uses OOF predictions throughout. Existing production estimators are untouched.

## 6. Experiment grid and metrics (`experiments/exp33_donsker_learners.py`)

Grid: **learner** ∈ {logistic, xgboost, LTB, HAR, HAL, oracle} × **crossfit** ∈ {off, on} × **surface** ∈ {jumpy, smooth}, at n = 700, default n_sims = 500 (CLI-configurable, matching other experiments). The same learner family fills both g (classifier) and Q (regressor/classifier) within a cell.

- **logistic** — parametric baseline, misspecified on both surfaces.
- **xgboost** — the non-Donsker ML baseline; theory predicts crossfit-off breaks here.
- **LTB / HAR** — the candidates; theory predicts crossfit-off holds (HAR only on smooth).
- **HAL** — reference via the existing `hal.py` rpy2 wrappers, run at reduced n_sims (default 50) and skipped with a warning when rpy2/hal9001 is unavailable, following the repo's R-optionality convention.
- **oracle** — true g₀/Q₀ injected; pins the remainder term at zero, isolating pure sampling variation.

Per-cell metrics, written to `results/` in the standard format:

1. Nuisance quality: RMSE of ĝ vs g₀ and Q̂ vs Q₀ on an independent draw.
2. Downstream: bias, RMSE, CI coverage, se_ratio (empirical SD of point estimates / mean IC-based SE) for AIPW and TMLE.
3. **Empirical-process term, measured directly**: EP = (P_n − P)(D(f̂) − D(f₀)) per simulation, where the population part is evaluated on a fixed independent Monte Carlo sample (N = 10⁵) using the known truth. Reported as √n·EP distributions per cell — the quantity the Donsker theory says is o_p(1) for LTB/HAL (and HAR-on-smooth) without cross-fitting.
4. **Remainder term**: R = ψ_plugin(f̂) + P·D(f̂) − ψ₀, same Monte Carlo evaluation, to confirm attribution (remainder should be small everywhere positivity is healthy, and exactly zero in the oracle arm).

Headline deliverable: a table/figure showing √n·EP for crossfit-off across learners × surfaces. Success for LTB looks like: crossfit-off ≈ crossfit-on on both surfaces while xgboost's crossfit-off EP term visibly fails to shrink; HAR matching that on smooth and degrading on jumpy would confirm the smoothness caveat empirically.

## 7. Tests (`tests/`)

- **LTB/HAR unit tests**: sklearn clone-ability; output shapes and probability ranges; each recovers a univariate step function with lower test MSE than a linear/logistic fit (the jump case is LTB's home turf and the concrete regression guard); HAR beats ridge-on-raw-features on a smooth nonlinear target.
- **DGP tests**: `true_tau` stable across cache rebuilds (seeded); g₀ bounds respected; jumpy surface actually discontinuous (finite-difference check across the gate).
- **Harness tests**: oracle arm recovers true τ within Monte Carlo tolerance at small n_sims; crossfit on/off produce identical results when handed the oracle (no fitting → no difference).
- **Smoke test**: exp33 end-to-end at n_sims = 3 with the HAL arm skipped.

## 8. Phase 2 (specified here, implemented in the follow-on PR)

1. **SuperLearner opt-in lists**: `ltb_classifiers()` / `ltb_regressors()` / `har_regressors()` in `super_learner.py`, mirroring `hal_classifiers()` — default libraries unchanged.
2. **TMLE-IPCW learner override**: `TMLEIPCWEstimator` gains optional `g_learner` / `q_learner` constructor arguments (defaults preserve current SuperLearner + logistic behavior). Cox `_fit_G` stays fixed, so the swap is attributable to g/Q alone — the same one-nuisance-at-a-time pattern as `tmle_ipcw_cv` and exp30.
3. **Evaluation**: rerun `edwards_realistic` (exp7 config) and the exp16 ENCIRCLE-calibrated scenario with LTB (and HAR, if phase 1 supports it) as g/Q, comparing against the existing `tmle_ipcw` / `tmle_ipcw_cv` rows, with particular attention to whether the crossfit-off IC SE closes any of the documented se_ratio gap.

**Phase 3 (future work, one paragraph, not scheduled):** an LTB discrete-time hazard model replacing Cox in `_fit_G`, which would make all three nuisances Donsker-class and complete the no-crossfit argument for the survival estimand.

**Candidate exp34 (future work, not scheduled): pooled-Q subgroup event rates.** Qiu et al. (arXiv:2605.15483) show within-trial borrowing for subgroup effects in RCTs via pooled outcome regression (TMLE-PR) and adaptive working-model selection (A-TMLE). The treatment-effect machinery does not transfer to single-arm ENCIRCLE (no within-trial A contrast; the S×A bias term does not exist), but the pooling principle does: fit Q on all trial patients, target the subgroup-specific composite event rate against the performance goal. A candidate exp34 would benchmark pooled-Q vs subgroup-only TMLE for subgroup event rates, extending exp21; subgroups must be pre-specified (forest-selected subgroups invalidate the CIs without sample splitting). A-TMLE's working models are HAL-based, so the LTB/HAR learners built here are drop-in candidates there.

## 9. Tracking and delivery

One GitHub issue covers both phases (created at implementation kickoff). One PR per phase; each PR body says "part of #N" — never a closing keyword, even negated — and the issue is closed manually after phase 2 lands. Nothing in this work gates on or references the `concrete` R package; the bridge is untouched.

## 10. Dependencies and non-goals

- New dependency: `xgboost>=2.0` added to core `dependencies` in `pyproject.toml` (verified absent as of this writing; the repo's existing boosting is sklearn's). Core rather than an extra because LTB is the experiment's central learner and phase 2 relies on xgboost's scalability for registry-sized fits; the package is pure-wheel and light. Precedent: rpy2 is already a core dependency with fit-time failure in `hal.py`.
- Non-goals: no changes to production estimators in phase 1; no survival/IPCW handling in exp33; no celer dependency (LassoCV is sufficient at this scale); no attempt to prove the LTB efficiency theorem — this is an empirical benchmark of a theoretical claim.
