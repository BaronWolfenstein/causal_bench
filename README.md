# causal_bench

Monte Carlo benchmarking of causal estimators for clinical trials with survival outcomes.

Generates synthetic randomized and observational trial data under controlled assumption violations — informative censoring, positivity violations, unmeasured confounding, time-varying post-treatment confounders, treatment crossover, enrollment drift, competing risks, and stratified randomization — then measures each estimator's bias, RMSE, coverage, and SE calibration across 22 experiments.

The core finding: the "right" estimator depends entirely on what's wrong with your data. This framework makes that concrete.

Designed for biostatisticians working on device trials (ENCIRCLE-scale, n≈700) who need to decide between TMLE, IPCW, LTMLE, and McCoy's `concrete` package. Estimand coverage includes risk difference, RMST difference, win ratio, restricted mean time in favorable state (RMT-IF), priority-standardized net benefit, and — for ordinal patient-reported outcomes — the cumulative log-OR.

ENCIRCLE's pre-specified primary estimand (Guerrero et al., *Lancet* 2025, SAP Section C) is a **non-hierarchical** KM composite event rate at 1 year tested against a 45% performance goal (one-sided Wald/Greenwood, α = 0.025) — not a hierarchical composite or hazard model. The synthetic control arm (SCA) augments this with an external TVT Registry comparator; the two estimands are reported side by side in `exp16_encircle_calibrated.py`.

---

## Quick start

```bash
git clone <repo> && cd causal_bench
pip install -e ".[dev,storage]"        # storage adds pyarrow for result persistence
pip install -e ".[bayes]"              # bambi/PyMC for the ordinal CLMM estimator (optional)

# Single scenario, 100 sims, MVP estimators
python -m causal_bench --scenario edwards_realistic --n-sims 100

# With diagnostics and sensitivity flags
python -m causal_bench --scenario edwards_realistic --n-sims 100 \
    --diagnostics --tipping-point --ess --convergence --overlap-map

# Export last sim's data for R/concrete benchmarking
python -m causal_bench --scenario edwards_realistic --n-sims 50 --export-r

# Full experiments (each ~2–5 min on 8 cores at n_sims=200)
python experiments/exp1_censoring.py --n-sims 200
python experiments/exp2_positivity.py --n-sims 200
python experiments/exp3_unmeasured.py --n-sims 200
python experiments/exp7_edwards.py    --n-sims 200
python experiments/exp8_mccoy.py      --n-sims 200   # R + concrete required for concrete_RMST
python experiments/exp11_strata.py    --n-sims 200   # R + concrete required for SE correction
```

---

## Estimators (30)

### Risk difference estimators (Python)

| Key | Method | DR | IPCW | Notes |
|-----|--------|:--:|:----:|-------|
| `naive` | Unadjusted mean difference | | | Maximum bias under informative censoring |
| `km` | Kaplan-Meier risk difference | | | Marginal, no covariate adjustment |
| `cox` | Cox G-computation | | | Biased under informative censoring |
| `cox_l1` | Cox + L1 covariate | | | ⚠ Intentionally biased — collider trap (Exp 5 only) |
| `ipw` | Horvitz-Thompson IPW | | | Weight truncation at 1st/99th pct |
| `overlap` | Overlap weighting | | | Targets ATO; stable near positivity violations |
| `aipw` | Augmented IPW | ✓ | | Doubly robust, no targeting step |
| `tmle_ipcw` | TMLE + IPCW | ✓ | ✓ | One-step Newton targeting, cross-fitted IC |
| `tmle_ipcw_comply` | TMLE + IPCW + compliance | ✓ | ✓ | Compliance score in censoring model |
| `ltmle` | Longitudinal TMLE | ✓ | ✓ | Sequential regression over L1; no collider bias |
| `tmle_ipcw_cv` | CV-TMLE (cross-fitted) | ✓ | ✓ | Cross-fitted censoring model → calibrated SE at finite n |
| `tmle_ipcw_cv_comply` | CV-TMLE + compliance | ✓ | ✓ | Cross-fitted, with compliance score in the censoring model |
| `tmle_ipcw_boot` | TMLE + IPCW (bootstrap SE) | ✓ | ✓ | Nonparametric bootstrap SE instead of the IC-based SE |

### RMST estimators (R bridge — requires `concrete`)

| Key | Method | Notes |
|-----|--------|-------|
| `concrete_RMST` | Direct RMST targeting | McCoy (2026); iid SE |
| `concrete_RMST_strata` | Direct RMST + BCS SE correction | Bugni-Canay-Shaikh / Ye-Shao correction for stratified randomization (concrete PR #29) |
| `rmst_k2 / k5 / k10 / k20` | Pointwise-then-integrate RMST | Bias O(1/K); K=20 near-exact |
| `concrete_simult` | Multi-horizon RMST, simultaneous bands | Joint TMLE across horizons (e.g. t=1,2) with simultaneous confidence bands |

### Win ratio estimators (R bridge — requires `concrete`)

| Key | Method | Notes |
|-----|--------|-------|
| `concrete_WR_direct` | Win ratio direct TMLE | `targetWinRatio()` (McCoy PR #30); jointly fluctuates both arms' cause-specific hazards |
| `concrete_WR_plugin` | Win ratio plug-in | `getWinRatio()` after `doConcrete()`; ~5× more bias than direct |

**True win ratio benchmark:** `compute_true_win_ratio(config)` computes P(T₁>T₀)/P(T₁<T₀) via U-statistic on 50k potential-outcome pairs using `np.searchsorted` (O(n log n)).

### Clinical composite estimators (R bridge — requires `concrete`)

| Key | Method | Notes |
|-----|--------|-------|
| `clinical_RMTIF` | Restricted mean time in favorable state | `clinicalRMTIF()` multistate engine (concrete PR #33) |
| `clinical_PSNB` | Priority-standardized net benefit | `clinicalPSNB()` priority-ranked win-ratio variant (concrete PR #34) |

### HTE / subgroup estimators (CATE)

| Key | Method | Notes |
|-----|--------|-------|
| `effectxshift` | Post-selection HTE subgroups | McCoy's EffectXShift R package; CV-TMLE subgroup effects |
| `bcf_bart` | BCF/BART CATE | Bayesian Causal Forest + rpart summary tree (Hahn et al. 2020) |

### Ordinal PRO estimator (Bayesian, Python)

| Key | Method | Notes |
|-----|--------|-------|
| `clmm_ordinal` | Bayesian cumulative-link mixed model | bambi/PyMC; **partial pooling** (random site intercept); posterior credible intervals; surfaces the site SD (τ) in `convergence_info`. Needs `pip install -e ".[bayes]"`. Assumption-adversary to the PRO win ratio — the head-to-head benchmark (exp25) is gated on `concrete#36` |
| `clmm_ordinal_slope` | CLMM, `(A \| site)` random slope | Correlated random intercept **and** treatment slope per site; surfaces τ_A. The stronger adversary when the DGP has site-varying treatment effects |
| `clmm_ordinal_nopool` | CLMM, **no pooling** (fixed site effects) | Reference arm: one intercept per site, no shrinkage |
| `clmm_ordinal_cpool` | CLMM, **complete pooling** (no site term) | Reference arm: ignores site clustering entirely |

---

## Experiments (22)

| Script | Swept parameter | Key story | Estimators |
|--------|-----------------|-----------|------------|
| `exp1_censoring.py` | `censoring_informativeness` 0→1 | Naive/KM degrade; TMLE+IPCW stays flat | MVP |
| `exp2_positivity.py` | `positivity_severity` 0→3 | IPW weight variance explodes; overlap stays stable | All Python |
| `exp3_unmeasured.py` | `unmeasured_confounding_strength` 0→0.8 | **ALL estimators biased** — identification failure | All Python |
| `exp4_crossover.py` | `crossover_rate` 0→0.3 | ITT attenuation; IPCW censoring at crossover helps | MVP |
| `exp5_collider.py` | `collider_strength` 0→1 | Opposite-direction biases; only LTMLE correct | cox, cox_l1, ltmle, tmle_ipcw |
| `exp6_drift.py` | `enrollment_drift` 0→0.5 | Learning-curve bias; Cox/TMLE adjust, KM/naive don't | MVP |
| `exp7_edwards.py` | Edwards scenarios (3) | Full benchmark, realistic device trial conditions | All Python |
| `exp8_mccoy.py` | RMST grid density K=2–20 | Direct targeting eliminates discretisation bias | tmle_ipcw, aipw, ltmle, rmst_k2–k20, concrete_RMST |
| `exp9_sample_size.py` | `n` 100→2000 | Where TMLE asymptotics hold for ENCIRCLE (n=700) | MVP |
| `exp10_win_ratio.py` | — | Direct TMLE cuts WR bias ~5× vs plug-in | concrete_WR_direct, concrete_WR_plugin |
| `exp11_strata.py` | — | BCS SE correction narrows CIs under stratified RCT | concrete_RMST, concrete_RMST_strata, tmle_ipcw |

**Extended experiments (design & operating-characteristics studies):**

| Script | Focus |
|--------|-------|
| `exp12_simultaneous.py` | Simultaneous coverage across a multi-estimand family |
| `exp13_censoring_sweep.py` | Censoring mechanism sweep (MAR / MNAR / informative) |
| `exp14_synthetic_augmentation.py` | Provenance-linked synthetic augmentation — cross-fitting independence |
| `exp15_sequential_monitoring.py` | Sequential CED monitoring — anytime-valid vs alpha-spending vs naive |
| `exp16_encircle_calibrated.py` | ENCIRCLE-calibrated replication — 14 estimators vs published marginals |
| `exp17_transport.py` | Transport decomposition — trial-to-commercial generalizability |
| `exp18_hawthorne.py` | Hawthorne decomposition — durable vs transient monitoring artifact |
| `exp19_hierarchical_oc.py` | Hierarchical borrowing operating characteristics |
| `exp20_tipping_point_borrowing.py` | Tipping-point sweep × borrowing strength |
| `exp21_hte_subgroup.py` | HTE subgroup benchmark — EffectXShift CV-TMLE vs BCF/BART posterior tree |
| `exp24_site_clustering.py` | Site clustering in registry comparator — undercoverage demonstration |

exp25 — the win-ratio-vs-CLMM ordinal-PRO benchmark — is planned and gated on `concrete#36`.

---

## Key findings

**Exp 1 (censoring):** Naive and KM bias grows monotonically with `censoring_informativeness`. TMLE+IPCW stays near zero. Adding compliance to the censoring model gives further advantage in the MNAR regime (informativeness > 0.5).

**Exp 2 (positivity):** IPW SE inflates and coverage collapses at `positivity_severity` ≥ 2. Overlap weighting targets a different estimand (ATO) and is robust by construction. TMLE+IPCW degrades less than IPW but is not immune.

**Exp 3 (unmeasured confounding — THE HONESTY EXPERIMENT):** Every estimator is biased. The bias grows linearly with `unmeasured_confounding_strength` regardless of how sophisticated the method is. Negative control outcome bias tracks primary outcome bias, confirming U as the source. Semiparametric efficiency is irrelevant when identification fails.

**Exp 4 (crossover):** As-treated analysis attenuates the apparent effect as `crossover_rate` rises. TMLE+IPCW censors at crossover and partially recovers the per-protocol effect. The compliance-based censoring model (TMLE+IPCW+comply) gives further improvement because compliance predicts who will switch.

**Exp 5 (collider trap):** At high `collider_strength`, Cox without L1 is biased toward the null (omitted-variable bias) and Cox with L1 is biased *away* from the null in the opposite direction (collider bias). There is no correct naive choice — both versions of Cox are wrong, in opposite directions. LTMLE marginalises over L1 rather than conditioning on it and stays near unbiased.

**Exp 6 (enrollment drift):** Under learning-curve conditions, early enrollees have worse outcomes independent of treatment. KM and naive diverge as `enrollment_drift` rises because they don't adjust for enrollment time. Cox and TMLE include `enrollment_time` as a covariate (the Senn fix) and maintain near-zero bias.

**Exp 7 (Edwards combined):** Under `edwards_realistic`, LTMLE and TMLE+IPCW have the smallest bias and best coverage. IPW and AIPW degrade under positivity stress. Naive and KM are unreliable across all but the optimistic scenario.

**Exp 8 (McCoy RMST):** Direct RMST targeting via `concrete` eliminates discretisation bias accumulated by pointwise estimators at coarse grids. The `concrete_RMST` estimator shows residual bias against the benchmark ATE, but this is an **estimand mismatch** — `compute_true_effects()` returns an all-cause counterfactual risk difference, while `concrete_RMST` estimates the cause-specific CIF difference for event 1. Python TMLE/AIPW treat competing events as independent censoring, inflating cause-1 risk toward the all-cause number. In a single-event scenario the three estimators converge.

**Exp 9 (sample size):** On `edwards_realistic` (the hardest scenario), TMLE+IPCW approaches near-unbiasedness only at n ≥ 700. At n=100 the Super Learner has too little data for nuisance model quality. Naive/KM bias is invariant to n — no asymptotic rescue for misspecification.

**Exp 10 (win ratio):** `concrete_WR_direct` (McCoy PR #30 — `targetWinRatio()`) cuts win ratio bias ~5× relative to the plug-in approach by solving the win/loss EIF estimating equations jointly rather than substituting targeted risk curves into the win functional.

**Exp 11 (stratified SE correction):** Under stratified block randomization (W2 × W4, block size 4), the iid SE is conservative (se_ratio > 1, wide CIs). The BCS-corrected SE from `concrete_RMST_strata` restores calibration (se_ratio ≈ 1) while maintaining coverage ≥ 0.95. The power gain is meaningful at trial-scale n.

---

## Data-generating process

AFT model with Gumbel noise (Weibull survival), unmeasured confounder U, post-treatment time-varying variable L1, informative censoring, optional competing risks, and optional stratified block randomization:

```
W1 ~ N(0,1)   W2 ~ Bern(0.5)   W3 ~ N(0,1)   W4 ~ Bern(0.3)   U ~ N(0,1) [latent]

Treatment (default Bernoulli):
  logit P(A=1|W,U) = logit(prev) + 0.3W1 + 0.2W2 - 0.2W3 + 0.1W4
                   + 0.5U·unmeasured_strength + 0.8W1·W3·positivity_severity

Treatment (stratified block, when strata_cols set):
  Permuted blocks of size strata_block_size within W2×W4 strata

Survival time (AFT):
  log T = 0 + 0.4W1 - 0.3W2 + 0.2W3 - 0.2W4 + 0.3U + τA
        + enrollment_drift·enrollment_time + nonlinearity·(W1²-1)
        + heterogeneity·A·W1 + ε     ε ~ Gumbel(0,1)

Post-treatment confounder (when collider_strength > 0):
  L1 = 0.5A + 0.4W3 + 0.3U·collider_strength + noise   (observed at t_L1)

Censoring:
  log C = 1.5 - 0.2W1 + 0.1W3 - 0.1A + 0.4U·censoring_informativeness
        + MNAR component for informativeness > 0.5

Competing risks (when enabled):
  log T2 = 0.3 + 0.2W1 - 0.1W3 + 0.2U + cause2_effect·A + ε2
  First event (T, T2, C, horizon) determines observed time and event type.
```

True ATE/ATT: G-computation on n=50,000 with shared Gumbel noise. True RMST: same population, trapezoidal integration. True win ratio: U-statistic on 50k potential-outcome pairs via `np.searchsorted`.

### Ordinal PRO DGP (`dgp/ordinal_pro.py`)

A separate thresholded-latent cumulative-logistic model for an ordinal patient-reported outcome (NYHA I–IV, KCCQ tertiles), used by the ordinal-PRO benchmark (issue #26):

```
P(Y<=j | W,A,site) = logistic(c_j + δ_site,j − f(W) − b_site − τ_eff,j·A),   j = 1..K-1
  f(W)      = 0.4W1 − 0.3W2 + 0.2W3 − 0.2W4     (shared with the survival DGP)
  b_site    ~ N(0, σ²_site)                      site random intercept (ICC-parameterized)
  τ_eff,j   = τ + offset_j                       proportional odds ⟺ offsets/floor/ceiling = 0
```

Both PO-respecting and PO-violating modes are supported so a Bayesian CLMM (targets the cumulative log-OR) and a GPC win ratio (targets the ordinal win ratio) can each be scored against known truth via `compute_true_cumulative_logOR` and `compute_true_ordinal_win_ratio`. Emits an `ordinal_pro` marker column consumable by `ConcretePROWinRatioEstimator`. Full writeup in the `index.qmd` Replication Data appendix.

---

## Scenarios

| Scenario | n | censor_info | positivity | collider | unmeasured | notes |
|----------|---|:-----------:|:----------:|:--------:|:----------:|-------|
| `clean` | 500 | 0.0 | 0.0 | 0.0 | 0.0 | Baseline |
| `censor_mild` | 500 | 0.3 | 0.0 | 0.0 | 0.0 | |
| `censor_moderate` | 500 | 0.6 | 0.0 | 0.0 | 0.0 | |
| `censor_severe` | 500 | 1.0 | 0.0 | 0.0 | 0.0 | |
| `positivity_mild/moderate/severe` | 500 | 0.0 | 1/2/3 | 0.0 | 0.0 | |
| `unmeasured_mild/mod/strong` | 500 | 0.0 | 0.0 | 0.0 | 0.2/0.5/0.8 | |
| `edwards_optimistic` | 700 | 0.3 | 0.5 | 0.2 | 0.1 | |
| `edwards_realistic` | 700 | 0.6 | 1.5 | 0.4 | 0.2 | ENCIRCLE-like |
| `edwards_pessimistic` | 700 | 0.9 | 2.5 | 0.7 | 0.4 | |
| `competing_risks_base` | 600 | 0.3 | 0.0 | 0.0 | 0.0 | event_type ∈ {0,1,2} |
| `stratified_base` | 500 | 0.0 | 0.0 | 0.0 | 0.0 | W2×W4 strata, block=4 |

---

## R integration (concrete)

```r
# Install concrete (McCoy's package — includes PR #29 BCS correction, PR #30 win ratio, PR #33 RMT-IF, PR #34 PSNB)
remotes::install_github("blind-contours/concrete", upgrade = "always")
install.packages(c("reticulate", "data.table"))

# SuperLearner libraries (may not be auto-installed from concrete's DESCRIPTION):
install.packages(c("glmnet", "ranger", "xgboost", "hal9001"))

# Direct use from RStudio via reticulate
source("r_scripts/concrete_bridge.R")
library(reticulate)
use_virtualenv(".venv")
cb  <- import("causal_bench.dgp.survival")
cfg <- import("causal_bench.dgp.config")$DGPConfig(n=600L, competing_risks=TRUE)
df  <- as.data.frame(cb$generate_data(cfg))
df$event_type <- as.integer(df$Delta)

result      <- run_concrete_bridge(df, horizon=1.0)           # RMST (iid SE)
result_bcs  <- run_concrete_bridge(df, horizon=1.0,           # RMST + BCS SE
                                   strata_cols=c("W2","W4"))
result_wr   <- run_concrete_win_ratio(df, horizon=1.0,        # win ratio (direct TMLE)
                                       method="direct")
```

From Python (requires `pip install -e ".[r]"`):
```python
from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
from causal_bench.estimators.concrete_win_ratio import ConcreteWinRatioEstimator

# Returns [] with warning if R unavailable — experiments handle this gracefully
rmst = ConcreteRMSTEstimator().estimate(df)
rmst_bcs = ConcreteRMSTEstimator(strata_cols=["W2", "W4"]).estimate(df)
wr   = ConcreteWinRatioEstimator(method="direct").estimate(df, estimand="WR")
```

---

## Diagnostics

All functions live in `causal_bench.diagnostics`. CLI flags activate them automatically after a run.

### Always-on (via `--diagnostics`)

| Function | Output |
|----------|--------|
| `plot_overlap(df)` | Propensity score histogram by arm, extreme weight %, ESS |
| `plot_love(df)` | Love plot: \|SMD\| before/after IPW weighting |
| `plot_se_calibration(results)` | Scatter: empirical SE vs median reported SE per estimator |

### Flag-enabled

| CLI flag | Function | Output |
|----------|----------|--------|
| `--tipping-point` | `tipping_point_table`, `plot_tipping_point` | Additive bias needed to explain away each estimate |
| `--ess` | `ess_across_sims`, `plot_ess_distribution` | IPW ESS histogram across 50 draws; flags ESS < 50% of n |
| `--convergence` | `convergence_table` | IC-based TMLE convergence: ic_mean (≈ε), ic_sd, ic_ratio per estimator |
| `--overlap-map` | `plot_overlap_map` | "Who are we borrowing for?" — treated patients in (W1,W3) space, size ∝ 1/g, control density in background |
| `--mnar-tipping-point` | `tipping_point_mnar`, `plot_tipping_point_mnar` | MNAR sensitivity grid: imputes censored outcomes across (δ_treated, δ_control) grid |
| `--export-r` | `export_for_r` | CSV + metadata JSON for loading into R/concrete directly |

```bash
python -m causal_bench --scenario edwards_realistic --n-sims 100 \
    --diagnostics \
    --tipping-point \
    --ess \
    --convergence \
    --overlap-map \
    --export-r
```

### MNAR and concrete sensitivity (Python API)

```python
from causal_bench.diagnostics import (
    tipping_point_mnar, plot_tipping_point_mnar,
    tipping_point_concrete, plot_tipping_point_concrete,
    convergence_table, plot_overlap_map, export_for_r,
)

# MNAR sensitivity grid
r = tipping_point_mnar(df, "km", horizon=cfg.horizon, n_grid=15)
plot_tipping_point_mnar(r, save_path="tipping_mnar.png")
r.to_parquet("tipping_mnar.parquet")    # attrs (MAR reference) survive roundtrip

# concrete MAR sensitivity (requires R + concrete)
r2 = tipping_point_concrete(df, horizon=cfg.horizon, deltas=[0, 0.05, 0.10, 0.15, 0.20])
plot_tipping_point_concrete(r2, save_path="tipping_concrete.png")
print(f"Tipping delta: {r2.attrs['tipping_delta']:.2f}")

# TMLE convergence (IC-based; no re-run needed)
conv = convergence_table(df, estimator_names=["tmle_ipcw", "tmle_ipcw_comply"])
print(conv)

# Overlap map
plot_overlap_map(df, save_path="overlap_map.png")

# Export for R/concrete benchmarking
paths = export_for_r(df, cfg, out_dir="results/export")
```

---

## Win ratio estimand

```python
from causal_bench.dgp.survival import compute_true_win_ratio
from causal_bench.estimators.concrete_win_ratio import ConcreteWinRatioEstimator

# True win ratio: P(T_treated > T_control) / P(T_treated < T_control)
wr_true = compute_true_win_ratio(cfg)
# Returns: {"ATE": wr, "ATT": wr_att, "p_win": ..., "p_loss": ..., "net_benefit": ...}

# Estimate (requires R + concrete PR #30)
est = ConcreteWinRatioEstimator(method="direct")   # or method="plugin"
results = est.estimate(df, estimand="WR")
```

Sign convention: `true_tau=-0.5` shortens T → T₁ < T₀ → WR < 1 (treated loses more often). Opposite sign from risk difference.

---

## Stratified randomization

```python
from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data

# Permuted-block randomization within W2×W4 strata (4 strata, block size 4)
cfg = DGPConfig(n=500, strata_cols=("W2", "W4"), strata_block_size=4)
df  = generate_data(cfg)
# df.attrs["strata_cols"] == ["W2", "W4"]  — passed automatically to concrete bridge

# BCS SE correction: pass strata_cols to ConcreteRMSTEstimator
from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
est = ConcreteRMSTEstimator(strata_cols=["W2", "W4"])
results = est.estimate(df)
```

Strata are defined by binarising each column at its median (binary columns need no transformation). Two strata_cols → 2² = 4 strata. Within each stratum, half-and-half blocks are shuffled; a partial final block gets Bernoulli draws.

---

## IC bootstrap CIs

TMLE+IPCW, LTMLE, and AIPW store influence curve values on `EstimatorResult.ic`. These can be bootstrapped cheaply without re-fitting:

```python
from causal_bench.bootstrap import ic_bootstrap_ci

result = TMLEIPCWEstimator().estimate(df)[0]
lo, hi = ic_bootstrap_ci(result, B=2000, method="bca")         # bias-corrected + accelerated
lo, hi = ic_bootstrap_ci(result, B=2000, method="t")           # Studentized
lo, hi = ic_bootstrap_ci(result, B=2000, method="percentile")  # plain quantiles
```

| Method | Skewness correction | Recommended for |
|--------|--------------------|-----------------|
| `percentile` | None | Large n (> 1000), symmetric IC |
| `t` | Empirical t-quantiles from per-resample SE* | Small–moderate n, asymmetric IC |
| `bca` | Bias-correction z₀ + jackknife acceleration a | Skewed estimators; best coverage in theory |

---

## Result persistence

```python
# Save
sr.to_parquet("results/exp1/tmle_ipcw.parquet")

# Load (next session, no re-run needed)
from causal_bench.metrics import SimResult
sr = SimResult.from_parquet("results/exp1/tmle_ipcw.parquet")
print(sr.summary())
```

---

## CLI reference

```
python -m causal_bench [OPTIONS]

  --scenario              Named DGP scenario (default: edwards_realistic)
  --n-sims                Monte Carlo replicates (default: 100)
  --n-jobs                Parallel workers, -1 = all CPUs (default: -1)
  --estimand              ATE or ATT (default: ATE)
  --estimators            Space-separated estimator keys
  --seed                  Random seed (default: 42)
  --out-dir               Output directory (default: results/)
  --no-plots              Skip plot generation

  Diagnostics (can combine freely):
  --diagnostics           Overlap, Love plot, SE calibration
  --tipping-point         Tipping-point sensitivity table + plot
  --ess                   ESS distribution (50 draws) + plot
  --convergence           IC-based TMLE convergence table (single dataset)
  --overlap-map           Propensity overlap map in (W1, W3) space
  --export-r              CSV + metadata JSON for R/concrete benchmarking
  --mnar-tipping-point    MNAR sensitivity grid (skipped if cens_informativeness=0)
  --mnar-estimator        Estimator for MNAR grid (default: km)
  --mnar-grid             Grid points per axis (default: 10, total = n²)
```

---

## References

- van der Laan & Rose (2011). *Targeted Learning*. Springer.
- van der Laan & Gruber (2012). Targeted minimum loss-based estimation of causal effects. *Int J Biostatistics*.
- Robins, Hernán & Brumback (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology*.
- Li, Morgan & Zaslavsky (2018). Balancing covariates via propensity score weighting. *JASA*.
- Bugni, Canay & Shaikh (2018). Inference under covariate-adaptive randomization. *JASA*.
- McCoy (2026). Direct RMST targeting for competing-risks TMLE. `concrete` R package.
- Hernán & Robins (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
