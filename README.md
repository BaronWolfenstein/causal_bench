# causal_bench

Monte Carlo benchmarking of causal estimators for clinical trials with survival outcomes.

Generates synthetic randomized trial data under controlled conditions — varying censoring informativeness, positivity violations, unmeasured confounding, and time-varying post-treatment variables — and measures each estimator's bias, RMSE, coverage, and SE calibration.

The core finding: the "right" estimator depends entirely on what's wrong with your data. This framework makes that concrete.

## Quick start

```bash
git clone <repo> && cd causal_bench
pip install -e ".[dev,storage]"        # storage adds pyarrow for result persistence

# Single scenario, 50 sims, all estimators
python -m causal_bench --scenario edwards_realistic --n-sims 50

# Full experiments (each ~2–5 min on 8 cores)
python experiments/exp1_censoring.py --n-sims 200
python experiments/exp5_collider.py  --n-sims 200
python experiments/exp7_edwards.py   --n-sims 200
python experiments/exp8_mccoy.py     --n-sims 200   # R + concrete required for concrete_RMST

# Render the Quarto walkthrough notebook
quarto render index.qmd
```

## Estimators

| Key | Method | Doubly robust | IPCW | Notes |
|-----|--------|:---:|:---:|-------|
| `naive` | Unadjusted mean difference | | | Baseline only |
| `km` | Kaplan-Meier risk difference | | | Marginal, no covariate adjustment |
| `cox` | Cox G-computation | | | Breaks under informative censoring |
| `ipw` | Horvitz-Thompson IPW | | | Weight truncation at 1st/99th pct |
| `overlap` | Overlap weighting | | | Targets ATO, stable under near-violations |
| `aipw` | Augmented IPW | ✓ | | Doubly robust, no targeting step |
| `tmle_ipcw` | TMLE + IPCW | ✓ | ✓ | One-step Newton targeting |
| `tmle_ipcw_comply` | TMLE + IPCW + compliance | ✓ | ✓ | Compliance in censoring model |
| `ltmle` | LTMLE | ✓ | ✓ | Marginalises over L1, no collider bias |
| `cox_l1` | Cox + L1 (collider) | | | ⚠ Intentionally biased — for Exp 5 only |
| `concrete_RMST` | concrete direct RMST | ✓ | ✓ | Requires R + concrete package |
| `rmst_k2/k5/k10/k20` | Pointwise RMST (K grid points) | ✓ | ✓ | Bias O(1/K); K=20 near-exact |

## Experiments

| Script | What it shows | Estimators |
|--------|--------------|------------|
| `exp1_censoring.py` | Bias as censoring informativeness increases 0→1 | All MVP |
| `exp5_collider.py` | Collider trap: Cox vs Cox+L1 vs LTMLE | cox, cox_l1, ltmle, tmle_ipcw |
| `exp7_edwards.py` | Full benchmark across 3 Edwards scenarios | All except cox_l1 |
| `exp8_mccoy.py` | RMST vs pointwise, competing risks | tmle_ipcw, aipw, ltmle, rmst_k2–k20, concrete_RMST |

## Key findings

**Exp 1 (censoring gradient):** Naive and KM bias grows monotonically with censoring informativeness. TMLE+IPCW stays near zero. Including compliance in the censoring model gives a small additional advantage at high informativeness (MNAR regime).

**Exp 5 (collider trap):** At high `collider_strength`, Cox without L1 is biased toward the null (missing-variable bias) and Cox with L1 is biased *away* from the null in the opposite direction (collider bias). LTMLE — which marginalises over L1 rather than conditioning on it — stays unbiased. There is no simple choice between the two naive approaches.

**Exp 7 (Edwards combined):** Under the realistic Edwards scenario, LTMLE and TMLE+IPCW have the smallest bias and best coverage. IPW and AIPW degrade under positivity stress. Naive and KM are unreliable across all but the optimistic scenario.

**Exp 8 (McCoy RMST):** Direct RMST targeting via `concrete` eliminates the discretisation bias accumulated by pointwise RMST estimators at coarse time grids. `concrete_RMST` shows apparent residual bias against the benchmark true ATE, but this is an **estimand difference**, not an estimator failure: `compute_true_effects()` computes an all-cause counterfactual risk difference, the Python TMLE/AIPW estimators treat competing events as independent censoring (inflating cause-1 risk toward the all-cause number), and `concrete_RMST` correctly estimates the cause-specific CIF difference for event 1 (lower, because competing events properly reduce the at-risk pool). The three are not measuring the same quantity. In a scenario without competing risks, or with a true ATE defined as the cause-specific CIF difference, `concrete_RMST` is unbiased.

**SE calibration (EIF-based estimators):** EIF-based SEs (TMLE+IPCW, LTMLE, AIPW) can undercover when nuisance models are trained and evaluated on the same data. All three estimators use cross-fitted (DML-style) influence function residuals: the propensity score `g` is obtained from the SuperLearner's own out-of-fold (OOF) predictions, and the outcome model `Q` is cross-fitted with K-fold logistic regression. The point estimate still uses full-data targeted Q* (TMLE targeting) or full-data SuperLearner Q (AIPW) for better finite-sample bias. If coverage remains insufficient in small samples, use an IC bootstrap CI instead — three methods are available (see below). There is no principled analytic multiplier: any constant correction factor is DGP-specific and not portable.

## Data-generating process

AFT model with Gumbel noise (Weibull survival), unmeasured confounder U, post-treatment time-varying variable L1, informative censoring, optional competing risks:

```
log T = 0 + 0.4W1 - 0.3W2 + 0.2W3 - 0.2W4 + 0.3U + τA + ε   ε ~ Gumbel(0,1)
L1    = 0.5A + 0.4W3 + 0.3U·collider_strength + noise         (at t_L1 = 0.5)
log C = 1.5 - 0.2W1 + 0.1W3 - 0.1A + 0.4U·informativeness    (MNAR for inf > 0.5)
```

True effects computed by g-computation on n=50,000 reference population with shared Gumbel noise across potential outcome arms.

## Scenarios

| Scenario | n | censor_info | positivity | collider | unmeasured |
|----------|---|-------------|------------|----------|------------|
| `clean` | 500 | 0.0 | 0.0 | 0.0 | 0.0 |
| `censor_mild/moderate/severe` | 500 | 0.3/0.6/1.0 | 0.0 | 0.0 | 0.0 |
| `edwards_optimistic` | 700 | 0.3 | 0.5 | 0.2 | 0.1 |
| `edwards_realistic` | 700 | 0.6 | 1.5 | 0.4 | 0.2 |
| `edwards_pessimistic` | 700 | 0.9 | 2.5 | 0.7 | 0.4 |
| `competing_risks_base` | 600 | 0.3 | 0.0 | 0.0 | 0.0 |

## R integration (concrete)

```r
# Install concrete (McCoy's package, actively developed)
remotes::install_github("blind-contours/concrete", upgrade = "always")
install.packages(c("reticulate", "data.table"))

# concrete's SuperLearner library requires these — install manually if the
# above doesn't pull them automatically (known gap in concrete's DESCRIPTION):
install.packages(c("glmnet", "ranger", "xgboost", "hal9001"))

# Use from RStudio — calls Python generate_data() directly via reticulate
source("r_scripts/concrete_bridge.R")
library(reticulate)
use_virtualenv(".venv")
cb  <- import("causal_bench.dgp.survival")
cfg <- import("causal_bench.dgp.config")$DGPConfig(n=600L, competing_risks=TRUE)
df  <- as.data.frame(cb$generate_data(cfg))
df$event_type <- as.integer(df$Delta)
result <- run_concrete_bridge(df, horizon=1.0)
```

From Python (requires `pip install -e ".[r]"`):
```python
from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
results = ConcreteRMSTEstimator().estimate(df)  # returns [] with warning if R unavailable
```

## IC bootstrap CIs

TMLE+IPCW, LTMLE, and AIPW all store the influence curve (IC) values on `EstimatorResult.ic`. These can be bootstrapped cheaply — O(B·n) instead of re-fitting the model — using `causal_bench.bootstrap.ic_bootstrap_ci`:

```python
from causal_bench.bootstrap import ic_bootstrap_ci

result = TMLEIPCWEstimator().estimate(df)[0]

lo, hi = ic_bootstrap_ci(result, B=2000, method="bca")    # bias-corrected + accelerated
lo, hi = ic_bootstrap_ci(result, B=2000, method="t")      # Studentized / bootstrap-t
lo, hi = ic_bootstrap_ci(result, B=2000, method="percentile")  # plain quantiles
```

All three methods bootstrap the IC values rather than re-fitting: each resample draws n rows of IC with replacement and computes θ̂* = θ̂ + mean(IC*). The methods differ in how they convert the bootstrap distribution into a CI:

| Method | Skewness correction | Recommended for |
|--------|--------------------|-----------------||
| `percentile` | None | Large n (>1000), symmetric IC |
| `t` | Via empirical t-quantiles from per-resample SE* | Small–moderate n, asymmetric IC; Hesterberg (2015) recommends this for general use |
| `bca` | Bias-correction z₀ + jackknife acceleration a | Skewed estimators; best coverage in theory (Efron & Tibshirani 1993) |


## Diagnostics

`causal_bench` ships a diagnostics module for inspecting positivity, covariate balance, and SE calibration. All functions are in `causal_bench.diagnostics`.

| Function | Output |
|----------|--------|
| `plot_overlap(df)` | Propensity score histogram by arm, extreme weight %, ESS |
| `plot_love(df)` | Love plot: \|SMD\| before and after IPW weighting |
| `plot_se_calibration(results)` | Scatter: empirical SE vs median reported SE |
| `plot_tipping_point(results)` | How much additive bias would explain away each estimate |
| `plot_ess_distribution(dgp_config)` | IPW ESS histogram across simulation draws |
| `tipping_point_mnar(df, estimator, horizon)` | MNAR sensitivity grid + heatmap (imputes censored outcomes) |
| `tipping_point_concrete(df, horizon, deltas)` | concrete::senseCensoring() delta-shift MAR sensitivity (requires R + concrete) |
| `plot_tipping_point_concrete(tipping_df)` | Line + CI ribbon of RD vs δ; marks tipping delta where CI crosses 0 |

CLI flags activate diagnostics automatically after a run:

```bash
python -m causal_bench --scenario edwards_realistic --n-sims 100 \
    --diagnostics       # overlap.png, love.png, se_calibration.png
    --tipping-point     # tipping_point.png + table to stdout
    --ess               # ess_distribution.png + summary to stdout
```

Python API:

```python
from causal_bench.diagnostics import (
    plot_overlap, plot_love,
    tipping_point_table, plot_tipping_point,
    ess_across_sims, plot_ess_distribution,
    tipping_point_mnar, plot_tipping_point_mnar,
    tipping_point_concrete, plot_tipping_point_concrete,
)

df = generate_data(cfg)
plot_overlap(df, save_path="overlap.png")
plot_love(df, save_path="love.png")

# After running simulations:
tipping_point_table(results)          # DataFrame: bias to explain away per estimator
ess_across_sims(cfg, n_draws=50)      # dict: median/min/max ESS, % of n

# MNAR sensitivity — pairs with censoring_informativeness in the DGP
cfg = DGPConfig(n=500, censoring_informativeness=0.6, censoring_rate=0.3, seed=42)
df  = generate_data(cfg)
r   = tipping_point_mnar(df, "km", horizon=cfg.horizon, n_grid=15)
plot_tipping_point_mnar(r, save_path="tipping_mnar.png")
r.to_parquet("tipping_mnar.parquet")  # attrs (MAR reference) survive the roundtrip

# concrete MAR sensitivity (requires R + blind-contours/concrete)
# L1 is already forwarded to CensoringTV, so the baseline (δ=0) uses
# L1-corrected IPCW; the delta-shift asks how robust that correction is.
r2  = tipping_point_concrete(df, horizon=cfg.horizon, deltas=[0, 0.05, 0.10, 0.15, 0.20])
plot_tipping_point_concrete(r2, save_path="tipping_concrete.png")
print(f"Tipping delta: {r2.attrs['tipping_delta']:.2f}")
```

## Result persistence

```python
# Save
sim_result.to_parquet("results/exp1/tmle_ipcw.parquet")

# Load (next session, no re-run needed)
from causal_bench.metrics import SimResult
sr = SimResult.from_parquet("results/exp1/tmle_ipcw.parquet")
print(sr.summary())
```

## CLI reference

```
python -m causal_bench [OPTIONS]

  --scenario      Named DGP scenario (default: edwards_realistic)
  --n-sims        Monte Carlo replicates (default: 100)
  --n-jobs        Parallel workers, -1 = all CPUs (default: -1)
  --estimand      ATE or ATT (default: ATE)
  --estimators    Space-separated estimator keys
  --seed          Random seed (default: 42)
  --out-dir       Output directory (default: results/)
  --no-plots      Skip plot generation
  --diagnostics         Overlap, Love plot, SE calibration after run
  --tipping-point       Tipping-point sensitivity table + plot
  --ess                 ESS distribution across 50 simulation draws + plot
  --mnar-tipping-point  MNAR sensitivity grid (skipped if censoring_informativeness=0)
  --mnar-estimator      Estimator for MNAR grid (default: km)
  --mnar-grid           Grid points per axis (default: 10, runs = n²)
  --concrete-sensitivity  concrete::senseCensoring() delta-shift analysis (requires R)
```

## References

- van der Laan & Gruber (2012). Targeted minimum loss-based estimation of causal effects. *Int J Biostatistics*.
- Li, Morgan & Zaslavsky (2018). Balancing covariates via propensity score weighting. *JASA*.
- McCoy (2026). Direct RMST targeting for competing-risks TMLE. `concrete` R package.
- van der Laan & Rose (2011). *Targeted Learning*. Springer.
- Hernán & Robins (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
