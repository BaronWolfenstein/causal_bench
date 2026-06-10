# causal_bench

`causal_bench` is a Monte Carlo benchmarking framework for causal estimators applied to clinical trial data with survival outcomes. It simulates randomized controlled trial datasets under a range of data-generating processes—varying degrees of informative censoring, positivity violations, and unmeasured confounding—and evaluates estimators on bias, RMSE, confidence interval coverage, and other metrics. The goal is to characterize when each estimator breaks down and to guide estimator selection for time-to-event endpoints in practical trial analysis.

## Quick Start

```bash
git clone <repo>
cd causal_bench
pip install -e ".[dev]"
python -m causal_bench --scenario clean --n-sims 20 --estimators naive km cox tmle_ipcw tmle_ipcw_comply
```

## Estimator Table

| Key | Name | Method | When It Fails |
|-----|------|--------|---------------|
| naive | Naive | Complete-case difference in means | Always under informative censoring |
| km | KM | Kaplan-Meier risk difference | Ignores covariates |
| cox | Cox PH | G-computation via Cox model | Informative censoring |
| tmle_ipcw | TMLE+IPCW | Targeted learning with IPCW | Extreme positivity violations |
| tmle_ipcw_comply | TMLE+IPCW+Comply | TMLE+IPCW with compliance in censoring model | Only helps when compliance predicts censoring |

## Key Findings

*(Run experiments to populate this section)*

- Under informative censoring (censoring_informativeness ≥ 0.6), naive and KM estimators show substantial bias
- TMLE+IPCW recovers unbiased estimates when censoring is at random (MAR)
- Including compliance in the censoring model (TMLE+IPCW+Comply) further reduces bias under MNAR censoring

## Available Scenarios

| Scenario | Description |
|----------|-------------|
| clean | No informative censoring, no positivity violations |
| censor_mild | Mild informative censoring |
| censor_moderate | Moderate informative censoring |
| censor_severe | Severe informative censoring |
| positivity_mild | Mild positivity violations |
| positivity_moderate | Moderate positivity violations |
| positivity_severe | Severe positivity violations |
| unmeasured_mild | Mild unmeasured confounding |
| unmeasured_mod | Moderate unmeasured confounding |
| unmeasured_strong | Strong unmeasured confounding |
| edwards_realistic | Edwards et al. realistic scenario |
| edwards_optimistic | Edwards et al. optimistic scenario |
| edwards_pessimistic | Edwards et al. pessimistic scenario |

## CLI Reference

```
python -m causal_bench [OPTIONS]

Options:
  --scenario       Named DGP scenario (default: edwards_realistic)
  --n-sims         Number of Monte Carlo replicates (default: 100)
  --n-jobs         Parallel workers, -1 = all CPUs (default: -1)
  --estimand       Target estimand: ATE or ATT (default: ATE)
  --estimators     Space-separated estimator keys to run
  --seed           Random seed (default: 42)
  --out-dir        Output directory (default: results/)
  --no-plots       Skip plot generation
```

Results are saved to `<out-dir>/<scenario>/summary.md` and `forest.png`.

## References

- van der Laan, M. J., & Rose, S. (2011). *Targeted Learning: Causal Inference for Observational and Experimental Data*. Springer.
- Robins, J. M., Hernán, M. A., & Brumback, B. (2000). Marginal structural models and causal inference in epidemiology. *Epidemiology*, 11(5), 550–560.
- Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
