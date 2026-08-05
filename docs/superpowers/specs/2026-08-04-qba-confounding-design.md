# exp49 — probabilistic QBA for unmeasured confounding

Issue #205. Adds probabilistic quantitative bias analysis for uncontrolled confounding
to the estimand-side sensitivity suite (which had M-bias, selection-into-R, and
Σ_ε/Σ_x measurement-error, but not QBA).

## Method

Fox, MacLehose & Lash (IJE 2023;52:1624), closed-form **v1** (n-independent
omitted-variable form). For a linear outcome the omitted-variable bias is exact:
`naive_beta_A = oracle_beta_A + beta_U * gamma`, where `beta_U` is the U→Y coefficient
and `gamma` is the A-coefficient of `U ~ A + X`. QBA samples the bias parameters
`(beta_U, gamma)` from priors (triangular / normal), giving the systematic-error
distribution; subtracting a random-error draw gives the total-error interval.

## DGP (`causal_bench/validation/qba_confounding.py`)

Measured covariates `X` (3-D); an UNMEASURED binary confounder `U` correlated with X and
driving both treatment `A` and a continuous linear outcome `Y`; true ATE `tau = 1`.

## The experiment (not a point adjustment): calibration + misspecification

Over replicates, measure coverage of the true `tau` by the systematic- and total-error
intervals under three prior regimes:
- **correct** — priors centered on the truth → recover (`adj_bias ≈ 0`) and cover.
- **misspec_half** — underestimate U's effect by half → partial correction, undercover.
- **misspec_null** — assume U harmless → stays at the naive bias, ~0 coverage.

Same discipline as exp41 borrowing-calibration: a bias tool is trustworthy only if its
interval is calibrated, and QBA's is exactly as good as its priors.

## Claims pinned by `tests/test_qba_confounding.py`

1. oracle (U measured) recovers tau; naive (U omitted) is biased;
2. correct priors → `adj_bias ≈ 0` and total-interval coverage;
3. misspecification (assume-U-harmless) undercovers and stays at the naive bias.

## Scope / next

v1 (closed-form, linear) here. **v2 (gated):** record-level, estimator-agnostic QBA via
an influence-function one-step update (avoids per-iteration refit), to certify QBA for
the nonlinear TMLE/AIPW-ML estimators at OC-sim scale — a small methods contribution.
Survival (HR/RMST) QBA composes with `concrete` later. Complements #206 (adjustment-set
validity) and Σ_ε/Σ_x; QBA bounds the residual a valid adjustment set can't capture.

**Relation to existing experiments.** `exp3_unmeasured` *demonstrates* that estimators
fail under unmeasured confounding (the honest null); exp49 is the QBA *tool that responds*
to exactly that failure (quantify + bound it). `exp40_bias_amplification` is the adjacent
OVB phenomenon. exp49 does not duplicate these — it adds the correction/sensitivity that
was missing.
