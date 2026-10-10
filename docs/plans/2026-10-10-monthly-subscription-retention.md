# Monthly subscription retention — exp59

Implemented as a Python-only prediction baseline, independent of McCoy's R
`concrete` fork and issue #223. It adapts exp34's interval-specific logistic
hazard and cumulative-product survival approach. Existing exp34/51/54 estimator
behavior is unchanged. This baseline is not a new TMLE or causal effect estimator.

## Run

From the repository root, with the existing Python 3.12 environment:

```sh
PYTHONPYCACHEPREFIX=/private/tmp/causal_bench_hazard_pycache \
MPLCONFIGDIR=/private/tmp/causal_bench_mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv312/bin/python -m experiments.exp59_subscription_retention \
  --n 24000 --seed 0 --bootstrap-reps 100
```

Outputs: `results/exp59_subscription_retention/REPORT.md` and `metrics.json`.
The latter includes configuration, package versions, source hashes, split ranges,
selection criterion, coefficient/model-based-SE results, calibration bins,
monthly Brier results, bootstrap successes/failures and aggregation checks.
Results are ignored by git, following the existing experiment convention.

## Observation and prediction contract

- One subscription row: `subscription_id`, integer acquisition `cohort`,
  `observed_time` in months, binary `event`, baseline predictors. Optional
  integer `entry_age` supports delayed-entry person-month construction; the
  simulation starts everyone at acquisition. Entry selection is not audited here.
- Cancellation is absorbing and at integer month ends. No reactivation or
  competing exits. An observed cancellation at 3 gives outcomes `[0, 0, 1]`.
- Censoring at 2.4 gives only months 1 and 2; the third outcome is unknown.
  Censoring at 2 gives two fully observed months. An event tied to a censoring
  boundary is observed, consistently with the simulation and evaluation rules.
- Predictors are acquisition-time binary channel, plan and engagement band.
  No future engagement or cancellation metadata enters the model design.
- Four prespecified candidates: tenure only; tenure plus covariates; those
  terms plus periodic calendar seasonality; additionally a plan-by-age term.
  Full age indicators have no redundant intercept. Seasonal terms are sine/cosine
  of calendar month, avoiding unrestricted age/period/cohort fixed-effect aliasing.
- Select by observed validation person-month log loss only. Freeze that choice
  before test evaluation. Report every prespecified candidate. No calibrator is
  selected or fitted using test outcomes.

## Chronology and censoring

Subscribers are split before expansion:

| Partition | Acquisition cohorts | As-of cutoff |
|---|---|---|
| Train | 0–23 | 36 |
| Validation | 36–41 | 54 |
| Test | 54–59 | 72 |

Gap cohorts are not simulated. The gaps let all 12-month outcomes mature before
the next partition starts. This avoids a subtle leak in merely splitting by
acquisition date while using labels arriving after the next cohort's entry.

Loss to observation is exponential with a scale depending on baseline channel
and plan. It is independent of cancellation conditional on those covariates.
The monthly likelihood uses fully observed at-risk intervals without censoring
weights. Evaluation reports IPCW Brier and horizon calibration using the
**known simulation censoring law**, plus separate complete-outcome/oracle metrics.
The Horvitz–Thompson score denominator is all evaluation subscribers, not the
sum of observed weights. Administrative observation supports the full horizon
for every cohort. A real-data version must estimate and validate censoring
survival, enforce positivity and flag unsupported horizons; this experiment
does not establish those capabilities.

## Aggregation and uncertainty

Fit the unpenalized binomial likelihood with explicit Newton/line-search
convergence checks. No class balancing, hidden regularization or fallback to an
all-surviving curve. Unsupported/rank-deficient or numerically separated fits
raise errors. Sparse late-tenure pooling/smoothing is future work, not automatic.

Aggregate only exact matches on the complete design inputs (age, modeled
predictors and calendar phase). Compute events and trials, not an unweighted
regression on cell event proportions. Aggregation retains the likelihood up to
parameter-independent combinatorial constants, coefficients and inverse-information
covariance. The reported likelihood omits those constants for direct row comparison.
Changing continuous predictors to bands would be a different model.

Coefficient SEs are **model-based**, not a cluster-robust sandwich covariance.
Optional bootstrap resamples whole training subscriptions and refits the locked
specification, forecasting onto fixed test covariates. Report failed resamples;
only report percentile intervals with at least 20 successes. This excludes model
selection and test-population sampling uncertainty. Shared-account dependence
would require account-level resampling. No nominal-coverage claim is made.

## Restricted lifetime and relation to causal experiments

`S(k) = P(T > k)`, including `S(0)=1`. Cancellation at month ends implies
`E[min(T,H)] = sum(S(k), k=0..H-1)`. Do not apply a continuous-time trapezoidal
rule or extrapolate this restricted quantity into full lifetime/LTV without
additional tail, billing, discount and margin assumptions.

Exp51 supplies the baseline-intervention survival-uplift setting; exp54 supplies
treatment-confounder feedback under sequential interventions. Those are future
extensions of this monthly baseline, requiring explicit intervention timing,
identification assumptions and interventional simulation truth. `concrete` can
be an optional comparator if its own validation is resolved; neither it nor
#223 blocks this baseline. The existing exp34 person-time nuisance is not silently
patched by this change: its partial-interval treatment remains a separate audit.

## Verification

Tests compare the fitter with analytic saturated-binomial solutions and an
independent unpenalized scikit-learn solver, verify row/cell covariance equality,
check censoring/entry boundaries and rejected malformed shapes, and validate
discrete lifetime against enumerated event probabilities. A simulation-oracle
check verifies the IPCW evaluation score. Test-label perturbation must leave
validation selection and coefficients unchanged. A regression case covers
objective rounding in a final Newton step without weakening the score tolerance.

References: [discrete survival likelihood and censoring conventions](https://grodri.github.io/glms/notes/c7s6),
[binomial cell input conventions](https://www.statsmodels.org/v0.14.4/generated/statsmodels.genmod.generalized_linear_model.GLM.html).
