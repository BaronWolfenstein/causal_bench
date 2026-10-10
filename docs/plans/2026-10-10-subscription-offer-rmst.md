# Baseline retention offers — causal extension to exp59

The prediction benchmark remains unchanged. This extension uses its monthly
hazard machinery to estimate the causal gain in restricted active months from
assigning an offer at eligibility. It connects exp51's baseline-intervention
question to explicit billing-month conventions. It is not a new implementation
of exp54's sequential-treatment setting or an industry deployment.

## Estimand and identification

`Delta_H = E[min(T^1,H)] - E[min(T^0,H)]`, positive for benefit.
Cancellation is absorbing and occurs at month ends, so each arm's restricted
mean is `sum(S^a(k), k=0..H-1)`. Units are months per eligible subscriber assigned
an offer, including subscribers who decline it (an assignment/ITT contrast).
No full-lifetime extrapolation, LTV conversion or redemption effect is implied.

Baseline columns: one unique `subscription_id`, `cohort`, binary offer `A`,
pre-offer `channel`, `plan`, `engaged`, `observed_time` and cancellation `event`.
The current specifications use these named predictors; new data requires an
explicit feature specification, not arbitrary post-treatment columns.

Identification assumes consistency/no interference, treatment exchangeability
conditional on baseline predictors, positivity and independent observation
censoring conditional on baseline and A. Randomized scenarios use the known
assignment probability .5; observational scenarios estimate it. Covariate
adjustment cannot repair unmeasured confounding. No delayed entry, repeated
offers, competing exits, reactivation or shared-account dependence is supported.

## Estimator

`estimate_offer` in `estimators/monthly_offer.py` is cross-fitted one-step/AIPW,
not TMLE. It reuses the exact binomial hazard likelihood and supports aggregated
outcome and censoring nuisance fits. Subscribers, not person-months, are assigned
to folds. These causal estimation folds are within an eligible population; they
do not replace the prediction experiment's future-cohort evaluation.

For each arm a and held-out subscriber i:

```
m_a(X) = sum_{k=0}^{H-1} S_a(k|X)
q_aj(X) = sum_{k=j}^{H-1} S_a(k|X) / S_a(j|X)
U_ai = m_a(X_i) - I(A_i=a)/g_a(X_i) *
       sum_{j=1}^{H-1} q_aj(X_i) R_ij/G_a(j|X_i) * (D_ij-h_aj(X_i))
Delta_hat = mean(U_1 - U_0)
```

`R_ij` means the month-end outcome is observed and cancellation has not occurred
earlier; `D_ij` indicates cancellation at j. q is evaluated backward without
dividing by small event-survival probabilities. The pseudo-outcome expectation
is correct if the outcome hazards are correct OR both assignment and censoring
nuisances are correct, subject to identification and positivity.

Observation dropout has its own interval likelihood. Censoring at 2.4 contributes
two observed cancellation outcomes and a dropout in interval 3. It does not
contribute a cancellation-negative label in interval 3. Administrative censoring
at H is not dropout before H. Cancellation takes precedence at integer ties.

The explicit propensity/observation-survival floor is .01; clipping counts and
unclipped minima are reported. Clipping active true probabilities can compromise
the consistency claim, so it is not hidden as a numerical fix. No clipping was
needed in the executed default scenarios.

Fully specified fits report empirical influence-function Wald intervals under
consistent nuisances/regularity and iid subscribers. Deliberately misspecified
arms report point estimates only: double robustness of a point estimate does
not automatically validate a plug-in IF variance under nuisance misspecification.
Account clustering and such inference would need additional methods.

## Simulation and verification

Four fixed scenarios: randomized/confounded assignment crossed with a beneficial
or null offer. Baseline predictors are binary; cohorts span 24 calendar months.
The hazard has tenure effects, periodic seasonality, plan-by-age and offer-by-age
effects. Loss to observation is a baseline/offer-dependent monthly Bernoulli
dropout before the cancellation outcome, matching the censoring likelihood.
Everyone has administrative support through H. All oracle quantities are returned
separately from observed fitting data.

Population truth is computed exactly over all eight baseline types and 24 cohorts.
Models are prespecified. Compare fully adjusted AIPW, correct-outcome-only AIPW,
correct-treatment/censoring-only AIPW, and naive unadjusted monthly hazards. No
evaluation outcomes choose an estimator. This is population uplift, not CATE ranking.

Executed `n=6000`, 20 seeds per scenario, H=12, three folds:

| Scenario | True months | Adjusted bias | Naive bias | Adjusted empirical coverage |
|---|---:|---:|---:|---:|
| Randomized benefit | .8662 | -.0096 | -.0464 | 19/20 |
| Randomized null | .0000 | .0046 | .0070 | 19/20 |
| Confounded benefit | .8662 | -.0261 | -.1765 | 20/20 |
| Confounded null | .0000 | -.0125 | -.1426 | 20/20 |

Small Monte Carlo samples do not establish 95% coverage. Wilson coverage intervals
are reported, including 20/20. JSON includes every seed, estimate, interval,
nuisance diagnostic, configuration, package version and source SHA-256.

Tests independently enumerate all event/dropout/assignment outcomes to establish
the double-robust pseudo-outcome identity; check a counterexample with both groups
wrong; audit fold separation, partial months, integer ties, H=1, broadcasting
rejection, oracle separation, known-truth recovery and the strict horizon option.

## Optional concrete integration

`--concrete` runs a separately labeled continuous-time exponential DGP with its
own analytic conditional RMST truth. This exercises the existing
`ConcreteRMSTEstimator(rmst_contrast=True)` path: positive means extra months,
without the legacy helper's negation. It uses 24 integration grid points, three
folds and a 120-second subprocess budget per companion. It is an API/numerical
comparator, not validation of exact monthly integration or resolution of #223.

New opt-in `horizon_policy="strict"` rejects horizons at/after the last observed
event; it never substitutes a shortened estimand. Both Python and R enforce it.
The R result exposes `effective_horizon`, checked again in Python. Legacy callers
keep `legacy_cap` behavior and receive a warning if the horizon changes. Follow-up
extends to 2H in the companion; no event time is fabricated to bypass the gate.

A live n=300 randomized-benefit smoke exercised the strict bridge: estimated
1.0682 months (SE .4571) versus its conditional sample oracle 1.2830 months.
Cox nuisance fits emitted small-sample separation warnings. This is a bridge
smoke, not a precision/coverage claim; the native monthly n=6000 study is separate.
The initial larger 96-grid exploratory bridge run was interrupted because of its
runtime; no incomplete result from it is reported as successful evidence.

## Run

```sh
PYTHONPYCACHEPREFIX=/private/tmp/causal_bench_hazard_pycache \
MPLCONFIGDIR=/private/tmp/causal_bench_mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv312/bin/python -m experiments.exp59_subscription_offer --n 6000 --seeds 20
```

Add `--concrete` for the optional R companion; unavailability/failures are recorded,
not replaced with Python output. The native default requires NumPy/SciPy/pandas
and no R or GPU. Artifacts are under `results/exp59_subscription_offer/` (gitignored).
Very small n can produce separated unpenalized nuisance fits and correctly fail;
there is no hidden ridge or all-surviving fallback.

Verification commands:

```sh
.venv312/bin/python -m pytest tests/test_subscription_offer.py tests/test_monthly_hazard.py \
  tests/test_concrete_bridge.py tests/test_concrete_config.py -q
Rscript tests/test_concrete_horizon.R
```

Related literature: [Bai, Tsiatis and O'Brien, contrasting treatment-specific survival](https://pmc.ncbi.nlm.nih.gov/articles/PMC4180679/);
[official concrete package and RMST API](https://github.com/blind-contours/concrete).
