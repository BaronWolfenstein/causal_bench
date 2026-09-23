# Concrete-RMST fixes — spec (2026-09-23)

> **CORRECTION (2026-09-23, later same day).** Empirical investigation overturned this doc's central claim.
> **Fix 2 (de-regularize the survival hazard learners) is UNNECESSARY** — introspection shows the hazards are
> already unregularized `Lrnr.Cox`; only the *propensity* is `SL.xgboost/glmnet`, and swapping it to `SL.glm`
> is a no-op (0.343→0.345). The real attenuation had two other causes: **(1)** an `Intervention=c(1L,0L)`
> slot-index bug in `run_concrete_bridge` (slot 0 doesn't exist; should be `c(1L,2L)`) + the missing dense
> TargetTime grid — together **0.221 → 0.343**; and **(2)** a positivity/overlap residual that grows with n,
> fixed by `MinNuisance` truncation — see `2026-09-23-concrete-positivity-fix.md`. Fix 1 (parse the `RMST Diff`
> row) below is still correct. Treat the "Fix 2" learner section here as **superseded**; kept for provenance.
> All tracked under #223 (retitled).


**Status: DEFERRED / gated.** The Python grid-TMLE RMST (`survival_uplift.rmst_tmle_cate`) is the working default
for exp51 (−0.05 additive bias, fast, no OOM). Pursue the concrete-native RMST **only if** a continuous-time
RMST becomes load-bearing on a real dataset. This spec records the three fixes and the diagnostic evidence so
the work is turnkey when/if needed.

## Motivation
exp51 wants a doubly-robust **RMST-difference** ("extra retention-days", time units) CATE. Two routes:
- **grid-TMLE** (`rmst_tmle_cate`): ∫₀^τ [S₁(t)−S₀(t)]dt via TMLE-IPCW survival-diff over a horizon grid. Works
  (V=0 est 0.41 vs truth 0.46; V=1 0.83 vs 0.88). Small **additive** bias −0.05 (statistical, *not* trapezoid —
  a 12-pt grid gave the same −0.05 as 6-pt). Fast, memory-safe.
- **concrete-native** (`getRMST`): continuous-time, no discretization — *should* be the principled one, but three
  problems below make it currently **worse** (RMST-diff 0.221 vs truth 0.461, ~half).

## Diagnostic evidence (2026-09-23 rabbit hole)
`getRMST(est, Horizon, Intervention=c(1L,0L))` returns a `ConcreteOut` data.table with columns
`Intervention, Estimand, Estimator, Event, Time, Pt Est, se, CI Low, CI Hi, pValue` and rows:
```
  A=0            RMST              1.843
  A=1            RMST              2.065
  [A=1]-[A=0]    RMST Diff         0.221   <- pre-computed contrast (truth 0.461)
  [A=1]-[A=0]    LYL Diff         -0.221
  A=0            Life Years Lost   1.147
  A=1            Life Years Lost   0.925
```
Control arm correct (1.84); **treated arm under-estimated** (2.07 vs ~2.26) → attenuated benefit. Default
`args$Model` learner library = **`SL.xgboost, SL.glmnet`** (both regularized) → multiplicative attenuation on
n≈4k. (Contrast: the grid-TMLE uses unregularized `LinearRegression` nuisances → additive bias only. Same
estimand, two estimators, two bias mechanisms — a clean additive-vs-multiplicative illustration.)

## The three fixes

### Fix 1 — parse the RMST Diff row (trivial, non-breaking)
In `r_scripts/concrete_bridge.R`, after the existing LYL extraction, add (additively — existing `ATE`/`SE`
untouched): extract the row with `Estimand == "RMST Diff"` and `Intervention == "[A=1] - [A=0]"`, return its
`Pt Est`/`se` as `RMST_ATE`/`RMST_SE`. In `estimators/concrete_rmst.py`, add `rmst_contrast: bool = False` to
`ConcreteRMSTEstimator.__init__` and, when set, read `RMST_ATE`/`RMST_SE` instead of `ATE`/`SE`. Sign: RMST Diff
= treated−control = positive survival benefit (NO negation, unlike the LYL/risk contrast). *(This part was
prototyped and verified to extract 0.221; reverted pending Fix 2 so we don't ship an attenuated number.)*

### Fix 2 — de-regularized survival learners (the accuracy fix; the real work)
The attenuation is the default SL library (`SL.xgboost, SL.glmnet`). Pass `Model=` to
`concrete::formatArguments` with a **less-regularized, survival-capable** hazard library. **`SL.glm` does NOT
work** ("All hazard learner candidates failed for event type") — concrete's *hazards* need survival learners,
not a plain binomial GLM. To do:
- Enumerate concrete's valid hazard-learner names (its `SL`-style survival wrappers; check `concrete`'s learner
  registry / vignette — candidates: a Cox/parametric survival learner, or `SL.glmnet` with a low/tuned penalty).
- Keep propensity (`"A"`) on a simple learner (there `SL.glm` is fine); relax only the hazard libraries.
- Verify RMST-diff → truth (0.46/0.88) on the exp51 DGP, multi-seed.
Model keys observed: `names(args$Model) = c("A","0","1")` (propensity + per-arm/event hazards); a stray `"C"`
key errored, so match the default keys exactly.

### Fix 3 — memory ceiling
concrete `doConcrete` **OOMs at n≳24k** ("vector memory limit of 48.0 Gb"). So you can't out-run Fix-2's
finite-sample regularization by scaling n on the current build. Options: (a) `mem.maxVSize()` raise if RAM
allows; (b) subsample/chunk per stratum; (c) fewer CV folds / a lighter SL library (couples with Fix 2);
(d) await a leaner concrete build (flag to McCoy — `blind-contours/concrete`).

## Verification / done-criterion
Multi-seed calibration of `ConcreteRMSTEstimator(rmst_contrast=True)` per stratum on the exp51 DGP recovers the
RMST-diff truth (0.461 / 0.876) within a few %, at a memory-feasible n. Then wire it as the CONCRETE column of
exp51's RMST table alongside the grid-TMLE.

Tracked as **issue #223** (default-learner attenuation + de-regularization + OOM).

## Gated follow-up experiments (blocked on #223 / this spec)
1. **Competing-risks retention RMTIF** — restricted mean time in the ACTIVE state under confounding, with
   competing exits (churn / upgrade-to-Enterprise / downgrade). This is *the one* case where continuous-time
   concrete is genuinely load-bearing for Figma: **grid-TMLE cannot represent time-in-state with competing
   exits.** Runs on `ClinicalRMTIFEstimator`.
   **Gating (CORRECTED 2026-09-23):** the old "gated on Fix 2 (de-regularize learners)" is void — Fix 2 was
   falsified. The real caveat is that RMTIF inherits the **residual-confounding contrast-compression** #223
   found (per-arm: control over, treated under; not positivity, not learners) — so it needs the *adjustment*
   fixes (force the confounders into the outcome hazard `Surv ~ .` not `Surv ~ A`; CV-TMLE), and it should ship
   with the **Kish-ESS overlap guardrail** (exp52 `policy_value_ess`; exp55/#224) so a poor-overlap regime is
   flagged rather than silently trusted. Even with those, concrete's reliability under confounding+overlap is
   in question, so validate against interventional MC truth before trusting it. NOTE: the cheap cause-specific-
   vs-naive teaching version (naive-treats-competing-as-censoring bias) needs NO concrete and is DONE —
   **exp53** (`validation/competing_risks.py`, Aalen–Johansen vs 1−KM, self-validating).

   **Figma datasets with *real* competing risks** (mutually-exclusive terminal transitions where one precludes
   the others AND the intervention shifts the competing-event rate — so treating it as censoring biases, per
   exp53). Ordered by cleanliness:
   - **Upgrade vs churn** (canonical): an onboarding/nudge raises retention *and* upgrade; upgrading removes a
     team from the churn risk set, so counting upgrades as censored inflates churn incidence and biases the
     effect. RMTIF = expected active-paid-days over the horizon.
   - **Free-trial: convert vs abandon** — the trial ends in paid-conversion or dormancy; a conversion nudge
     shifts both. Competing, not censoring.
   - **Seat expansion vs contraction vs full cancellation** — competing plan-size transitions; RMTIF in the
     growing/stable-seats state.
   - **Absorbed-by-acquisition vs independent churn** — an acquired customer is consolidated onto the
     acquirer's plan (leaves the cohort for a *non-churn* reason) vs churns on its own; the acquisition exit
     must not be scored as churn.
   - **First cross-surface adoption: FigJam vs Dev Mode vs neither** — competing first-adoption events; RMTIF =
     active time before a team lands on its first adjacent product.
   - **Renewal vs non-renewal vs early-termination** at contract boundaries — the active-contract RMTIF.
   The estimand where continuous-time concrete is load-bearing is the **RMTIF (restricted mean time in the
   ACTIVE state)** — e.g. "expected active-paid-days over the year, accounting for competing exits" — which
   grid-TMLE cannot represent.
2. **Survival variant of exp45 (time-varying treatment + time-to-event outcome)** — do NOT bolt onto exp45.
   exp45 is the point-outcome linear-SNMM case; a survival version (W→A0→L1→A1→T with informative censoring)
   needs longitudinal-survival g-methods (survival-SNMM / structural nested failure-time, or multi-period
   survival-LTMLE) beyond the repo's two-timepoint `LTMLEEstimator` — a NEW experiment and a bigger build.
   Also couples to the continuous-time DR survival machinery, so partially gated on #223 too. File as its own
   follow-up when a real time-varying-treatment survival question motivates it.

## Decision
Grid-TMLE stays the exp51 RMST default. The concrete fixes are a ~1-day project (mostly Fix 2's learner spec +
a memory workaround); do them only when continuous-time RMST accuracy on a real (non-simulated) survival
dataset justifies it, or bundle as a request to McCoy (learner defaults + memory) since it's his fork.
