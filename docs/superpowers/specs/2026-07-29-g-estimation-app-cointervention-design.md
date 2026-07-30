# exp45 — estimators for the time-varying app co-intervention: multi-period LTMLE vs g-estimation

**Date:** 2026-07-29
**Issues:** #71 (app-followed cohort / app co-intervention), #144-adjacent (methods bake-off).
**Reference:** Loh & Ren, "A Tutorial on Causal Inference in Longitudinal Data With Time-Varying
Confounding Using G-Estimation," *AMPPS* 2023 (10.1177/25152459231174029); Naimi, Cole &
Kennedy, "An introduction to g methods," *IJE* 2017; Vansteelandt & Joffe 2014. See memory
`reference_dynamic_borrowing_conflict` is unrelated; this is the g-methods thread.

**Status:** design only — no implementation until approved and a plan is written.
**Independent of** the exp41 composite-null box run (different thread entirely).

---

## Motivation — a gap, not just an alternative

The app co-intervention (#71) is a **genuinely multi-period time-varying treatment**: the app
nudges / engages the patient repeatedly over follow-up (A₀, A₁, …, A_T), and app engagement
affects health state / adherence (time-varying confounders L_t) which affect *future*
engagement and the outcome — **treatment-confounder feedback**.

The repo's `LTMLEEstimator` is **two-timepoint**: a *fixed baseline* treatment A plus a
*single* intermediate confounder L1 (at t_L1=0.5), IPCW-weighted. It correctly handles the
one-mediator collider (its docstring: "avoids collider bias"), but it **cannot represent a
multi-period time-varying treatment** — there is no A₀…A_T, no per-period blip. So for the app
exposure, LTMLE is not merely less convenient; it is **structurally inapplicable**.

Two **doubly-robust** routes close that gap, targeting **different estimands** — so this is
not either/or on correctness, it is which question:

- **Multi-period LTMLE** (van der Laan–Gruber longitudinal TMLE): sequential regression over
  the time-ordered nodes (A₀, L₁, A₁, …, Y) with a targeting step at each, estimating the
  **mean outcome under an app *regime*** E[Y_ā] (e.g. high- vs low-engagement policy). It
  **extends the repo's existing two-timepoint LTMLE pattern** and plugs `SuperLearner` — the
  more codebase-continuous route, and the regime-contrast is arguably the more decision-relevant
  estimand for "deploy the app policy."
- **G-estimation of a linear SNMM** (Robins; Vansteelandt & Joffe): the blip-down procedure
  targeting the per-period **structural blip ψ** — the effect of an app increment given history,
  natively accommodating **effect modification by time-varying covariates** (the app helping
  more/less depending on evolving engagement), which a marginal-regime target does not directly
  surface.

Because causal_bench *is* a benchmark, exp45 **evaluates both** on one controlled DGP —
demonstrating the estimand distinction, not crowning a winner. Multi-period LTMLE is the
primary for the regime question; g-estimation is the effect-modification complement.

## Estimands (two, one per route)

- **Regime mean** E[Y_ā] and regime contrasts (multi-period LTMLE): the expected outcome under
  a static or dynamic app-engagement regime — the policy question.
- **SNMM blip ψ** (g-estimation; Robins, Vansteelandt & Joffe): the conditional average effect
  of the app increment at each period given history, holding later exposure at a reference, and
  — with covariate×blip interactions — its **modification** by time-varying covariates.

Self-validating: a linear-additive SNMM DGP has **closed-form ψ** *and* closed-form regime
means, so both estimators are checked against a known truth (as in Loh & Ren).

---

## Part A — methodological core (general)

### Components
1. **DGP: multi-period time-varying app exposure with feedback** —
   `causal_bench/dgp/app_cointervention.py` (or extend `user_sim.py`, the existing app-engagement
   sim). Periods t=0,1,2: app exposure A_t (∈{0,1} or continuous engagement) drawn from history
   (A_{t-1}, L_{t-1}); time-varying confounder L_t (health state / adherence) with feedback
   (A_{t-1}→L_t, L_t→A_t, L_t→Y); outcome Y at horizon. **Linear-additive blip ψ known in
   closed form**; optional covariate×blip interaction for the effect-modification arm.
2. **Estimator (primary): multi-period LTMLE** — `causal_bench/estimators/ltmle_longitudinal.py`
   (generalizes the two-timepoint `LTMLEEstimator`). Sequential regression / iterated conditional
   expectations over (A₀, L₁, A₁, …, Y) with a targeting step at each time (clever covariate =
   cumulative inverse propensity of the regime), `SuperLearner` nuisances. Estimates E[Y_ā] and
   regime contrasts. Reuses the repo's crossfit + SuperLearner + IPCW machinery — the continuous
   route.
3. **Estimator (complement): g-estimation of the linear SNMM** —
   `causal_bench/estimators/g_estimation.py`. Blip-down / "peel-off" (sequential, **latest period
   first**): estimate the last blip, subtract to form the counterfactual-absent-later-treatment
   outcome, regress on earlier exposure adjusting for pre-treatment history. **Doubly robust**
   with a per-period propensity model. **Bootstrap SE** (Loh & Ren: the naive last-stage SE
   under-covers — it ignores first-stage blip-estimate uncertainty). Optional: the
   residual-correlation **ρ-sweep** unmeasured-confounding sensitivity (an *alternative* to the
   repo's E-value/M-bias tools, not a replacement).
4. **Comparison / the demonstration** — a four-way bake-off on the multi-period DGP:
   - **Naive** (regress Y on A with L_t as covariates) is biased — conditioning on the feedback
     confounder L_t blocks part of the effect / opens a collider (Ch-3, `exp5`).
   - **Two-timepoint `LTMLEEstimator`** — reported *not applicable* (or applied to a collapsed A
     and shown to miss the per-period structure): the gap made explicit, not a horse-race it was
     never built for.
   - **Multi-period LTMLE** recovers the regime means / contrasts, doubly-robustly.
   - **G-estimation** recovers ψ incl. the effect-modification interaction — the piece the
     regime-mean does not directly surface.
   The read-out is the **estimand distinction** (regime-mean vs blip vs the biased naive), the
   causal_bench thesis that the right estimator depends on the question.

### Tests
- **Multi-period LTMLE** recovers the closed-form regime means / contrast within CI.
- **G-estimation** recovers the closed-form ψ within CI.
- **Double robustness** (both): consistent when the outcome model is misspecified but the
  treatment/PS model is right, and vice versa (two arms each).
- **Effect-modification arm**: g-estimation recovers the covariate×blip interaction; the LTMLE
  regime-mean does not surface it directly — the estimand distinction, made concrete.
- **Naive** is biased under feedback; **two-timepoint LTMLE** cannot represent the multi-period
  exposure (the gap).
- Bootstrap SE ~95%; the naive last-stage g-est SE under-covers (the Loh-Ren point).

### exp45 driver
`experiments/exp45_app_cointervention.py` — self-validating report (closed-form regime means +
ψ), the four-way comparison, and the DR + effect-modification arms. **Verify the number is still
free at build time** (`ls experiments/` + `gh issue list`); exp45 is next-free-forward after
exp44 as of 2026-07-29.

### Build phasing
The DGP + **multi-period LTMLE** + naive comparison is phase 1 (extends existing machinery, the
regime estimand). **G-estimation + the effect-modification arm** is phase 2 (the new paradigm).
Each is its own PR; both land under this spec.

---

## Part B — ENCIRCLE / #71 application layer (**clearly marked; specified, not built here**)

> How the Part-A method transfers to the ENCIRCLE app-followed cohort. **Not** part of the
> exp45 deliverable.

- The app co-intervention *is* this structure: the app's **logged actions** (the front-door
  logging #71 already requires — "app must log its own actions") supply the per-period exposure
  A_t; **both estimators** consume them as the time-varying treatment.
- Estimand for #71: **either** the outcome under an app-engagement **regime** (multi-period
  LTMLE — the "deploy the policy" question) **or** the app's **structural effect + its
  modification by engagement** (g-estimation) — both doubly robust, richer than one marginal
  number. Which leads depends on the clinical/payer question.
- For the **device** analysis, the two-timepoint `LTMLEEstimator` stays correct (one-time
  implant + a single mediator L1 — `exp5`); the multi-period estimators are for the **app**
  exposure only.
- The app-as-**second-comparator** (IPCW-light, #71's main body) is a *separate* estimand from
  the app-**exposure-effect** this spec targets; both can coexist.

## Out of scope (future)

- The full #71 second-comparator (IPCW-light external cohort) build.
- Formal front-door mediator *removal* of the app's effect from the device estimand (distinct
  identification strategy).
- Continuous-time / irregular-visit exposure (discrete periods here).
- Interim monitoring.

## Non-goal / honest scoping

This does **not** replace the existing two-timepoint `LTMLEEstimator` anywhere in the repo — it
stays the device / single-mediator tool. What's added is **multi-period** capability for the
app's time-varying exposure, via **two complementary estimators** targeting different estimands
(regime-mean LTMLE and structural-blip g-estimation). The "replace LTMLE" framing is explicitly
rejected; the two-timepoint LTMLE is *generalized* (multi-period LTMLE) and *complemented*
(g-estimation), not swapped out.
