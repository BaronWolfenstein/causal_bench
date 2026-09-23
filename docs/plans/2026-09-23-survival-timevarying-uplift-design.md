# Survival variant of exp45 — time-varying treatment + time-to-event outcome (design)

**Status: DESIGN, gated / not built.** Captures the scenario + DGP + method so it's turnkey when a real
sequential-intervention *retention* question motivates it. Fuses exp45 (sequential treatment + feedback,
scalar outcome) with exp51 (survival CATE, point treatment). Gated on the longitudinal-survival g-machinery
build + the CONCRETE learner fix (#223) for the DR survival part.

## Motivation — Figma use case
A *sequential* lifecycle program on new teams, with churn (time-to-event) as the outcome:
- **A0** — week-1 intervention (guided tutorial / starter-template push / "invite your team" prompt).
- **L1** — engagement after week 1 (files created, collaborators invited, DAU). **Treatment-affected
  confounder**: A0 raises L1, and L1 both *drives who gets the next nudge* and *predicts churn*.
- **A1** — week-4 re-engagement nudge, **targeted by L1** (low-engagement teams get the retention offer → L1
  confounds A1→churn).
- **T** — time until the team churns (cancels / goes dormant), with **censoring**: administrative (study ends)
  and *competing* (upgrade to Enterprise = leaves the cohort; can reuse exp53's competing-risks structure).

**The trap** (same as exp45, now on a survival outcome): L1 is simultaneously a *mediator* of A0 (the week-1
nudge works partly by raising engagement) and a *confounder* of A1. Adjusting for L1 in the outcome model
blocks A0's benefit-through-engagement; not adjusting biases A1. → needs g-methods, not ordinary adjustment.

## DGP (sketch — concrete-schema compatible, extends survival_uplift + longitudinal_cointervention)
```
W ~ N(0,1)
A0 ~ Bern(expit(γ·W))
L1 = b_L0·A0 + b_LW·W + noise                       # engagement, affected by A0
A1 ~ Bern(expit(a0 + a_L·L1))                        # re-nudge targeted by (low) engagement
log λ_churn = β0 + βW·W + βL·L1 + ψ0·A0 + ψ1·A1      # churn hazard; sequential treatment effects
T ~ Exp(λ_churn);  C ~ informative (in W or L1);  [optional competing upgrade cause 2]
T_obs = min(T, C, τ);  event ∈ {0 censored, 1 churn, (2 upgrade)}
```
Self-validating: interventional MC truth for the regime survival curve / RMST (set A0=a0, A1=a1, integrate).

## Estimand
Per **dynamic regime** ā=(a0,a1): the survival curve S(t|ā), the **RMST** ("expected active-days over τ"), and
the regime contrast S(t|1,1)−S(t|0,0) / RMST(1,1)−RMST(0,0). Plus the A0 blip (total effect incl. the
L1-mediated path) and its modification — the survival analogues of exp45's ψ.

## Methods
- **Longitudinal-survival g-formula / ICE-survival** — sequential regression with a survival outcome (the
  survival analogue of exp45's `ice_regime_mean`), IPCW for censoring; the g-computation backbone.
- **Structural nested failure-time / survival-SNMM** — g-estimation of the sequential treatment blips on the
  hazard/RMST (the survival analogue of exp45's `g_estimation`), recovers the A0 total effect + effect
  modification. This is beyond the repo's **two-timepoint** `LTMLEEstimator` (which exp45 already flags cannot
  represent multi-period exposure). Multi-period survival-LTMLE is the DR target (phase 2).
- **Baseline (biased)** — naive survival model adjusting L1 in the outcome (blocks A0's mediated effect), and a
  per-arm KM that ignores the sequential structure — the "why g-methods" contrast, mirroring exp45's
  `naive_effects`.

## Aspirational endpoint
**Optimal dynamic treatment regime on a survival outcome** — *when* to send the second nudge and *to whom*
(as a function of L1) to maximize retention time. A sequential policy learned/evaluated on time-to-event —
the senior causal-DS artifact. Policy value via the DR survival machinery (ties to exp52's AIPW policy value,
lifted to survival).

## Gates / un-gate condition
- **#223** — the DR survival part (survival-LTMLE / CONCRETE) inherits the default-learner attenuation + OOM.
- **Build** — the longitudinal-survival g-methods (survival-SNMM / ICE-survival) don't exist in the repo yet.
- **Un-gate when**: a real *sequential*-intervention retention dataset (or a strong portfolio reason) justifies
  the build. Until then, exp45 (sequential/scalar) + exp51 (survival/point) + exp53 (competing risks) cover
  the pieces separately, and this is the "fuses them" story to *describe*, not build.

## Relation
Fuses exp45 (`longitudinal_cointervention.py`) + exp51 (`survival_uplift.py`); competing-exit option reuses
exp53 (`competing_risks.py`); policy-value endpoint reuses exp52 (`uplift_policy.py`). Concrete-RMST fixes:
`docs/plans/2026-09-23-concrete-rmst-fixes.md` + #223.
