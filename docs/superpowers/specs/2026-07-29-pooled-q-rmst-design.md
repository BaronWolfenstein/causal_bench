# Pooled-Q subgroup RMST (#189) + optional RP-spline nuisance (#188)

**Date:** 2026-07-29
**Issues:** #189 (subgroup RMST via pooled Q), #188 (Royston-Parmar flexible-parametric
survival as a nuisance), #77 (single-arm subgroup estimands).
**Builds on:** PR #192 (`feat/pooled-q-subgroup`) — the landed pooled-Q *event-rate* core.

## Decision (route)

Route **A**: a self-contained **single-arm Python** pooled-Q RMST estimator, with RP-splines
as an **optional flexsurv nuisance backend**. NOT route B (concrete's two-arm ATE RMST + an
`sl3::Lrnr_flexsurvspline`). Rationale: pooled-Q exists for *single-arm* subgroup borrowing,
which concrete's two-arm contrast structurally cannot express; route A keeps the estimator
single-arm-safe and CI-green without the R stack, and makes RP a clean optional nuisance
rather than a hard R dependency. The landed core's docstring claim ("reuses the SAME pooled Q
via `concrete_RMST`") is aspirational and will be corrected — `concrete_RMST` is a two-arm
external bridge, not the pooled-Q vehicle.

## Estimand

For each pre-specified subgroup `s`:

    RMST_s(τ) = ∫₀^τ S(t | S=s) dt,   S(t|S=s) = P(T > t | S=s)  (composite event)

the single-arm subgroup restricted mean survival time — the RMST sibling of the event rate
`psi_s = E[Y|S=s]` already implemented in `pooled_q_subgroup.py`. Marginal over subgroup s's
own W distribution; comparable against a performance goal, like `psi_s`.

## Mechanism (lifts the landed core to a time grid)

Time grid `t₁<…<t_K in (0,τ]` via `linspace(0,τ,K+1)[1:]` (reuse the `pointwise_rmst` pattern).

1. **Pooled time-resolved survival — pooled discrete-time hazard.** Fit ONE pooled logistic
   hazard `λ_k(W,S)` on person-time (long) format: shared W coefficients + subgroup one-hot
   (`drop_first`), IPCW-weighted (the SAME Cox-censoring IPCW as the core, one model).
   `S(t_k) = ∏_{j≤k}(1 − λ_j)`. This yields **monotone survival by construction** — preferred
   over K independent logistic cuts, which would need post-hoc isotonic patching.
2. **Per-subgroup TMLE targeting** at each `t_k` with the same membership clever covariate
   `H = 1{S=s}/π_s · ipcw`, fluctuating only inside `s` → targeted `Ŝ_s(t_k)`.
3. **Integrate:** `RMST_s = Σ_k Ŝ_s(t_k)·Δt`.
4. **SE via IC summation:** `IC_RMST_s = Δt · Σ_k IC_{S_s(t_k)}` (per-time targeted ICs summed
   over shared patients), `se = sqrt(var(IC_RMST_s, ddof=1)/n)` — the integrated-estimand IC
   pattern `pointwise_rmst` uses.

`pooled=False` fits the hazard within each subgroup (borrows nothing) — the high-variance
baseline the pooled version should beat on small subgroups.

## RP-spline nuisance (#188, route-A form)

A pluggable survival backend replacing the pooled logistic-hazard:
- `flexsurvspline(Surv(T_obs, event) ~ W + S, scale="hazard", anc=list(gamma1=~S))` via a thin
  **rpy2 bridge** (`r_scripts/flexsurv_bridge.R`), predicting `S(t_k | W, S)` on our grid.
- Predictions feed the SAME per-subgroup TMLE targeting — RP is a **nuisance debiased by our
  one-step**, matching #188's "RP as nuisance, not plug-in; DR gap filled", with *our*
  single-arm targeting filling the gap instead of concrete's.
- **Gated:** falls back to the Python pooled-hazard when rpy2/flexsurv is unavailable (the same
  graceful pattern as the `concrete_*` estimators). No hard R dependency; CI stays green.

## exp44 — the finding

Non-PH small-subgroup survival DGP: early-harm/late-benefit **crossing hazards** (per #188's
motivation), covariate-defined subgroups of unequal size. Swept by subgroup size `n_s`, on
**RMSE↓ and CI coverage→95%**:
- pooled-Q RMST **(borrow)** vs subgroup-only `pooled=False` **(no borrow)** — the borrowing win.
- vs naive per-subgroup **KM-RMST** — the no-nuisance baseline.
- pooled-Q with **logistic-hazard** vs **RP-spline** nuisance — RP's non-PH payoff.

Read-out: a table where pooled beats subgroup-only in small-`n_s` RMSE at ~nominal coverage,
and RP beats logistic-hazard under the non-PH shape.

## Files

- `causal_bench/estimators/pooled_q_subgroup.py` — **add an RMST path in the same file**
  (`estimand="subgroup_rmst"` switch + a `_estimate_rmst` helper sharing `_ipcw` and the
  pooling logic); a `nuisance="logistic"|"rp_spline"` selector. Correct the `concrete_RMST`
  docstring. (Not a sibling class — IPCW/pooling are shared, estimand family stays in one place.)
- `causal_bench/estimators/rp_spline_nuisance.py` *(new)* — rpy2/flexsurv predict-`S` bridge +
  graceful unavailability (`_flexsurv_available()`), returning an `S[i, k]` matrix.
- `r_scripts/flexsurv_bridge.R` *(new)* — `flexsurvspline` fit + predict on a grid.
- `experiments/exp44_pooled_q_rmst.py` *(new)* + a `causal_bench/validation/` DGP/report
  helper mirroring `hazard_selection.py`.
- `tests/test_pooled_q_subgroup.py` — extend.

## Testing

- **RMST recovery:** closed-form Weibull/exponential per-subgroup RMST recovered within CI;
  ~95% IC-based coverage over reps.
- **Borrowing win:** pooled beats subgroup-only RMSE on a small covariate-defined subgroup
  (the ENCIRCLE regime), matching the event-rate test's design.
- **IPCW path:** informative-censoring DGP recovers truth.
- **RP nuisance:** `skipif` no rpy2/flexsurv; when present, RP-backed RMST recovers truth on a
  non-PH DGP where the logistic-hazard backend is biased.

## Out of scope / non-goals

- The concrete/sl3 route-B (`Lrnr_flexsurvspline` into concrete's `Model`) — documented as a
  secondary path for the two-arm debiased-RMST setting, not built here.
- Competing-risks cause-specific RP hazards (#188 mentions; single composite event here).
- Bayesian survival nuisance (M-splines / `stan_surv`) — a separate branch (memory:
  survival-nuisance-concrete); RP is frequentist-only and fine for this frequentist TMLE.
