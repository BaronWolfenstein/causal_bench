# Concrete-RMST positivity fix — spec (2026-09-23)

**Status: STAGED, gated on the positivity confirmation run.** Companion to `2026-09-23-concrete-rmst-fixes.md`
and issue #223. This spec captures the *corrected* diagnosis of the concrete-RMST residual bias and the fix to
apply, pending the `getPositivityDx` numbers from the n=4000 × 3-seed confirmation now running
(`/tmp/concrete_positivity.R`). Finalize the numbers below when it lands.

## What the gridn sweep actually found (2026-09-23)

exp51 V=0 stratum, truth RMST-diff = **0.461**, default learners, corrected `Intervention=c(1L,2L)` + dense grid:

| config | RMST-diff | err |
|---|---|---|
| n=2000, grid=12 | 0.343 | −0.118 |
| n=2000, grid=24 | 0.342 | −0.119 |
| n=4000, grid=12 | **0.235** | **−0.226** |

Two clean negatives that **overturn the original spec's diagnosis**:

1. **Grid density is not the lever** — 12→24 points moves it 0.001. RMST integration is fine.
2. **Not benign finite-sample bias either — it GROWS with n** (0.343 → 0.235, back toward the broken 0.221).
   Finite-sample bias shrinks with n; bias that *grows* with n is the signature of a **positivity / TMLE-
   stability** failure, not a statistical one.
3. Earlier: swapping the propensity learner (`Model$A = "SL.glm"`) was a **no-op at n=2000** (0.343 vs 0.345) —
   so the residual is NOT the propensity *library* (the original spec's "de-regularize learners" call). It only
   bites at larger n.

## Corrected diagnosis: overlap tails

The DGP has `A ~ Bern(expit(0.7·W1))`, so P(A=1|W1) → 0 or 1 in the W1 tails. Larger n samples **more extreme-
propensity units**, the TMLE clever covariate ∝ 1/g(W1) blows up on them, and the fluctuation step attenuates
toward the plug-in — worsening as n grows. This is the classic TMLE-with-poor-overlap failure, and it explains
why n=4000 is *worse* than n=2000. (The Python grid-TMLE bounds/truncates its clever covariate, which is why it
sits at a stable 0.41 vs 0.46 — hence it stays the exp51 default, now empirically vindicated, not just by
convenience.)

## Confirmation run (gating this spec) — FILL IN WHEN IT LANDS

`/tmp/concrete_positivity.R`: n=4000 × seeds {1,2,3}, default learners, reporting RMST-diff **and**
`getPositivityDx()$summary` per arm (ESS, max weight, min observation probability, truncation share).

- [x] **Mechanism CONFIRMED (seed 1, n=4000, 2026-09-23):** RMST-diff = 0.235 (worsens vs n=2000's 0.343),
      and `getPositivityDx` shows the overlap signature directly:
      | arm | min_obs_prob | max_weight | ESS | pct_at_bound |
      |-----|-------------|-----------|-----|--------------|
      | A=1 | **0.0173**  | **57.8**  | 0.815 | 0 |
      | A=0 | 0.0273      | 36.6      | 0.846 | 0 |
      `min_obs_prob ≈ 0.017` ⇒ 1/g ≈ 58 (clever covariate blown up); `pct_at_bound = 0` ⇒ no truncation is
      being applied (min_nuisance=None = concrete default). This is the overlap-tail failure, confirmed.
- [ ] Seeds 2–3 (pending, running) — expected to reproduce; seed 1 is already decisive on the mechanism.

## The fix — bound the nuisance (positivity truncation)

`formatArguments()` exposes **`MinNuisance`** (seen in `names(args)`): the lower bound on the propensity g used
in the clever covariate. The default is too permissive for this overlap, so 1/g explodes on tail units. Fix:

- Pass **`MinNuisance = 0.025`** (bound g ∈ [0.025, 0.975]) to `formatArguments` in the bridge — caps the clever
  covariate at 40, the standard TMLE positivity truncation. Tune in {0.01, 0.025, 0.05} against the truth.
- This is a **one-argument change** in the shared `.concrete_args` helper (see the `ConcreteConfig` refactor in
  `2026-09-23-concrete-rmst-fixes.md`): `min_nuisance` becomes a validated `ConcreteConfig` field, default 0.025.
- **Not** a learner change — the original spec's Fix 2 (de-regularize hazards) is unnecessary (hazards are
  already `Lrnr.Cox`); the propensity swap is a no-op. The lever is truncation, not the library.

## Verification / done-criterion

Rerun n=4000 (and n=8000 if memory allows) with `MinNuisance=0.025`: RMST-diff should **stop worsening with n**
and move toward 0.461 (within a few % or matching the Python grid-TMLE's ~0.41). If truncation alone does not
recover it, escalate to CV-TMLE / a collaborative-double-robust update (deeper, separate work).

## Relation
- Primary #223 fix (the real one): `Intervention=c(1L,0L)→c(1L,2L)` slot-index + dense TargetTime grid
  (0.221 → 0.343). See `2026-09-23-concrete-rmst-fixes.md`.
- This spec: the *residual* after that fix. Both land together in the shared `.concrete_args` / `ConcreteConfig`.
- Decision unchanged: **grid-TMLE stays the exp51 default**; concrete-RMST is usable only once this positivity
  fix verifies. The gated CR-RMTIF experiment inherits this too.
