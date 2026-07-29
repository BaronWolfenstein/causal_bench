# exp44 — two-epoch borrowing under prior–data conflict (δ-offset)

**Date:** 2026-07-29
**Issues:** #195 (exp41 composite-null validation — sibling), #180 (auditable
exchangeability / conflict diagnostic + protocol response), #173 (calendar-time
confounding), #144 (borrowing-calibration program).
**Reference:** Qian, *Evidence in the Wild* / JSM 2026 (prior–data conflict, composite-null
validation); FDA draft guidance *Use of Bayesian Methodology in Clinical Trials* (Jan 2026);
Schmidli et al. 2014 (robust MAP); Wang/Chen/Ibrahim/Yue et al. (propensity-score-integrated
power prior). See memory `reference_dynamic_borrowing_conflict`.

**Status:** design only — no implementation until this spec is approved and a plan is written.

---

## Motivation

exp41 (#144) calibrates borrowing on the between-subgroup **heterogeneity τ** and, per #195,
sweeps the composite null over τ. What neither exercises is the article's core hazard: a
**data-generating departure** (historical population drifts from the current one) that
produces **prior–data conflict** — the borrowed prior disagrees with the accruing current
data. exp41 cannot express this, because its "historical" information is a *fixed prior*
(van Zwet on τ), not *data* that can drift. This experiment adds that axis.

## Estimand & the null-preserving resolution

exp41's decided quantity is the population mean μ. A naive "shift the current level by δ"
does **not** keep the null true — it moves μ by δ, so rejecting would be correct, not a
Type-I error. (The article's drift keeps the null true only because two arms drift together
and the contrast cancels; exp41 has no control arm.)

**Resolution — put the departure in the epoch gap, not the level:**
- **Historical cohort** carries a true effect **μ_hist = δ** (its own heterogeneity τ_hist).
- **Current cohort** is **truly null: μ_curr = 0** (τ_curr).
- Borrow a **prior built from the historical data** into the current analysis and test
  **H0: μ_curr = 0**.

Then **δ is exactly the prior–data-conflict magnitude**: δ = 0 = exchangeable (historical
also null, no conflict); δ > 0 = historical says "effect", current says "null", borrowing
drags the current estimate off zero. The read-out is **Type-I(δ)** — the size a
point-calibration (δ = 0 only) would miss. Single-estimand, no control arm, no per-unit time
coordinate needed.

---

## Part A — methodological core (abstract, general)

The reusable validation of the borrowing-under-conflict mechanism. Historical "studies" are
abstract exchangeable pseudo-studies with a **marginal** rate; no propensity layer. This is
what the result *proves*, independent of any trial.

### Components
1. **`make_two_epoch_spec(..., delta, tau_hist, tau_curr, seed)`** → `(spec_hist, spec_curr)`
   from the same grammar/hierarchy: `spec_hist` at μ = δ, `spec_curr` at μ = 0.
2. **MAP-prior-from-historical-data** (the machinery exp41 lacks): fit the 3-level model to
   the historical cohort → the meta-analytic-**predictive** distribution for a new
   exchangeable study's μ → approximate as a prior on μ (a small Normal mixture). This is a
   **μ-borrowing** prior (the control-arm-MAP analog), distinct from exp41's τ-borrowing.
   Needs ≥ a few historical pseudo-studies so the between-study τ_hist is estimable — that
   heterogeneity is what robustifies the prior. Reuse `three_level_bhm` for the fit.
3. **Policies on the μ-borrowing axis:**
   - `flat` — no borrowing (reference; nominal by construction).
   - `map` — the MAP prior from historical data (borrows fully; inflates under conflict).
   - `robust_map` — Schmidli mixture: MAP + a vague component with weight `w_vague`; the
     posterior weight shifts to the vague component as conflict grows (the "model adapts").
   - `pooled` — naive full pooling of historical + current (the adversary; maximal inflation).
4. **Sweep δ** (the conflict axis) × θ₀ × K (existing axes), **crossed** (not marginal — the
   article's departure×trend interaction). Report **Type-I(δ)** per policy, plus the
   supremum over δ (the #195 discipline).

### The payoff figure
Type-I as a function of δ per policy: `flat` flat at α; `pooled`/`map` climb with δ;
`robust_map` stays near α by down-weighting — *if* the vague weight and the between-study
τ_hist are adequate. Where `robust_map` still inflates is the honest finding (adaptation is
not automatic protection — the article's central point).

### Tests
- δ = 0: all policies ~ nominal (no conflict).
- δ large: `pooled` > `map` > `robust_map` in Type-I (ordering).
- `robust_map` Type-I is bounded well below `pooled` across the δ sweep.
- MAP-from-historical recovers the historical μ within CI (nuisance-fit sanity).

---

## Part B — ENCIRCLE application layer (**clearly marked; specified, not built in the core**)

> This section describes how the Part-A method transfers to ENCIRCLE. It is **not** part of
> the exp44 core deliverable; it is the documented mapping so the transfer is designed-in.
> It requires machinery (PS integration, registry partitioning) that is a separate build.

Plain MAP does **not** fit ENCIRCLE directly, for two reasons, each with a fix:

1. **Single registry, not multiple trials.** MAP needs exchangeable historical *studies* with
   between-study heterogeneity; ENCIRCLE has one TVT Registry. **Fix:** induce **pseudo-studies**
   by partitioning the registry (**site × calendar-era × region**). The between-partition
   heterogeneity supplies τ_hist — and it is exactly where **standard-of-care drift** shows up,
   so **δ in Part A ≙ the era gap between registry vintages and the trial enrollment window**
   (ties to #173 calendar-time confounding).
2. **Case-mix mismatch.** A marginal registry rate is confounded (TVT patients ≠ trial
   patients). **Fix:** borrow a **propensity-score-adjusted** rate — a **PS-integrated
   power/MAP prior** (Wang/Chen/Ibrahim/Yue): the PS layer fixes case-mix, the power/MAP layer
   does the dynamic down-weighting under conflict. This composes MAP with the SCA's *existing*
   TMLE/IPCW propensity machinery rather than replacing it.

**Three borrowing paradigms to keep straight** (only the third is what exp44 validates):
- ENCIRCLE *primary* = a **fixed 45% performance goal** (a frozen, non-adaptive prior).
- ENCIRCLE *SCA* = a **propensity/TMLE external control** from TVT (what the trial uses today).
- **Bayesian dynamic borrowing** (MAP/robust-MAP/power) = what exp44 validates — relevant to
  ENCIRCLE **only if the SCA is reframed as dynamic borrowing**, the world the conflict article
  is about.

**Application-layer build (future, own spec):** registry partitioning → PS-integrated
robust-MAP → the conflict diagnostic + protocol response of #180. exp44 does none of this; it
validates the borrowing-under-conflict mechanism the application layer would rely on.

---

## Files

- `causal_bench/validation/two_epoch_borrowing.py` — `make_two_epoch_spec`, the
  MAP-from-historical fit, the policy priors, and the δ-sweep engine (reuses
  `three_level_bhm`). *(Part A only.)*
- `experiments/exp44_borrowing_conflict.py` — driver + Type-I(δ) report. **(Verify the number
  is still free at build time — `ls experiments/` + `gh issue list` — per the numbering rule;
  exp44 is next-free-forward as of 2026-07-29.)**
- Sibling to exp41's τ-heterogeneity engine — **not a modification of exp41** (different
  borrowing target: μ-from-historical vs τ-prior).

## Out of scope (future)

- Interim looks / SSR / RAR adaptive monitoring (exp41 and exp44 are single-analysis).
- A per-unit enrollment-time coordinate (δ = the epoch gap suffices).
- The AnCred/reverse-Bayes **diagnostic** + **protocol-response** layers (#180) — consumed
  here only as the `robust_map` policy; their prespecifiable form is #180's build.
- The full Part-B application (PS-integrated, partitioned registry) — its own spec.

## Open item to finalize from the running validation

Whether exp44 is the **primary** conflict-demonstration vehicle (if exp41's existing
composite-null / partial-null validation comes back benign) or **secondary** (if the
partial-null already inflates clearly). One line, set when `results/exp41_composite_null_validation/summary.md` lands.
