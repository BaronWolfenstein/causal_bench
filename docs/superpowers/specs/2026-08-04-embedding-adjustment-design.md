# exp50 — high-dim learned-embedding covariates as a fraught causal adjustment set

Part of the research direction in issue #206. This is the **minimal cut**: a
controlled synthetic reproduction of the failure mode + a scoping of what does and
does not fix it.

## Motivation

On real 8B foundation-model embeddings (SMB rare-subpopulation demo, `payoff_v7`),
adjusting for the embedding as a causal covariate **attenuated the treatment effect
toward null** — even with cross-fit, doubly-robust AIPW/TMLE and even with the
confounder measured (oracle bias +0.152 vs true −0.155). We need a controlled,
reproducible artifact that isolates *why*, so the fix can be studied without the
box/8B in the loop.

## DGP (`causal_bench/validation/embedding_positivity.py`)

- Low-dim true confounder `Ustar ~ N(0, I_2)` drives treatment `A` (logistic in
  `Ustar`, strength `conf`) and a continuous linear outcome `Y = τ·A + Ustar·β + ε`
  with `τ = 1` the constant true effect (so **every subpopulation ATE == τ**).
- Observed covariate `W = Ustar·B + noise/fidelity` — a high-fidelity high-dim
  (d=50) linear encoding of the confounder.
- `conf` is the **positivity-severity** knob (higher ⇒ `A` near-deterministic given
  the confounder ⇒ propensity → 0/1).

## Estimators

`oracle_Ustar` (adjust for true Ustar), `naive_fullW` (adjust for W), `trimmed_W`
(AIPW on the propensity-overlap region), `ato_W` (overlap-weighted ATO). The outcome
model is switchable: **Ridge (linear)** or **HistGradientBoosting (flexible)**.

## Claims the experiment establishes (pinned by `tests/test_embedding_positivity.py`)

1. **conf=0 ⇒ unbiased** for every method (sanity).
2. **Linear outcome ⇒ no pathology** — adjustment recovers the effect even at severe
   positivity. So the failure is *not* "positivity from confounder encoding" alone.
3. **Flexible outcome ⇒ attenuation toward null, monotone in positivity**, and it
   hits the **oracle** too — so it is the **flexible-Q × positivity interaction**,
   not high-dimensionality per se. The embedding's role in the real case is to
   *create* the severe positivity (near-perfect propensity separation).
4. **Tier-1 partial, tier-2 recovers.** Augmented **DR-ATO** (`dr_ato_W`,
   positivity-robust, ATO≠ATE) ~halves the bias; the **prognostic-score reduction**
   (`prog_score_W`, tier-2) *largely recovers* the effect — because the prognostic
   score (control-outcome surface, Hansen 2008) is a balancing score that is **not
   treatment-degenerate**, unlike the propensity score, which *is* the positivity
   direction and so cannot escape the trap.

## What this scopes for #206 (out of scope here)

Remaining: characterize *when* prognostic sufficiency holds vs. when a joint
(prognostic + a non-degenerate confounding direction) is needed; a proper SDR of the
confounding subspace; and real-8B validation. Plus the honest ceiling: under genuine
near-determinism there is no overlap and the ATE is unrecoverable — only a restricted
estimand (ATO/local) is identified.

## Non-goals

No dependence on the box or real 8B embeddings; no HAL (needs R); no attempt to
*solve* #206 — only to reproduce and scope it.
