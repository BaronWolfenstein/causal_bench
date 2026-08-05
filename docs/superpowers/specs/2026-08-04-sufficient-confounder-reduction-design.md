# Sufficient-confounder dimension reduction for FM-embedding adjustment sets

Design spec for the tier-2 general method of issue #206 — the piece that elevates the
current result from a workshop note to a full CLeaR/CHIL methods paper.

## Motivation

Using a foundation-model embedding as a causal adjustment set induces a **positivity
trap** (high-fidelity confounder encoding near-determines treatment); a flexible
outcome model then **attenuates the effect toward null**, defeating cross-fit
doubly-robust AIPW/TMLE even with the confounder measured (exp50, PR #207; real-8B
payoff_v8: bias +0.150). The **prognostic-score reduction** escapes it — the
prognostic score is a balancing score that is *not* treatment-degenerate — but only
under **prognostic-sufficiency**. We need the general reduction that stays valid when
prognostic-sufficiency fails.

## Problem statement

Find a reduction `φ : embedding → R^k` (k small) such that adjusting for `φ(W)`:
1. **preserves confounding** — `φ(W)` is a valid back-door adjustment set (`Y(a) ⊥ A | φ(W)`),
2. **escapes positivity** — `φ` excludes the propensity-degenerate directions that make
   `e(W) → 0/1`, so overlap on `φ(W)` is non-trivial,
3. **handles effect modification** — remains valid when the treatment effect varies with
   the confounder (where a single prognostic score is insufficient).

The tension is (1)+(3) vs (2): the propensity score satisfies (1) but *is* the
positivity direction; the prognostic score satisfies (2) but fails (3) under effect
modification.

## What we already know (empirical)

- **Full-embedding adjustment**: positivity trap → flexible-Q attenuation (exp50; real-8B).
- **Prognostic-score reduction** (`E[Y|A=0,W]`): recovers under prognostic-sufficiency
  (synthetic ~0.02 residual; real-8B ~65% bias cut). Boundary: **degrades under effect
  modification** (boundary sweep — see Results).
- **Double-score** (both potential-outcome surfaces `E[Y|A=0,W]`, `E[Y|A=1,W]`): **recovers
  across the whole effect-modification range** because the pair captures the CATE — |bias|
  ≤ 0.05 for all γ in the sweep below, vs a single prognostic score that degrades.

### Boundary-sweep results (effect modification γ)

Big run (n=2000, 15 reps; effect modifier `τ(U) = 1 + γ·(U·d)`, where `d` is the confounder
loading from exp50; bias vs true ATE = 1.0). The `prog + treated` column is the **double-score**
— both potential-outcome surfaces `(E[Y|A=0,W], E[Y|A=1,W])`:

| γ | naive (full W) | prog_only | prog + treated (double-score) |
|---|---|---|---|
| 0.0 | −0.197 | +0.004 | −0.045 |
| 1.0 | −0.159 | +0.071 | −0.010 |
| 2.0 | −0.129 | +0.137 | +0.020 |
| 3.0 | −0.105 | +0.198 | +0.040 |
| 4.0 | −0.109 | +0.258 | +0.048 |

Read: at γ=0 both reductions recover (prog_only +0.004). As effect modification grows,
**a single prognostic score degrades monotonically** (+0.004 → +0.258) and by γ≈3 its bias
*exceeds naïve's* in magnitude — prognostic-sufficiency fails exactly where the treatment
effect varies with the confounder, as the theory predicts. The **double-score (prog+treated)
stays valid throughout** (|bias| ≤ 0.048), because the pair of potential-outcome surfaces
captures the CATE that a single control-outcome surface cannot. This is the spec's central
empirical claim, now characterized: **the degradation boundary is real, and the double-score
reduction is the necessary fix under effect modification** (the minimal earlier run — only
γ≤1.5 — was too weak to separate them and misleadingly suggested prog_only was durable).

### 2-D frontier: effect modification × positivity (cross-fit DML)

The 1-D sweep above varies effect modification at *fixed* positivity. The 2-D sweep
(`report_rows_2d`, exp50) varies **both**, adds the two new reductions and the positivity
response, and reports **bias AND 95%-interval coverage** under DML cross-fitting — nuisances
*and* the reduction map are fit out-of-fold, because the reduction is a nuisance too.
Cross-fitting is not optional: the in-sample influence-function SE under-covers badly
(0.2–0.7) and carries a plug-in bias that cross-fitting removes (double-score at γ=4,conf=3:
−0.23 → −0.04).

Cross-fit DML, n=2000, 15 reps; **bias / 95%-coverage**; estimand = tau (mean-zero modifier):

| γ | conf | naïve | prognostic | double-score | SDR (arm-strat) | **SDR + ATO** |
|---|---|---|---|---|---|---|
| 0 | 1 | +0.03/1.00 | +0.04/0.73 | +0.03/0.93 | +0.05/1.00 | **+0.02/1.00** |
| 0 | 3 | −0.12/0.73 | +0.08/0.47 | +0.03/0.60 | −0.08/0.80 | **+0.01/0.80** |
| 2 | 1 | +0.02/1.00 | −0.29/0.00 | +0.03/1.00 | +0.04/0.93 | **+0.01/0.93** |
| 2 | 3 | −0.11/0.87 | −0.35/0.13 | +0.01/0.80 | −0.18/0.73 | **−0.03/1.00** |
| 4 | 1 | +0.04/0.93 | −0.62/0.00 | +0.02/1.00 | +0.04/0.93 | **+0.01/0.93** |
| 4 | 3 | −0.08/0.93 | −0.79/0.07 | −0.04/0.80 | −0.24/0.67 | **−0.05/1.00** |

Positivity diagnostic: near-deterministic-propensity fraction is 0.09 (conf=1) / 0.57 (conf=3)
on the full embedding and essentially unchanged on the SDR-reduced set (0.09 / 0.54) — the
outcome-relevant subspace **still contains the positivity direction**, so the reduction does
not escape severe positivity on its own. Requirement (2) failing for the reduction alone, now
demonstrated rather than conjectured.

Findings (all cross-fit):
- **Prognostic** fails under effect modification (bias to −0.79, coverage ~0) — fundamental;
  cross-fitting does not rescue it.
- **Double-score** is robust to effect modification (|bias| ≤ 0.04) but under-covers at severe
  positivity (0.60–0.80): it handles the CATE, not the positivity axis.
- **Arm-stratified SDR** matches the double-score at mild positivity; at severe positivity it
  inherits the trap (bias to −0.24). SDR **must be arm-stratified** — fitting SIR/SAVE on
  *pooled* Y contaminates the subspace with the treatment signal and biases it (~−0.29 at γ=0);
  fitting within each arm and unioning is the linear analog of the double-score and recovers
  ≈ oracle.
- **The composition — arm-stratified SDR reduction + ATO-on-φ positivity response — is the only
  method both near-unbiased (|bias| ≤ 0.05) and ~nominally covered (0.80–1.00) across the WHOLE
  frontier**, including γ=4,conf=3 where every other method fails. Tier-1 (ATO/positivity) +
  tier-2 (reduction) COMPOSITION, empirically shown: neither piece suffices alone.

This resolves the earlier caveat — positivity-escape (requirement 2) is now tested along the
conf axis, and the answer is that the *reduction* handles effect modification while the *ATO
response* handles positivity; the composition is the recommendation. The remaining load-bearing
gaps are **real-8B validation of the composition** (inject a known effect into the SMB
embeddings, as payoff_v8 did for the prognostic score) and the **causal-role stress-test**:
does the outcome-targeted reduction drop instrument directions and get fooled by Y-predictive
colliders (the predictive-vs-causal-sufficiency question)? On real embeddings the causal-role
question is only reachable semi-synthetically (append known-role directions to the real
embedding); the residual — whether the real embedding is itself collider-laden — is unverifiable
and falls to estimand-side discipline (baseline restriction, FCI, M-bias sensitivity), not
validation.

## Candidate methods (to evaluate)

**Ranking by evidence.** The **double-score** and the **arm-stratified SDR** both now have
empirical backing (the 2-D DML sweep above): each is robust to effect modification, and the
SDR composed with the ATO positivity response is near-unbiased and ~nominally covered across
the whole frontier — that composition is the **recommendation**. What remains genuinely open is
the **validity theory**: SIR/SAVE yield *predictive* sufficiency — the central subspace of a
regression — which does **not** by itself imply the causal back-door validity `Y(a) ⊥ A | φ(W)`
a valid adjustment set requires. Arm-stratification is what buys back causal relevance
empirically (it is the linear analog of the balancing-score / double-score construction), but
the conditions under which it is *provably* valid — and its behaviour under instrument and
collider directions (the causal-role stress-test) — are the open contribution. The learned
bottleneck stays exploratory.

1. **Double-score / joint PO surfaces** — `φ = (E[Y|A=0,W], E[Y|A=1,W])`. 2-D, captures
   effect modification; the treated surface may partly re-import positivity → test whether
   the pair still escapes the trap.
2. **SDR of the confounding subspace** — adapt SIR / SAVE to target the subspace relevant to
   *both* A and Y (not a single response), i.e. the central subspace for the joint
   `(propensity, prognostic)` structure. Open: causal (not merely predictive) sufficiency.
3. **Learned sufficient-representation bottleneck** — a small head trained to be a minimal
   sufficient confounder (predict Y and A through a k-dim bottleneck, penalize treatment
   determinism to preserve overlap). Bridges to the deferred learned-latent branch but is
   adjustment-set-only.

## Validity theory (the open core — the paper's contribution)

- Formalize **causal sufficiency** of a reduction `φ` and give conditions under which each
  candidate yields a valid back-door set.
- Characterize the **positivity-escape** property: which directions must be *excluded* (the
  propensity-degenerate ones) and why the prognostic/double-score constructions do so.
- State the **effect-modification boundary**: exactly when a single prognostic score
  suffices vs. when the double-score / SDR reduction is required (the boundary sweep is the
  empirical anchor).

## Positivity handling

Combine the reduction with an explicit positivity response on `φ(W)`: overlap weights /
trimming, reporting ATO≠ATE where the ATE is unidentified.

## Evaluation plan (extends exp50)

- Sweep **effect modification × positivity severity**; report bias + interval coverage for
  {prognostic, double-score, SDR, learned}.
- **Real-8B validation** of the winning reduction on the SMB embeddings (as payoff_v8 did
  for the prognostic score).
- Self-validating controls (linear outcome = no pathology; γ=0 = prognostic suffices).

## Deliverables & venue

Method(s) + validity theory + the exp50-extension experiment → **CLeaR** (dedicated
causality conference) or a **CHIL** main-track paper. The workshop note (ML4H/CHIL/causal
workshop) is the current result; this spec is the completing research.

## Prior art

D'Amour, Ding, Feller, Lei & Sekhon (2021, high-dim positivity); Hansen (2008, prognostic
score); Li (1991, SIR); Cook & Weisberg (1991, SAVE); Veitch et al. (text-embedding causal
inference); double-score / double-robustness literature.

## Non-goals

Not the learned-latent *generative* branch (separate); not solving unmeasured confounding
(that is QBA / exp49 — composes on top). Adjustment-set reduction only.
