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
- **Double-score** (both potential-outcome surfaces `E[Y|A=0,W]`, `E[Y|A=1,W]`): expected
  to recover under effect modification because the pair captures the CATE (boundary sweep).

### Boundary-sweep results (effect modification γ)

Minimal run (n=1200; effect modifier `τ(U) = 1 + γ·(U·d)`; bias vs true ATE = 1.0):

| γ | naive (full W) | prog_only | prog + treated |
|---|---|---|---|
| 0.0 | −0.28 | −0.04 | −0.08 |
| 1.5 | −0.17 | +0.06 | −0.05 |

Read: at γ=0 the prognostic score recovers (as expected). At **moderate** effect
modification (γ=1.5) prognostic-only stays roughly valid (bias ~0.06) — **prognostic-
sufficiency is more robust than feared**, and the double-score shows no clear advantage
*yet*. The precise degradation boundary (where a single prognostic score clearly breaks
and the double-score / SDR reduction becomes necessary) needs a **larger run at stronger
γ + more replicates** — that characterization is the deferred work this spec scopes. (A
useful practical implication already: on this synthetic setting the cheap prognostic-only
fix is surprisingly durable.)

## Candidate methods (to evaluate)

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
(that is QBA / #205 — composes on top). Adjustment-set reduction only.
