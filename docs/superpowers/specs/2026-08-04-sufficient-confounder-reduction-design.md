# Sufficient-confounder dimension reduction for FM-embedding adjustment sets

Design spec for the tier-2 general method of issue #206 — the piece that elevates the
current result from a workshop note to a full CLeaR/CHIL methods paper.

**Status — design + results, now with real-8B.** Built and validated in exp50 (PR #213): the
effect-modification × positivity 2-D frontier with cross-fit DML coverage; the double-score,
arm-stratified SDR, and ATO-on-φ reductions; the causal-role stress-test; and **real-8B validation
on the SMB Qwen3-8B embeddings**. The synthetic frontier and the real data disagree on which
reduction wins, and the real data is decisive:

- **Synthetic (linear-Gaussian embedding):** the composition (arm-stratified SIR/SAVE SDR +
  ATO-on-φ) was near-unbiased and best-covered across the frontier.
- **Real 8B (nonlinear embedding), the decisive test:** the **regression-based double-score
  recovers (bias +0.02 on a true −1.22 effect); the SIR/SAVE-SDR fails (+0.70)** because linear
  moment methods cannot find the confounding subspace of a nonlinear embedding. Kernelizing SDR
  (RBF features) confirms the diagnosis (+0.70 → +0.24) but plateaus ~10× worse than the
  double-score; pooling SDR is worse still (treatment contamination).

**Recommendation (revised by real-8B): the flexible regression-based DOUBLE-SCORE** — it recovers
on real embeddings, handles effect modification, and escapes the positivity trap as a low-dim
balancing score (no ATO needed on this cohort). The SIR/SAVE-SDR is retained only as the
linear-regime method; the ATO-on-φ positivity response composes on top of *any* reduction when the
positivity axis bites. The one role no reduction handles is a Y-predictive collider (estimand-side
discipline). **Remaining:** the formal causal-sufficiency theory. (The nonlinear-SDR salvage was
tested exhaustively — RKS kernel SDR plateaus at +0.24, a learned MLP bottleneck reaches +0.16,
neither matching the double-score's +0.02 — so the "general reduction" resolves to the double-score
rather than remaining an open salvage.)

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
response* handles positivity. On this synthetic **linear** DGP the SDR+ATO composition is the
winner — **but see Real-8B validation below: on real nonlinear embeddings the regression-based
double-score wins and the SIR/SAVE-SDR composition fails, so the composition is NOT the final
recommendation.** The remaining load-bearing gaps were **real-8B validation** (now done) (inject a known effect into the SMB
embeddings, as payoff_v8 did for the prognostic score) and the **causal-role stress-test**:
does the outcome-targeted reduction drop instrument directions and get fooled by Y-predictive
colliders (the predictive-vs-causal-sufficiency question)? On real embeddings the causal-role
question is only reachable semi-synthetically (append known-role directions to the real
embedding); the residual — whether the real embedding is itself collider-laden — is unverifiable
and falls to estimand-side discipline (baseline restriction, FCI, M-bias sensitivity), not
validation.

## Real-8B validation (the decisive test)

The synthetic sweeps use a *linear* embedding (`W = Ustar @ B`); a real FM embedding is nonlinear,
and that difference decides which reduction wins. payoff_v9 (demo `real8b/`) encodes a 1600-patient
oncology cohort through the SMB Qwen3-8B model (baseline, pre-treatment; PCA-64 + ZCA), injects a
known heterogeneous continuous effect (`tau_cont`, stronger in the rare subgroup; rare are ~76%
treated vs ~36% common — a real positivity trap), and runs every reduction cross-fit against the
true ATE (−1.222).

| method | bias vs true | read |
|---|---|---|
| naïve (full embedding) | +0.967 | pathology — attenuates a −1.22 effect to ~−0.25 |
| oracle (embedding + U) | +0.842 | positivity trap: still attenuated *with* U measured |
| **prognostic** | **−0.011** | recovers |
| **double-score** | **+0.024** | recovers |
| SIR/SAVE-SDR (arm-strat, k=2) | +0.701 | **fails** on the nonlinear embedding |
| SDR + ATO composition | +0.606 | fails (inherits the SDR base) |

**The regression-based reductions recover; the SIR/SAVE-SDR fails** — linear moment methods cannot
find the confounding subspace of a nonlinear encoding. This inverts the synthetic-frontier result,
where the SDR composition won on the *linear* DGP, and it is exactly why real-8B validation was
load-bearing.

SDR salvage attempts (real-8B), all cross-fit — the explicit reductions improve monotonically with
nonlinearity but none reach the double-score:

| reduction | bias | note |
|---|---|---|
| double-score | **+0.02** | the winner — the two PO surfaces, fit directly with HGB |
| learned bottleneck (MLP, k=4) | +0.16 | best explicit reduction; still 6× the double-score |
| RKS kernel SDR (SIR/SAVE on random Fourier features) | +0.24 | D×γ sweep plateaus; more features give no gain |
| SIR/SAVE SDR, k=6 | +0.63 | more components barely help |
| SIR/SAVE SDR, k=2 | +0.70 | the failing baseline |
| pooled SIR/SAVE | +1.00 | worse — treatment contamination, as predicted |

**Conclusion — the "general SDR method" resolves to the double-score.** The double-score already *is*
the minimal outcome-sufficient reduction — the two potential-outcome surfaces, fit with the strongest
flexible learner. SIR/SAVE (linear or RKS) and the learned MLP bottleneck are all more elaborate ways
of approximating those two surfaces, and each loses something the direct fit does not; the monotone
+0.70 → +0.24 → +0.16 → +0.02 progression is the evidence. The double-score is the real-data
recommendation; explicit dimension reduction adds complexity without benefit on real embeddings, and
the SIR/SAVE-SDR is retained only for the linear regime.

## Causal-role stress-test (validity beyond confounding)

The sweeps above use a *friendly* DGP — the embedding is a pure encoding of the confounder. A
real FM embedding is an undifferentiated mix of causal roles, and (unlike decoded variables) you
cannot select the adjustment set by role. `simulate_roles` (exp50) injects two **pre-treatment**
roles a baseline embedding can legitimately contain, alongside the confounder, and asks whether
the outcome-targeted reduction handles them:

  * **instrument** `Zi → A` only, with an unmeasured confounder `Uh` for it to amplify;
  * **M-bias collider** `Cm ← Ha, Hy` (`Ha → A`, `Hy → Y`, both hidden) — a *pre-treatment*
    collider, so baseline-restriction does not exclude it.

Cross-fit DML, n=3000, 12 reps; **excess bias over the oracle** (adjust the true confounder `U`
only), which nets out the common unmeasured-`Uh` baseline:

| scenario | naïve | prognostic | double-score | SDR |
|---|---|---|---|---|
| base | +0.05 | +0.07 | +0.07 | +0.05 |
| +instrument | +0.06 | +0.18 | +0.16 | +0.08 |
| +collider | −0.68 | −0.58 | −0.59 | −0.67 |

- **Instrument — the reduction is *not* automatically instrument-proof, and *which* reduction
  matters.** The moment-based SDR (+0.08) and naïve (+0.06) barely move, but the **arm-conditional**
  reductions — prognostic (+0.18), double-score (+0.16) — *leak* a strong instrument: conditioning
  on `A` opens the `Zi → A ← U` collider path, so within an arm the instrument becomes spuriously
  Y-predictive and survives the outcome surface. A real caution, and a mild point for the SDR over
  the double-score in instrument-heavy settings — and it interacts with the earlier
  arm-stratification result (arm-stratification fixes treatment contamination but is the very
  conditioning that leaks the instrument).

- **Y-predictive collider — every embedding method is fooled (−0.58…−0.68 excess); only the oracle
  is clean.** `Cm` predicts `Y` (through `Hy`), so it survives outcome-targeting; the reduction
  cannot drop it, and adjusting for it opens the M-bias path. No reduction is a de-biasing wand
  here — the recourse is **estimand-side discipline** (baseline restriction, FCI orientation,
  M-bias sensitivity; `project_collider_estimand_discipline`), *not* a better reduction. On a real
  embedding the collider question is only reachable **semi-synthetically** (append a known-role
  direction to the real embedding); whether the real embedding is itself collider-laden is
  unverifiable.

**Bottom line for validity.** Adjusting on an FM embedding is valid only under
unconfoundedness-given-`W` **and** a confounder-rich (not collider-laden) baseline representation.
The outcome-targeted reduction buys back the instrument-exclusion an analyst would do by hand with
decoded variables — but only the moment-based SDR, and it cannot buy back collider-avoidance, which
stays an assumption managed on the estimand side. That is what separates the embedding program from
decoded-patient inference (where the DAG makes role-selection explicit) and bounds the claim to
"as-valid-as-the-reduction-plus-estimand-discipline", not unconditional.

## Candidate methods (to evaluate)

**Ranking by evidence (synthetic + real-8B).** The **double-score is the recommendation.** On the
synthetic linear frontier the SDR+ATO composition tied it; on **real nonlinear 8B embeddings the
double-score recovers (+0.02) while every explicit dimension reduction underperforms** — SIR/SAVE
+0.70, RKS kernel +0.24, learned MLP bottleneck +0.16 (Real-8B validation). The double-score *is*
the minimal outcome-sufficient reduction (the two PO surfaces, flexibly fit), so there is nothing
better to reduce to; the other methods are more elaborate approximations of it. The **ATO-on-φ**
positivity response composes on top of the double-score when the positivity axis bites (though the
double-score escaped it unaided on the real cohort). What remains genuinely open is the **validity
theory**: SIR/SAVE yield *predictive* sufficiency, which does not by itself imply back-door validity
`Y(a) ⊥ A | φ(W)` — but since the double-score won empirically, the theory question is now "why is
the PO-surface pair the right sufficient statistic", not "which reduction". SDR (linear/RKS) is
retained only for the linear regime; the learned bottleneck confirmed nonlinearity is the axis but
did not beat the direct double-score.

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

**Done** (ATO-on-φ, exp50). Combine the reduction with an explicit positivity response on
`φ(W)` — overlap weighting (ATO), reporting ATO≠ATE where the ATE is unidentified. On the 2-D
frontier the reduction *alone* inherits severe positivity (the outcome subspace still contains
the positivity direction, `posv_sdr ≈ posv_full`); the **SDR+ATO composition recovers there**.
The ATO targets the overlap estimand — equal to the ATE under a constant effect, and the honest
identified target otherwise.

## Evaluation plan (extends exp50)

- **Done** — swept effect modification × positivity severity; reported bias + cross-fit DML
  coverage for {prognostic, double-score, SDR}; the SDR+ATO composition is near-unbiased and
  ~nominally covered across the frontier (2-D frontier table above).
- **Done** — causal-role stress-test (instrument vs Y-predictive collider).
- **Done** — self-validating controls (linear outcome = no pathology; γ=0 = prognostic
  suffices; conf=0 unbiased).
- **Done** — **real-8B validation** on the SMB Qwen3-8B embeddings (payoff_v9): the regression
  double-score recovers (+0.02); the SIR/SAVE-SDR fails on the nonlinear embedding (+0.70, RKS
  salvage plateaus +0.24). See the Real-8B validation section — this reversed the recommendation.
- **Done** — nonlinear-SDR salvage swept (RKS kernel +0.24; learned MLP bottleneck +0.16); neither
  matches the double-score (+0.02), which is the resolution.
- **Remaining** — formal causal-sufficiency theory; the semi-synthetic role injection was
  inconclusive (SDR already failed for the nonlinearity reason).

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
