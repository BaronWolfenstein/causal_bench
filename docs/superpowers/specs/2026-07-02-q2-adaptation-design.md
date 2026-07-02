# Q2 Adaptation — Design Pass (#46 follow-up)

**Date:** 2026-07-02
**Status:** Design exploration (pre-implementation). Not scheduled. *Revised after review — §1 information structure, §4 three-arm design with an oracle ceiling, §6/§7 corrected.*
**Context:** The eₜ detector (#46, merged #51) answers the Collinear post's **Q1** — *detect an unobserved shift*. This design pass scopes the unbuilt **Q2** — *how should the agent adapt once it detects the shift*: "let go of a now-stale goal without discarding still-useful context." exp26 currently overclaimed this contrast in a docstring (fixed in #53); this is what it would actually take.

## 1. The gap that makes this non-trivial

The user-sim DGP has **no agent**: the action `a_t` is exogenous `N(0,1)` noise, and there is no belief state. So Q2 is not plumbing on top of the detector — it requires introducing the agent-side model the DGP lacks. Three decisions, each with a cheap-v1 and a faithful-but-heavier option.

**Information structure (load-bearing — this is what makes Q2 a real problem).** The agent's belief filter sees the emission `u_t`; the detector operates on the negative control `n_t`; and the detector's flag is the *only* channel by which shift information reaches the agent's plan. This must be stated explicitly, because `n_t` is a near-direct readout of `z`: if the belief filter could consume `n_t` directly it would track `z` through shocks automatically and the whole flag apparatus would be pointless. Q2 is coherent precisely when the shift-revealing signal reaches the plan *only* via the (imperfect) detector — otherwise the problem is either trivial (filter `n` directly) or rigged (the belief can never self-correct).

## 2. Decision A — what is the agent's belief?

- **(A1) Continuous latent estimate `ẑ_t`.** The agent maintains a filtered estimate of the user's latent state. Naive integrates it under the endogenous model; NC-flag corrects it on detection. Adaptation error is directly measurable as `|ẑ_t − z_t|`. **Recommended v1** — cleanest, quantitative, reuses the existing continuous `z`.
- **(A2) Discrete goal/plan `g_t`.** The agent pursues one of K goals; an exogenous shock switches the true goal; naive keeps pursuing the stale one, NC-flag re-infers. More faithful to the post's "total pivot" story, but needs a discrete-goal DGP layer and a goal-inference model. **Deferred** — richer, and the right *second* step if v1 shows the effect.

## 3. Decision B — does the agent act on its belief?

- **(B1) Belief-tracking only (action stays exogenous).** Measure how well `ẑ` tracks `z` after a shock under naive vs NC-flag updating; `a` remains exogenous noise. Isolates the *detection→adaptation* value without a policy/reward loop. **Recommended v1.**
- **(B2) Full act-on-belief loop.** `a_t` is chosen from `ẑ_t` (a policy), so a stale belief produces wrong actions and a measurable reward penalty — the post's actual concern (Claude Code wasting tokens on a stale plan). **Deferred** — needs a policy, a reward, and closes the loop between belief error and outcome; substantially more.

## 4. Decision C — the arms, the update rules, and the metric

Given A1 + B1, **three arms** (not two — a two-arm naive-vs-flag contrast rigs the result):

- **Naive (self-correcting filter, no flag).** A proper Kalman-style filter that BOTH predicts under the endogenous model AND measurement-updates on the emission: predict `ẑ_t⁻ = ẑ_{t-1} + γ̂·tanh(a_{t-1})`, then correct toward `u_t`. Crucially it *does* incorporate `u_t`, so after a shock it partially self-corrects on its own (the emission reflects the shifted `z`) — it is **not** a strawman that drifts forever. It simply has no shock term and no flag. (A prediction-only naive baseline that ignores `u_t` would drift permanently and hand the flag a tautological win — do not use it.)
- **NC-flag.** Identical filter, plus: when `|nc_residual_t| > c` the agent admits the exogenous jump — inflate belief variance so `u_t` dominates the next update (or reset toward it). `c` from **exp26's detection ROC** at a target FPR (a clean tie-back to Q1, not a free parameter).
- **Oracle (`eₜ` known).** Same filter, but conditions on the *true* shock indicator instead of the detector's flag — the **ceiling** on achievable adaptation.

**Metric — the marginal, bounded quantity (not "flag beats naive"):**
- Post-shock tracking error (mean `|ẑ_t − z_t|` over the K turns after each shock) and time-to-recover, for all three arms.
- **Headline: the fraction of achievable adaptation the imperfect detector captures** — `(naive − NC-flag) / (naive − oracle)`. "NC-flag beats naive" alone is near-tautological (one arm has shift info, the other doesn't) and is **not** the reported result.
- Predicted behavior, tying to #46's observability curve: as the control degrades (`nc_coupling`↓), the NC-flag arm slides from oracle toward naive; the achievable gap (`naive − oracle`) is large at high δ and collapses at low δ.

## 5. What this validates (and its honest limit)

It would show, against known ground truth, that acting on the exogenous-shift flag recovers the user's state faster than treating every turn as endogenous continuation — the post's Q2, measured rather than asserted. **Limit:** under A1/B1 it measures *belief-tracking*, a proxy for adaptation, not task outcome. The claim "the agent adapts better" is only as good as "belief-tracking error is the right proxy for adaptation" — which B2 (reward loop) would close but v1 does not.

## 6. Decisions that come first (ours), and the one seam that's Guga's

**Two decisions are ours and are more fundamental than the Collinear seam — they are what make the experiment non-trivial, and they must be settled before building:**
- the **information structure** (§1): the flag is the *only* shift channel to the plan;
- the **three-arm design with an oracle ceiling** (§4): without it, "flag beats naive" is a tautology.

Neither needs Collinear. **Guga's input matters at exactly one *later* seam:** does Collinear frame adaptation as belief-tracking (A1) or goal/plan-switching (A2), and what does "correct adaptation" mean to them — recover the latent state, re-elicit from the user, or preempt (their Q3)? If their notion is goal-switching or outcome-based, v1's belief-tracking metric is a defensible demo but not their target, and the effort routes to A2/B2. So: settle the two internal decisions and build the three-arm v1 without him; validate the *framing* against his before the faithful version.

## 7. Effort

- **v1 (A1 + B1, three arms):** a proper measurement-updating filter (naive) + the flag and oracle variants + the marginal-capture metric + an experiment reusing exp26's threshold, DGP, and observability knob. **~2 focused days** — the earlier "~1 day" assumed the under-designed two-arm contrast; a real self-correcting filter plus the oracle arm is the difference. No new DGP primitives (reuses `z`, `u`, `a`, `nc_residual`, `nc_coupling`).
- **Faithful (A2 and/or B2):** discrete-goal layer and/or policy+reward loop. Multi-day, and the piece that most benefits from the §6 Guga seam first.

**Recommendation:** if this line is picked up, settle §1 (information structure) and §4 (three arms) — both ours — then build the three-arm v1 (~2 days, reuses everything, produces the marginal-capture-vs-observability result), then stop at the Guga seam before A2/B2. Do **not** build the two-arm naive-vs-flag version — it rigs its own result.
