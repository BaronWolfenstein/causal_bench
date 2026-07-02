# Q2 Adaptation — Design Pass (#46 follow-up)

**Date:** 2026-07-02
**Status:** Design exploration (pre-implementation). Not scheduled.
**Context:** The eₜ detector (#46, merged #51) answers the Collinear post's **Q1** — *detect an unobserved shift*. This design pass scopes the unbuilt **Q2** — *how should the agent adapt once it detects the shift*: "let go of a now-stale goal without discarding still-useful context." exp26 currently overclaimed this contrast in a docstring (fixed in #53); this is what it would actually take.

## 1. The gap that makes this non-trivial

The user-sim DGP has **no agent**: the action `a_t` is exogenous `N(0,1)` noise, and there is no belief state. So Q2 is not plumbing on top of the detector — it requires introducing the agent-side model the DGP lacks. Three decisions, each with a cheap-v1 and a faithful-but-heavier option.

## 2. Decision A — what is the agent's belief?

- **(A1) Continuous latent estimate `ẑ_t`.** The agent maintains a filtered estimate of the user's latent state. Naive integrates it under the endogenous model; NC-flag corrects it on detection. Adaptation error is directly measurable as `|ẑ_t − z_t|`. **Recommended v1** — cleanest, quantitative, reuses the existing continuous `z`.
- **(A2) Discrete goal/plan `g_t`.** The agent pursues one of K goals; an exogenous shock switches the true goal; naive keeps pursuing the stale one, NC-flag re-infers. More faithful to the post's "total pivot" story, but needs a discrete-goal DGP layer and a goal-inference model. **Deferred** — richer, and the right *second* step if v1 shows the effect.

## 3. Decision B — does the agent act on its belief?

- **(B1) Belief-tracking only (action stays exogenous).** Measure how well `ẑ` tracks `z` after a shock under naive vs NC-flag updating; `a` remains exogenous noise. Isolates the *detection→adaptation* value without a policy/reward loop. **Recommended v1.**
- **(B2) Full act-on-belief loop.** `a_t` is chosen from `ẑ_t` (a policy), so a stale belief produces wrong actions and a measurable reward penalty — the post's actual concern (Claude Code wasting tokens on a stale plan). **Deferred** — needs a policy, a reward, and closes the loop between belief error and outcome; substantially more.

## 4. Decision C — the update rules and the metric

Given A1 + B1, concretely:

- **Naive update:** `ẑ_{t+1} = ẑ_t + γ̂·tanh(a_t)` — the endogenous model only; never admits an exogenous jump. (A Kalman filter with fixed process noise and no shock term.)
- **NC-flag update:** identical, except when `|nc_residual_t| > c` the agent lets the exogenous move in — reset `ẑ_{t+1}` toward the current observation (or inflate its belief variance so the next observation dominates). `c` is the detection threshold.
- **Threshold `c`:** **reuse exp26's detection ROC** to pick `c` at a target FPR — a clean tie-back to the Q1 work, not a free parameter pulled from nowhere.
- **Metric:** post-shock tracking error — mean `|ẑ_t − z_t|` over the K turns following each shock — plus **time-to-recover** (turns until error returns below a band). Expected result: NC-flag has lower post-shock error and faster recovery; the *gap* between them is the measured value of detection-driven adaptation, and it should **grow with shock magnitude δ and shrink as the control degrades** (tying to the #46 observability curve).

## 5. What this validates (and its honest limit)

It would show, against known ground truth, that acting on the exogenous-shift flag recovers the user's state faster than treating every turn as endogenous continuation — the post's Q2, measured rather than asserted. **Limit:** under A1/B1 it measures *belief-tracking*, a proxy for adaptation, not task outcome. The claim "the agent adapts better" is only as good as "belief-tracking error is the right proxy for adaptation" — which B2 (reward loop) would close but v1 does not.

## 6. Where Guga's input refines it — the checkpoint

**This design is buildable now without Collinear** — it's a self-contained modeling choice in our idiom. His input matters at exactly one seam, *before* investing in the richer version: **does Collinear frame adaptation as belief-tracking (our A1) or goal/plan-switching (our A2), and what does "correct adaptation" mean to them** — recover the latent state, re-elicit from the user, or preempt (their Q3)? If their notion is goal-switching or outcome-based, v1's belief-tracking metric is a defensible demo but not their target, and the effort should route to A2/B2. So: **draft and even build v1 without him; validate the formulation against his framing before building the faithful version.**

## 7. Effort

- **v1 (A1 + B1):** the Kalman-style belief filter + naive/NC-flag updates + tracking-error/recovery metric + an experiment reusing exp26's threshold and DGP. ~1 focused day. No new DGP primitives (reuses `z`, `a`, `nc_residual`).
- **Faithful (A2 and/or B2):** discrete-goal layer and/or policy+reward loop. Multi-day, and the piece that most benefits from the §6 checkpoint first.

**Recommendation:** if this line is picked up, build v1 (belief-tracking) — it's a day, reuses everything, and produces a real measured Q2 result — then stop at the §6 checkpoint and get Guga's read before the faithful version. Do not build A2/B2 blind.
