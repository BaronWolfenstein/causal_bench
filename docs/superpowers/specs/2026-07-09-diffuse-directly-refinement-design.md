# diffuse_directly — Plan Refinement Design (2026-07-09)

**Status:** design delta to the existing plan `docs/superpowers/plans/2026-07-07-diffuse-directly.md`. Refines it against three references that postdate it (ELF, transport-geometry, dispersive loss). This is **not** a redesign — the 2026-07-07 plan already folds ELF discretization and the #88 metric-hacking guard into Task 9 and defers two-space matching (#87). This document records five targeted decisions and is the basis for the plan update.

**Not yet started.** The numpy core is agreed buildable now (see Decision 5), but implementation is paused pending explicit go-ahead.

---

## Context

`diffuse_directly` is the embedding-space generative model behind the localization terminal of the same name: ZCA-whitened score diffusion on frozen-encoder EHR embeddings, tail-aware training, CFG-guided generation. It produces the `recon_*` / `rare_guided` arrays that `causal_bench/diagnostics/localization.py::run_diagnostic` consumes, and it feeds the twisted-SMC inference reranker for the `smc_required` terminal.

The 2026-07-07 plan builds it **numpy-first** (Tasks 1–7: whiten → VP-SDE → round-trip → tail-aware → CFG → integration → stand-in encoder), with torch (Task 8, score net) and the ELF render + #88 guard (Task 9) as CPU-`importorskip` / GPU-deferred additions. Real MOTOR/CLMBR is the only truly on-box piece.

Three references prompted this refinement:
- **ELF** (He et al. 2026, arXiv:2605.10938) — embedding-space flow matching; validates the embedding-space bet and supplies the weight-tied final-step discretization pattern.
- **Transport geometry** (Van Assel et al. 2026, arXiv:2606.00514) — "generate in reconstruction space, match in semantic space"; theoretical backing for the embedding-space bet **and** the decoupled-`E_eval` metric-hacking guard (causal_bench #87/#88).
- **Dispersive loss** (Diffuse-and-Disperse; LM-Dispersion) — a self-contained anti-collapse training regularizer.

---

## Decision 1 — Core parameterization: keep DDPM/VP-SDE score-diffusion

**Unchanged from the plan.** ELF and transport-geometry are cast in the flow-matching / one-step idiom, but their headline gains come from **embedding-space geometry and two-space matching, not from flow-matching over diffusion** — those gains are orthogonal to the core parameterization. For the Gaussian noising paths used here, ε-prediction (DDPM), score `∇log pₜ`, and the flow-matching velocity are affine-equivalent parameterizations of the same object; the genuine degrees of freedom are the noising path (curved VP vs straight OT), SDE vs ODE sampling, and step count.

"Flow matching" here means its practical variants — **conditional flow matching (CFM)** and the straight-path **OT-CFM / rectified-flow** family — not a strawman. Their one advantage, few/one-step sampling from straighter trajectories, is **irrelevant to ENCIRCLE**, which generates synthetic patient embeddings **offline, in batches, on the A100 box** (not interactive, not latency-bound, not rendered live). Minibatch-OT coupling further *adds* training-time cost (an OT assignment per batch) for that same unused benefit.

**Score and flow are the same object here, not rival models.** For any Gaussian path, ε-prediction (DDPM), the score `∇log pₜ`, and the FM velocity are affine-related; the genuine degrees of freedom are the **noising path** (curved VP vs straight OT), **SDE vs ODE sampling**, and **step count**. Concretely, when the FM probability path is chosen to match the VP marginals, the FM marginal ODE *is* the VP **probability-flow ODE** — `v(x,t) = f(x,t) − ½ g(t)² s(x,t)` — the deterministic reverse ODE with matched marginals (the reverse *SDE* shares marginals but not trajectories). This equivalence is **path-specific**: it holds for the VP path, not for the OT/straight path, whose different marginals are exactly what buy the shorter trajectories.

**Corollary — "keep DDPM" forecloses little.** A trained VP score net already *defines* its probability-flow ODE, so choosing the score/DDPM core does **not** lock out deterministic or few-step ODE sampling later — the PF-ODE (DDIM is its discrete instance) runs from the *same trained net*, no retraining, one affine transform from the score.

**On CPU-validatability, precisely.** OT-CFM is **not inherently unvalidatable on CPU** — a Gaussian target has a closed-form optimal velocity too, so an analytic no-trained-net validation path exists for it as well. The real cost of switching is not lost analyticity; it is **rewriting the plan's existing VP/score machinery** (schedule, Tweedie denoiser, DDPM reverse, Tasks 2–5/8), and the specific analytic asset the plan *currently* exploits is the VP score. With no fidelity reason to switch (many-step DDPM is plenty faithful offline; rare-region coverage is handled by tail-aware weighting + the metric-hacking guard, both core-agnostic), the weight-tied ELF discretization (Task 9) is the only borrow worth taking. **DDPM core stays.**

## Decision 2 — Complete the #88 metric-hacking guard across all fidelity gates

The **consumer side is already done**: `run_diagnostic` accepts decoupled-`E_eval` arrays for every fidelity gate — `recon_b_eval`, `recon_b_prime_eval`, `recon_c_eval`, and `rare_guided_eval` / `common_ref_eval` for the B″ twist-landing — and raises `metric_hacking_flag` (True ⟺ generation space passes but `E_eval` fails) on each (`localization.py`).

The **generator side is incomplete**: the plan's Task 9 `eval_space_inputs` helper and its test only produce and exercise `recon_b_eval` (Test B). **Refinement:** Task 9 must produce the full eval-space set (`recon_b_eval`, `recon_b_prime_eval`, `recon_c_eval`, `rare_guided_eval`, `common_ref_eval`) via the ELF render→re-encode bridge, and add tests asserting `metric_hacking_flag` fires on **Test B′ and the B″ landing gate**, not just Test B. This closes the loop the transport-geometry note calls "the load-bearing bit." Still MOTOR-free (runs on the decoupled stand-in encoder pair).

## Decision 3 — Two-space matching (#87): stays deferred, scope sharpened

The transport-geometry paper contributes two separable things: (a) a **validation correction** — decouple the eval encoder from the generation encoder — and (b) a **training objective** — compute the distribution-matching loss (Sinkhorn divergence) in a separate *semantic* space.

Part (a) is **already delivered** by the render→re-encode `E_eval` guard (Decision 2), so the near-term metric-hacking concern is covered. Only part (b) — the two-space **Sinkhorn training objective** — remains deferred, and genuinely so: it is a training-space change with no near-term need, warranted only if matching stability becomes an observed problem. It stays in "Remaining deferred" as a design axis gated on the diagnostic verdict, not promoted into the core.

## Decision 4 — Dispersive loss: recorded as a deferred, self-contained option

**What it is (self-contained).** Dispersive loss is a **training-time regularizer that counters representation collapse** — the tendency of learned representations to condense into a narrow cone / low effective dimensionality, which manifests as mode collapse and poor support coverage. Given a batch of internal representations `{h_i}`, it adds a **repulsion term** to the training objective that penalizes over-similarity among them — conceptually an InfoNCE-style contrastive loss **with no positive pairs**, only the repulsive denominator (e.g. penalize `log Σ_{i≠j} exp(−‖h_i − h_j‖² / τ)`, or equivalently reward large mean pairwise distance). Intuitively: "spread the batch's representations apart so they don't all collapse together."

**Why it is cheap and self-contained.** It requires **no external data, no labels, and no pretrained reference encoder** — it operates purely on the model's own batch activations. This is its key advantage over REPA-style alignment, which would need a pretrained medical encoder we may not have. Cost is a single extra loss term. Origin: "Diffuse and Disperse" (image diffusion); "LM-Dispersion" transfers it to autoregressive language models — the transfer across modalities (images → discrete/sequential) is the evidence that de-risks trying it on clinical embeddings.

**Relevance and status here.** Representation condensation is exactly the **support-risk / rare-region coverage** failure the SCA cares about (the same axis the twisted-SMC and localization-diagnostic threads address). So dispersive loss is the **cheap first knob to try IF `diffuse_directly` synthetic patients show diversity/coverage collapse** — not adopted preemptively. Reported gains elsewhere are modest and "hard to separate from noise" (low cost, low expected gain). It is a **training-time knob orthogonal to the estimator work** and to inference-time guidance (CFG / twisted-SMC). **Filed as deferred**, to be reached for only on observed collapse.

## Decision 5 — Gate wording: verdict-independent core is buildable now

The plan's current Global Constraint ("do not start until the localization diagnostic returns `diffuse_directly` (or `tail_aware`) on real embeddings") reads as blocking everything, but that overstates the gate. Distinguish two layers:

- **Verdict-independent and buildable now (CPU-validatable):** the numpy core (Tasks 1–7), the CPU-torch score net (Task 8, via `importorskip`), and the ELF render / #88 bridge (Task 9). These validate on the **stand-in encoder** and do not depend on which terminal the diagnostic returns — their correctness is round-trip fidelity, tail reweighting, guidance landing, and the metric-hacking flag, all checkable on synthetic data.
- **Gated on the real-embedding verdict:** the **architectural commitment** (whether a separate latent is warranted at all — the diagnostic's job) and the **real MOTOR/CLMBR + A100 run**. Only these wait on the on-box localization terminal.

**Reword the plan's gate accordingly.** This reconciles "start now" with the ordering discipline: we build and CPU-validate the generator's machinery ahead of the verdict, but do not commit the architecture or run the real encoder until the diagnostic speaks.

---

## Plan changes this design implies (to be applied when the plan is updated)

1. **Global Constraints:** replace the single blocking-gate bullet with the two-layer wording from Decision 5.
2. **Task 9:** expand `eval_space_inputs` to emit all five eval-space arrays; add tests asserting `metric_hacking_flag` on Test B′ and the B″ landing gate (Decision 2).
3. **Remaining deferred:** keep #87 as the two-space **training objective** only (Decision 3); **add dispersive loss** as a new deferred item with the self-contained definition from Decision 4; **note the tangent-DSM torch score net (causal_bench #108)** — a manifold-aware training-time score-net variant (`generative/tangent_dsm.py`, prototyped CPU/closed-form in PR #106) that shares Task 8's torch-score-net infrastructure. Its penalty becomes a differentiable auxiliary loss; the CPU closed-form stays the validation oracle. Gated on real embeddings **and** a curvature / STRUCT-S trigger (manifold-aware propensity spec, PR #99). Deployment details live in the A100 spec §6.
4. **Core (Tasks 1–8):** unchanged (Decision 1).

## Non-goals

- Switching the core to flow matching / one-step generation.
- Promoting the two-space Sinkhorn **training** objective into the core.
- Adopting dispersive loss preemptively (before observed collapse).
- Any real-MOTOR / on-box work (remains gated on the diagnostic verdict).
