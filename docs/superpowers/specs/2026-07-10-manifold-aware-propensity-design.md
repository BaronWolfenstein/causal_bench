# Manifold-Aware Propensity ↔ Riemannian TDS — Design Note (2026-07-10)

**Status:** design-only, speculative, **gated** on evidence that the flat-embedding
approximation fails (see the decision rule). NOT a plan; nothing here is scheduled.

## Motivation

Wu et al. 2023 (Twisted Diffusion Sampler) is formulated to extend to Riemannian
manifolds. Everything we have built — VP-SDE, analytic/learned score, Tweedie,
the twist reward, the annealed `β_t`, and (implicitly) the propensity / positivity
region `R` — lives in **flat, ZCA-whitened Euclidean** embedding space. If the
frozen-encoder EHR embedding is a **curved manifold**, then generation and
propensity must share that geometry, or the synthetic-control-arm is a **silent
bias generator**. This note specs the Riemannian-consistent version and, crucially,
**when it is worth the complexity**.

## The load-bearing principle

**Generation geometry ≡ propensity / positivity geometry.** One metric `g` on both
sides, or neither. Generating on-manifold (Riemannian transport) while scoring
positivity off-manifold (flat propensity) — or the reverse — produces synthetic
patients whose treatment/positivity structure is inconsistent with how they were
generated, distorting the estimand. `R` is a *single* object: the manifold
positivity-violation region the twist steers **into** and the propensity **flags**.
If defined under different metrics, the two `R`s diverge.

## Generation side — Riemannian TDS (three components, adopted together)

1. **Riemannian SDE.** Reverse VP-SDE driven by Brownian motion on `(M, g)` with
   geodesic drift and a *manifold* score (Riemannian score-based diffusion is the
   base machinery). The forward marginal and Tweedie estimate become the
   Riemannian (Fréchet/Karcher) mean under `g`, not the Euclidean mean.

2. **Geodesic transport.** Particles propagate along **geodesics** of `g`, not
   Euclidean straight lines. **Riemannian Neural OT** (Micheli, Cao, Monod, Bhatt,
   arXiv:2602.03566) is directly usable here: continuous, amortized neural transport
   maps *on* the manifold. Their load-bearing result for us — **discrete
   approximations of manifold OT suffer the curse of dimensionality; a continuous
   map is sub-exponential** — means the transport step for a high-dimensional
   patient embedding must be a continuous/neural map, **not** a discretized
   Sinkhorn/GW plan (which is exactly what the SGA OT layer is today: correct and
   cheap for small claim-graphs, but curse-bound on a patient manifold).

3. **Metric-aware reward (the twist).** The twist reward becomes **geodesic
   distance to `R`** under `g`, not Euclidean distance to `R`. Everything else in
   the twist machinery carries over: `make_twist`/`run_twisted_smc` are agnostic to
   how `reward_fn` is computed, and the annealed `β_t` temperature is a scalar on
   the potential — **metric-agnostic, reused unchanged**. So the *only* generation
   change at the twist layer is swapping the Euclidean reward for a geodesic one.

> **Continuous embeddings throughout — no discrete-latent relaxations (terminology
> guard).** "Discrete OT" above means the transport is represented as an
> *empirical/discrete plan over sample points* (an N×M coupling matrix) — that is
> what suffers the curse of dimensionality — **not** discrete latent variables. The
> embeddings stay **continuous end-to-end**; nothing in this note reintroduces
> token/categorical latents, **Gumbel-softmax, or Concrete** relaxations. That is
> precisely the ELF / `diffuse_directly` bet (diffuse on continuous embeddings;
> CFG/guidance is clean there *because* the space is continuous). RNOT's continuous
> transport **map** is *more* aligned with the continuous-embedding philosophy than
> a discrete plan — it removes a discretization, it does not add one. The only
> discretization anywhere in the pipeline remains the **optional ELF final-step
> token projection** for human-legible MEDS output (the #88 render path), which is
> off the estimator path and unchanged by this note.

## Propensity side — manifold-aware

- **Propensity** `e(X) = P(T | X)` estimated respecting `g`: heat-kernel / geodesic
  features of the embedding manifold rather than raw-coordinate features. This ties
  directly to the SGA spectral **heat-kernel** machinery (`heat_kernel_cost`) and
  the cuGraph backend (SGA issue #15) — the same diffusion-geometry on both repos.
- **Positivity region `R`** defined by **geodesic-neighborhood overlap** (manifold
  positivity), with the positivity-ESS (Kish on the causal weights) computed on
  manifold-consistent weights.

## Interplay / consistency (what makes it one system)

- **One metric `g`, both sides — never one.** The generation twist steers into `R`;
  the propensity flags `R`; if `R` is defined under two different metrics the
  augmentation mis-serves the estimand. This is the whole point of the note.
- **Detectors we already have.** The **#88 `E_eval` render→re-encode guard** is a de
  facto manifold-consistency check: a sample faithful in `E_gen` but collapsing
  under a decoupled `E_eval` is one that left the data manifold. The
  STRUCT/localization screen signals whether compositional/manifold structure is
  real. So we can *observe* flat-approximation failure cheaply, before paying for
  Riemannian machinery.

## Decision rule — when to go Riemannian

- **Default: flat (ZCA-Euclidean).** ZCA is chosen precisely to flatten toward
  isotropy; it is cheaper and avoids all of the above.
- **Trigger (any of):** persistent #88 `metric_hacking_flag` firing; reconstruction
  faithful only in `E_gen`'s own space; measured curvature/anisotropy the flat
  approximation cannot absorb; or discrete-OT curse-of-dimensionality symptoms on
  the patient embedding (the RNOT premise).
- **If triggered:** adopt **all three** generation components **and** the manifold
  propensity **together**, on the **same `g`**. Adopting one side alone reintroduces
  the inconsistency this note exists to prevent.

## What to measure first (cheap, before committing)

- Embedding **curvature / anisotropy**: local intrinsic dimension, divergence
  between geodesic and Euclidean distances, heat-kernel vs Euclidean-kernel
  discrepancy.
- **#88 flag rate** as the operational trigger.
- Whether the SGA discrete Sinkhorn/GW shows scaling pathology on the patient
  embedding — if so, that alone argues for the continuous RNOT map regardless of the
  causal question.

## On the two references (honest scope)

- **Riemannian Neural OT — arXiv:2602.03566:** directly usable for the
  geodesic-transport component and for manifold-aware matching; the
  curse-of-dimensionality result is a concrete constraint (discrete manifold-OT
  won't scale → continuous/neural map). Also a candidate upgrade to SGA's discrete
  GW.
- **Diffusion Models in Simulation-Based Inference (tutorial review) —
  arXiv:2512.20685:** frames the synthetic-control-arm as simulation-based inference
  and catalogs sampler / guidance / schedule design (relevant to our CFG landing +
  the annealed `β_t`), but does **not** address manifold geometry — it informs the
  diffusion/inference framing, **not** the Riemannian question.

## Non-goals / honesty

- Not implementing. Large complexity jump; benefit unproven for the frozen EHR
  encoder (its embedding geometry is an empirical unknown).
- Not claiming a specific published "Riemannian-TDS + manifold-propensity" result —
  this is a **design synthesis** of TDS's Riemannian extension, RNOT, and the SCA
  positivity machinery.

## Cross-repo synergy

The manifold metric via the heat kernel is the *same* diffusion-geometry the SGA
spectral + GW/OT layer already uses (issue #15, cuGraph backend). A Riemannian move
here and a continuous-OT move in SGA are the same underlying upgrade — worth
sequencing together if the triggers fire.
