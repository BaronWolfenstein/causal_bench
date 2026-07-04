# diffuse_directly Test-B/B″ Feeder + localization Test-B″ Extension — Design

**Date:** 2026-07-04
**Status:** Approved design, pre-implementation
**Repos:** `causal_bench` (localization extension) + new `diffuse_directly` package (GPU feeder)
**Validated against:** `Localization diagnostic-2026-07-02` and `SMC Twisted Diffusion-2026-07-02` mermaids (authoritative; supersede the 2026-06-22 build spec `localization.py` was built from).

## 1. Motivation & scope

`localization.py` decides the synthetic-patient generative architecture from a frozen-encoder embedding space: encoder capacity (Test A), reconstruction faithfulness (Test B / tail-aware B′), and — per the 2026-07-02 diagram — **CFG generative landing (Test B″)** before awarding a production arch. The GPU pieces (diffusion training, encoder/MEDS loading) are explicitly external to `localization.py`. This spec builds:

1. **Prerequisite (causal_bench):** extend `localization.py` to the 2026-07-02 diagram — add **Test B″ (CFG generative landing)** and the `pending_cfg_landing_check` / `smc_required` terminals. This reconciles the code (A/B/B′/C only) to the diagram; it is the same act as "folding the corrections in."
2. **Task 2 (`diffuse_directly`):** the GPU feeder that produces the reconstruction round-trip (Test B) **and** the CFG generative-landing samples (Test B″), routes them through `run_diagnostic`, and persists a reusable base sampler. Toy-validated on the M4 Pro first; SMB-v1/TVT (MIMIC as public prototype) on the A100s later.

**Out of scope (downstream twist project):** the SMC loop, scatter/gather all-to-all, bulk/tail-stratified ESS, and SMC profiling — §7 records their seams and deferred stance so nothing is retrofitted.

## 2. Prerequisite: localization Test-B″ extension (causal_bench)

Add to `causal_bench/diagnostics/localization.py`, matching the 2026-07-02 diagram:

- **Test B″ — CFG generative landing.** Reconstruction (Test B) tests denoising *near existing points*; B″ tests **generation from noise under rare-cohort classifier-free guidance (CFG, after ELF)** — held-out generated samples, NOT round-tripped. Two metrics, never collapsed:
  - **Fidelity AUC** (guided-generated vs REAL rare) — *lower is better*; high ⟹ poor score in the tail.
  - **Drift AUC** (guided-generated vs COMMON) — *higher is better*; low ⟹ conditioning too weak, collapsed to bulk.
  - Pass = fidelity low AND drift high.
- **New terminals / routing:**
  - Test B pass (or B′ pass) now routes to `pending_cfg_landing_check` (not directly to `diffuse_directly`/`tail_aware`).
  - B″ pass → award the pending arch (`diffuse_directly` if reached via B, `tail_aware` if via B′).
  - B″ fail (reconstruction faithful, CFG cannot land in R) → **`smc_required`**: twisted-diffusion SMC resampler is the REQUIRED inference-time fix (asymptotically unbiased; not optional). `diffuse_directly`/`tail_aware` stay unreachable until CFG passes B″ or SMC-guided samples pass the check.
- **Signature:** `run_diagnostic` gains `cfg_landing: Optional[tuple] = None` carrying `(rare_guided, common_ref)` held-out generated samples; when Test B/B′ passes and `cfg_landing is None`, return `pending_cfg_landing_check`.
- **Characterization tests** (extend `tests/test_localization.py`): B″ pass → arch terminal; B″ fail → `smc_required`; missing cfg_landing → `pending_cfg_landing_check`. Constructed arrays only, no GPU.

## 3. diffuse_directly feeder (new package)

**Home & footprint.** New `diffuse_directly` package (torch/GPU deps) depending on `causal_bench` for `diagnostics.localization` and `diagnostics.embedding_eda.zca_whiten`. Keeps causal_bench's lean numpy/sklearn footprint intact.

**Device-agnostic compute.** PyTorch with `device = "mps"` (M4 Pro toy validation, entirely off the GPU server) or `"cuda"` (A100 real run) — one model implementation. The tiny toy model trains in minutes on MPS. (MLX rejected: native but duplicates the score net for no gain at toy scale.)

**Components:**
1. **`EmbeddingSource → (rare_emb, common_emb)`** — the swappable seam (matches `localization.py`'s interface AND the toy/real device split):
   - `PlantedSource(config)` — toy GMM: bulk `N(0,I)` + rare cluster with separation/thinness/fraction knobs, CPU/MPS. **Toy uses embedding-defined rare/common purely as a machinery-validation device;** the real path defines rare/common by **clinical/outcome label, never embedding clustering** (the diagram's circularity guard).
   - `SMBSource(meds, encoder)` — frozen SMB-v1-1.7B forward passes, GPU. Interface-level now; built when MIMIC (public prototype) / TVT (production) access lands.
2. **Whitening** — `zca_whiten` before diffusion, `zca_unwhiten` after (frozen-encoder conditioning fix).
3. **Base score diffusion** — an MLP denoiser (embeddings are vectors → no U-Net) trained with denoising score matching. Persisted as a **replicable, data-parallel sampler artifact** (the SMC diagram replicates the score net across GPUs — §7 seam).
4. **Reconstruction round-trip (Test B)** — noise-and-reconstruct rare & common → `(rare_recon, common_recon)`.
5. **CFG generative landing (Test B″)** — CFG-guided generation from noise conditioned on the rare cohort → `(rare_guided, common_ref)` held-out samples. Any validity filter applied to generated samples must record a survival probability (§7 kill rule).
6. **Diagnostic glue** — call `run_diagnostic(rare_emb, common_emb, recon_b=…, cfg_landing=…)`; record the terminal (`diffuse_directly` / `tail_aware` / `smc_required` / …).
7. **Persisted artifacts** — trained sampler + rare/common strata membership (for the future tail-ESS) + lineage-ready outputs (§7).

**Incremental (diagram-faithful):** Test-B feeder first (reaches `pending_cfg_landing_check`), then Test-B″ feeder (reaches an arch verdict or `smc_required`). B′ (tail-aware importance-weighted loss) and C (latent) built only if real-data `run_diagnostic` asks (`pending_B_prime` / `pending_C`).

## 4. Data flow

```
EmbeddingSource → (rare, common)
   → zca_whiten
   → train base score-diffusion  (persist replicable sampler)
   → Test B:  round-trip reconstruct → (rare_recon, common_recon)
   → Test B″: CFG generate from noise → (rare_guided, common_ref)
   → run_diagnostic(rare, common, recon_b=…, cfg_landing=…)
   → terminal ∈ {diffuse_directly, tail_aware, smc_required, pending_*, …}
```

## 5. Validation (the point of phase 1)

Planted known-answer cases as the regression suite, on the M4 Pro (MPS):
- **Easy case** (well-separated, non-thin rare) → assert terminal `diffuse_directly` (after B and B″ pass).
- **Reconstruction-faithful-but-CFG-fails case** (rare recoverable in reconstruction but guided generation collapses to bulk / can't land) → assert `smc_required`. This is the case that *proves the B″ layer earns its keep* — reconstruction alone would have wrongly said `diffuse_directly`.
- **Tail-collapse case** (thin low-density rare) → assert `pending_B_prime`.

Without B″, the easy and CFG-fail cases are indistinguishable by reconstruction — which is exactly why the diagram added B″.

## 6. Testing

- **causal_bench:** extend `tests/test_localization.py` for B″/`smc_required`/`pending_cfg_landing_check` (constructed arrays, no GPU).
- **diffuse_directly:** the planted known-answer cases as CPU/MPS regression tests (tiny model); a source-seam test (PlantedSource shape/interface); a device-agnostic smoke (runs on CPU in CI, MPS locally).

## 7. Downstream twist project — seams & deferred concerns (NOT built here)

The twisted-SMC loop is a separate downstream project. Recorded here so task 2 leaves the seams and nothing is retrofitted:

- **`smc_required` is a real, possibly-mandated terminal** — the twist is the required inference-time fix when reconstruction is faithful but CFG cannot land in R; not an optional enhancement.
- **Scatter/gather / all-to-all:** the SMC resampling barrier is a global reduce (ESS) + ancestor selection (cumsum + searchsorted) + gather (`particles[ancestor_indices]`); multi-GPU turns the gather into an all-to-all as ancestors cross devices, skewed under rare-event degeneracy. Mitigations for the twist project: adaptive resampling (barrier only when ESS < threshold), island/local resampling (per-GPU sub-populations, occasional exchange — small bias/variance cost to remove the all-to-all), and index-indirection (defer the physical gather). **Task-2 seams:** persist the base sampler as a replicable module; keep localization outputs lineage-ready (the ancestor-index tensor is the lineage-collapse diagnostic's raw material — cheap int32).
- **Bulk/tail-stratified ESS:** the SMC diagram resamples on global Kish ESS < N/2 — a false-pass that hides tail collapse (the exp29 region-R analog). The twist project must report ESS_bulk and ESS_tail separately (same rare/common split localization uses) and floor ESS_tail. `tail_aware` training (B′) and tail-stratified ESS are the train-time and sample-time versions of the same fix. **Task-2 seam:** persist the rare/common strata membership.
- **IPCW every-kill rule:** in exact SMC, resampling-with-recorded-weights *is* IPCW — redundant to bolt on. But every kill *outside* the weight bookkeeping (mid-trajectory validity/safety filters, heuristic pruning) is informative censoring: model survival G(survive|state), weight by 1/G (discrete-time IPCW, stabilized weights). Positivity caveat: where a filter near-deterministically kills a region, G→0, 1/G explodes, no reweighting recovers support — the fix is upstream (twist earlier), truncation is last-resort bias-for-variance. **Design rule:** every point in the pipeline either enters the SMC weights or gets a recorded survival probability — no third option; the localization diagnostic consumes those weights. **Task-2 seam:** any validity filter on generated samples records a survival probability.
- **Profiling stance (twist project):** profile the *algorithm* off-A100 (resample-trigger rate — dominant hardware-independent risk, meaningful only against a realistic R from the positivity map; per-particle linear scaling; ESS-trajectory/numerical health; cost ratios). Do NOT conclude absolute wall-clock/communication off the NVLink fabric (structurally uninformative). Keep N small (N≈50). Task 2 itself has no SMC loop to profile.
- **Bias-variance framing (twist design context):** in exact SMC, *choices* cost variance, *bias* enters only through the shortcuts that control it — twist choice (learned vs analytic Tweedie), truncation/tempering, resampler-vs-reranker (variance risk vs support risk), the projected-clever-covariate analog (constrain the twist family vs truncate the value), island resampling. These bound the twist project's design, not task 2.

## 8. Deliverables

- causal_bench: `localization.py` Test-B″ extension + tests (the code↔diagram reconciliation).
- `diffuse_directly`: device-agnostic torch feeder (Test B + B″), toy-validated on M4 Pro, with the SMB-v1/TVT source as the documented data-swap follow-on, and the §7 seams in place.
- A persisted, replicable base sampler + rare/common strata + lineage-ready localization outputs for the downstream twist.
