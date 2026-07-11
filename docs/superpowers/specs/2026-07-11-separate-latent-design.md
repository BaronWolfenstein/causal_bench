# Separate-Latent Generative Path — Design (draft, gated)

**Status:** design-only draft. **GATED** — build only if the localization diagnostic returns `separate_latent_justified` (Test C). The default remains **diffuse_directly** (diffuse directly in the frozen embedding; no separate latent). This document fills the architecture that was previously only *triggered-and-gated* — the localization diagnostic (`2026-07-10-localization-diagnostic.mermaid` Test C) and the ENCIRCLE architecture diagram (`DIFF_LAT`, `RTRIP`) name the decision and the round-trip gate, but no design existed for the latent itself. This is that design.

**Relation to the gate:** this is the "architectural commitment" on the gated side of the diffuse_directly refinement Decision 5. Per that decision the *machinery* is CPU-validatable ahead of the verdict; only the *commitment to adopt it* and the real-encoder run wait on the on-box terminal. So this spec can be prototyped on the stand-in encoder now, but is not adopted into the pipeline until the diagnostic on real embeddings says the embedding is insufficient.

---

## 1. Context — what triggers this path

`diffuse_directly` diffuses directly in the frozen encoder's embedding space `Z = E_gen(x)`. That is correct **when the rare clinical detail is localizable in that embedding** — the localization diagnostic's Tests A/B pass. `E_gen` is frozen and trained for a general objective, so a rare detail can collapse into directions it does not preserve (low variance / off-manifold); diffusing in `Z` then cannot generate or reconstruct that detail.

**Test C is the residual probe:** encode `Z → Z'` into a *learned* latent, diffuse, decode back, and check (via a permanent round-trip validator) whether the rare detail survives. If it does — and only in `Z'`, not in `Z` directly — the diagnostic returns `separate_latent_justified`. This spec designs `Z'` and that validator.

## 2. Architecture

The separate-latent path adds exactly **two** new components; everything else is reused from diffuse_directly (T1–T9).

1. **Latent autoencoder** `g_enc: Z → Z'`, `g_dec: Z' → Z` — a small MLP AE. `dim(Z')` is a hyperparameter; **default same-or-larger than `dim(Z)`** to avoid re-inducing the same collapse. The AE is trained (§3) then **frozen**, matching the frozen-encoder discipline.
2. **Round-trip validator** (§4) — the permanent gate.

**Data flow (generation):**
```
sample Z' ~ diffusion(in Z')  →  g_dec  →  Ẑ (embedding space)
   →  ELF render (T9)  →  tokens  →  E_eval  (the #88 guard, unchanged)
```

**Reused unchanged, now operating on `Z'` instead of `Z`:**
- VP-SDE/DDPM core (T2), ZCA whitening of `Z'` (T1), tail-aware `1/p(z')` weights (T4), CFG (T5), the torch score net (T8), ELF render + `E_eval` #88 guard (T9). The score net trains on ZCA-whitened `Z'` exactly as it trains on `Z` today — `Z'` is a drop-in for `Z` downstream of `g_enc`.

So the *only* genuinely new code is `g_enc`/`g_dec` and the validator; the generative stack is otherwise identical.

## 3. Training objective for `Z'`

Train the AE (before, and separately from, the score net) with:

1. **Reconstruction:** `‖ g_dec(g_enc(Z)) − Z ‖²` — `Z'` must be a faithful re-representation of the embedding.
2. **Rare-detail preservation:** the rare-vs-common separation AUC in `Z'` must be **≥** the AUC in `Z` (the Test-A invariant, carried into the latent — do not lose what `E_gen` kept; ideally recover detail it smeared). Realized as an auxiliary separation term (e.g. a small supervised or contrastive head on the rare/common label) if plain reconstruction under-preserves it.
3. *(optional)* a light **whitening/isotropy** term so diffusion in `Z'` is well-conditioned (or just ZCA `Z'` post-hoc, as T1 already does).

**Decision:** deterministic AE by default — **not** a VAE. Escalate to a probabilistic/flow latent only if diffusion in a deterministic `Z'` proves unstable or if the collapse evidence specifically calls for a stochastic latent. Rationale: YAGNI; the VAE's KL/regularization is unnecessary machinery unless observed instability demands it.

## 4. Round-trip validator — the permanent gate

For **held-out** data, run `Z → g_enc → Z' → (diffuse-reconstruct) → g_dec → Ẑ` and require **all** of:

- **Recon fidelity (per mode):** median `‖Ẑ − Z‖ / ‖Z‖` below a calibrated threshold for **both** rare and common (a good common-mode recon that destroys the rare mode must fail).
- **Separation preserved:** `AUC(rare, common | Ẑ) ≥ AUC(rare, common | Z) − ε`.
- **#88 decoupled check:** round-tripped samples, rendered→re-encoded in `E_eval`, must not collapse — `metric_hacking_flag` off. A latent that looks faithful in `E_gen` but collapses in `E_eval` is exactly the metric-hacking failure and must not pass.

If any criterion fails, the separate latent is **not** justified — the gate fails and the diagnostic re-routes (it does not silently ship a lossy latent). Thresholds are calibrated on synthetic rare/common like the other localization tests.

## 5. Relationship to existing pieces

- **Trigger:** localization diagnostic **Test C** → `separate_latent_justified`.
- **Gate:** the §4 round-trip validator (already named `RTRIP` "separate-latent path only" in the architecture diagram).
- **Eval:** `E_eval` / #88 unchanged — still a decoupled evaluation encoder; the AE does not touch it.
- **Twisted-SMC inference:** if the terminal is `smc_required`, the §1 SMC reranker operates in `Z'` (the score/twist contracts are latent-agnostic).
- **Only additions:** `g_enc`, `g_dec`, the validator. No new estimator, no change to the diagnostic's consumer side.

## 6. Gating / non-goals

- **Design only.** Adopt only on `separate_latent_justified`. Default is diffuse_directly (no `Z'`).
- Not a VAE by default (§3). Not a new hard dependency beyond what diffuse_directly already uses (torch for the AE + score net; numpy core for validation).
- The AE + score-net *training run* is the same **box-gated** tier as the rest of diffuse_directly §6; the design + a synthetic-encoder prototype of the AE and validator are **CPU-validatable now** (like T1–T9).

## 7. Open questions (for review)

1. **`dim(Z')`** — same as `dim(Z)`, larger, or task-structured? Default: same-or-larger, to avoid re-collapse. Revisit if the round-trip validator shows the larger latent overfits.
2. **Joint vs staged training** — train the AE jointly with the score net, or freeze the AE first then train the score net? Default: **staged** (frozen AE, then score net) — simpler and matches the frozen-encoder discipline; joint training only if staged under-preserves the rare detail.
3. **Deterministic AE vs VAE vs normalizing flow** — start deterministic (§3); escalate only on observed instability.
4. **Validator thresholds** — recon-ratio and `ε` for the AUC floor; calibrate on synthetic, same methodology as the localization Tests.

## 8. CPU-validatable-now scope (if prototyped ahead of the verdict)

Following the two-layer gate: a synthetic prototype — `g_enc`/`g_dec` MLP AE + the round-trip validator on the **stand-in encoder pair** with synthetic rare/common — is buildable and CPU-validatable now, exactly like T1–T9. Correctness checks: reconstruction fidelity, rare-separation preservation through the latent, and the round-trip validator firing/passing on planted lossy-vs-faithful latents. Only the **architectural commitment** (adopt `Z'` into the pipeline) and the **real MOTOR/CLMBR + A100 run** wait on the on-box localization terminal.
