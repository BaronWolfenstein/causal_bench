# FCI-with-latents for the collider audit (design spec, issue #216)

Design spec for the estimand-side **collider-audit** residual left by the SDR detection layer
(PR #215, `role_detection`) — the audit that certifies the "confounders, not colliders" assumption the
double-score reduction depends on. **`pcalg` is already installed via rpy2**, so this is a *bridge +
Python fallback + read-off*, not a from-scratch FCI. Built in causal_bench, but designed as a **shared
capability**: SGA's plan never scoped causal discovery (it consumes CI verdicts only as KG edge
evidence today), yet it is the natural second consumer of the PAG output (edge orientation + latent-
confounding flags), so the interface is kept dependency-free for cross-repo reuse.

## Motivation

The detection layer orients the **instrument** v-structure cleanly (`A` collider of `Zi, U1` — all
observed) via the two-test signature. But the **M-bias collider `Cm` has hidden parents**
(`Ha → A`, `Hy → Y`): PC skeleton + `orient_colliders` cannot orient it, because latent confounding
is exactly what PC assumes away. `causal_discovery.py`'s docstring already flags this as "the
latent-confounder failure that motivates FCI". FCI is the algorithm that *does* handle latents — it
outputs a **PAG** (partial ancestral graph) whose **bidirected edge `X ↔ Y`** encodes "latent common
cause", distinguishing a genuine collider/confounded pair from a direct edge.

## What already exists (reuse inventory — do NOT reinvent)

- `detectors/zero_flow_ci.zero_flow_ci_test(X, Y, Z)` — the nonparametric CI **oracle**
  (residualize-then-permutation; CPU, no torch). Returns `supports | refutes | underpowered`.
- `detectors/zero_flow_ci.markov_blanket(target, data)` — MB routine (prediction object; documents
  the collider caveat).
- `validation/causal_discovery.pc_skeleton(data, ...)` — adjacency **edges** + **`sepset`** (the `Z`
  that d-separated each non-adjacent pair — already retained precisely for orientation).
- `validation/causal_discovery.orient_colliders(edges, sepset, p)` — v-structure rule
  `X → C ← Y` when `C ∉ sepset(X,Y)`.
- `sim_fork / sim_collider / sim_chain` — self-validation DGPs (extend for the latent case).
- **`pcalg` (R, via rpy2) — installed and verified**: exposes `fci`, `rfci`, `fciPlus`, `skeleton`,
  and pluggable `indepTest`s. This is canonical FCI-with-latents — **we do not hand-roll the
  algorithm; we bridge it**, exactly mirroring the existing `r_scripts/{flexsurv,concrete,cobalt}_bridge.R`
  + rpy2 pattern (and the dual-backend pattern of `estimators/rp_spline_nuisance.py`).

So the build is a **bridge + a fallback + a read-off**, not a from-scratch FCI.

## Design — dual-backend `fci_audit`, mirroring `rp_spline_nuisance`

A new `validation/fci_audit.py` (or `detectors/`) with the **same dual-backend shape** as the
survival nuisance: canonical R implementation preferred, pure-Python fallback so it never hard-fails
on the R-free box.

**Primary backend — pcalg via rpy2 (`r_scripts/pcalg_bridge.R`).** Call `pcalg::fci` (or `rfci` for
speed / `fciPlus` for completeness) with our own CI oracle injected as the `indepTest`: wrap
`zero_flow_ci_test` behind pcalg's `indepTest(x, y, S, suffStat)` signature (returning a p-value), so
FCI runs **nonparametrically** on the same residualize-then-permutation test the rest of the pipeline
uses — not restricted to `gaussCItest`'s linear-Gaussian assumption. Return the PAG (amat) to Python.

**Fallback backend — minimal Python FCI** (only if R absent, e.g. the box): PC skeleton (reuse
`pc_skeleton`) → **possible-d-sep** pruning → v-structures (reuse `orient_colliders`) → FCI rules
**R1–R4** (Zhang 2008 subset) producing a PAG with bidirected edges. More work, so build it only if a
box-side audit is actually needed; the R backend covers Mac/production.

**Read-off (shared, backend-agnostic).** From the PAG: a **bidirected `X ↔ Y`** ⇒ latent common cause
(the `Cm ← Ha,Hy` signature); an arrowhead **into** a candidate covariate ⇒ collider/descendant ⇒
**exclude from adjustment**. Emit (i) a **backdoor-/FCI-valid adjustment set** (pre-treatment, no
arrowhead-into, not on a bidirected/collider path) and (ii) the **audit-flagged exclusions**. This is
the object `role_detection` needs but cannot produce for the hidden-parent case.

## Cross-repo — SGA consumes the PAG

Build it in causal_bench with a clean, importable interface; **SGA reuses it directly.** SGA already
consumes `zero_flow_ci` verdicts as *association* evidence on KG claim edges
(`kg/confidence_toolkit.EmpiricalCIResult`). The FCI PAG is the natural upgrade from *association* to
*causal-structure* annotation: a **directed** endpoint orients a claim edge's causal direction; a
**bidirected** endpoint flags a **latent-common-cause** on that edge (a confounding warning the KG
should carry). So the same `fci_audit` output feeds (a) causal_bench's collider audit and (b) SGA's
KG edge orientation / confounding flags — one build, two consumers. Keep the return type a plain PAG +
verdict list (no causal_bench-only deps) so SGA can import it.


## Validation (self-validating, extends causal_discovery)

- **M-structure DGP:** `Ha → A`, `Hy → Y`, `Cm ← Ha, Hy` with `Ha, Hy` **dropped** (latent). FCI must
  (i) NOT orient a spurious direct `A → Y` through `Cm`, and (ii) flag `Cm` as a collider / the
  `A ↔ Y`-via-`Cm` structure as latent-confounded — where PC + `orient_colliders` fail. Compare
  against `simulate_roles` (the exp50 role DGP) with the named roles.
- **Known controls:** `sim_fork` (no collider), `sim_chain` (no v-structure) must yield no false
  collider flags; the pure `sim_collider` must still orient.
- Wire the resulting safe-adjustment-set into `role_detection` and show the M-bias `Cm` is now
  *excluded* (closing the residual the instrument case already closes).

## Scope / non-goals

- This is an **estimand-side AUDIT diagnostic**, not the reduction. The double-score stays the
  ATE-sufficient reduction (under confounders-not-colliders); FCI *certifies* that assumption where
  variables are named.
- **Named / derived variables only.** On the raw dense embedding there are no variables to test —
  that residual is unchanged (issue #216 part 2: keep roles named / use derived interpretable
  directions). FCI does not make the raw embedding auditable.
- Not full FCI completeness (all of Zhang's rules) unless a case needs it — R1–R4 + bidirected
  read-off suffices for the collider/latent-confounding audit.

## Priority

The **primary (R/pcalg) backend is a short rpy2 bridge**, not an algorithm build — much cheaper than
a from-scratch FCI, so the first increment (pcalg bridge + zero_flow_ci indepTest + read-off, R-only)
is small. Gate the *Python fallback* and any box-side audit on actual need. Build the whole thing when
the collider audit becomes load-bearing (an ENCIRCLE regulatory context, a reviewer challenge to the
confounders-not-colliders assumption, or SGA wanting causal-structure KG edges). The paper's claim is
honestly bounded without it; this strengthens the audit from "instrument case caught, latent-collider
case flagged-but-not-oriented" to "latent-collider case oriented and excluded".

## Prior art

Spirtes, Glymour, Scheines (FCI); Zhang 2008 (completeness / orientation rules R1–R4);
Colombo et al. 2012 (RFCI / possible-d-sep efficiency). Implementations: **`pcalg` (R, installed via
rpy2 — the primary backend)**; `causal-learn` (Python, a fallback option). Reuse note:
`project_collider_estimand_discipline` in session memory.
