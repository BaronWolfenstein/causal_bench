# Latent-aware collider audit — RCD (LiNGAM) primary + FCI cross-check (design spec, issue #216)

> **Revised after empirical checks (2026-08-05).** Constraint-based FCI is *uninformative* on the
> tangled latent-collider: the M-structure `A–Y–Cm` is a fully-connected triangle (A→Y, A↔Cm via
> latent `Ha`, Cm↔Y via latent `Hy`), so there are **no unshielded triples** and pcalg `fci` returns
> an **all-circle PAG** (nothing oriented) — structural, not tuning. That is LiNGAM's sweet spot:
> non-Gaussianity (our treatment `A` is binary → strongly non-Gaussian) orients a unique DAG where
> CI-based methods are stuck. So the **latent-aware LiNGAM variant RCD** is the *primary* orienter and
> pcalg **FCI is the assumption-light cross-check** (which honestly reports "unidentified" as circles).
> Verified: RCD flags `A↔Cm` latent (NaN) where plain DirectLiNGAM misattributes it as a direct edge;
> RCD is imperfect (missed the `Cm↔Y` pair) → the cross-check + "identified only under assumptions"
> reporting is load-bearing. `lingam` (RCD) installed; `pcalg` installed via rpy2.

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
- **`lingam` (Python) — installed (1.13.0), the PRIMARY orienter**: `RCD` (latent-aware, Maeda–Shimizu)
  outputs a directed adjacency **plus explicit latent-confounded pairs** (NaN entries) — exactly the
  collider/latent flag the audit needs, and it orients the fully-connected structures FCI cannot.
  Pure Python ⇒ runs everywhere including the R-free box. `DirectLiNGAM` is the no-latent special case
  (do NOT use it alone — it misattributes latent confounding as direct edges).
- **`pcalg` (R, via rpy2) — installed on R-capable machines, the CROSS-CHECK**: `fci`/`rfci`/`fciPlus`
  with a pluggable `indepTest` (inject `zero_flow_ci_test` → nonparametric). Assumption-light (no
  non-Gaussianity needed) but returns an equivalence class (often all-circles on tangled structures).
  **Bridge it, don't hand-roll**, mirroring `r_scripts/{flexsurv,concrete}_bridge.R` + the dual-backend
  pattern of `estimators/rp_spline_nuisance.py`.

So the build is **RCD (primary, Python) + a pcalg-FCI bridge (cross-check, R) + a shared read-off** —
no from-scratch algorithm.

## Design — `causal_role_audit.py`, RCD primary + FCI cross-check, backends by environment

A new `validation/causal_role_audit.py` that runs **both** discovery methods and reconciles them.
Backends are selected by what the environment has, mirroring `rp_spline_nuisance`'s
degrade-gracefully shape:

**Primary orienter — RCD (`lingam`, Python, runs everywhere incl. the box).** `lingam.RCD().fit(X)`
returns a directed adjacency **plus latent-confounded pairs** (NaN entries). Read-off: a NaN pair ⇒
**latent common cause** (the `Cm ← Ha,Hy` signature); an arrowhead into a candidate covariate ⇒
collider/descendant. Exploits non-Gaussianity to orient the tangled structures FCI leaves as circles.
Assumes (near-)linearity + non-Gaussian noise — so it is *strong where those hold*, and the
cross-check flags where they don't.

**Cross-check — pcalg FCI (`r_scripts/pcalg_bridge.R`, R, Mac/production only).** `pcalg::fci`/`rfci`
with `zero_flow_ci_test` injected as the `indepTest` (nonparametric; behind pcalg's
`indepTest(x,y,S,suffStat)` p-value signature). Assumption-light (no non-Gaussianity needed) but
returns an equivalence-class PAG that is **often all-circles on fully-connected latent-collider
structures** — which is itself informative ("CI structure alone does not identify this; the RCD
orientation rests on non-Gaussianity"). Degrades to `None` when R is absent (the box), exactly like
`rp_spline_nuisance`.

**Ensemble of latent-aware LiNGAM variants — RCD *and* ParceLiNGAM.** They have complementary error
modes (verified on the tangled M-structure): **RCD misses latent pairs (false negatives); ParceLiNGAM
over-flags (false positives, up to flagging everything).** For an audit the asymmetry is decisive — a
**missed collider (false negative) lets bias through**, the dangerous error; over-exclusion is merely
inefficient. So take the **union of the latent-confounded flags** (conservative: flag if *either*
flags) and rank by agreement.

**Reconciliation / read-off (the audit verdict) — confidence by agreement, never a single method.**
For each candidate covariate, combine RCD ∪ ParceLiNGAM (latent-confounded pairs + arrowheads-into)
with the FCI cross-check:
- **high-confidence exclude** — flagged by a latent-aware method **and** FCI orients-and-agrees;
- **exclude (method-dependent)** — flagged by RCD and/or ParceLiNGAM but FCI is all-circles
  ("not CI-identifiable; rests on LiNGAM's non-Gaussianity/linearity");
- **conflict** — RCD and ParceLiNGAM disagree ⇒ report both, lean exclude (conservative).
Emit (a) a **safe adjustment set** (pre-treatment, not latent-confounded, no arrowhead-into) and (b)
**flagged exclusions with a confidence tag**. **Honest bound baked in:** the fully-tangled
latent-collider is near the identifiability boundary — the audit reports *which methods say what*, not
a false-confident single verdict.

**Deferred (filed as #218):** a from-scratch minimal Python FCI as a box-side FCI cross-check —
small, because `pc_skeleton` + `orient_colliders` + `zero_flow_ci_test` already exist (the delta is
possible-d-sep + R1–R4 + the bidirected read-off). Only if a box-side *nonparametric* cross-check is
needed and installing R via conda is undesirable. RCD already covers the box; the pcalg cross-check
covers Mac/production.

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

## ENCIRCLE application (the real payoff — more applicable than the embedding case)

The audit *runs* on ENCIRCLE because ENCIRCLE has **named baseline covariates** (LVEDD, stage,
clinical flags), unlike the raw embedding. Value: (a) **certify the adjustment set** — flag any
candidate covariate that is actually a collider or an instrument and *exclude* it, giving a
**defensible, regulatory-credible covariate selection** rather than a data-dredged one; (b) the
**selection-into-trial collider** (a known single-arm synthetic-control concern — `project_collider_
estimand_discipline` point 5) is exactly what the audit + M-bias sensitivity address.

**Caveat — mixed types.** ENCIRCLE covariates are mixed (continuous LVEDD, categorical stage, binary
flags), where LiNGAM's non-Gaussianity + linearity is strained. There the **FCI-nonparametric**
backend (`zero_flow_ci` handles mixed types) is likely *more* applicable, with RCD as the cross-check
where continuous non-Gaussian covariates dominate. So the dual-backend earns its keep on ENCIRCLE:
run whichever's assumptions hold, and agreement is the confidence signal.

## Deployment (box vs Mac) — a real constraint

- **`lingam` / RCD** is pure Python → installs on the **box** (`~/venv`) and Mac alike; RCD is the
  everywhere-available audit backend.
- **`pcalg` FCI needs R**, and the **box has no R** (same reason the survival nuisance fell back to
  lifelines). So on the box the audit is **RCD-only**; the pcalg FCI cross-check runs on Mac/production
  (or via a conda-R env if a box-side cross-check is ever wanted, or the deferred Python-FCI). The
  dual-backend split is therefore **Python-everywhere (RCD) + R-where-available (pcalg)** — degrade
  gracefully, never hard-fail.

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
