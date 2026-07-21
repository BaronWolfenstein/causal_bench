"""Exp 41: borrowing calibration — identifiability-set `tau_sd` operating characteristics.

Drives the joint-DGP fidelity engine (`causal_bench.validation.joint_fidelity`) over the
design grid the #144 audit calls for, to answer: does setting the between-subgroup SD
prior `tau_sd` from **identifiability** (decode accuracy at θ₀, via the correctly-signed
`canonical_tau_prior`) keep Type-I nominal while retaining power — and where does it fail?

The pipeline is the BP-decoded-labels one: the estimator pools over subgroup labels
DECODED from the representation at working corruption θ₀ (not the true labels), so
identifiability genuinely bites on the fine-level effect (the #144 learnability license).

Grid: **level** (effect-on-coarse `group` vs effect-on-fine `member`) × **θ₀** ×
**scenario** (global null μ=τ=0; heterogeneous null μ=0, τ>0 — where borrowing threatens
size; alternative μ≠0) × **policy** (`flat` / `oracle` / `canonical`).

The story to look for:
- **global null**: every policy keeps Type-I ≈ nominal (identifiability is orthogonal to
  the outcome — no selection-induced inflation);
- **heterogeneous null at a well-decoded level**: `canonical` ≈ `oracle` (weak pooling,
  honest μ SE); at a **poorly-decoded** level (low θ₀ / the fine `member` coordinate),
  `canonical` shrinks harder — the adversarial cell where over-pooling a real τ can
  inflate Type-I / drop coverage;
- **alt**: power of `canonical` vs `flat`/`oracle`.

Requires the 3.12 `[bayes]` stack (PyMC/NumPyro) — run in `.venv312` (or the box).

**Scope caveats (independent review, #144).**
- At the default g=4/b=3 there are only 3–4 subgroups, so τ is barely identified and the
  credible-interval-as-test is deeply conservative: ``reject ≈ 0`` in the nulls is a
  **size-≈0** result, NOT a certification of 0.05-level Type-I. The sweep can catch
  *catastrophic* over-pooling (a real hetero-null τ pooled to ~0 → μ SE collapses →
  inflation), not fine size control. Read **interval width + coverage** as the primary
  operating characteristics, and put K on the grid before trusting "nominal".
- This engine decides only on the **population μ**; the per-subgroup partial-null size
  (the borrowing-inflation mechanism) is not yet exercised.
- **This experiment licenses NOTHING about the frozen-encoder embedding pipeline.** It is
  an internal-validity result about *this* grammar DGP with exact BP and known labels.
  Transfer to embeddings requires (among others) subgroups DECODED from the
  representation (not observed covariates), an estimable decode accuracy, near-symmetric
  Y-independent misclassification, and an encoder preserving the coarse→fine ordering —
  none checked here. Only the structural guidance transfers. See #144.

Run: python -m experiments.exp41_borrowing_calibration          # small illustrative grid
     python -m experiments.exp41_borrowing_calibration --full   # the real run
"""
from pathlib import Path

import numpy as np

from causal_bench.validation.joint_fidelity import joint_fidelity, make_scenario_spec

OUT_DIR = Path("results/exp41_borrowing_calibration")
SCENARIOS = {"global_null": (0.0, 0.0), "hetero_null": (0.0, 0.6), "alt": (0.5, 0.3)}
POLICIES = ["flat", "oracle", "canonical"]


def iter_cells(levels, thetas):
    """Deterministic enumeration of the (level, θ₀, scenario, policy) grid — the
    stable cell order the multi-GPU sharder partitions over."""
    for level in levels:
        for theta0 in thetas:
            for scen in SCENARIOS:
                for policy in POLICIES:
                    yield level, theta0, scen, policy


def run_grid(*, levels, thetas, n_reps, n_units, depth, draws, tune, chains, seed,
             tail_ess_threshold=100.0, g=4, b_size=3, s=2, m=2,
             chain_method="sequential", shard=None, fast=False) -> list[dict]:
    """Sweep level × θ₀ × scenario × policy, one fidelity run per cell. `shard`
    = (worker_id, n_workers): run only cells with `cell_index % n_workers ==
    worker_id` (the multi-GPU partition). `chain_method` threads to the NumPyro
    sampler ('vectorized' runs chains in one vmap on the device)."""
    rows = []
    for idx, (level, theta0, scen, policy) in enumerate(iter_cells(levels, thetas)):
        if shard is not None and idx % shard[1] != shard[0]:
            continue
        mu, tau = SCENARIOS[scen]
        spec = make_scenario_spec(g, b_size, s, m, level=level, mu=mu, tau=tau, seed=seed)
        r = joint_fidelity(spec, level=level, policy=policy, theta0=theta0,
                           n_reps=n_reps, n_units=n_units, depth=depth,
                           draws=draws, tune=tune, chains=chains, seed=seed,
                           chain_method=chain_method, fast=fast,
                           tail_ess_threshold=tail_ess_threshold)
        rows.append({"cell": idx, "level": level, "theta0": theta0, "scenario": scen,
                     "policy": policy, **r})
    return rows


def report(rows: list[dict]) -> str:
    hdr = ("| level | θ₀ | scenario | policy | reject | coverage | mean τ_sd | τ_true | used |\n"
           "|-------|----|----------|--------|--------|----------|-----------|--------|------|")
    lines = [hdr]
    for r in rows:
        lines.append(
            f"| {r['level']} | {r['theta0']:.2f} | {r['scenario']} | {r['policy']} | "
            f"{r['reject_rate']:.2f} | {r['coverage']:.2f} | {r['mean_tau_sd']:.3f} | "
            f"{r['tau_true']:.2f} | {r['n_used']} |")
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 41: identifiability-set tau_sd calibration")
    p.add_argument("--full", action="store_true", help="the real run (θ₀ sweep, more reps/draws)")
    p.add_argument("--levels", nargs="+", default=["group", "member"])
    p.add_argument("--n-reps", type=int, default=None)
    p.add_argument("--n-units", type=int, default=3000)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--tune", type=int, default=None)
    p.add_argument("--chains", type=int, default=2)
    p.add_argument("--tail-ess", type=float, default=None, help="tail-ESS gate (drops flagged fits)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chain-method", default="sequential",
                   choices=["sequential", "vectorized", "parallel"],
                   help="NumPyro chain method; 'vectorized' runs chains in one vmap on the device")
    p.add_argument("--shard", default=None,
                   help="'i/n' — run only cells with cell_index %% n == i (multi-GPU partition)")
    p.add_argument("--fast", action="store_true",
                   help="compile-once direct-NumPyro fit (fit_three_level_meta_fast, ~2.3x)")
    p.add_argument("--out", default=None,
                   help="write raw rows as JSON here (worker mode, for the multi-GPU sharder)")
    a = p.parse_args()

    thetas = [0.5, 0.7, 0.9] if a.full else [0.7]
    n_reps = a.n_reps if a.n_reps is not None else (100 if a.full else 8)
    draws = a.draws if a.draws is not None else (800 if a.full else 600)
    tune = a.tune if a.tune is not None else (800 if a.full else 600)
    # illustrative uses a looser ESS gate (short chains) so cells populate; full is strict.
    tail_ess = a.tail_ess if a.tail_ess is not None else (100.0 if a.full else 40.0)
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None

    n_cells = len(a.levels) * len(thetas) * len(SCENARIOS) * len(POLICIES)
    print(f"Exp 41 borrowing calibration | {'FULL' if a.full else 'illustrative'} | "
          f"{n_cells} cells × {n_reps} reps (draws={draws}, tail-ESS≥{tail_ess:g})"
          + (f" | shard {shard[0]}/{shard[1]}" if shard else ""))
    rows = run_grid(levels=a.levels, thetas=thetas, n_reps=n_reps, n_units=a.n_units,
                    depth=a.depth, draws=draws, tune=tune, chains=a.chains, seed=a.seed,
                    tail_ess_threshold=tail_ess, chain_method=a.chain_method, shard=shard,
                    fast=a.fast)

    if a.out:                                   # worker mode: dump raw rows for the sharder
        import json
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows))
        print(f"shard wrote {len(rows)} cells → {a.out}")
        return

    rep = report(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ("summary_full.md" if a.full else "summary.md")).write_text(rep + "\n")
    print("\n" + rep)
    print(f"\nSaved → {OUT_DIR}")
    print("\nRead-out: global-null reject ≈ nominal for all policies (identifiability ⊥ "
          "outcome). Compare `canonical` vs `oracle` under hetero_null across θ₀/level — "
          "watch the poorly-decoded (low-θ₀ / member) cells for over-pooling → Type-I "
          "inflation or coverage drop. `alt` shows the power the prior buys.")


if __name__ == "__main__":
    main()
