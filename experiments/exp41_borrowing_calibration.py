"""Exp 41: borrowing calibration — identifiability-set `tau_sd` operating characteristics.

Drives the joint-DGP fidelity engine (`causal_bench.validation.joint_fidelity`) over the
design grid the #144 audit calls for, to answer: does setting the between-subgroup SD
prior `tau_sd` from **identifiability** (decode accuracy at θ₀, via the correctly-signed
`canonical_tau_prior`) keep Type-I nominal while retaining power — and where does it fail?

The pipeline is the BP-decoded-labels one: the estimator pools over subgroup labels
DECODED from the representation at working corruption θ₀ (not the true labels), so
identifiability genuinely bites on the fine-level effect (the #144 learnability license).

Grid: **level** (effect-on-coarse `group` vs effect-on-fine `member`) × **θ₀** × **K**
(subgroups pooled over — see KS) × **scenario** (global null μ=τ=0; heterogeneous null
μ=0, τ>0 — where borrowing threatens size; alternative μ≠0) × **policy** (`flat` /
`oracle` / `canonical` / `empirical`).

The story to look for:
- **global null**: every policy keeps Type-I ≈ nominal (identifiability is orthogonal to
  the outcome — no selection-induced inflation);
- **heterogeneous null at a well-decoded level**: `canonical` ≈ `oracle` (weak pooling,
  honest μ SE); at a **poorly-decoded** level (low θ₀ / the fine `member` coordinate),
  `canonical` shrinks harder — the adversarial cell where over-pooling a real τ can
  inflate Type-I / drop coverage;
- **alt**: power of `canonical` vs `flat`/`oracle`.

The honest exp41 question (see memory reference_vanzwet_single_trial_prior): does the
identifiability-aware `canonical` discount beat the FIXED `empirical` van Zwet prior — the
reference-class baseline — at all? Expected: marginally, and only where a level is decoded
well enough that its accuracy carries information the pooled-Cochrane prior does not.

Requires the 3.12 `[bayes]` stack (PyMC/NumPyro) — run in `.venv312` (or the box).

**Scope caveats (independent review, #144).**
- At K≈3–4 subgroups τ is barely identified and the credible-interval-as-test is deeply
  conservative. The v2 run (72 cells, 4 policies) confirmed this empirically and showed
  it extends to coverage: **coverage 0.96–1.00 in ALL 72 cells**, reject ≈ 0 in the
  nulls — i.e. BOTH headline OCs were degenerate and `canonical` vs `empirical` was
  therefore undecidable (their width gap merely tracked whether canonical's τ_sd sat
  above or below the fixed 0.081). **K is now on the grid (KS)** precisely so a regime
  with non-degenerate coverage exists; read the K trend before reading any policy
  contrast. Read **interval width + coverage** as the primary OCs.
- K also enlarges the grammar alphabet (g·b_size). Empirically decode accuracy *rises*
  with K (0.62 at K=4 → ~0.95 for K≥8) rather than falling, so canonical's τ_sd
  converges to ≈tau_base and canonical ≈ flat at large K; `mean_decode_acc` is reported
  per cell so this is visible rather than assumed.
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

from causal_bench.validation.joint_fidelity import (
    joint_fidelity, make_scenario_spec, binom_ci_from_rate)

OUT_DIR = Path("results/exp41_borrowing_calibration")
SCENARIOS = {"global_null": (0.0, 0.0), "hetero_null": (0.0, 0.6), "alt": (0.5, 0.3)}
POLICIES = ["flat", "oracle", "canonical", "empirical"]
# K = number of subgroups the meta-analysis pools over. The v2 run showed BOTH headline
# OCs are degenerate at K≈3–4 (coverage 0.96–1.00 across all 72 cells, reject ≈ 0 in the
# nulls), so K must be swept for the experiment to discriminate at all — see #144.
KS = [4, 8, 16, 32]


def dims_for_K(level, K, *, g, b_size):
    """Map a grid ``K`` onto the DGP's ``(g, b_size)``. K is the number of subgroups the
    meta-analysis pools over, so it sets the cardinality of the level UNDER TEST; the other
    level keeps its default. Group level pools over ``g`` groups, member level over
    ``b_size`` members (see ``joint_fidelity``'s ``n_sub``)."""
    return (K, b_size) if level == "group" else (g, K)


def iter_cells(levels, thetas, Ks):
    """Deterministic enumeration of the (level, θ₀, K, scenario, policy) grid — the
    stable cell order the multi-GPU sharder partitions over. Every worker must be given
    the SAME Ks, or the shards stop being a partition of one grid."""
    for level in levels:
        for theta0 in thetas:
            for K in Ks:
                for scen in SCENARIOS:
                    for policy in POLICIES:
                        yield level, theta0, K, scen, policy


def run_grid(*, levels, thetas, Ks, n_reps, n_units, depth, draws, tune, chains, seed,
             tail_ess_threshold=100.0, g=4, b_size=3, s=2, m=2,
             chain_method="sequential", shard=None, fast=False) -> list[dict]:
    """Sweep level × θ₀ × K × scenario × policy, one fidelity run per cell. `shard`
    = (worker_id, n_workers): run only cells with `cell_index % n_workers ==
    worker_id` (the multi-GPU partition). `chain_method` threads to the NumPyro
    sampler ('vectorized' runs chains in one vmap on the device).

    `K` sets the tested level's subgroup count via `dims_for_K`; `g`/`b_size` supply the
    UNtested level's default. Note K also enlarges the grammar alphabet (g·b_size), which
    can depress decode accuracy — read `mean_decode_acc` alongside any K trend."""
    rows = []
    for idx, (level, theta0, K, scen, policy) in enumerate(iter_cells(levels, thetas, Ks)):
        if shard is not None and idx % shard[1] != shard[0]:
            continue
        mu, tau = SCENARIOS[scen]
        g_eff, b_eff = dims_for_K(level, K, g=g, b_size=b_size)
        spec = make_scenario_spec(g_eff, b_eff, s, m, level=level, mu=mu, tau=tau, seed=seed)
        r = joint_fidelity(spec, level=level, policy=policy, theta0=theta0,
                           n_reps=n_reps, n_units=n_units, depth=depth,
                           draws=draws, tune=tune, chains=chains, seed=seed,
                           chain_method=chain_method, fast=fast,
                           tail_ess_threshold=tail_ess_threshold)
        rows.append({"cell": idx, "level": level, "theta0": theta0, "K": K,
                     "scenario": scen, "policy": policy, **r})
    return rows


def report(rows: list[dict]) -> str:
    """Markdown table. Coverage and CI width are the headline OCs (#144: reject≈0 at small
    K is a size-≈0 test, not "nominal"); `decode` is the canonical policy's input, shown so
    a K trend can be separated from a decode-difficulty trend."""
    hdr = ("| level | θ₀ | K | scenario | policy | reject | coverage [95% CI] | width ±SE |"
           " mean τ_sd | decode | τ_true | used |\n"
           "|-------|----|---|----------|--------|--------|-------------------|-----------|"
           "-----------|--------|--------|------|")
    lines = [hdr]
    for r in rows:
        # Rows produced before MC error was added (e.g. the v3 run, launched earlier)
        # carry only the rate and n_used — a binomial CI needs nothing else, so
        # recover it post-hoc rather than demanding a re-run. Width SE is not
        # recoverable from an aggregate and renders as nan.
        lo, hi = (r["coverage_lo"], r["coverage_hi"]) if "coverage_lo" in r else \
            binom_ci_from_rate(r["coverage"], r.get("n_used", 0))
        lines.append(
            f"| {r['level']} | {r['theta0']:.2f} | {r.get('K', '')} | {r['scenario']} | "
            f"{r['policy']} | {r['reject_rate']:.2f} | {r['coverage']:.2f} ["
            f"{lo:.2f}-{hi:.2f}] | "
            f"{r.get('mean_ci_width', float('nan')):.3f}±{r.get('mean_ci_width_se', float('nan')):.3f} | "
            f"{r['mean_tau_sd']:.3f} | "
            f"{r.get('mean_decode_acc', float('nan')):.3f} | "
            f"{r['tau_true']:.2f} | {r['n_used']} |")
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 41: identifiability-set tau_sd calibration")
    p.add_argument("--full", action="store_true", help="the real run (θ₀ sweep, more reps/draws)")
    p.add_argument("--levels", nargs="+", default=["group", "member"])
    p.add_argument("--thetas", nargs="+", type=float, default=None,
                   help="working corruptions θ₀ to sweep (default: [0.5,0.7,0.9] on --full, "
                        "[0.7] otherwise). Must match across shard workers.")
    p.add_argument("--Ks", nargs="+", type=int, default=None,
                   help="subgroup counts to sweep (default: KS on --full, [4] otherwise). "
                        "ALL workers must get the same value or the shards stop partitioning "
                        "one grid.")
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

    thetas = a.thetas if a.thetas is not None else ([0.5, 0.7, 0.9] if a.full else [0.7])
    Ks = a.Ks if a.Ks is not None else (KS if a.full else [4])
    n_reps = a.n_reps if a.n_reps is not None else (100 if a.full else 8)
    draws = a.draws if a.draws is not None else (800 if a.full else 600)
    tune = a.tune if a.tune is not None else (800 if a.full else 600)
    # illustrative uses a looser ESS gate (short chains) so cells populate; full is strict.
    tail_ess = a.tail_ess if a.tail_ess is not None else (100.0 if a.full else 40.0)
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None

    n_cells = len(a.levels) * len(thetas) * len(Ks) * len(SCENARIOS) * len(POLICIES)
    print(f"Exp 41 borrowing calibration | {'FULL' if a.full else 'illustrative'} | "
          f"{n_cells} cells × {n_reps} reps (draws={draws}, tail-ESS≥{tail_ess:g}, K={Ks})"
          + (f" | shard {shard[0]}/{shard[1]}" if shard else ""))
    rows = run_grid(levels=a.levels, thetas=thetas, Ks=Ks, n_reps=n_reps, n_units=a.n_units,
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
    print("\nRead-out: the v2 run found BOTH headline OCs degenerate at K≈3-4 (coverage "
          "0.96-1.00 everywhere, reject≈0 in the nulls), so read the K trend FIRST: "
          "coverage should fall toward nominal as K grows, and only in that regime is a "
          "canonical-vs-empirical width/coverage difference interpretable. Read "
          "`decode` alongside — larger K enlarges the alphabet and can depress decode "
          "accuracy, confounding K with decode difficulty.")


if __name__ == "__main__":
    main()
