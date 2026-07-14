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

Run: python -m experiments.exp41_borrowing_calibration          # small illustrative grid
     python -m experiments.exp41_borrowing_calibration --full   # the real run
"""
from pathlib import Path

import numpy as np

from causal_bench.validation.joint_fidelity import joint_fidelity, make_scenario_spec

OUT_DIR = Path("results/exp41_borrowing_calibration")
SCENARIOS = {"global_null": (0.0, 0.0), "hetero_null": (0.0, 0.6), "alt": (0.5, 0.3)}
POLICIES = ["flat", "oracle", "canonical"]


def run_grid(*, levels, thetas, n_reps, n_units, depth, draws, tune, chains, seed,
             tail_ess_threshold=100.0, g=4, b_size=3, s=2, m=2) -> list[dict]:
    """Sweep level × θ₀ × scenario × policy, one fidelity run per cell."""
    rows = []
    for level in levels:
        for theta0 in thetas:
            for scen, (mu, tau) in SCENARIOS.items():
                spec = make_scenario_spec(g, b_size, s, m, level=level, mu=mu, tau=tau, seed=seed)
                for policy in POLICIES:
                    r = joint_fidelity(spec, level=level, policy=policy, theta0=theta0,
                                       n_reps=n_reps, n_units=n_units, depth=depth,
                                       draws=draws, tune=tune, chains=chains, seed=seed,
                                       tail_ess_threshold=tail_ess_threshold)
                    rows.append({"level": level, "theta0": theta0, "scenario": scen,
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
    a = p.parse_args()

    thetas = [0.5, 0.7, 0.9] if a.full else [0.7]
    n_reps = a.n_reps if a.n_reps is not None else (100 if a.full else 8)
    draws = a.draws if a.draws is not None else (800 if a.full else 600)
    tune = a.tune if a.tune is not None else (800 if a.full else 600)
    # illustrative uses a looser ESS gate (short chains) so cells populate; full is strict.
    tail_ess = a.tail_ess if a.tail_ess is not None else (100.0 if a.full else 40.0)

    n_cells = len(a.levels) * len(thetas) * len(SCENARIOS) * len(POLICIES)
    print(f"Exp 41 borrowing calibration | {'FULL' if a.full else 'illustrative'} | "
          f"{n_cells} cells × {n_reps} reps (draws={draws}, tail-ESS≥{tail_ess:g})")
    rows = run_grid(levels=a.levels, thetas=thetas, n_reps=n_reps, n_units=a.n_units,
                    depth=a.depth, draws=draws, tune=tune, chains=a.chains, seed=a.seed,
                    tail_ess_threshold=tail_ess)
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
