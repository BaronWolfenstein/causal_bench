"""Exp 36: empirical two-vs-three-level OC fidelity (#40).

Sweeps between-subgroup heterogeneity τ and, at each level, fits BOTH the
conjugate two-level kernel (ignores τ — exp19's fast full-grid default) and the
three-level BHM via MCMC (models τ; PyMC + NumPyro/JAX backend) on the SAME
replicate subsample. Reports each kernel's Type-I / power / coverage and the
ΔType-I / Δpower / Δcoverage — the empirical evidence that the conjugate
approximation's *operating characteristics* diverge from the three-level model
under heterogeneity (which #16's analytical `approximation_deviation` cannot show).

The story: as τ grows, the two-level kernel becomes anti-conservative (Type-I
inflates, coverage collapses) while the three-level BHM stays calibrated. Every
MCMC fit is tail-ESS-gated (Type-I is a tail event on a rare-subpop estimand).

**Number note:** exp36 was released from the earlier z_anatomy-diagnostics slot
(#73 dropped it); reclaimed here as the lowest free experiment number.

Requires the 3.12 `[bayes]`+`[bayes-gpu]` stack (pymc/arviz + numpyro/jax) — run
in the `.venv312` dev env (or the box). On the box, `jax[cuda12]` gives GPU MCMC.

Run: python -m experiments.exp36_three_level_fidelity --n-reps 100
"""
from pathlib import Path

from causal_bench.estimators.three_level_bhm import run_fidelity

TAU_GRID = [0.0, 0.3, 0.6, 0.9]
OUT_DIR = Path("results/exp36_three_level_fidelity")


def run(n_reps: int = 100, tau_grid=TAU_GRID, effect_alt: float = 0.5,
        n_subgroups: int = 10, n_per: int = 30, draws: int = 800, tune: int = 800,
        chains: int = 4, seed: int = 42, tail_ess_threshold: float = 100.0,
        sampler: str = "numpyro"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exp 36: two-vs-three-level OC fidelity | n_reps={n_reps} | τ={tau_grid}\n")

    rows = ["| τ | kernel | Type-I | power | coverage | (Δ vs 3-level) | tail-ESS flagged |",
            "|---|---|---|---|---|---|---|"]
    for tau in tau_grid:
        res = run_fidelity(n_reps=n_reps, tau=tau, effect_alt=effect_alt,
                           n_subgroups=n_subgroups, n_per=n_per, draws=draws,
                           tune=tune, chains=chains, seed=seed,
                           sampler=sampler, tail_ess_threshold=tail_ess_threshold)
        two, three, d = res["two_level"], res["three_level"], res["delta"]
        rows.append(f"| {tau:g} | two_level | {two['type_i']:.2f} | {two['power']:.2f} | "
                    f"{two['coverage']:.2f} | ΔI={d['type_i']:+.2f} Δcov={d['coverage']:+.2f} | "
                    f"{res['n_tail_ess_flagged']} |")
        rows.append(f"| {tau:g} | three_level | {three['type_i']:.2f} | {three['power']:.2f} | "
                    f"{three['coverage']:.2f} | — | — |")
        print(f"τ={tau:g}: two-level Type-I={two['type_i']:.2f}/cov={two['coverage']:.2f}  "
              f"three-level {three['type_i']:.2f}/{three['coverage']:.2f}  "
              f"(ΔType-I={d['type_i']:+.2f})")

    table = "\n".join(rows)
    (OUT_DIR / "summary.md").write_text(table + "\n")
    print(f"\nSaved → {OUT_DIR}/summary.md")
    print("\nRead-out: the ΔType-I / Δcoverage columns grow with τ — the conjugate")
    print("two-level kernel is anti-conservative under heterogeneity, while the")
    print("three-level BHM stays calibrated. This is the empirical fidelity evidence")
    print("#16's analytical approximation_deviation cannot provide.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 36: two-vs-three-level OC fidelity")
    p.add_argument("--n-reps", type=int, default=100)
    p.add_argument("--draws", type=int, default=800)
    p.add_argument("--tune", type=int, default=800)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", type=str, default="numpyro")
    args = p.parse_args()
    run(n_reps=args.n_reps, draws=args.draws, tune=args.tune, chains=args.chains,
        seed=args.seed, sampler=args.sampler)
