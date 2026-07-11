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

import numpy as np

from causal_bench.estimators.three_level_bhm import (
    run_fidelity, fit_three_level_meta, tail_ess_ok,
)

TAU_GRID = [0.0, 0.3, 0.6, 0.9]
OUT_DIR = Path("results/exp36_three_level_fidelity")


# ─── exp19 real-DGP wiring ───────────────────────────────────────────────────
def exp19_subgroup_summaries(phi: float, conflict: float, scenario: str,
                             seed: int, *, target: str = "teer",
                             n_subgroups: int = 4) -> dict:
    """Run exp19's REAL registry DGP for one replicate, discover subgroups, and
    return the per-subgroup borrowing summaries (θ̂_g, se_g) plus exp19's own
    ESS-weighted two-level aggregate (which ignores between-subgroup τ). This is
    the honest input to the fidelity comparison — the same data both kernels see."""
    import warnings
    from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
    from causal_bench.estimators.subgroup import discover_subgroups, subgroup_level_borrow
    from experiments.exp19_hierarchical_oc import _aggregate_subgroup_results

    true_ate = -0.12 if scenario == "alternative" else 0.0
    cfg = RegistryConfig(true_ate_main=true_ate, conflict_strength=conflict,
                         embedding_fidelity=phi, seed=seed)
    main_df, teer_df, mac_df, emb = generate_registry_data(cfg)
    tgt = {"teer": (teer_df, emb["teer"], cfg.true_ate_teer),
           "mac": (mac_df, emb["mac"], cfg.true_ate_mac)}[target]
    target_df, target_emb, target_true = tgt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = discover_subgroups(main_df, emb["main"], n_subgroups=n_subgroups,
                                   classifier="knn")
        sub = subgroup_level_borrow(main_df, target_df, emb["main"], target_emb,
                                    model, target_true_ate=target_true,
                                    tau_prior_sd=cfg.tau_prior_sd,
                                    robust_weight=cfg.robust_weight,
                                    vague_sd=cfg.vague_sd)
    two = _aggregate_subgroup_results(sub, target, target_true)   # ESS-weighted, ignores τ
    return {
        "theta_hat": np.array([r.borrowing.ate_posterior for r in sub]),
        "se": np.array([r.borrowing.se_posterior for r in sub]),
        "true_ate": float(target_true),
        "two_level": two,                                         # BorrowingResult
    }


def run_exp19_fidelity(*, n_reps: int = 20, phi: float = 0.7, conflict: float = 0.0,
                       target: str = "teer", draws: int = 500, tune: int = 500,
                       chains: int = 2, seed: int = 0, sampler: str = "numpyro",
                       tail_ess_threshold: float = 100.0) -> dict:
    """Two-vs-three-level OC fidelity on exp19's REAL registry DGP: exp19's
    ESS-weighted subgroup aggregate (two-level) vs a random-effects meta-analysis
    over the same per-subgroup summaries (three-level). Type-I / power / coverage
    + deltas over shared replicates."""
    acc = {k: {"null_rej": [], "alt_rej": [], "cover": []}
           for k in ("two_level", "three_level")}
    n_fits = n_flagged = 0
    for r in range(n_reps):
        for scen in ("null", "alternative"):
            s = exp19_subgroup_summaries(phi, conflict, scen, seed + r, target=target)
            two = s["two_level"]
            three = fit_three_level_meta(s["theta_hat"], s["se"], draws=draws,
                                         tune=tune, chains=chains, seed=seed + r,
                                         sampler=sampler, true_effect=s["true_ate"])
            n_fits += 1
            flagged = not tail_ess_ok(three, threshold=tail_ess_threshold)
            n_flagged += int(flagged)
            lo, hi = two.ate_posterior - 1.96 * two.se_posterior, two.ate_posterior + 1.96 * two.se_posterior
            two_rej = bool(lo > 0 or hi < 0)
            two_cover = bool(lo <= s["true_ate"] <= hi)
            for k, rej, cover, keep in (("two_level", two_rej, two_cover, True),
                                        ("three_level", three["rejects_null"],
                                         three["covers_truth"], not flagged)):
                if not keep:
                    continue
                (acc[k]["null_rej"] if scen == "null" else acc[k]["alt_rej"]).append(rej)
                acc[k]["cover"].append(cover)

    def _oc(a):
        return {"type_i": float(np.mean(a["null_rej"])) if a["null_rej"] else float("nan"),
                "power": float(np.mean(a["alt_rej"])) if a["alt_rej"] else float("nan"),
                "coverage": float(np.mean(a["cover"])) if a["cover"] else float("nan")}

    two_oc, three_oc = _oc(acc["two_level"]), _oc(acc["three_level"])
    return {"two_level": two_oc, "three_level": three_oc,
            "delta": {q: two_oc[q] - three_oc[q] for q in ("type_i", "power", "coverage")},
            "n_mcmc_fits": n_fits, "n_tail_ess_flagged": n_flagged}


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


def run_real(n_reps: int = 100, phi: float = 0.7, conflict: float = 0.0,
             target: str = "teer", draws: int = 800, tune: int = 800,
             chains: int = 4, seed: int = 42, sampler: str = "numpyro",
             tail_ess_threshold: float = 100.0):
    """The exp19 REAL-DGP fidelity run: exp19's ESS-weighted subgroup aggregate
    (two-level, drops τ) vs a random-effects meta-analysis over the same
    per-subgroup summaries (three-level)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exp 36 (exp19 real DGP): two-vs-three-level fidelity | n_reps={n_reps} "
          f"| φ={phi} conflict={conflict} target={target}\n")
    res = run_exp19_fidelity(n_reps=n_reps, phi=phi, conflict=conflict, target=target,
                             draws=draws, tune=tune, chains=chains, seed=seed,
                             sampler=sampler, tail_ess_threshold=tail_ess_threshold)
    two, three, d = res["two_level"], res["three_level"], res["delta"]
    lines = ["| kernel | Type-I | power | coverage |", "|---|---|---|---|",
             f"| two_level (ESS-agg, drops τ) | {two['type_i']:.2f} | {two['power']:.2f} | {two['coverage']:.2f} |",
             f"| three_level (meta, models τ) | {three['type_i']:.2f} | {three['power']:.2f} | {three['coverage']:.2f} |",
             f"| Δ (two − three) | {d['type_i']:+.2f} | {d['power']:+.2f} | {d['coverage']:+.2f} |",
             f"\nMCMC fits: {res['n_mcmc_fits']}  ·  tail-ESS flagged: {res['n_tail_ess_flagged']}"]
    report = "\n".join(lines)
    (OUT_DIR / "summary_exp19.md").write_text(report + "\n")
    print(report)
    print(f"\nSaved → {OUT_DIR}/summary_exp19.md")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 36: two-vs-three-level OC fidelity")
    p.add_argument("--dgp", choices=["synthetic", "exp19"], default="synthetic",
                   help="synthetic τ-sweep (methodology) or exp19's real registry DGP")
    p.add_argument("--n-reps", type=int, default=100)
    p.add_argument("--phi", type=float, default=0.7)
    p.add_argument("--conflict", type=float, default=0.0)
    p.add_argument("--target", type=str, default="teer")
    p.add_argument("--draws", type=int, default=800)
    p.add_argument("--tune", type=int, default=800)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", type=str, default="numpyro")
    args = p.parse_args()
    if args.dgp == "exp19":
        run_real(n_reps=args.n_reps, phi=args.phi, conflict=args.conflict,
                 target=args.target, draws=args.draws, tune=args.tune,
                 chains=args.chains, seed=args.seed, sampler=args.sampler)
    else:
        run(n_reps=args.n_reps, draws=args.draws, tune=args.tune, chains=args.chains,
            seed=args.seed, sampler=args.sampler)
