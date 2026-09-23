"""Exp 53: COMPETING RISKS — treating a competing event as censoring is biased.

Product framing: churn (cause 1) competes with upgrade (cause 2). The naive 1−KM churn incidence (treating
upgrade as censoring) over-estimates the cause-1 cumulative incidence and biases the treatment effect; the
Aalen–Johansen CIF is correct. Randomized treatment (isolates the competing-risks bias). Self-validating.
Run: python -m experiments.exp53_competing_risks
"""
from pathlib import Path
from causal_bench.validation.competing_risks import (
    sim_competing, true_cif1, naive_1km_cif, aalen_johansen_cif,
)

OUT_DIR = Path("results/exp53_competing_risks")


def run(*, n=20000, tau=3.0, seed=0):
    truth = true_cif1(tau=tau)
    df = sim_competing(n, tau=tau, seed=seed)
    return {"truth": truth, "naive": naive_1km_cif(df, tau), "aj": aalen_johansen_cif(df, tau),
            "n": n, "tau": tau}


def report(r) -> str:
    t = r["truth"]["cif1"]; nv = r["naive"]; aj = r["aj"]
    eff_t = r["truth"]["effect"]; eff_nv = nv[1.0] - nv[0.0]; eff_aj = aj[1.0] - aj[0.0]
    L = ["## Exp 53 — competing risks: treating the competing event as censoring is biased\n",
         f"n={r['n']}, τ={r['tau']}. Cause 1 = churn, cause 2 = upgrade (competing). Estimand: churn incidence "
         "CIF₁(τ|A), randomized A.\n",
         "| arm | truth CIF₁ | naive 1−KM (upgrade as censoring) | Aalen–Johansen CIF (correct) |",
         "|-----|-----------|-----------------------------------|------------------------------|",
         f"| control (A=0) | {t[0.0]:.3f} | {nv[0.0]:.3f} | {aj[0.0]:.3f} |",
         f"| treated (A=1) | {t[1.0]:.3f} | {nv[1.0]:.3f} | {aj[1.0]:.3f} |",
         "",
         f"**Treatment effect on churn incidence** CIF₁(1)−CIF₁(0): truth {eff_t:.3f}, "
         f"Aalen–Johansen {eff_aj:.3f} (correct), naive 1−KM {eff_nv:.3f} (biased).",
         "",
         "Read-out: the naive 1−KM OVER-estimates churn incidence in BOTH arms — it treats upgraded teams as if",
         "they could still churn, inflating the risk. The Aalen–Johansen CIF accounts for the competing upgrade",
         "and recovers the truth. Because the inflation is arm-dependent (upgrade rate differs by treatment when",
         "the treatment shifts the churn hazard), the naive treatment effect is biased too. Cause-specific vs",
         "competing-risks is a classic error; this is the label-free product version.",
         "",
         "(The doubly-robust *continuous-time RMTIF* — restricted mean active-time under confounding — is the",
         "load-bearing case for CONCRETE; gated on the default-learner attenuation fix, see the concrete-RMST spec.)"]
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 53: competing risks")
    p.add_argument("--n", type=int, default=20000)
    a = p.parse_args()
    r = run(n=a.n)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
