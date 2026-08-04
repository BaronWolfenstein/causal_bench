"""Exp 50: high-dim learned-embedding covariates as a fraught causal adjustment set.

Demonstrates, in a controlled synthetic DGP, the failure mode found on real 8B
foundation-model embeddings (the SMB rare-subpopulation demo, payoff_v7): when the
observed covariate is a high-fidelity high-dim encoding of the confounders,
treatment becomes near-deterministic given the adjustment set (severe positivity),
and a FLEXIBLE outcome model then attenuates the treatment effect toward null.

Setup (`causal_bench.validation.embedding_positivity`): a low-dim true confounder
Ustar drives A and Y; the observed W is a high-fidelity linear encoding of Ustar.
Confounding strength `conf` is the positivity-severity knob. Two self-validating
controls make the read-out unambiguous:
  * conf=0 (no confounding) -> every method unbiased.
  * LINEAR outcome model -> NO attenuation at any conf (adjustment recovers the
    effect); the pathology is specific to a FLEXIBLE outcome model x positivity,
    and it attenuates even the ORACLE that adjusts for the true low-dim Ustar --
    so it is the flexible-Q x positivity interaction, not high-dimensionality per se.

Read-out: with a flexible (HistGradientBoosting) outcome model the estimate
attenuates toward null monotonically in positivity severity, and doubly-robust
AIPW does not rescue it. The cheap positivity-robust responses (trim to the overlap
region; overlap-weighted ATO) do NOT fully recover either -- which motivates the
sufficient-confounder / dimension-reduction research program (issue #206). This
experiment is what licenses the claim that a raw FM embedding is not a safe causal
adjustment set, and scopes what does and does not fix it.

Run: python -m experiments.exp50_embedding_adjustment
"""
import json
from pathlib import Path

from causal_bench.validation.embedding_positivity import report_rows

OUT_DIR = Path("results/exp50_embedding_adjustment")
CONFS = (0.0, 1.0, 2.0, 3.0, 5.0)


def run(*, n=2500, n_reps=12, seed=0):
    return {
        "flexible": report_rows(n=n, n_reps=n_reps, confs=CONFS, flex=True, seed=seed),
        "linear_control": report_rows(n=n, n_reps=n_reps, confs=CONFS, flex=False, seed=seed),
    }


def _table(rows) -> list[str]:
    hdr = f"  {'conf':>5} {'frac_ext':>8} {'oracle':>8} {'naive_W':>8} {'dr_ato':>8} {'prog_sc':>8}"
    lines = [hdr, "  " + "-" * (len(hdr) - 2)]
    for r in rows:
        lines.append(f"  {r['conf']:>5.1f} {r['frac_extreme']:>8.2f} "
                     f"{r['oracle_Ustar']:>+8.3f} {r['naive_fullW']:>+8.3f} "
                     f"{r['dr_ato_W']:>+8.3f} {r['prog_score_W']:>+8.3f}")
    return lines


def report(res) -> str:
    lines = ["Exp 50: embedding-as-adjustment-set (bias vs true ATE = 1.0; every-subpop ATE == 1.0)",
             "", "FLEXIBLE (HistGradientBoosting) outcome  -- the pathology:"]
    lines += _table(res["flexible"])
    lines += ["", "LINEAR (Ridge) outcome  -- control (no pathology):"]
    lines += _table(res["linear_control"])
    lines += ["",
              "Read: flexible-Q attenuates toward null, monotone in positivity, even for the",
              "oracle; linear control does not; conf=0 unbiased. dr_ato (tier-1, positivity-robust)",
              "~halves the bias; prog_score (tier-2, prognostic-score reduction) LARGELY RECOVERS",
              "-- the propensity direction can't escape positivity but the prognostic score can.",
              "See issue #206."]
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 50: embedding as causal adjustment set")
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--n_reps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    res = run(n=a.n, n_reps=a.n_reps, seed=a.seed)
    print(report(res))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
