"""Exp59 causal extension: extra active months from a baseline retention offer.

python -m experiments.exp59_subscription_offer --n 6000 --seeds 20
Add --concrete for a separate strict-horizon continuous-time comparator (needs R).
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from causal_bench.validation.subscription_offer import run


def report(result):
    lines = ["# Exp59 — causal retention offers", "", str(result["configuration"]), "", result["units"], "",
             "Prespecified baseline offers, benefit and null effects, randomized and confounded assignment. "
             "No model selection on evaluation labels. Oracle data are separate from fitting inputs.", "",
             "| Case | Estimator | True months | Bias | RMSE | Wald coverage |",
             "|---|---|---:|---:|---:|---:|"]
    for case, methods in result["summaries"].items():
        for method, m in methods.items():
            coverage = f"{m['wald_coverage']:.3f}" if "wald_coverage" in m else "not reported"
            lines.append(f"| {case} | {method} | {m['truth_months']:.4f} | {m['bias_months']:.4f} | "
                         f"{m['rmse_months']:.4f} | {coverage} |")
    lines += ["", "AIPW is cross-fitted one-step estimation, not TMLE. The two misspecification arms test "
              "correct outcome nuisance versus jointly correct treatment/censoring nuisances. The naive "
              "monthly-hazard model ignores targeting covariates and conditional censoring. "
              "Misspecified arms test point-estimate consistency only; their plug-in IF intervals are suppressed.", "",
              "Wald intervals target a superpopulation of iid eligible subscribers, not individual treatment effects. "
              "Coverage is a Monte Carlo diagnostic; small replicate counts do not establish nominal coverage. "
              "Coverage Wilson intervals are in metrics.json, including when all replicates cover. "
              "Per-replicate clipping counts and raw overlap minima are in metrics.json. No account clustering, "
              "repeated offers, hidden confounding, competing exits, or transport guarantee is claimed.", "",
              "Causal cross-fitting does not replace exp59's chronological prediction evaluation. "
              "The time origin here is eligibility and baseline offer assignment; engagement is measured before the offer.", "",
              "## Optional concrete companion", "",
              "The existing concrete integration computes grid-integrated continuous-time RMST. Its comparator "
              "uses a separate exponential-time DGP and its own oracle, with observation beyond the target horizon. "
              "It must not be labeled the exact monthly sum. Strict horizon mode rejects a requested horizon "
              "at/after the last observed cancellation rather than silently shortening it. Legacy callers retain "
              "their default horizon policy. R failures/unavailability are reported, never filled with Python results.", "",
              json.dumps(result["concrete_continuous_companions"], indent=2), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--concrete", action="store_true")
    p.add_argument("--output", type=Path, default=Path("results/exp59_subscription_offer"))
    a = p.parse_args()
    result = run(n=a.n, seeds=a.seeds, horizon=a.horizon, folds=a.folds, include_concrete=a.concrete)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "causal_bench/validation/subscription_offer.py",
               root / "causal_bench/estimators/monthly_offer.py", root / "causal_bench/estimators/monthly_hazard.py",
               root / "causal_bench/estimators/concrete_rmst.py", root / "r_scripts/concrete_bridge.R"]
    result["provenance"] = {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__,
                            "source_sha256": {str(x.relative_to(root)): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (a.output / "REPORT.md").write_text(report(result))
    print(report(result))
    print(f"Executed causal retention benchmark: {a.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
