"""exp39 calibration — Type-I (false-positive rate) and power of the zero-flow
CI test over many DGP replicates (#85).

Type-I is a property of the test's *size*, driven by residualization quality and
the permutation null — NOT by n_perm (which only sets the p-value's Monte-Carlo
precision). This measures the empirical false-positive rate under H0 (should sit
near alpha) and the power under H1, over independent DGP replicates.

    python experiments/exp39_ci_calibration.py            # default 200 reps
    python experiments/exp39_ci_calibration.py 60         # faster smoke run

CPU only; ~1-2 min at 200 reps.
"""
import sys

import numpy as np

from causal_bench.detectors.zero_flow_ci import zero_flow_ci_test


def _null_dgp(n, rng):
    """X ⫫ Y | Z (CI holds) — both driven by Z, no direct link."""
    Z = rng.standard_normal((n, 1))
    X = Z[:, 0] + 0.5 * rng.standard_normal(n)
    Y = Z[:, 0] + 0.5 * rng.standard_normal(n)
    return X, Y, Z


def _alt_dgp(n, rng):
    """X → Y | Z (CI violated)."""
    Z = rng.standard_normal((n, 1))
    X = Z[:, 0] + 0.5 * rng.standard_normal(n)
    Y = X + Z[:, 0] + 0.5 * rng.standard_normal(n)
    return X, Y, Z


def rejection_rate(dgp, reps, *, n, n_perm, alpha, seed0):
    rej = 0
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        res = zero_flow_ci_test(*dgp(n, rng), n_perm=n_perm, alpha=alpha, rng=rng)
        rej += int(res.verdict == "refutes")
    return rej / reps


def main(reps=200):
    n, n_perm, alpha = 300, 100, 0.05
    fpr = rejection_rate(_null_dgp, reps, n=n, n_perm=n_perm, alpha=alpha, seed0=0)
    power = rejection_rate(_alt_dgp, reps, n=n, n_perm=n_perm, alpha=alpha, seed0=1_000_000)
    print(f"reps={reps}  n={n}  n_perm={n_perm}  alpha={alpha}")
    print(f"Type-I (false-positive rate under H0): {fpr:.3f}   (target ≈ {alpha})")
    print(f"Power  (rejection rate under H1):      {power:.3f}")
    # rough MC 95% band on the FPR estimate
    se = (alpha * (1 - alpha) / reps) ** 0.5
    print(f"(FPR MC ±1.96·SE ≈ ±{1.96 * se:.3f}; inflation if FPR ≫ alpha + that band)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
