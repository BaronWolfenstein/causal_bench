"""exp39 — zero-flow conditional-independence test on synthetic DGPs (#85).

Demonstrates the estimator on data with known CI structure: CI-holds → supports,
CI-violated → refutes, plus Markov-blanket recovery on a chain. CPU-only
(numpy/sklearn); no torch, no GPU. The verdict maps 1:1 onto SGA's
EmpiricalCIResult (the confidence toolkit's empirical leg).

    python experiments/exp39_zero_flow_ci.py
"""
import numpy as np

from causal_bench.detectors.zero_flow_ci import markov_blanket, zero_flow_ci_test


def main():
    rng = np.random.default_rng(0)
    n = 400
    Z = rng.standard_normal((n, 1))
    X = Z[:, 0] + 0.5 * rng.standard_normal(n)
    Y_ci = Z[:, 0] + 0.5 * rng.standard_normal(n)          # X ⫫ Y | Z holds
    Y_dep = X + Z[:, 0] + 0.5 * rng.standard_normal(n)     # X → Y given Z

    r_ci = zero_flow_ci_test(X, Y_ci, Z, rng=rng)
    r_dep = zero_flow_ci_test(X, Y_dep, Z, rng=rng)
    print(f"X ⫫ Y | Z  (true CI):  verdict={r_ci.verdict:9s} p={r_ci.p_value:.3f}")
    print(f"X ⫫ Y | Z  (violated): verdict={r_dep.verdict:9s} p={r_dep.p_value:.3f}")

    # chain X0 -> X1 -> X2 ; Markov blanket of the middle node is {0, 2}
    x0 = rng.standard_normal(n)
    x1 = x0 + 0.5 * rng.standard_normal(n)
    x2 = x1 + 0.5 * rng.standard_normal(n)
    data = np.column_stack([x0, x1, x2])
    print(f"Markov blanket of X1: {markov_blanket(1, data, rng=rng)}  (expect [0, 2])")


if __name__ == "__main__":
    main()
