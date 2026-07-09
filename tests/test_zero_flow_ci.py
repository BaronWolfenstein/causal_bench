"""Zero-flow CI test — validated on synthetic DGPs with known CI structure."""
import numpy as np

from causal_bench.detectors.zero_flow_ci import (
    markov_blanket, zero_flow_ci_test, zero_flow_statistic)


def _indep(n, rng):
    """X ⫫ Y | Z: both driven by Z, no direct X–Y link."""
    Z = rng.standard_normal((n, 1))
    X = Z[:, 0] + 0.5 * rng.standard_normal(n)
    Y = Z[:, 0] + 0.5 * rng.standard_normal(n)
    return X, Y, Z


def _dep(n, rng):
    """Y depends on X even given Z (CI violated)."""
    Z = rng.standard_normal((n, 1))
    X = Z[:, 0] + 0.5 * rng.standard_normal(n)
    Y = X + Z[:, 0] + 0.5 * rng.standard_normal(n)
    return X, Y, Z


def test_supports_when_conditionally_independent():
    rng = np.random.default_rng(0)
    X, Y, Z = _indep(300, rng)
    r = zero_flow_ci_test(X, Y, Z, n_perm=40, rng=rng)
    assert r.verdict == "supports"
    assert r.test == "zero-flow-ci" and r.effective_n == 300


def test_refutes_when_conditionally_dependent():
    rng = np.random.default_rng(1)
    X, Y, Z = _dep(300, rng)
    r = zero_flow_ci_test(X, Y, Z, n_perm=40, rng=rng)
    assert r.verdict == "refutes"
    assert r.p_value < 0.05


def test_underpowered_below_min_n():
    rng = np.random.default_rng(2)
    X, Y, Z = _indep(30, rng)
    r = zero_flow_ci_test(X, Y, Z, min_n=50, rng=rng)
    assert r.verdict == "underpowered"


def test_statistic_larger_for_shifted_distribution():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((300, 2))
    B = rng.standard_normal((300, 2))          # same distribution
    C = rng.standard_normal((300, 2)) + 3.0    # shifted
    s_same = zero_flow_statistic(A, B, rng=rng)
    s_diff = zero_flow_statistic(A, C, rng=rng)
    assert s_diff > s_same


def test_markov_blanket_of_chain_middle():
    # chain X0 -> X1 -> X2 ; MB(X1) = {X0, X2}
    rng = np.random.default_rng(4)
    n = 300
    x0 = rng.standard_normal(n)
    x1 = x0 + 0.5 * rng.standard_normal(n)
    x2 = x1 + 0.5 * rng.standard_normal(n)
    data = np.column_stack([x0, x1, x2])
    mb = set(markov_blanket(1, data, n_perm=40, rng=rng))
    assert mb == {0, 2}


def test_result_maps_onto_sga_empirical_slot():
    # the fields SGA's EmpiricalCIResult needs are all present
    rng = np.random.default_rng(5)
    X, Y, Z = _indep(120, rng)
    r = zero_flow_ci_test(X, Y, Z, n_perm=20, rng=rng)
    assert r.verdict in ("supports", "refutes", "underpowered")
    assert isinstance(r.effective_n, int) and r.test == "zero-flow-ci"
