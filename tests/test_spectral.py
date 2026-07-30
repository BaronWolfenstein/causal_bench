"""Off-box (scipy, CPU) tests for the #99 graph-Laplacian spectral machinery.
The GPU scaling story is measured on-box by scripts/spectral_validate.py; here we
lock in the two correctness invariants that need no GPU: the Chebyshev filter
equals the dense matrix function, and the k-NN Laplacian recovers the known
Laplace-Beltrami spectrum of the ring.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
sla = pytest.importorskip("scipy.linalg")

from causal_bench.geometry import spectral as S


def test_chebyshev_matches_dense_heat_and_matern():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5))
    L = S.build_knn_laplacian(X, k=12)
    Ld = L.toarray()
    sig = rng.standard_normal(300)
    heat = S.heat_apply(L, sig, t=0.5, order=50)
    assert np.linalg.norm(heat - sla.expm(-0.5 * Ld) @ sig) / np.linalg.norm(sig) < 5e-3
    mat = S.matern_apply(L, sig, kappa=1.0, nu=1.5, order=50)
    ref = sla.fractional_matrix_power(np.eye(300) + Ld, -0.75) @ sig
    assert np.linalg.norm(mat - ref) / np.linalg.norm(ref) < 5e-3


def test_ring_recovers_laplace_beltrami_spectrum():
    n = 1500
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.c_[np.cos(th), np.sin(th)]
    L = S.build_knn_laplacian(X, k=8)
    vals, _ = S.lanczos_smallest(L, k=7)
    ratios = np.sort(np.real(vals)) / np.sort(np.real(vals))[1]
    known = np.array([0, 1, 1, 4, 4, 9, 9], float)      # S¹: m² with degeneracy 2
    assert np.max(np.abs(ratios - known) / np.maximum(known, 1)) < 0.15
