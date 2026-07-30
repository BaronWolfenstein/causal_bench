"""On-box validation of the graph-Laplacian spectral machinery (#99, Tier-1).

    PYTHONPATH=~/causal_bench CUDA_VISIBLE_DEVICES=0 python scripts/spectral_validate.py

Three checks, all on synthetic manifolds with KNOWN geometry — no real data:
  1. Chebyshev filter == dense matrix-function (heat + Matérn), to tolerance.
  2. k-NN graph Laplacian recovers the KNOWN Laplace-Beltrami spectrum of the
     ring S¹ (eigenvalues ∝ 0,1,1,4,4,9,9,... — m²) and the sphere S² (degeneracy
     1,3,5,7 with ratios 0:2:6:12).
  3. Scaling: Chebyshev heat-apply CPU (scipy) vs GPU (cupy) at N up to 1e5 —
     O(edges), sidestepping the O(N³) dense-eigendecomposition wall.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.linalg as sla

from causal_bench.geometry import spectral as S


def check_chebyshev_vs_dense():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5))
    L = S.build_knn_laplacian(X, k=12)
    Ld = L.toarray()
    sig = rng.standard_normal(400)
    for name, dense, applied in [
        ("heat  exp(-0.5 L)", sla.expm(-0.5 * Ld) @ sig, S.heat_apply(L, sig, t=0.5, order=50)),
        ("matern (1+L)^-0.75", sla.fractional_matrix_power(np.eye(400) + Ld, -0.75) @ sig,
         S.matern_apply(L, sig, kappa=1.0, nu=1.5, order=50)),
    ]:
        rel = np.linalg.norm(applied - dense) / np.linalg.norm(dense)
        print(f"  [cheb==dense] {name}: rel err = {rel:.2e}  {'OK' if rel < 5e-3 else 'HIGH'}")


def check_ring_spectrum():
    n = 2000
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.c_[np.cos(th), np.sin(th)]
    L = S.build_knn_laplacian(X, k=8)
    vals, _ = S.lanczos_smallest(L, k=7)
    vals = np.sort(np.real(vals))
    base = vals[1]                                      # first non-zero (m=±1)
    ratios = vals / base
    # known S¹: 0, 1,1, 4,4, 9,9  (m² with degeneracy 2)
    known = np.array([0, 1, 1, 4, 4, 9, 9], float)
    err = np.max(np.abs(ratios - known) / np.maximum(known, 1))
    print(f"  [ring S¹] eigenvalue ratios {np.round(ratios, 2)}")
    print(f"            known m² pattern  {known}   max rel err = {err:.2f}  "
          f"{'OK' if err < 0.15 else 'HIGH'}")


def check_sphere_degeneracy():
    n = 4000
    v = np.random.default_rng(1).standard_normal((n, 3))
    X = v / np.linalg.norm(v, axis=1, keepdims=True)   # uniform-ish on S²
    L = S.build_knn_laplacian(X, k=12)
    vals, _ = S.lanczos_smallest(L, k=9)
    vals = np.sort(np.real(vals))
    base = vals[1]
    print(f"  [sphere S²] first 9 eigenvalues / base: {np.round(vals / base, 2)}")
    print("             known LB: l(l+1) with degeneracy 2l+1 -> "
          "1 zero, then a triple (~1), then a quintuple (~3)")


def scaling_profile():
    print(f"\n  {'N':>8s} {'edges':>10s} {'build s':>8s} {'cpu ms':>8s} {'gpu ms':>8s} {'speedup':>8s}")
    for n in (2000, 20000, 100000):
        rng = np.random.default_rng(0)
        t = rng.uniform(0, 4 * np.pi, n)               # swiss roll
        X = np.c_[t * np.cos(t), 20 * rng.random(n), t * np.sin(t)]
        t0 = time.perf_counter()
        L = S.build_knn_laplacian(X, k=15)
        build = time.perf_counter() - t0
        sig = rng.standard_normal(n)
        # cpu
        t0 = time.perf_counter()
        for _ in range(5):
            S.heat_apply(L, sig, t=0.3, order=40, lmax=2.0)
        cpu_ms = (time.perf_counter() - t0) / 5 * 1e3
        # gpu
        import cupy as cp
        Lg = S.to_gpu(L); sg = cp.asarray(sig)
        for _ in range(2):
            S.heat_apply(Lg, sg, t=0.3, order=40, lmax=2.0)   # warmup
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            S.heat_apply(Lg, sg, t=0.3, order=40, lmax=2.0)
        cp.cuda.runtime.deviceSynchronize()
        gpu_ms = (time.perf_counter() - t0) / 5 * 1e3
        print(f"  {n:8d} {L.nnz:10d} {build:8.2f} {cpu_ms:8.2f} {gpu_ms:8.2f} {cpu_ms/gpu_ms:7.1f}x")
    print("  (dense eigendecomposition is O(N³) -> intractable at N=1e5; Chebyshev "
          "is O(order·edges), linear in the sparse graph)")


if __name__ == "__main__":
    import cupy as cp
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print("[1] Chebyshev vs dense matrix-function")
    check_chebyshev_vs_dense()
    print("[2] known-manifold Laplace-Beltrami spectrum recovery")
    check_ring_spectrum()
    check_sphere_degeneracy()
    print("[3] scaling profile (Chebyshev heat-apply, CPU vs GPU)")
    scaling_profile()
