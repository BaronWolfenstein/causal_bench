"""q-ary (Potts) BP density evolution — the shared #131 primitive: reconstruction
threshold + the KS-vs-reconstruction (hard-phase) gap. Synthetic; exact BP."""
import numpy as np

from causal_bench.diagnostics.tree_reconstruction import (
    qary_ks_threshold, qary_bp_magnetization, has_reconstruction_gap,
)


def test_reconstruct_above_ks_fail_below():
    for q in (2, 5):
        tks = qary_ks_threshold(3)                        # = 1/√3 ≈ 0.577
        assert qary_bp_magnetization(16, 3, q, 0.80, pop=4000, seed=0) > 0.3   # θ>θ_KS
        assert qary_bp_magnetization(16, 3, q, 0.40, pop=4000, seed=0) < 0.03  # θ<θ_KS


def test_overlap_non_decreasing_in_theta():
    theta = np.linspace(0.35, 0.9, 8)
    m = [qary_bp_magnetization(14, 3, 5, float(t), pop=3000, seed=1) for t in theta]
    assert np.all(np.diff(m) >= -1e-6)                    # easier channel ⇒ ≥ overlap


def test_binary_qary_matches_binary_recursion_threshold():
    # q=2 with θ=1−2ε must reconstruct on the same side of KS as the binary code.
    from causal_bench.diagnostics.tree_reconstruction import ks_threshold
    eps = 0.10                                            # below binary KS(3)=0.211
    m = qary_bp_magnetization(16, 3, 2, 1 - 2 * eps, pop=4000, seed=2)
    assert eps < ks_threshold(3) and m > 0.3


def test_hard_phase_gap_opens_for_large_q_only():
    b = 5
    tks = qary_ks_threshold(b)                            # ≈ 0.447
    theta_gap = 0.95 * tks                                # just below KS
    # q=15: planted reconstructs, BP-from-scratch fails → the hard phase.
    assert has_reconstruction_gap(b, 15, theta_gap, pop=5000, seed=1) is True
    # binary (q=2): no gap — both fail below KS.
    assert has_reconstruction_gap(b, 2, theta_gap, pop=5000, seed=1) is False
