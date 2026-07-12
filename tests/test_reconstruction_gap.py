"""#131 Part A (finished): the KS-vs-reconstruction gap boundary-mapper."""
import numpy as np

from causal_bench.diagnostics.tree_reconstruction import reconstruction_gap


def test_no_gap_for_binary_symmetric_channel():
    r = reconstruction_gap(5, 2, depth=26, pop=5000, seed=1)
    assert r["has_gap"] is False                         # binary: recon = KS (BRZ)
    assert abs(r["theta_recon"] - r["theta_ks"]) < 0.02


def test_gap_opens_for_large_alphabet():
    for q in (5, 15):
        r = reconstruction_gap(5, q, depth=26, pop=5000, seed=1)
        assert r["has_gap"] is True                      # q ≥ 5: hard phase
        assert r["theta_recon"] < r["theta_ks"]          # reconstruct below the KS closeness


def test_gap_widens_with_alphabet_size():
    w5 = reconstruction_gap(5, 5, depth=26, pop=5000, seed=1)["gap_width"]
    w15 = reconstruction_gap(5, 15, depth=26, pop=5000, seed=1)["gap_width"]
    assert w15 > w5                                      # bigger alphabet ⇒ wider gap


def test_empirical_ks_near_analytic():
    r = reconstruction_gap(5, 2, depth=26, pop=5000, seed=1)
    assert abs(r["theta_ks"] - r["theta_ks_analytic"]) < 0.07   # ≈ 1/√b (finite-depth bias)
