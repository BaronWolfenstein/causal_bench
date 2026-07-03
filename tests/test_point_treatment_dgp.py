import numpy as np
import pytest

from causal_bench.dgp.point_treatment import (
    GATE, SURFACES, draw_point_treatment, true_Q, true_g, true_tau)


@pytest.mark.parametrize("surface", SURFACES)
def test_draw_columns_and_types(surface):
    df = draw_point_treatment(n=500, surface=surface, seed=1)
    assert list(df.columns) == ["W1", "W2", "W3", "W4", "A", "Y"]
    assert set(df["A"].unique()) <= {0, 1}
    assert set(df["Y"].unique()) <= {0, 1}
    assert len(df) == 500


@pytest.mark.parametrize("surface", SURFACES)
def test_g_bounds(surface):
    rng = np.random.default_rng(2)
    W = rng.normal(size=(2000, 4)) * 3  # deliberately wide tails
    g = true_g(W, surface)
    assert g.min() >= 0.1 - 1e-12 and g.max() <= 0.9 + 1e-12


def test_true_tau_cached_and_stable():
    t1 = true_tau("jumpy")
    t2 = true_tau("jumpy")
    assert t1 == t2  # cached
    assert -1.0 < t1 < 1.0 and t1 != 0.0


def test_jumpy_surface_is_discontinuous_smooth_is_not():
    base = np.zeros((1, 4))
    lo, hi = base.copy(), base.copy()
    lo[0, 0], hi[0, 0] = GATE - 1e-6, GATE + 1e-6
    jump_q = abs(true_Q(1, hi, "jumpy") - true_Q(1, lo, "jumpy"))[0]
    jump_g = abs(true_g(hi, "jumpy") - true_g(lo, "jumpy"))[0]
    smooth_q = abs(true_Q(1, hi, "smooth") - true_Q(1, lo, "smooth"))[0]
    assert jump_q > 0.05 and jump_g > 0.05
    assert smooth_q < 1e-4


def test_empirical_tau_matches_true_tau():
    # Oracle G-computation on a big draw should land near the cached truth.
    df = draw_point_treatment(n=200_000, surface="smooth", seed=7)
    W = df[["W1", "W2", "W3", "W4"]].values
    emp = float(np.mean(true_Q(1, W, "smooth") - true_Q(0, W, "smooth")))
    assert abs(emp - true_tau("smooth")) < 0.01
