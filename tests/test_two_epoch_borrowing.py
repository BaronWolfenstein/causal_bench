"""Tests for exp44 two-epoch borrowing under prior–data conflict (needs the 3.12 [bayes] stack)."""
import numpy as np
import pytest

pytest.importorskip("pymc")            # skip on CPU/3.10 installs without PyMC/NumPyro

from causal_bench.validation.two_epoch_borrowing import (  # noqa: E402
    map_prior_from_historical, sim_epoch, type_i_curve,
)


def test_map_prior_recovers_historical_effect():
    """The MAP prior for the current μ is the historical posterior of μ — it should sit on the
    historical effect δ."""
    rng = np.random.default_rng(0)
    th, se = sim_epoch(12, 60, mu=0.5, tau=0.15, sigma=1.0, rng=rng)
    m, s = map_prior_from_historical(th, se, draws=300, tune=300, seed=0)
    assert abs(m - 0.5) < 3 * s + 0.1


def test_no_conflict_is_nominal():
    """δ=0 (historical also null, no conflict): robust_map keeps Type-I near nominal."""
    curve = type_i_curve([0.0], "robust_map", n_reps=12, draws=200, tune=200, seed=1)
    assert curve[0]["type_i"] <= 0.25


def test_conflict_map_inflates_robust_map_adapts():
    """δ=0.6 (strong conflict): naive MAP is dragged to the historical effect and inflates
    Type-I; robust-MAP down-weights the conflicting prior; flat ignores history."""
    ti = {p: type_i_curve([0.6], p, n_reps=12, draws=200, tune=200, seed=2)[0]["type_i"]
          for p in ("flat", "map", "robust_map")}
    assert ti["map"] > ti["robust_map"]        # robust-MAP adapts under conflict
    assert ti["flat"] <= 0.25                  # flat borrows nothing -> nominal
    assert ti["map"] >= 0.5                    # map dragged to the historical effect
