"""Residual-conflict gate (exp44 Part-B first step): Y_control ⫫ Source | embedding."""
import numpy as np

from causal_bench.validation.residual_conflict_gate import gate_by_region, residual_conflict_gate


def test_no_residual_conflict_skips_borrowing():
    """Outcome depends only on the embedding, source assigned independently → Y ⫫ Source | Z →
    matched SCA suffices, gate does not trigger."""
    rng = np.random.default_rng(0)
    n = 2000
    Z = rng.normal(size=(n, 2))
    source = (rng.random(n) < 0.5).astype(float)
    outcome = Z[:, 0] + 0.5 * Z[:, 1] + rng.normal(size=n)          # no residual source effect
    g = residual_conflict_gate(outcome, source, Z, n_perm=80, rng=rng)
    assert g["residual_conflict"] is False


def test_residual_conflict_triggers_borrowing():
    """A residual source effect given the embedding (the modality-gap / era-drift analog) →
    Y ⫫̸ Source | Z → gate triggers, borrowing warranted."""
    rng = np.random.default_rng(1)
    n = 2000
    Z = rng.normal(size=(n, 2))
    source = (rng.random(n) < 0.5).astype(float)
    outcome = Z[:, 0] + 0.8 * source + rng.normal(size=n)           # residual dependence on source
    g = residual_conflict_gate(outcome, source, Z, n_perm=80, rng=rng)
    assert g["residual_conflict"] is True


def test_gate_by_region_localizes_conflict():
    """Conflict confined to one era: the region-wise gate flags that era and clears the other."""
    rng = np.random.default_rng(2)
    n = 3000
    Z = rng.normal(size=(n, 2))
    era = (rng.random(n) < 0.5).astype(int)
    source = (rng.random(n) < 0.5).astype(float)
    outcome = Z[:, 0] + 0.9 * source * era + rng.normal(size=n)     # conflict only in era==1
    out = gate_by_region(outcome, source, Z, era, n_perm=80, rng=rng)
    assert out[0]["residual_conflict"] is False
    assert out[1]["residual_conflict"] is True
