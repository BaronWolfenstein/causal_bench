"""exp44 Part-B residual-borrowing layer: gate → protocol response → PS-integrated robust-MAP."""
import numpy as np
import pytest

pytest.importorskip("pymc")

from causal_bench.validation.residual_borrowing_layer import (  # noqa: E402
    gated_borrow, protocol_response, ps_integrated_map_prior,
)
from causal_bench.validation.two_epoch_borrowing import sim_epoch  # noqa: E402


def test_protocol_response_mapping():
    """#180 response keyed to the gate verdict (pure logic)."""
    assert protocol_response({"residual_conflict": False}) == "map"          # exchangeable
    assert protocol_response({"residual_conflict": True}) == "robust_map"     # conflict
    assert protocol_response({"residual_conflict": None}) == "flat"           # underpowered


def test_ps_integration_downweights_noncomparable():
    """A conflicting historical source (μ=0.6) pulls the borrowed prior less when its
    pseudo-studies are propensity-INcomparable (low ps_weight → se inflated → borrow less)."""
    rng = np.random.default_rng(0)
    th, se = sim_epoch(10, 40, mu=0.6, tau=0.15, sigma=1.0, rng=rng)
    m_hi, _ = ps_integrated_map_prior(th, se, np.ones(10), draws=250, tune=250, seed=0)
    m_lo, _ = ps_integrated_map_prior(th, se, np.full(10, 0.05), draws=250, tune=250, seed=0)
    assert abs(m_lo) < abs(m_hi)                                 # down-weighting → prior toward 0


def test_gated_borrow_fires_robust_map_on_conflict():
    """Residual conflict in the gate outcome → gate refutes → the layer fires robust_map."""
    rng = np.random.default_rng(1)
    n = 1500
    Z = rng.normal(size=(n, 2))
    source = (rng.random(n) < 0.5).astype(float)
    gate_outcome = Z[:, 0] + 0.8 * source + rng.normal(size=n)   # residual dependence on source
    th_h, se_h = sim_epoch(10, 40, mu=0.6, tau=0.15, sigma=1.0, rng=rng)
    th_c, se_c = sim_epoch(10, 40, mu=0.0, tau=0.15, sigma=1.0, rng=rng)
    res = gated_borrow(th_c, se_c, th_h, se_h, np.ones(10),
                       gate_outcome=gate_outcome, gate_source=source, gate_embedding=Z,
                       draws=200, tune=200, gate_n_perm=60, seed=2)
    assert res["policy_fired"] == "robust_map"
    assert res["residual_conflict"] is True
