"""Tests for exp22 — M-bias sensitivity (adjust-and-still-open collider).

Pins the collider-caveat story end-to-end: in a pure M-structure the *unadjusted*
estimate is unbiased, adjusting for the pre-treatment collider M *introduces* bias
(coverage collapses), and a collider-aware backdoor set excludes M. Also pins the
concrete "MB != adjustment set" claim: markov_blanket() returns M but the backdoor
set does not.
"""
import numpy as np
import pytest

from experiments.exp22_mbias_sensitivity import (
    A_IDX, M_IDX, Y_IDX,
    simulate_m_structure, estimate_ate, conditioning_opens_path,
    backdoor_set, run_mbias_sweep,
)
from causal_bench.detectors.zero_flow_ci import markov_blanket


def test_m_is_a_marginal_lookalike_confounder():
    # M looks like a confounder: correlated with BOTH A and Y...
    d = simulate_m_structure(collider_strength=1.0, tau=0.0, n=8000, seed=1)
    A, M, Y = d[:, A_IDX], d[:, M_IDX], d[:, Y_IDX]
    assert abs(np.corrcoef(A, M)[0, 1]) > 0.2
    assert abs(np.corrcoef(M, Y)[0, 1]) > 0.2
    # ...yet A and Y are (near-)independent MARGINALLY under tau=0 (U1 ⟂ U2).
    assert abs(np.corrcoef(A, Y)[0, 1]) < 0.06


def test_adjusting_for_M_introduces_bias():
    d = simulate_m_structure(collider_strength=1.0, tau=0.0, n=12000, seed=2)
    unadj = estimate_ate(d, adjust_cols=())["tau_hat"]
    adj_m = estimate_ate(d, adjust_cols=(M_IDX,))["tau_hat"]
    # unadjusted ~ unbiased (nothing to adjust for); adjusting for M opens the path
    assert abs(unadj) < 0.03
    assert abs(adj_m) > 0.10
    assert abs(adj_m) > abs(unadj) + 0.07


def test_effect_recovered_when_tau_nonzero_only_without_M():
    tau = 0.5
    d = simulate_m_structure(collider_strength=1.0, tau=tau, n=12000, seed=3)
    unadj = estimate_ate(d, adjust_cols=())["tau_hat"]
    adj_m = estimate_ate(d, adjust_cols=(M_IDX,))["tau_hat"]
    assert abs(unadj - tau) < 0.03            # unbiased for the true effect
    assert abs(adj_m - tau) > 0.07            # biased by adjusting for the collider


def test_conditioning_on_M_opens_the_path():
    d = simulate_m_structure(collider_strength=1.2, tau=0.0, n=4000, seed=4)
    assert conditioning_opens_path(d, rng=np.random.default_rng(0)) is True
    # a genuine confounder-free flat pair should NOT open
    rng = np.random.default_rng(5)
    flat = rng.normal(size=(4000, 3))         # all independent
    assert conditioning_opens_path(flat, rng=np.random.default_rng(1)) is False


def test_backdoor_set_excludes_the_collider():
    d = simulate_m_structure(collider_strength=1.0, tau=0.0, n=4000, seed=6)
    S = backdoor_set(d, candidates=(M_IDX,), rng=np.random.default_rng(2))
    assert M_IDX not in S


def test_markov_blanket_includes_M_but_adjustment_set_does_not():
    # The concrete MB != adjustment set claim, end-to-end.
    d = simulate_m_structure(collider_strength=1.2, tau=0.3, n=1500, seed=7)
    mb = markov_blanket(A_IDX, d, n_perm=40, rng=np.random.default_rng(3))
    S = backdoor_set(d, candidates=(M_IDX, Y_IDX), rng=np.random.default_rng(4))
    assert M_IDX in mb                        # association object keeps M
    assert M_IDX not in S                     # backdoor-valid set drops it


def test_sweep_bias_grows_with_collider_strength():
    res = run_mbias_sweep([0.0, 0.6, 1.2], tau=0.0, n=6000, n_sims=12, seed=11)
    bias_adj = [abs(b) for b in res["adjust_M"]["bias"]]
    # adjust-for-M bias is monotone-ish increasing in collider strength
    assert bias_adj[0] < bias_adj[-1]
    assert bias_adj[0] < 0.03                 # no collider -> no M-bias
    # unadjusted stays ~unbiased across the sweep; its coverage stays near nominal
    assert max(abs(b) for b in res["unadjusted"]["bias"]) < 0.04
    assert min(res["unadjusted"]["coverage"]) > 0.80
    # adjusting for M collapses coverage at the strong end
    assert res["adjust_M"]["coverage"][-1] < 0.80
