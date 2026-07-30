"""Ambiguity knob for the RHM (#137): non-injective leaf emission.

Ambiguity IS non-injective emission — distinct latent symbols emitting one surface
token, so several parses fit one string. The paper motivating this
(Parley/Cagnetta/Wyart, arXiv:2602.06065) generalises the RHM along exactly this axis,
and the consequence for us is that an unambiguous grammar makes reconstruction failure
attributable to NOISE, which is what licenses reading a threshold as identifiability.
"""
import numpy as np
import pytest

from causal_bench.diagnostics.rhm_grammar import (
    make_rhm, make_emission, emission_matrix, _leaf_channel, _bp_belief,
    _bp_belief_batch, rhm_class_overlap, make_lowrank_corruption,
)

V, S, M, D = 8, 2, 3, 3


def test_p_merge_zero_is_the_identity_control():
    # Prespecified control: at p_merge=0 nothing may move, asserted not eyeballed.
    assert np.array_equal(make_emission(V, 0.0), np.arange(V))


@pytest.mark.parametrize("corr", [None, "lowrank"])
def test_identity_emission_reproduces_the_unambiguous_path_exactly(corr):
    C = make_lowrank_corruption(V, 2, seed=1)[0] if corr == "lowrank" else None
    ident = make_emission(V, 0.0)
    a = _leaf_channel(V, 0.7, C, None)
    b = _leaf_channel(V, 0.7, C, ident)
    assert np.allclose(a, b, atol=0, rtol=0)          # bit-identical, not merely close


def test_leaf_channel_is_a_proper_conditional_distribution():
    # Rows are P(observed token | true symbol) and must normalise for ANY merge level.
    for pm in (0.0, 0.5, 1.0):
        E = make_emission(V, pm, seed=3)
        for C in (None, make_lowrank_corruption(V, 2, seed=2)[0]):
            ch = _leaf_channel(V, 0.6, C, E)
            assert np.allclose(ch.sum(axis=1), 1.0)
            assert (ch >= 0).all()


def test_merged_symbols_are_observationally_identical_under_uniform_corruption():
    # The mechanism: if E(a)==E(b) no observation distinguishes a from b, so their
    # channel rows coincide. This is what makes the floor IRREDUCIBLE rather than noisy.
    E = make_emission(V, 1.0, seed=5)
    ch = _leaf_channel(V, 0.6, None, E)
    pairs = [(a, b) for a in range(V) for b in range(a + 1, V) if E[a] == E[b]]
    assert pairs, "p_merge=1 must produce merged pairs"
    for a, b in pairs:
        assert np.allclose(ch[a], ch[b])


def test_batch_and_scalar_bp_agree_under_ambiguity():
    rules = make_rhm(V, S, M, seed=0)
    E = make_emission(V, 0.5, seed=7)
    rng = np.random.default_rng(0)
    leaves = rng.integers(0, V, size=(4, S ** D))
    batch = _bp_belief_batch(leaves, D, rules, V, 0.7, None, E)
    for i in range(leaves.shape[0]):
        one = _bp_belief(leaves[i], D, rules, V, 0.7, None, E)
        assert np.allclose(batch[i], one / (one.sum() + 1e-300))


def test_ambiguity_lowers_class_overlap_at_low_corruption():
    # The scientific claim: even at theta close to 1 (little corruption), merging caps
    # recoverable class information. A pure noise knob cannot produce this.
    kw = dict(n_trees=120, seed=0, grammar_seed=0)
    clean = rhm_class_overlap(V, S, M, D, 0.95, emission=make_emission(V, 0.0), **kw)
    merged = rhm_class_overlap(V, S, M, D, 0.95, emission=make_emission(V, 1.0, seed=5), **kw)
    assert clean > merged, f"ambiguity must reduce overlap: {clean:.3f} -> {merged:.3f}"
