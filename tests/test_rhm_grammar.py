"""RHM grammar + exact rule-BP — canonical SFW Part B (#131). The grammar produces
the genuine class-overlap transition the symmetric broadcast could not. Synthetic,
exact BP."""
import numpy as np

from causal_bench.diagnostics.rhm_grammar import (
    make_rhm, rhm_class_overlap,
)
from causal_bench.diagnostics.tree_reconstruction import diffusion_class_overlap


def test_grammar_has_expected_shape():
    rules = make_rhm(8, 2, 3, seed=0)
    assert rules.shape == (8, 3, 2) and rules.min() >= 0 and rules.max() < 8


def test_rhm_shows_a_class_overlap_transition():
    hi = rhm_class_overlap(8, 2, 2, 7, 1.0, n_trees=300, seed=1)   # clean
    lo = rhm_class_overlap(8, 2, 2, 7, 0.3, n_trees=300, seed=1)   # heavily corrupted
    assert hi > 0.5                                                # class recoverable clean
    assert lo < 0.15                                              # destroyed under corruption
    assert hi - lo > 0.5                                          # a genuine transition


def test_overlap_monotone_non_decreasing_in_theta():
    th = [0.3, 0.5, 0.7, 0.9, 1.0]
    m = [rhm_class_overlap(8, 2, 2, 7, t, n_trees=250, seed=2) for t in th]
    assert np.all(np.diff(m) >= -0.03)                            # less corruption ⇒ ≥ overlap


def test_grammar_transition_vs_broadcast_flatness():
    # THE canonical-SFW contrast: the grammar has a large-range transition; the
    # symmetric broadcast is flat (no transition) at the same corruption sweep.
    g_hi = rhm_class_overlap(8, 2, 2, 7, 1.0, n_trees=250, seed=3)
    g_lo = rhm_class_overlap(8, 2, 2, 7, 0.3, n_trees=250, seed=3)
    b_hi = diffusion_class_overlap(12, 3, 8, 0.85, 1.0, pop=4000, seed=3)
    b_lo = diffusion_class_overlap(12, 3, 8, 0.85, 0.3, pop=4000, seed=3)
    assert (g_hi - g_lo) > 0.5                                    # grammar: real transition
    assert abs(b_hi - b_lo) < 0.10                               # broadcast: flat
