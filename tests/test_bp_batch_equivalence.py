"""Locks the batched BP to the exact per-unit oracle. `decode_cohort_labels` now
uses `_bp_belief_batch` for speed; this asserts it stays byte-identical to
`_bp_belief` row-wise, so the RHM / STRUCT-S / SFW (Sclocchi–Favero–Wyart)
analysis that rests on the exact belief-propagation is never silently changed."""
import numpy as np

from causal_bench.dgp.joint_hierarchy import make_joint_hierarchy, sample_joint_cohort
from causal_bench.diagnostics.rhm_grammar import _bp_belief, _bp_belief_batch


def test_bp_batch_equals_loop_beliefs_and_argmax():
    spec = make_joint_hierarchy(4, 3, 2, 2)
    coh = sample_joint_cohort(spec, 25, 4, seed=0)
    v, depth, rules = spec["v"], coh["depth"], spec["rules"]
    leaves = np.stack([np.asarray(l) for l in coh["leaves"]])          # (25, n_leaf)

    batch = _bp_belief_batch(leaves, depth, rules, v, 0.7)             # (25, v)
    loop = np.stack([_bp_belief(coh["leaves"][i], depth, rules, v, 0.7)
                     for i in range(len(coh["leaves"]))])
    assert np.allclose(batch, loop, atol=1e-12)                        # beliefs identical
    assert (batch.argmax(1) == loop.argmax(1)).all()                  # decoded labels identical


def test_bp_batch_matches_at_multiple_theta():
    spec = make_joint_hierarchy(3, 2, 2, 3)
    coh = sample_joint_cohort(spec, 15, 5, seed=1)
    v, depth, rules = spec["v"], coh["depth"], spec["rules"]
    leaves = np.stack([np.asarray(l) for l in coh["leaves"]])
    for theta in (0.3, 0.6, 1.0):
        b = _bp_belief_batch(leaves, depth, rules, v, theta)
        lp = np.stack([_bp_belief(coh["leaves"][i], depth, rules, v, theta)
                       for i in range(len(coh["leaves"]))])
        assert np.allclose(b, lp, atol=1e-12), f"theta={theta}"
