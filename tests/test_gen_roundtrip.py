import numpy as np
from causal_bench.generative.vpsde import Schedule, gaussian_score
from causal_bench.generative.roundtrip import per_mode_roundtrip


def test_faithful_roundtrip_reconstructs_both_modes():
    sch = Schedule(n_steps=150)
    rng = np.random.default_rng(0)
    rare = rng.standard_normal((40, 1)) + 4.0
    common = rng.standard_normal((200, 1))
    # analytic score of the pooled 2-component structure, approximated per-mode:
    score_fn = lambda x, t: gaussian_score(x, t, np.array([0.0]), np.eye(1), sch)
    rr, cr = per_mode_roundtrip(rare, common, score_fn, sch, t_start=20, rng=rng)
    assert rr.shape == rare.shape and cr.shape == common.shape
    assert np.linalg.norm(rr - rare, axis=1).mean() < 0.3      # rare reconstructs
    assert np.linalg.norm(cr - common, axis=1).mean() < 1.0     # common reconstructs
