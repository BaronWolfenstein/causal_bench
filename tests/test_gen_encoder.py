import numpy as np
from causal_bench.generative.encoder import RandomProjectionEncoder, make_encoder_pair


def test_encoder_is_deterministic_and_shaped():
    enc = RandomProjectionEncoder(in_dim=8, out_dim=6, seed=0)
    X = np.random.default_rng(1).standard_normal((10, 8))
    assert enc(X).shape == (10, 6)
    assert np.allclose(enc(X), enc(X))                 # frozen/deterministic


def test_encoder_pair_are_distinct_geometries():
    e_gen, e_eval = make_encoder_pair(in_dim=8, out_dim=6)
    X = np.random.default_rng(2).standard_normal((20, 8))
    assert not np.allclose(e_gen(X), e_eval(X))        # decoupled for the #88 guard
