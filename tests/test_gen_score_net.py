import numpy as np
import pytest
torch = pytest.importorskip("torch")
from causal_bench.generative.score_net import ScoreMLP, make_torch_score_fn, train_score
from causal_bench.generative.vpsde import Schedule

def test_torch_score_fn_matches_contract_and_shape():
    sch = Schedule(n_steps=50)
    model = ScoreMLP(dim=2, hidden=32)
    score_fn = make_torch_score_fn(model, sch)
    x = np.random.default_rng(0).standard_normal((16, 2))
    s = score_fn(x, 10)
    assert s.shape == (16, 2) and np.isfinite(s).all()

def test_train_score_reduces_loss_on_a_gaussian():
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((512, 2)) + 3.0
    model = ScoreMLP(dim=2, hidden=32)
    losses = []
    train_score(model, X, sch, epochs=3, rng=np.random.default_rng(0), _loss_log=losses)
    assert losses[-1] < losses[0]                       # learns something
