import numpy as np
import pytest
torch = pytest.importorskip("torch")
from causal_bench.generative.score_net import ScoreMLP, make_torch_score_fn, train_score
from causal_bench.generative.vpsde import Schedule

# Force device='cpu': deterministic, and it dodges the torch MPS-backend Adam
# segfault on Apple Silicon (MPS is beta/opportunistic — never the test device).
def test_torch_score_fn_matches_contract_and_shape():
    sch = Schedule(n_steps=50)
    model = ScoreMLP(dim=2, hidden=32)
    score_fn = make_torch_score_fn(model, sch, device="cpu")
    x = np.random.default_rng(0).standard_normal((16, 2))
    s = score_fn(x, 10)
    assert s.shape == (16, 2) and np.isfinite(s).all()

def test_train_score_reduces_loss_on_a_gaussian():
    torch.manual_seed(0)                                # eps ~ torch global RNG; seed it
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((512, 2)) + 3.0
    model = ScoreMLP(dim=2, hidden=32)
    losses = []
    # enough epochs that the training signal beats per-epoch DSM noise; compare
    # windowed means, not two single noisy samples (the old flaky assertion).
    train_score(model, X, sch, epochs=60, rng=np.random.default_rng(0),
                device="cpu", _loss_log=losses)
    assert np.mean(losses[-10:]) < np.mean(losses[:10]) - 0.02   # learns something
