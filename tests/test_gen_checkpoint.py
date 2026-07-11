"""Tests for score-net training checkpoints (A100 spec §6 deliverable).

Save/load/resume of the torch score net — model weights + optimizer state + meta
— so a long GPU run can crash-recover or roll back. Validated on CPU (device
forced to 'cpu' for determinism; the code path is identical on cuda/mps).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from causal_bench.generative.vpsde import Schedule
from causal_bench.generative.score_net import (
    ScoreMLP, train_score, make_optimizer, make_torch_score_fn,
)
from causal_bench.generative.checkpoint import save_checkpoint, load_checkpoint


def _probe(model, sch):
    """Deterministic score-fn output on a fixed probe batch — a fingerprint of
    the model's weights."""
    fn = make_torch_score_fn(model, sch, device="cpu")
    x = np.linspace(-1.0, 1.0, 12).reshape(4, 3)
    return fn(x, t=10)


def _train(seed_torch, rng_seed, epochs, model, opt, X, sch):
    torch.manual_seed(seed_torch)
    train_score(model, X, sch, opt=opt, epochs=epochs,
                rng=np.random.default_rng(rng_seed), device="cpu")


def test_save_load_roundtrip_reproduces_scores(tmp_path):
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((128, 3))
    m = ScoreMLP(3)
    opt = make_optimizer(m)
    _train(0, 1, 5, m, opt, X, sch)
    before = _probe(m, sch)

    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, opt, meta={"dim": 3, "epoch": 5})

    m2 = ScoreMLP(3)                                   # fresh weights
    meta = load_checkpoint(p, m2, map_location="cpu")
    after = _probe(m2, sch)

    assert np.allclose(before, after, atol=1e-6)       # restored weights → same scores
    assert meta == {"dim": 3, "epoch": 5}


def test_load_restores_optimizer_state(tmp_path):
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((128, 3))
    m = ScoreMLP(3)
    opt = make_optimizer(m)
    _train(0, 1, 5, m, opt, X, sch)

    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, opt)

    m2 = ScoreMLP(3)
    opt2 = make_optimizer(m2)
    load_checkpoint(p, m2, opt2, map_location="cpu")

    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert len(s2) > 0 and s1.keys() == s2.keys()      # Adam moments actually restored
    for k in s1:
        assert torch.allclose(s1[k]["exp_avg"], s2[k]["exp_avg"])
        assert s1[k]["step"] == s2[k]["step"]


def test_resume_equals_uninterrupted_training(tmp_path):
    # The decisive test: loading a checkpoint and continuing must reproduce, bit
    # for bit, training that was never interrupted — proof the checkpoint captured
    # everything needed (weights AND optimizer moments).
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((128, 3))

    m = ScoreMLP(3)
    opt = make_optimizer(m)
    _train(0, 1, 5, m, opt, X, sch)                    # first 5 epochs
    p = tmp_path / "ck.pt"
    save_checkpoint(p, m, opt)

    # Path A — continue on the original, in memory
    _train(42, 2, 5, m, opt, X, sch)
    a = _probe(m, sch)

    # Path B — fresh model+opt, load checkpoint, continue with identical randomness
    m2 = ScoreMLP(3)
    opt2 = make_optimizer(m2)
    load_checkpoint(p, m2, opt2, map_location="cpu")
    _train(42, 2, 5, m2, opt2, X, sch)
    b = _probe(m2, sch)

    assert np.allclose(a, b, atol=1e-5)                # resume == uninterrupted


def test_rollback_to_earlier_checkpoint(tmp_path):
    # Two checkpoints; loading the earlier one discards later training.
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((128, 3))
    m = ScoreMLP(3)
    opt = make_optimizer(m)
    _train(0, 1, 5, m, opt, X, sch)
    early = _probe(m, sch)
    p_early = tmp_path / "early.pt"
    save_checkpoint(p_early, m, opt, meta={"epoch": 5})

    _train(7, 3, 5, m, opt, X, sch)                    # train further
    late = _probe(m, sch)
    assert not np.allclose(early, late, atol=1e-6)     # training moved the weights

    m2 = ScoreMLP(3)
    meta = load_checkpoint(p_early, m2, map_location="cpu")
    rolled = _probe(m2, sch)
    assert np.allclose(rolled, early, atol=1e-6)       # rolled back to epoch 5
    assert meta["epoch"] == 5
