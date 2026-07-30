"""On-box (A100/CUDA) validation of the torch score net + checkpoints (issue #116).

CUDA-gated regression test: skipped everywhere without a CUDA device (so CI and
macOS stay green), runs the four #116 acceptance checks when a GPU is present.
Mirrors the SMC §1a cuda-parity acceptance pattern.
"""
import copy
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required (on-box only)"
)

from causal_bench.generative.vpsde import Schedule
from causal_bench.generative.score_net import (
    ScoreMLP, train_score, make_optimizer, make_torch_score_fn,
)
from causal_bench.generative.checkpoint import save_checkpoint, load_checkpoint

PROBE = np.linspace(-1.0, 1.0, 12).reshape(4, 3)


def _data():
    return np.random.default_rng(0).standard_normal((256, 3))


def test_cuda_cpu_forward_parity():
    """Same weights -> score-fn outputs match cpu vs cuda to fp tolerance."""
    sch = Schedule(n_steps=50)
    torch.manual_seed(100)
    m = ScoreMLP(3)
    torch.manual_seed(0)
    train_score(m, _data(), sch, opt=make_optimizer(m), epochs=5,
                rng=np.random.default_rng(1), device="cpu")
    s_cpu = make_torch_score_fn(copy.deepcopy(m), sch, device="cpu")(PROBE, t=10)
    s_cuda = make_torch_score_fn(copy.deepcopy(m), sch, device="cuda")(PROBE, t=10)
    assert np.max(np.abs(s_cpu - s_cuda)) < 1e-4


def test_checkpoint_resume_cpu_to_cuda(tmp_path):
    """save(cpu) -> load(map_location=cpu) -> resume train(cuda); the Adam
    moment re-align must avoid a device mismatch on opt.step()."""
    sch = Schedule(n_steps=50)
    X = _data()
    p = tmp_path / "ck.pt"
    m_a = ScoreMLP(3); opt_a = make_optimizer(m_a)
    torch.manual_seed(7)
    train_score(m_a, X, sch, opt=opt_a, epochs=5, rng=np.random.default_rng(2), device="cpu")
    save_checkpoint(p, m_a, opt_a, meta={"dim": 3, "epoch": 5})

    m_b = ScoreMLP(3); opt_b = make_optimizer(m_b)
    meta = load_checkpoint(p, m_b, opt_b, map_location="cpu")
    torch.manual_seed(8)
    train_score(m_b, X, sch, opt=opt_b, epochs=5, rng=np.random.default_rng(3), device="cuda")
    out = make_torch_score_fn(m_b, sch, device="cuda")(PROBE, t=10)
    assert np.isfinite(out).all()
    assert meta == {"dim": 3, "epoch": 5}


def test_checkpoint_rollback_on_cuda(tmp_path):
    """Load an earlier checkpoint on cuda -> weights roll back."""
    sch = Schedule(n_steps=50)
    X = _data()
    p_early = tmp_path / "early.pt"
    m = ScoreMLP(3); opt = make_optimizer(m)
    torch.manual_seed(11)
    train_score(m, X, sch, opt=opt, epochs=3, rng=np.random.default_rng(4), device="cuda")
    save_checkpoint(p_early, m, opt)
    early = make_torch_score_fn(copy.deepcopy(m), sch, device="cuda")(PROBE, t=10)

    train_score(m, X, sch, opt=opt, epochs=5, rng=np.random.default_rng(5), device="cuda")
    moved = make_torch_score_fn(copy.deepcopy(m), sch, device="cuda")(PROBE, t=10)

    load_checkpoint(p_early, m, opt, map_location="cuda")
    back = make_torch_score_fn(m, sch, device="cuda")(PROBE, t=10)
    assert np.allclose(early, back, atol=1e-5)
    assert not np.allclose(early, moved, atol=1e-5)


def test_fp32_baseline_on_cuda():
    """Params are fp32 on cuda and the forward pass is finite (pre mixed-prec)."""
    sch = Schedule(n_steps=50)
    m = ScoreMLP(3)
    out = make_torch_score_fn(m, sch, device="cuda")(PROBE, t=10)
    assert next(m.parameters()).dtype == torch.float32
    assert np.isfinite(out).all()
