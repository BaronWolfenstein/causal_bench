"""Score-net perf layer (A100 spec §6b): bf16 / Tensor Cores / torch.compile.

fp32 is the code default and must stay bit-for-bit unchanged; bf16 is opt-in and
only asserted numerically close on-box (needs cuda + Tensor Cores).  Throughput
(the ~1.8x at real-embedding width) is measured by scripts/score_net_perf.py.
"""
import copy
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from causal_bench.generative.vpsde import Schedule
from causal_bench.generative.score_net import ScoreMLP, train_score, make_torch_score_fn


def test_fp32_default_unchanged():
    """precision='fp32' (the default) trains identically to omitting it — the
    opt-in perf layer must not perturb the default path. Runs on CPU/CI."""
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((128, 3))
    probe = np.linspace(-1, 1, 12).reshape(4, 3)

    def trained_scores(**kw):
        torch.manual_seed(0)
        m = ScoreMLP(3)
        train_score(m, X, sch, epochs=5, rng=np.random.default_rng(1), device="cpu", **kw)
        return make_torch_score_fn(m, sch, device="cpu")(probe, t=10)

    assert np.array_equal(trained_scores(), trained_scores(precision="fp32"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 Tensor Cores need CUDA")
def test_bf16_forward_close_to_fp32():
    """Same weights -> bf16 forward within bf16 tolerance (~a few % of scale)."""
    sch = Schedule(n_steps=100)
    X = np.random.default_rng(0).standard_normal((2048, 256)).astype(np.float32)
    probe = np.random.default_rng(1).standard_normal((256, 256)).astype(np.float32)
    torch.manual_seed(0)
    m = ScoreMLP(256, 1024)
    train_score(m, X, sch, epochs=3, device="cuda", precision="fp32")
    s32 = make_torch_score_fn(copy.deepcopy(m), sch, device="cuda", precision="fp32")(probe, t=10)
    s16 = make_torch_score_fn(copy.deepcopy(m), sch, device="cuda", precision="bf16")(probe, t=10)
    assert np.abs(s16 - s32).max() < 0.05 * np.abs(s32).mean() + 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mode", ["default", "reduce-overhead"])
def test_bf16_and_compile_train_runs(mode):
    """bf16 + torch.compile (default and CUDA-graph modes) run end-to-end finite.
    (Perf finding: compile adds ~0 over eager bf16 for this MLP — bf16/TF32 are
    the win; this test just guards that the modes don't crash.)"""
    sch = Schedule(n_steps=50)
    X = np.random.default_rng(0).standard_normal((512, 64)).astype(np.float32)
    torch.manual_seed(0)
    m = ScoreMLP(64, 256)
    log = []
    train_score(m, X, sch, epochs=4, device="cuda", precision="bf16", compile=True,
                compile_mode=mode, _loss_log=log)
    assert np.isfinite(log).all() and len(log) == 4
