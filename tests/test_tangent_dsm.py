"""Tests for the #99 tangent-space-penalty DSM (Layer-2, gated).
Local-normal estimation is CPU (runs anywhere); the penalty's effect on the
learned score is torch/cuda-gated (measured on-box by the validate script)."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from causal_bench.geometry.tangent_dsm import estimate_local_normals

torch = pytest.importorskip("torch")


def test_local_normals_perpendicular_to_a_plane():
    """Points on the xy-plane in R^3 -> the estimated normal is the z-axis."""
    rng = np.random.default_rng(0)
    X = np.c_[rng.standard_normal((800, 2)), np.zeros(800)].astype(np.float32)
    N = estimate_local_normals(X, k=15, intrinsic_dim=2)
    z = np.abs(N[:, 0, 2])
    assert np.median(z) > 0.99


@pytest.mark.skipif(not torch.cuda.is_available(), reason="training needs a GPU (on-box)")
def test_tangent_penalty_makes_score_more_tangent():
    from causal_bench.generative.vpsde import Schedule
    from causal_bench.generative.score_net import ScoreMLP
    from causal_bench.geometry.tangent_dsm import (
        train_score_tangent, normal_component_fraction)
    rng = np.random.default_rng(1)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, 1500)
    X = np.c_[t * np.cos(t), rng.uniform(0, 21, 1500), t * np.sin(t)].astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    sch = Schedule(n_steps=100)
    normals = estimate_local_normals(X, k=15, intrinsic_dim=2)
    torch.manual_seed(0); base = ScoreMLP(3, 128)
    train_score_tangent(base, X, sch, normals, lam=0.0, epochs=150, device="cuda")
    torch.manual_seed(0); pen = ScoreMLP(3, 128)
    train_score_tangent(pen, X, sch, normals, lam=3.0, epochs=150, device="cuda")
    nb = normal_component_fraction(base, X, normals, sch, device="cuda")
    npn = normal_component_fraction(pen, X, normals, sch, device="cuda")
    assert npn < 0.5 * nb
