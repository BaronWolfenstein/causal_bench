"""Multimodal manifold alignment (#160). Load-bearing test:
`test_alignment_recovers_correspondence_from_anchors` — two warped views of one
manifold, a few anchor correspondences, and the shared eigenbasis places
corresponding points on top of each other (generalizing beyond the anchors).
Metric is normalized alignment error + top-k retrieval, NOT exact rank-1 NN,
which is ill-posed under dense sampling."""
import numpy as np

from causal_bench.geometry.alignment import (
    align, alignment_error, retrieval_accuracy, make_aligned_pair,
    joint_laplacian, _knn_weights)


def test_alignment_recovers_correspondence_from_anchors():
    Xa, Xb = make_aligned_pair(n=200, seed=0, warp=True)
    rng = np.random.default_rng(0)
    anchors = rng.choice(200, size=30, replace=False)          # 15% anchored
    corr = [(int(i), int(i)) for i in anchors]

    Fa, Fb, evals = align(Xa, Xb, corr, d=2, k=12, mu=80.0)
    assert alignment_error(Fa, Fb) < 0.12                       # corresponding pts coincide
    assert retrieval_accuracy(Fa, Fb, k=10) > 0.7               # correspondent retrievable
    # both views share the SAME frame (not two separate embeddings)
    assert abs(np.corrcoef(Fa[:, 0], Fb[:, 0])[0, 1]) > 0.9


def test_alignment_robust_across_seeds():
    rets = []
    for seed in range(5):
        Xa, Xb = make_aligned_pair(n=200, seed=seed, warp=True)
        rng = np.random.default_rng(seed)
        anchors = rng.choice(200, size=30, replace=False)
        Fa, Fb, _ = align(Xa, Xb, [(int(i), int(i)) for i in anchors], d=2, k=12, mu=80.0)
        rets.append(retrieval_accuracy(Fa, Fb, k=10))
    assert np.mean(rets) > 0.75, f"mean retrieval@10 {np.mean(rets):.2f} across seeds"


def test_no_warp_pure_rotation_aligns_tightly():
    """Pure rotation (no nonlinear lift): the two views are isometric → alignment
    error should be very small."""
    Xa, Xb = make_aligned_pair(n=200, seed=4, warp=False)
    rng = np.random.default_rng(4)
    anchors = rng.choice(200, size=30, replace=False)
    Fa, Fb, _ = align(Xa, Xb, [(int(i), int(i)) for i in anchors], d=2, k=12, mu=80.0)
    assert alignment_error(Fa, Fb) < 0.1
    assert retrieval_accuracy(Fa, Fb, k=10) > 0.75


def test_more_anchors_do_not_hurt():
    Xa, Xb = make_aligned_pair(n=200, seed=1, warp=True)
    rng = np.random.default_rng(1)
    errs = []
    for m in (15, 40):
        anchors = rng.choice(200, size=m, replace=False)
        Fa, Fb, _ = align(Xa, Xb, [(int(i), int(i)) for i in anchors], d=2, k=12, mu=80.0)
        errs.append(alignment_error(Fa, Fb))
    assert errs[1] <= errs[0] + 0.05                            # more anchors don't hurt


def test_joint_laplacian_structure():
    Xa, Xb = make_aligned_pair(n=30, seed=3, warp=False)
    Wa, Wb = _knn_weights(Xa, 5), _knn_weights(Xb, 5)
    L, D = joint_laplacian(Wa, Wb, [(0, 0), (1, 1)], mu=2.0)
    assert L.shape == (60, 60)
    assert L[0, 30] < 0 and L[30, 0] < 0                        # cross-link A0 ↔ B0
    assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0.0, atol=1e-8)  # Laplacian rows ~0
