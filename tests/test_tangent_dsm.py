"""Tests for the tangent-space-penalty DSM + gap-sampler prototype.

Instantiates the PR #99 spec loss on a synthetic curved 1-manifold (an arc in
R^2). Pins: (1) the manifold geometry / tangent projector, (2) that the
gap-sampler draws near-manifold points inside an under-covered arc, and (3) the
payoff — a score trained with the tangent penalty on gap-sampled points denoises
gap-region points back onto the manifold, where a plain-DSM score (no data in the
gap) drifts off.
"""
import numpy as np
import pytest

from causal_bench.generative.tangent_dsm import (
    ArcManifold, SwissRoll, Helix, Plane, RFF, dsm_target, fit_linear_score,
    score_fn, denoise, gap_noise_points, offmanifold_dist, estimate_local_normals,
)


def test_projection_lands_on_manifold_with_unit_tangent():
    man = ArcManifold(R=1.0, lo=0.0, hi=np.pi)
    rng = np.random.default_rng(0)
    _, X0 = man.sample(200, rng)
    Xoff = X0 + 0.15 * rng.normal(size=X0.shape)
    feet, tang = man.project(Xoff)
    assert np.allclose(np.linalg.norm(feet, axis=1), 1.0, atol=1e-6)   # on radius-1 arc
    assert np.allclose(np.linalg.norm(tang, axis=1), 1.0, atol=1e-6)   # unit tangents
    # tangent ⟂ radius (radius is the normal for a circle)
    assert np.all(np.abs(np.sum(tang * feet, axis=1)) < 1e-6)


def test_tangent_projector_is_a_projector():
    man = ArcManifold()
    rng = np.random.default_rng(1)
    _, X0 = man.sample(50, rng)
    _, tang = man.project(X0)
    P = np.einsum("ni,nj->nij", tang, tang)          # t tᵀ, projects onto tangent
    assert np.allclose(np.einsum("nij,njk->nik", P, P), P, atol=1e-6)  # idempotent
    assert np.allclose(np.einsum("nij,nj->ni", P, tang), tang, atol=1e-6)  # fixes t
    N = np.eye(2)[None] - P                            # normal projector
    assert np.allclose(np.einsum("nij,nj->ni", N, tang), 0.0, atol=1e-6)  # kills t


def test_gap_sampler_draws_inside_the_gap_near_manifold():
    man = ArcManifold(R=1.0, lo=0.0, hi=np.pi)
    gap = (1.2, 1.9)
    rng = np.random.default_rng(2)
    xt, feet, normals, r = gap_noise_points(man, gap, n=300, sigma=0.1, rng=rng)
    ang = np.arctan2(feet[:, 1], feet[:, 0])
    assert np.all((ang >= gap[0] - 1e-6) & (ang <= gap[1] + 1e-6))     # feet in the gap
    assert offmanifold_dist(xt, man).mean() < 0.4                      # but near the arc
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_both_models_fit_the_supported_region():
    man = ArcManifold(R=1.0, lo=0.0, hi=np.pi)
    rng = np.random.default_rng(3)
    sigma = 0.1
    _, X0 = man.sample(1500, rng, gaps=[(1.2, 1.9)])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)
    rff = RFF(n_features=200, dim=2, scale=2.0, seed=0)
    Phi = rff.transform(Xt)
    W_plain = fit_linear_score(Phi, T, lam=0.0)
    # in-support test points denoise well under the plain model
    _, Xs = man.sample(300, rng, gaps=[(1.2, 1.9)])
    Xts = Xs + sigma * rng.normal(size=Xs.shape)
    Xhat = denoise(Xts, W_plain, rff, sigma)
    assert offmanifold_dist(Xhat, man).mean() < offmanifold_dist(Xts, man).mean()


def test_tangent_penalty_beats_plain_dsm_in_the_gap():
    man = ArcManifold(R=1.0, lo=0.0, hi=np.pi)
    gap = (1.2, 1.9)
    rng = np.random.default_rng(4)
    sigma = 0.12

    # DSM data: covers the arc EXCEPT the gap
    _, X0 = man.sample(2000, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)

    rff = RFF(n_features=250, dim=2, scale=2.0, seed=1)
    Phi = rff.transform(Xt)

    # plain DSM (no gap data) vs tangent-penalized on gap-sampled points
    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, gfeet, gnorm, gr = gap_noise_points(man, gap, n=400, sigma=sigma, rng=rng)
    Phi_pen = rff.transform(gxt)
    W_tan = fit_linear_score(Phi, T, Phi_pen=Phi_pen, normals=gnorm,
                             r_pen=gr, lam=5.0)

    # evaluate: denoise fresh gap-region points, measure off-manifold error
    gxt_test, _, _, _ = gap_noise_points(man, gap, n=400, sigma=sigma,
                                         rng=np.random.default_rng(9))
    err_plain = offmanifold_dist(denoise(gxt_test, W_plain, rff, sigma), man).mean()
    err_tan = offmanifold_dist(denoise(gxt_test, W_tan, rff, sigma), man).mean()
    assert err_tan < err_plain                        # the payoff
    assert err_tan < 0.7 * err_plain                  # and materially so


# ----------------------------------------------------------- Swiss roll (R³, 2-manifold)
def test_swiss_roll_geometry():
    man = SwissRoll()
    rng = np.random.default_rng(0)
    params, X0 = man.sample(200, rng)
    t, h = params[:, 0], params[:, 1]
    # normal is orthogonal to BOTH surface tangents ∂P/∂t and ∂P/∂h
    n = man.normal(t)
    dPdt = np.column_stack([np.cos(t) - t * np.sin(t), np.zeros_like(t),
                            np.sin(t) + t * np.cos(t)])
    dPdh = np.tile([0.0, 1.0, 0.0], (len(t), 1))
    assert np.all(np.abs(np.sum(n * dPdt, axis=1)) < 1e-8)
    assert np.all(np.abs(np.sum(n * dPdh, axis=1)) < 1e-8)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-8)
    # projection of on-manifold points is (near) identity
    feet, _ = man.project(X0)
    assert offmanifold_dist(X0, man).max() < 0.05


def test_swiss_roll_tangent_penalty_beats_plain_dsm_in_the_gap():
    man = SwissRoll()
    gap = (2.5 * np.pi, 2.8 * np.pi)                   # a held-out band of the roll
    rng = np.random.default_rng(7)
    sigma = 0.15

    _, X0 = man.sample(4000, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)

    rff = RFF(n_features=400, dim=3, scale=0.6, seed=2)
    Phi = rff.transform(Xt)

    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gnorm, gr = gap_noise_points(man, gap, n=800, sigma=sigma, rng=rng)
    W_tan = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt), normals=gnorm,
                             r_pen=gr, lam=8.0)

    gt, _, _, _ = gap_noise_points(man, gap, n=3000, sigma=sigma,
                                   rng=np.random.default_rng(21))
    err_plain = offmanifold_dist(denoise(gt, W_plain, rff, sigma), man).mean()
    err_tan = offmanifold_dist(denoise(gt, W_tan, rff, sigma), man).mean()
    assert err_tan < 0.5 * err_plain                  # payoff generalizes to R³


# --------------------------------------------- Helix (curve in R³, codimension 2)
def test_helix_normal_basis_is_orthonormal_and_normal():
    man = Helix()
    rng = np.random.default_rng(0)
    t, X0 = man.sample(200, rng)
    N = man.frenet_normals(t)                          # (n, 2, 3)
    assert N.shape == (200, 2, 3)
    assert np.allclose(np.linalg.norm(N, axis=2), 1.0, atol=1e-8)      # each unit
    assert np.all(np.abs(np.einsum("nk,nk->n", N[:, 0], N[:, 1])) < 1e-8)  # ⟂ each other
    Tg = man.tangent(t)
    assert np.all(np.abs(np.einsum("njk,nk->nj", N, Tg)) < 1e-8)       # both ⟂ tangent
    assert offmanifold_dist(X0, man).max() < 0.05                      # projection ≈ id


def test_fit_linear_score_normal_basis_matches_single_normal():
    # codim-1 given as (n, D) vs (n, 1, D) must be identical (back-compat), and a
    # (n, 2, D) basis must run and return the right shape.
    rng = np.random.default_rng(1)
    Phi = rng.normal(size=(60, 12))
    T = rng.normal(size=(60, 3))
    Phi_pen = rng.normal(size=(25, 12))
    r = rng.normal(size=(25, 3))
    n1 = rng.normal(size=(25, 3))
    W_2d = fit_linear_score(Phi, T, Phi_pen=Phi_pen, normals=n1, r_pen=r, lam=1.0)
    W_3d = fit_linear_score(Phi, T, Phi_pen=Phi_pen, normals=n1[:, None, :],
                            r_pen=r, lam=1.0)
    assert np.allclose(W_2d, W_3d, atol=1e-9)
    basis = rng.normal(size=(25, 2, 3))
    W_basis = fit_linear_score(Phi, T, Phi_pen=Phi_pen, normals=basis, r_pen=r, lam=1.0)
    assert W_basis.shape == (3, 12)


def test_helix_tangent_penalty_beats_plain_dsm_in_the_gap_codim2():
    man = Helix()
    gap = (1.6 * np.pi, 2.0 * np.pi)                   # a held-out turn of the helix
    rng = np.random.default_rng(5)
    sigma = 0.10

    _, X0 = man.sample(4000, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)

    rff = RFF(n_features=500, dim=3, scale=1.2, seed=3)
    Phi = rff.transform(Xt)

    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gnorm, gr = gap_noise_points(man, gap, n=900, sigma=sigma, rng=rng)
    assert gnorm.shape[1] == 2                         # codim-2 normal basis
    W_tan = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt), normals=gnorm,
                             r_pen=gr, lam=8.0)

    gt, _, _, _ = gap_noise_points(man, gap, n=3000, sigma=sigma,
                                   rng=np.random.default_rng(31))
    err_plain = offmanifold_dist(denoise(gt, W_plain, rff, sigma), man).mean()
    err_tan = offmanifold_dist(denoise(gt, W_tan, rff, sigma), man).mean()
    assert err_tan < 0.5 * err_plain                  # payoff holds at codim 2

    # ABLATION — why codim-2 needs the FULL basis: constraining only ONE normal
    # leaves the score free to drift along the OTHER normal direction.
    W_one = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt),
                             normals=gnorm[:, 0, :], r_pen=gr, lam=8.0)
    err_one = offmanifold_dist(denoise(gt, W_one, rff, sigma), man).mean()
    assert err_tan < 0.5 * err_one                    # two normals materially beat one


# ------------------------------------------- learned metric (data-driven normals)
def test_local_pca_recovers_the_analytic_normal_space():
    # On a dense clean helix cloud, local PCA should recover the Frenet normal
    # SPACE (basis-invariant, compared via projectors) almost exactly.
    man = Helix()
    rng = np.random.default_rng(0)
    _, ref = man.sample(6000, rng)
    t = rng.uniform(man.t_lo + 0.6, man.t_hi - 0.6, 200)
    q = man.point(t)
    N_learn = estimate_local_normals(ref, q, k=40, intrinsic_dim=1)   # (200, 2, 3)
    N_true = man.frenet_normals(t)
    assert N_learn.shape == N_true.shape
    P_learn = np.einsum("njk,njl->nkl", N_learn, N_learn)             # normal projectors
    P_true = np.einsum("njk,njl->nkl", N_true, N_true)
    frob = np.linalg.norm(P_learn - P_true, axis=(1, 2))
    assert frob.mean() < 0.05                                         # (actual ≈ 0.003)


def test_learned_normals_recover_the_gap_payoff():
    # The bridge to real embeddings: estimate the normal basis from the point
    # cloud (no analytic manifold) and recover the same gap payoff.
    man = Helix()
    gap = (1.6 * np.pi, 2.0 * np.pi)
    sigma = 0.10
    rng = np.random.default_rng(5)
    _, X0 = man.sample(4000, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)
    rff = RFF(n_features=500, dim=3, scale=1.2, seed=3)
    Phi = rff.transform(Xt)

    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gn_analytic, gr = gap_noise_points(man, gap, n=900, sigma=sigma, rng=rng)
    _, cloud = man.sample(6000, np.random.default_rng(123))           # geometry cloud
    gn_learned = estimate_local_normals(cloud, gxt, k=40, intrinsic_dim=1)
    W_analytic = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt),
                                  normals=gn_analytic, r_pen=gr, lam=8.0)
    W_learned = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt),
                                 normals=gn_learned, r_pen=gr, lam=8.0)

    gt, _, _, _ = gap_noise_points(man, gap, n=3000, sigma=sigma,
                                   rng=np.random.default_rng(31))
    ep = offmanifold_dist(denoise(gt, W_plain, rff, sigma), man).mean()
    ea = offmanifold_dist(denoise(gt, W_analytic, rff, sigma), man).mean()
    el = offmanifold_dist(denoise(gt, W_learned, rff, sigma), man).mean()
    assert el < 0.25 * ep                              # learned normals rescue the gap
    assert el < 1.5 * ea + 0.005                       # ≈ as good as analytic ground truth


# ------------------------------------------------ flat manifold (negative control)
def test_plane_flat_penalty_is_harmless_codim3():
    # Gated-design control: on a FLAT manifold the plain Euclidean DSM already
    # handles the gap, and the tangent penalty (codim 3) is a near-no-op.
    man = Plane(dim=5, intrinsic=2, seed=0)
    gap = (-0.8, 0.8)                                  # gap on the first tangent coord
    sigma = 0.12
    rng = np.random.default_rng(2)
    _, X0 = man.sample(4000, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)
    rff = RFF(n_features=400, dim=5, scale=0.7, seed=4)
    Phi = rff.transform(Xt)

    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gnorm, gr = gap_noise_points(man, gap, n=800, sigma=sigma, rng=rng)
    assert gnorm.shape[1] == 3                         # codimension 3
    W_tan = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt), normals=gnorm,
                             r_pen=gr, lam=8.0)

    gt, _, _, _ = gap_noise_points(man, gap, n=3000, sigma=sigma,
                                   rng=np.random.default_rng(9))
    ep = offmanifold_dist(denoise(gt, W_plain, rff, sigma), man).mean()
    et = offmanifold_dist(denoise(gt, W_tan, rff, sigma), man).mean()
    assert ep < 0.02                                   # flat → plain DSM already good
    assert et < ep + 0.01                              # penalty is harmless (no distortion)
