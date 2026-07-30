"""Constraint-based discovery on the zero-flow CI detector (exp46, ADAfaEPoV Ch 22)."""
import numpy as np

from causal_bench.validation.causal_discovery import (
    orient_colliders, pc_skeleton, sim_chain, sim_collider, sim_fork,
)


def test_chain_and_fork_delete_the_XY_edge():
    """Chain X→M→Y and fork X←S→Y both imply X ⫫ Y | {M or S}, so the X–Y edge must be
    deleted by the conditional zero-flow CI test."""
    rng = np.random.default_rng(0)
    for data, truth, _ in (sim_chain(2000, rng), sim_fork(2000, rng, beta_xy=0.0)):
        edges, _ = pc_skeleton(data, n_perm=60, seed=0)
        assert frozenset((0, 2)) not in edges or frozenset((0, 1)) not in edges  # some X–Y-type edge gone
        assert edges == truth


def test_collider_is_oriented():
    """X→C←Y: skeleton {X-C, Y-C} and the v-structure oriented (C not in sepset(X,Y))."""
    rng = np.random.default_rng(1)
    data, truth, _ = sim_collider(2000, rng)
    edges, sepset = pc_skeleton(data, n_perm=60, seed=1)
    assert edges == truth
    directed = orient_colliders(edges, sepset, 3)
    assert (0, 2) in directed and (1, 2) in directed          # X→C and Y→C


def test_hidden_confounder_yields_spurious_edge():
    """The FCI motivation: drop the confounder S and X–Y (radon–lung-cancer) reappears as a
    spurious direct edge — confounding masquerading as a direct effect."""
    rng = np.random.default_rng(2)
    data, _, _ = sim_fork(2000, rng, beta_xy=0.0)
    edges, _ = pc_skeleton(data[:, :2], n_perm=60, seed=2)     # only X, Y observed
    assert frozenset((0, 1)) in edges
