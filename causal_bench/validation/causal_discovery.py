"""Constraint-based causal discovery on causal_bench's own CI detector (ADAfaEPoV Ch 22).

Wires `detectors.zero_flow_ci.zero_flow_ci_test` — a nonparametric residualize-then-
conditional-permutation CI oracle — into the **PC/SGS skeleton** + **collider (v-structure)
orientation**, and demonstrates the **latent-confounder failure** (radon/smoking) that motivates
FCI: drop the confounder and the confounding masquerades as a spurious direct edge.

The point: causal_bench already ships the *primitives* of constraint-based discovery (a CI test
+ Markov-blanket routine); this composes them into the classic skeleton search. Self-validating
against known linear-Gaussian DGP structure.
"""
from __future__ import annotations

import itertools

import numpy as np

from causal_bench.detectors.zero_flow_ci import zero_flow_ci_test


def _ci_holds(data, i, j, Z, *, alpha, n_perm, rng) -> bool:
    """True if X_i ⫫ X_j | X_Z. Conditional tests (Z non-empty) use the zero-flow CI oracle
    (residualize-then-permute; verdict 'supports' ⇒ CI holds). The MARGINAL case (empty Z) can't
    residualize on nothing, so it uses a permutation test on |correlation| — the standard PC
    r=0 step. (Linear-Gaussian DGPs here, so correlation captures the marginal dependence.)"""
    x, y = data[:, i], data[:, j]
    if len(Z) == 0:
        obs = abs(np.corrcoef(x, y)[0, 1])
        null = [abs(np.corrcoef(x, y[rng.permutation(len(y))])[0, 1]) for _ in range(n_perm)]
        p = (1.0 + sum(t >= obs for t in null)) / (1.0 + n_perm)
        return p >= alpha                                   # CI (independence) NOT rejected
    res = zero_flow_ci_test(x, y, data[:, list(Z)], alpha=alpha, n_perm=n_perm, rng=rng)
    return res.verdict == "supports"


def pc_skeleton(data, *, alpha=0.05, n_perm=80, max_cond=3, seed=0):
    """PC/SGS skeleton phase. Start from the complete graph; delete edge (i,j) as soon as
    SOME conditioning set Z (|Z| ≤ max_cond) renders i ⫫ j | Z, recording the separating set.

    Returns (edges, sepset): `edges` a set of frozenset({i,j}); `sepset` maps each *deleted*
    pair to the Z that d-separated it (needed for collider orientation)."""
    rng = np.random.default_rng(seed)
    p = data.shape[1]
    edges = {frozenset((i, j)) for i, j in itertools.combinations(range(p), 2)}
    sepset: dict = {}
    for i, j in itertools.combinations(range(p), 2):
        rest = [k for k in range(p) if k not in (i, j)]
        for r in range(min(max_cond, len(rest)) + 1):
            hit = None
            for Z in itertools.combinations(rest, r):
                if _ci_holds(data, i, j, Z, alpha=alpha, n_perm=n_perm, rng=rng):
                    hit = set(Z)
                    break
            if hit is not None:
                edges.discard(frozenset((i, j)))
                sepset[frozenset((i, j))] = hit
                break
    return edges, sepset


def orient_colliders(edges, sepset, p):
    """Orient unshielded triples X–C–Y (X,Y NOT adjacent) as X→C←Y when C ∉ sepset(X,Y) —
    the only orientation rule identifiable from CI structure alone. Returns a set of directed
    (parent, child) pairs (the partially-oriented CPDAG's v-structures)."""
    nbr = {k: {m for e in edges if k in e for m in e if m != k} for k in range(p)}
    directed = set()
    for c in range(p):
        for x, y in itertools.combinations(sorted(nbr[c]), 2):
            if frozenset((x, y)) not in edges:                      # unshielded pair
                if c not in sepset.get(frozenset((x, y)), set()):   # collider signature
                    directed.add((x, c))
                    directed.add((y, c))
    return directed


# ----------------------------------------------------------------- known-structure DGPs

def sim_fork(n, rng, *, beta_xy=0.0):
    """Confounder / radon-smoking fork: S→X, S→Y (and optional X→Y).
    Columns 0=X (radon), 1=Y (lung cancer), 2=S (smoking). With beta_xy=0 there is NO direct
    X–Y edge — X ⫫ Y | S — the pure-confounding case."""
    S = rng.normal(size=n)
    X = -0.8 * S + rng.normal(size=n)
    Y = 1.0 * S + beta_xy * X + rng.normal(size=n)
    data = np.column_stack([X, Y, S])
    truth = {frozenset((0, 2)), frozenset((1, 2))}            # X-S, Y-S
    if beta_xy != 0.0:
        truth.add(frozenset((0, 1)))
    return data, truth, ["X", "Y", "S"]


def sim_collider(n, rng):
    """Collider: X→C←Y with X ⫫ Y marginally. Columns 0=X, 1=Y, 2=C.
    True skeleton {X-C, Y-C}; the v-structure X→C←Y should be oriented."""
    X = rng.normal(size=n)
    Y = rng.normal(size=n)
    C = 0.9 * X + 0.9 * Y + rng.normal(size=n)
    data = np.column_stack([X, Y, C])
    return data, {frozenset((0, 2)), frozenset((1, 2))}, ["X", "Y", "C"]


def sim_chain(n, rng):
    """Chain X→M→Y with X ⫫ Y | M. Columns 0=X, 1=M, 2=Y. Skeleton {X-M, M-Y}."""
    X = rng.normal(size=n)
    M = 0.9 * X + rng.normal(size=n)
    Y = 0.9 * M + rng.normal(size=n)
    data = np.column_stack([X, M, Y])
    return data, {frozenset((0, 1)), frozenset((1, 2))}, ["X", "M", "Y"]
