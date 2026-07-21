"""Joint hierarchical DGP — prerequisite A for causal_bench #144.

Couples the two synthetic worlds that were disconnected: a hierarchical latent
structure whose **identifiability** is measured by exact rule-BP (representation
side) AND per-unit **outcomes** whose subgroup treatment-effect heterogeneity is
generated at chosen levels (effect side), so the "does identifiability inform
borrowing?" question can actually be tested on one object.

Built on the **product grammar** (`rhm_grammar.make_product_grammar`, #138): each
symbol factors as ``(group, member)``, group and member each evolving by their own
identifiable RHM. Group is the genuinely coarse coordinate (survives more corruption),
member the fine one — the symmetric RHM has no such per-level ordering, so the product
grammar is the right substrate. Identifiability is measured **canonically** (exact
rule-BP reconstruction threshold), NOT the phenomenological embedding crossover — for
synthetic objects we own the grammar (see #144 audit).

The **coupling knob** is where effect heterogeneity is attached: ``w_group`` vs
``w_member``. Effect at the well-identified coarse level (``w_group`` large) = the
*coupled* regime where identifiability legitimately informs borrowing; effect at the
fragile fine level (``w_member`` large) = the *orthogonal* regime where an
identifiability-set prior would miss it. That contrast is what #144-B/D need.

numpy only; reuses exact BP from `rhm_grammar`.
"""
from __future__ import annotations

import numpy as np

from causal_bench.diagnostics.rhm_grammar import (
    make_product_grammar, _generate, _bp_belief, _bp_belief_batch)


def make_joint_hierarchy(g: int, b_size: int, s: int, m: int, *, w_group: float = 1.0,
                         w_member: float = 1.0, mu0: float = 0.0, seed: int = 0) -> dict:
    """Build the joint DGP spec: a product grammar plus a per-group and per-member
    treatment-effect table. A unit's true effect is ``w_group·group_effect[group] +
    w_member·member_effect[member]`` of its root symbol, so ``w_group``/``w_member``
    set how much between-subgroup effect heterogeneity (τ) lives at the coarse vs fine
    level — the coupling knob. Returns the spec dict."""
    rules, groups = make_product_grammar(g, b_size, s, m, seed=seed)
    rng = np.random.default_rng(seed + 1)
    return {
        "rules": rules, "groups": groups, "g": g, "b_size": b_size, "s": s, "m": m,
        "v": g * b_size, "mu0": mu0, "w_group": w_group, "w_member": w_member,
        "group_effect": rng.normal(size=g), "member_effect": rng.normal(size=b_size),
    }


def _effect(spec: dict, group: np.ndarray, member: np.ndarray) -> np.ndarray:
    return (spec["w_group"] * spec["group_effect"][group]
            + spec["w_member"] * spec["member_effect"][member])


def sample_joint_cohort(spec: dict, n: int, depth: int, *, treatment_frac: float = 0.5,
                        sigma: float = 0.5, seed: int = 0) -> dict:
    """Sample ``n`` units. Each has a root symbol → coarse ``group`` and fine ``member``
    subgroup labels, a randomized treatment ``A``, and outcome ``Y = mu0 + effect·A +
    sigma·noise`` (effect from the level-attached tables). Also returns each unit's
    corrupted-free ``leaves`` (for optional BP reconstruction / rendering). Treatment is
    randomized (⊥ subgroup), so crude arm means are unconfounded — the borrowing/τ
    question is about effect *heterogeneity across subgroups*, not confounding."""
    rng = np.random.default_rng(seed)
    rules, b_size, v = spec["rules"], spec["b_size"], spec["v"]
    roots = rng.integers(0, v, n)
    group, member = roots // b_size, roots % b_size
    effect = _effect(spec, group, member)
    A = (rng.random(n) < treatment_frac).astype(float)
    Y = spec["mu0"] + effect * A + sigma * rng.normal(size=n)
    leaves = [np.array(_generate(int(r), depth, rules, rng)) for r in roots]
    return {"roots": roots, "group": group, "member": member, "A": A, "Y": Y,
            "effect": effect, "leaves": leaves, "depth": depth}


def decode_cohort_labels(spec: dict, cohort: dict, *, theta0: float, seed: int = 0) -> dict:
    """The **label-observation model** for #144 B/C: the estimator does NOT see true
    subgroup labels — it sees labels **BP-decoded from the representation at a working
    corruption θ₀** (see the #144 decision). For each unit, corrupt its clean leaves at
    ``theta0``, run exact rule-BP, and take the MAP root → decoded ``group`` / ``member``.

    This is what makes identifiability *bite* on estimation (misclassification at θ₀
    attenuates the recoverable between-subgroup variance) while staying orthogonal to the
    outcome — decoding reads only the leaves, never Y/A/effects. ``theta0 = 1`` ⇒ decoded
    ≈ true; lower ``theta0`` ⇒ more misclassification, with the coarse (group) coordinate
    decoded better than the fine (member) one. The per-level **decode accuracy** is the
    operating-point learnability scalar the ``tau_sd`` prior should track (measured at the
    SAME θ₀ used for decoding — non-circular). True labels are retained by the caller for
    oracle-τ and scoring only. Returns ``{group_decoded, member_decoded, theta0,
    group_decode_acc, member_decode_acc}``."""
    rng = np.random.default_rng(seed)
    rules, b_size, v, depth = spec["rules"], spec["b_size"], spec["v"], cohort["depth"]
    leaves_list = cohort["leaves"]
    n, n_leaf = len(leaves_list), len(leaves_list[0])
    # corruption loop kept per-unit (cheap; preserves the RNG stream → bit-identical
    # to the old loop); only the expensive BP is batched over units below.
    Y = np.empty((n, n_leaf), int)
    for i, leaves in enumerate(leaves_list):
        keep = rng.random(len(leaves)) < theta0
        Y[i] = np.where(keep, leaves, rng.integers(0, v, size=len(leaves)))
    root_hat = _bp_belief_batch(Y, depth, rules, v, float(theta0)).argmax(1)  # (n,), one batched BP
    dg, dm = root_hat // b_size, root_hat % b_size
    return {"group_decoded": dg, "member_decoded": dm, "theta0": theta0,
            "group_decode_acc": float((dg == cohort["group"]).mean()),
            "member_decode_acc": float((dm == cohort["member"]).mean())}


def true_tau_by_level(spec: dict) -> dict:
    """The ground-truth between-subgroup effect SD contributed at each level:
    ``tau_group = |w_group|·std(group_effect)``, ``tau_member = |w_member|·std(member_
    effect)``. This is what a borrowing prior's ``tau_sd`` should track — and it is set
    by the coupling knob, NOT by identifiability (the #144 crux)."""
    return {
        "tau_group": abs(spec["w_group"]) * float(np.std(spec["group_effect"])),
        "tau_member": abs(spec["w_member"]) * float(np.std(spec["member_effect"])),
    }


def _half_clean_threshold(theta_grid: np.ndarray, overlap: np.ndarray) -> float:
    """Retention θ at which the overlap first reaches half its clean (θ=max) value.
    LOWER θ_c ⇒ the coordinate is recoverable under MORE corruption ⇒ more robust."""
    target = 0.5 * overlap[-1]
    hits = np.where(overlap >= target)[0]
    return float(theta_grid[hits[0]]) if len(hits) else float(theta_grid[-1])


def joint_reconstruction_scan(spec: dict, depth: int, *, theta_grid: np.ndarray | None = None,
                              n_trees: int = 250, seed: int = 0) -> dict:
    """Canonical per-level identifiability via **exact rule-BP** (not the embedding
    crossover). Sweep uniform corruption θ; for each sampled tree run rule-BP and record:
    the **group** (coarse) overlap — belief marginalized to the true group — and the
    **member** (fine) overlap — the conditional belief on the true member given the group
    (``belief[root]/group_belief``, normalized over ``b_size``). The coarse coordinate is
    recoverable under more corruption, so its half-clean threshold ``theta_c_group`` sits
    at LOWER θ than the fine ``theta_c_member``. Returns ``{theta, group_overlap,
    member_overlap, theta_c_group, theta_c_member}``."""
    theta_grid = np.linspace(0.3, 0.98, 16) if theta_grid is None else np.asarray(theta_grid, float)
    rules, b_size, v, g = spec["rules"], spec["b_size"], spec["v"], spec["g"]
    g_ov, mem_ov = [], []
    for theta in theta_grid:
        rng = np.random.default_rng(seed)
        tg = tm = 0.0
        for _ in range(n_trees):
            root = int(rng.integers(v))
            leaves = np.array(_generate(root, depth, rules, rng))
            keep = rng.random(len(leaves)) < theta
            y = np.where(keep, leaves, rng.integers(0, v, size=len(leaves)))
            belief = _bp_belief(y, depth, rules, v, float(theta))
            gbelief = belief.reshape(g, b_size).sum(1)          # marginalize to group
            tg += gbelief[root // b_size]
            tm += belief[root] / (gbelief[root // b_size] + 1e-12)   # member | group
        g_ov.append((g * tg / n_trees - 1.0) / (g - 1))
        mem_ov.append((b_size * tm / n_trees - 1.0) / (b_size - 1))
    g_ov, mem_ov = np.asarray(g_ov), np.asarray(mem_ov)
    return {"theta": theta_grid, "group_overlap": g_ov, "member_overlap": mem_ov,
            "theta_c_group": _half_clean_threshold(theta_grid, g_ov),
            "theta_c_member": _half_clean_threshold(theta_grid, mem_ov)}
