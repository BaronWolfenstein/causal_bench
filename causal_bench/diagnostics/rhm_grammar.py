"""Random Hierarchy Model (RHM) grammar + exact rule-BP — canonical SFW, #131 Part B.

The symmetric broadcast (tree_reconstruction.py) has no diffusion transition: a
supercritical tree just amplifies leaf information. The sharp SFW class-flip
transition comes from the RHM **grammar** — a probabilistic context-free grammar
where each symbol has ``m`` production rules, each a fixed tuple of ``s`` child
symbols over a vocabulary ``v``. A corrupted leaf can make a subtree inconsistent
with *every* rule (a nonlinear rule-consistency / CSP effect), so information about
the root **class** is destroyed sharply as corruption grows.

The exact denoiser is **belief propagation over the rules** (not the symmetric
channel, and not a trained net — 'BP' = belief propagation here). Leaf message =
the corrupted-token likelihood; a node's belief combines its children through the
allowed rules: ``belief(a) ∝ Σ_{rule r of a} Π_i child_i(r[i])``. Sweeping the
diffusion corruption traces the genuine class-overlap phase transition (arXiv
references: Sclocchi-Favero-Wyart; Cagnetta/Favero/Wyart RHM; Favero et al. 2502.12089).
numpy only, exact BP on sampled trees.
"""
from __future__ import annotations

import numpy as np


def make_rhm(v: int, s: int, m: int, *, seed: int = 0) -> np.ndarray:
    """A random grammar: ``rules[a]`` is ``m`` production rules for symbol ``a``,
    each a tuple of ``s`` child symbols over the vocabulary ``v``. Shape ``(v, m, s)``."""
    return np.random.default_rng(seed).integers(0, v, size=(v, m, s))


def _generate(sym: int, depth: int, rules: np.ndarray, rng: np.random.Generator) -> list:
    """Sample a leaf-token string by expanding ``sym`` down ``depth`` levels."""
    if depth == 0:
        return [sym]
    rule = rules[sym, rng.integers(rules.shape[1])]      # pick one of m rules → (s,)
    out = []
    for c in rule:
        out.extend(_generate(int(c), depth - 1, rules, rng))
    return out


def _bp_belief(leaves, depth: int, rules: np.ndarray, v: int, theta: float) -> np.ndarray:
    """Exact rule-BP upward pass → the posterior belief (v-vector) over this node's
    symbol given its (corrupted) subtree leaves."""
    if depth == 0:
        y = int(leaves[0])
        msg = np.full(v, (1.0 - theta) / v)
        msg[y] += theta                                  # P(observed y | true ·)
        return msg
    s = rules.shape[2]
    csz = len(leaves) // s
    child = np.stack([_bp_belief(leaves[i * csz:(i + 1) * csz], depth - 1, rules, v, theta)
                      for i in range(s)])                # (s, v)
    # belief(a) ∝ Σ_{rule r of a} Π_i child[i, rules[a,r,i]]
    gathered = child[np.arange(s)[None, None, :], rules]  # (v, m, s)
    belief = gathered.prod(2).sum(1)                      # (v,)
    return belief / (belief.sum() + 1e-300)


def rhm_class_overlap(v: int, s: int, m: int, depth: int, theta: float, *,
                      n_trees: int = 200, seed: int = 0, grammar_seed: int = 0) -> float:
    """The SFW **class-overlap** order parameter on the RHM grammar, via exact
    rule-BP. For each sampled tree: generate leaves from a root class, corrupt each
    leaf through the uniform channel (keep w.p. ``theta``, else uniform over ``v``),
    run rule-BP → posterior on the true root. Returns the normalized overlap
    ``(v·E[posterior(root)] − 1)/(v − 1)`` (0 = no class info, 1 = certain).
    Sweeping ``theta`` traces the genuine diffusion phase transition."""
    rules = make_rhm(v, s, m, seed=grammar_seed)
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(n_trees):
        root = int(rng.integers(v))
        leaves = np.array(_generate(root, depth, rules, rng))
        keep = rng.random(len(leaves)) < theta
        y = np.where(keep, leaves, rng.integers(0, v, size=len(leaves)))
        tot += _bp_belief(y, depth, rules, v, theta)[root]
    mean_p = tot / n_trees
    return float((v * mean_p - 1.0) / (v - 1))
