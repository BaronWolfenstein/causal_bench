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

**Refinements (#131 Part B follow-up).** ``rhm_transition_scan`` locates the
transition via the susceptibility (``dm/dθ``) peak; ``rhm_finite_size`` shows the
transition **sharpens** with depth (width ∝ 1/max-susceptibility shrinks) — the FSS
signature of a genuine phase transition, not a smooth crossover. ``rhm_bp_density_
evolution`` is the **population-dynamics** predictor: it iterates the rule-BP
recursion over a population of (true symbol, belief) pairs — no full trees sampled —
and its transition matches the empirical exact-BP ``theta_star`` (the grammar analog
of the KS/reconstruction density evolution in tree_reconstruction.py).

**Large-K corruption channels.** The uniform channel used here is rank-1/identity-
structured, so BP stays ``O(v)`` and no K×K transition matrix is ever stored — the
D3PM ``O(K²T)`` cost (see tree_reconstruction.py) never arises for large ``v``.
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


def rhm_transition_scan(v: int, s: int, m: int, depth: int, *,
                        theta_grid: np.ndarray | None = None, n_trees: int = 250,
                        seed: int = 0, grammar_seed: int = 0) -> dict:
    """Scan corruption θ (exact rule-BP on sampled trees): class-overlap +
    susceptibility ``dm/dθ``. The **susceptibility peak** locates the transition
    ``theta_star``. Returns ``{theta, overlap, susceptibility, theta_star}``."""
    theta_grid = np.linspace(0.3, 0.98, 16) if theta_grid is None else np.asarray(theta_grid, float)
    m_ov = np.array([rhm_class_overlap(v, s, m, depth, float(t), n_trees=n_trees,
                                       seed=seed, grammar_seed=grammar_seed)
                     for t in theta_grid])
    susc = np.diff(m_ov) / np.diff(theta_grid)            # overlap rises with θ
    mid = 0.5 * (theta_grid[:-1] + theta_grid[1:])
    return {"theta": theta_grid, "overlap": m_ov, "susceptibility": susc,
            "theta_star": float(mid[int(np.argmax(susc))])}


def rhm_finite_size(v: int, s: int, m: int, *, depths=(3, 5, 7), n_trees: int = 250,
                    seed: int = 0, grammar_seed: int = 0) -> dict:
    """Finite-size scaling on the grammar: transition **width** (= 1/max
    susceptibility) and location ``theta_star`` vs depth. A genuine transition
    **sharpens** (width shrinks) as depth grows. Returns ``{depths, widths, theta_stars}``."""
    widths, stars = [], []
    for d in depths:
        r = rhm_transition_scan(v, s, m, int(d), n_trees=n_trees, seed=seed,
                                grammar_seed=grammar_seed)
        widths.append(1.0 / float(np.max(r["susceptibility"])))
        stars.append(r["theta_star"])
    return {"depths": np.asarray(depths), "widths": np.asarray(widths),
            "theta_stars": np.asarray(stars)}


def _sample_by_symbol(syms: np.ndarray, targets: np.ndarray, rng, v: int) -> np.ndarray:
    """For each target symbol, draw a population index whose symbol matches it."""
    idx = np.zeros(len(targets), dtype=int)
    for a in range(v):
        g = np.where(syms == a)[0]
        mask = targets == a
        n = int(mask.sum())
        if n:
            idx[mask] = (g[rng.integers(0, len(g), n)] if len(g)
                         else rng.integers(0, len(syms), n))
    return idx


def rhm_bp_density_evolution(v: int, s: int, m: int, depth: int, theta: float, *,
                             pop: int = 4000, seed: int = 0, grammar_seed: int = 0) -> float:
    """Rule-BP **density evolution** (population dynamics) — predicts the class
    overlap from the recursion WITHOUT sampling full trees (the grammar analog of
    ``bp_magnetization``). Tracks (true symbol, belief message) pairs; the leaf init
    is the corrupted-observation likelihood; each step draws children by symbol,
    combines through the rules, and conditions on a fresh node symbol. Returns the
    normalized class overlap; the theta at which it transitions matches the
    empirical exact-BP ``theta_star``."""
    rules = make_rhm(v, s, m, seed=grammar_seed)
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, v, pop)
    y = np.where(rng.random(pop) < theta, syms, rng.integers(0, v, pop))
    msgs = np.full((pop, v), (1.0 - theta) / v)
    msgs[np.arange(pop), y] += theta
    ri = np.arange(v)
    for _ in range(depth):
        new_syms = rng.integers(0, v, pop)
        child_syms = rules[new_syms, rng.integers(0, m, pop)]      # (pop, s)
        belief = np.ones((pop, v, m))
        for i in range(s):
            cm = msgs[_sample_by_symbol(syms, child_syms[:, i], rng, v)]  # (pop, v)
            belief *= cm[:, rules[:, :, i]]                        # (pop, v, m)
        belief = belief.sum(2)                                     # (pop, v)
        belief /= belief.sum(1, keepdims=True) + 1e-300
        syms, msgs = new_syms, belief
    mean_p = float(msgs[np.arange(pop), syms].mean())
    return (v * mean_p - 1.0) / (v - 1)
