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

**FSS collapse + exponent (#136).** ``rhm_fss_collapse`` fits the mean-field-style
ansatz ``m(theta, L) = L^{-beta/nu} f((theta - theta_c) L^{1/nu})`` and reports a
``collapse_residual`` (curves genuinely land on one master curve). ``theta_c`` is
anchored from ``rhm_density_evolution_threshold`` (density evolution, stable across
seeds) rather than extrapolated from the noisier finite-tree ``theta_star(L)`` — with
only a handful of feasible exact-BP depths (cost grows as ``s**depth``), that
extrapolation is unstable enough to land outside ``[0, 1]``.

**Large-K corruption channels.** The uniform channel used here is rank-1/identity-
structured, so BP stays ``O(v)`` and no K×K transition matrix is ever stored — the
D3PM ``O(K²T)`` cost (see tree_reconstruction.py) never arises for large ``v``.

**Structured (low-rank) corruption (#138).** ``make_lowrank_corruption`` builds a
zero-diagonal, row-stochastic channel (``O(v·r)`` parameterization) where a corrupted
symbol is replaced by a *similar* one; ``corruption=C`` threads through ``_bp_belief``
/ ``rhm_class_overlap`` / ``rhm_transition_scan`` (``None`` = uniform). Finding
(``structured_corruption_shift``): a concentrated channel **washes out** the
transition — a corrupted leaf still points at a small neighbourhood, stays
informative, and the root class survives even under heavy corruption (the overlap
floor lifts from ~0 to ~0.75). The transition needs corruption that genuinely
destroys leaf information (uniform/masking); this parallels the broadcast having no
transition because it *amplifies*. Gates the trainable low-rank channel (see #131).
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


def _bp_belief(leaves, depth: int, rules: np.ndarray, v: int, theta: float,
               corruption: np.ndarray | None = None) -> np.ndarray:
    """Exact rule-BP upward pass → the posterior belief (v-vector) over this node's
    symbol given its (corrupted) subtree leaves. ``corruption`` = an optional
    row-stochastic channel ``C`` (``C[a,b] = P(replacement b | true a)``); ``None``
    is the uniform channel. Leaf likelihood ``P(observed y | true a) = theta·[a==y]
    + (1-theta)·C[a,y]`` → ``msg = (1-theta)·C[:,y]; msg[y] += theta``."""
    if depth == 0:
        y = int(leaves[0])
        if corruption is None:
            msg = np.full(v, (1.0 - theta) / v)
        else:
            msg = (1.0 - theta) * corruption[:, y].copy()
        msg[y] += theta                                  # P(observed y | true ·)
        return msg
    s = rules.shape[2]
    csz = len(leaves) // s
    child = np.stack([_bp_belief(leaves[i * csz:(i + 1) * csz], depth - 1, rules, v, theta,
                                 corruption)
                      for i in range(s)])                # (s, v)
    # belief(a) ∝ Σ_{rule r of a} Π_i child[i, rules[a,r,i]]
    gathered = child[np.arange(s)[None, None, :], rules]  # (v, m, s)
    belief = gathered.prod(2).sum(1)                      # (v,)
    return belief / (belief.sum() + 1e-300)


def make_lowrank_corruption(v: int, r: int, *, beta: float = 1.0, seed: int = 0,
                            features: np.ndarray | None = None):
    """A low-rank, zero-diagonal, row-stochastic corruption matrix ``C`` (v×v): when
    a symbol corrupts, its replacement is drawn from ``C[a,:]``, concentrated on
    symbols SIMILAR to ``a`` in an ``r``-dim feature space (``r ≪ v``), never on
    ``a`` itself. Score ``S[a,b] = beta·⟨u_a, u_b⟩`` (rank ≤ r), diagonal masked to
    ``-inf``, row-softmaxed. ``beta = 0`` ⇒ uniform over the other v−1 symbols (the
    structure-free baseline); larger ``beta`` ⇒ more concentrated (a corrupted leaf
    still points at a small neighbourhood, so it leaks more information than a flat
    uniform draw). Parameterization is ``O(v·r)``; ``C`` is materialized at v² only
    because the synthetic ``v`` is small. Returns ``(C, U)``."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(v, r)) if features is None else np.asarray(features, float)
    S = beta * (U @ U.T)
    np.fill_diagonal(S, -np.inf)
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)                                        # exp(-inf) = 0 on the diagonal
    return E / E.sum(1, keepdims=True), U


def _sample_corruption(leaves: np.ndarray, C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a replacement symbol for each leaf from its corruption row ``C[leaf,:]``
    (vectorized inverse-CDF sampling)."""
    cdf = np.cumsum(C[leaves], axis=1)                   # (n, v)
    u = rng.random(len(leaves))
    return (u[:, None] <= cdf).argmax(1)


def rhm_class_overlap(v: int, s: int, m: int, depth: int, theta: float, *,
                      n_trees: int = 200, seed: int = 0, grammar_seed: int = 0,
                      corruption: np.ndarray | None = None) -> float:
    """The SFW **class-overlap** order parameter on the RHM grammar, via exact
    rule-BP. For each sampled tree: generate leaves from a root class, corrupt each
    leaf (keep w.p. ``theta``, else replace — uniform over ``v`` if ``corruption``
    is ``None``, else drawn from the row-stochastic channel ``corruption[leaf,:]``),
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
        repl = (rng.integers(0, v, size=len(leaves)) if corruption is None
                else _sample_corruption(leaves, corruption, rng))
        y = np.where(keep, leaves, repl)
        tot += _bp_belief(y, depth, rules, v, theta, corruption)[root]
    mean_p = tot / n_trees
    return float((v * mean_p - 1.0) / (v - 1))


def rhm_transition_scan(v: int, s: int, m: int, depth: int, *,
                        theta_grid: np.ndarray | None = None, n_trees: int = 250,
                        seed: int = 0, grammar_seed: int = 0, n_reps: int = 1,
                        corruption: np.ndarray | None = None) -> dict:
    """Scan corruption θ (exact rule-BP on sampled trees): class-overlap +
    susceptibility ``dm/dθ``. The **susceptibility peak** locates the transition
    ``theta_star``. ``n_reps > 1`` averages the overlap curve over independent
    tree-sampling seeds — cheaper noise reduction than deepening trees (whose cost
    grows as ``s**depth``), used by ``rhm_fss_collapse`` to stabilize the fit.
    ``corruption`` threads a structured channel through (``None`` = uniform).
    Returns ``{theta, overlap, susceptibility, theta_star}``."""
    theta_grid = np.linspace(0.3, 0.98, 16) if theta_grid is None else np.asarray(theta_grid, float)
    m_ov = np.mean([
        [rhm_class_overlap(v, s, m, depth, float(t), n_trees=n_trees,
                           seed=seed + rep, grammar_seed=grammar_seed, corruption=corruption)
         for t in theta_grid]
        for rep in range(n_reps)
    ], axis=0)
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


def rhm_density_evolution_threshold(v: int, s: int, m: int, *, de_depth: int = 20,
                                    pop: int = 6000, seed: int = 0, grammar_seed: int = 0,
                                    lo: float = 0.05, hi: float = 0.98, tol: float = 1e-3,
                                    max_iter: int = 30) -> float:
    """Bisect the rule-BP **density-evolution** overlap (population dynamics, cost
    additive in ``pop`` rather than exponential in depth — so ``de_depth`` can be
    pushed much deeper than exact-BP trees) to find the theta where the overlap
    crosses half its clean-signal (``theta=hi``) value. Used as the stable
    infinite-depth anchor for ``theta_c`` in ``rhm_fss_collapse``, since it does not
    depend on extrapolating the noisier finite-tree ``theta_star(L)``."""
    target = 0.5 * rhm_bp_density_evolution(v, s, m, de_depth, hi, pop=pop, seed=seed,
                                            grammar_seed=grammar_seed)
    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        ov = rhm_bp_density_evolution(v, s, m, de_depth, mid, pop=pop, seed=seed,
                                      grammar_seed=grammar_seed)
        if ov < target:
            a = mid
        else:
            b = mid
        if b - a < tol:
            break
    return 0.5 * (a + b)


def rhm_fss_collapse(v: int, s: int, m: int, depths, *, theta_grid: np.ndarray | None = None,
                     n_trees: int = 250, seed: int = 0, grammar_seed: int = 0,
                     n_reps: int = 1, theta_c: float | None = None) -> dict:
    """Finite-size scaling collapse for the RHM class-overlap transition (#136).

    Fits the mean-field-style ansatz ``m(theta, L) = L^{-beta/nu} f((theta -
    theta_c) L^{1/nu})`` using depth ``L`` as the scaling variable (the tree's
    "volume" grows as ``s**depth``, so depth plays the role of log-volume).

    ``theta_c`` is **anchored from density evolution**
    (``rhm_density_evolution_threshold``), not extrapolated from the noisy
    finite-tree ``theta_star(L)`` — with only a handful of feasible depths (exact-BP
    tree cost grows as ``s**depth``), a linear extrapolation of ``theta_star(L)`` is
    too unstable (it can even land outside ``[0, 1]``). Density evolution's
    population-dynamics cost is additive in population size, so it can run at a much
    larger effective depth and gives a stable estimate; pass ``theta_c`` explicitly to
    override. The convergence of ``theta_star(L)`` toward this anchor as depth grows
    is reported as a **consistency check**, not the estimator.

    Fit procedure:
    1. ``nu`` from a log-log fit of transition width vs depth (``width ~
       L**(-1/nu)``, the susceptibility-peak widths from ``rhm_transition_scan``).
    2. ``beta_over_nu`` from a log-log fit of ``m(theta_c, L)`` vs depth.

    Then rescales every depth's curve onto ``x = (theta - theta_c) L**(1/nu)``,
    ``y = m * L**beta_over_nu`` and reports ``collapse_residual`` = the mean
    cross-depth spread on the common rescaled domain — small residual means the
    curves genuinely collapse onto one master curve, confirming a real
    transition (not a per-depth artifact).

    Returns ``{theta_c, nu, beta_over_nu, collapse_residual, theta_stars, widths}``.
    """
    depths = np.asarray(list(depths), dtype=float)
    if theta_c is None:
        theta_c = rhm_density_evolution_threshold(v, s, m, seed=seed, grammar_seed=grammar_seed)
    curves = [rhm_transition_scan(v, s, m, int(L), theta_grid=theta_grid, n_trees=n_trees,
                                  seed=seed, grammar_seed=grammar_seed, n_reps=n_reps)
              for L in depths]
    theta_stars = np.array([c["theta_star"] for c in curves])
    widths = np.array([1.0 / np.max(c["susceptibility"]) for c in curves])

    slope, _ = np.polyfit(np.log(depths), np.log(widths), 1)
    nu = -1.0 / slope

    m_at_c = np.array([np.interp(theta_c, c["theta"], c["overlap"]) for c in curves])
    slope2, _ = np.polyfit(np.log(depths), np.log(np.clip(m_at_c, 1e-6, None)), 1)
    beta_over_nu = -slope2

    xs_resc, ys_resc = [], []
    for L, c in zip(depths, curves):
        xs_resc.append((c["theta"] - theta_c) * L ** (1.0 / nu))
        ys_resc.append(c["overlap"] * L ** beta_over_nu)
    lo = max(x.min() for x in xs_resc)
    hi = min(x.max() for x in xs_resc)
    grid = np.linspace(lo, hi, 20)
    stacked = np.array([np.interp(grid, *zip(*sorted(zip(x, y))))
                        for x, y in zip(xs_resc, ys_resc)])
    collapse_residual = float(np.mean(np.std(stacked, axis=0)))

    return {"theta_c": float(theta_c), "nu": float(nu), "beta_over_nu": float(beta_over_nu),
            "collapse_residual": collapse_residual, "theta_stars": theta_stars, "widths": widths}


def structured_corruption_shift(v: int, s: int, m: int, depth: int, *, r: int = 4,
                                beta: float = 6.0, n_trees: int = 250, seed: int = 0,
                                grammar_seed: int = 0, n_reps: int = 1,
                                features: np.ndarray | None = None) -> dict:
    """How does a CONCENTRATED (low-rank) corruption channel change the transition vs
    the structure-free (uniform-over-others) baseline? Both use the same zero-diagonal
    corruption-matrix code path with the SAME features ``U`` — only ``beta`` differs
    (``beta = 0`` = uniform-over-others), isolating the effect of concentration.

    **Finding (why the headline metric is the floor, not theta_star).** A concentrated
    channel does not merely *shift* the transition — it can *wash it out*: a corrupted
    leaf still points at a small neighbourhood of similar symbols, so it stays
    informative and the root class survives even under heavy corruption (low θ). The
    overlap curve goes flat and high, so its susceptibility ``theta_star`` is
    ill-defined (peak at the grid edge). The meaningful quantity is the **overlap
    floor** at heaviest corruption (θ = min of the grid): concentrated corruption
    *lifts the floor* (class stays recoverable). This parallels the founding
    observation of this line — the symmetric broadcast has no transition because it
    amplifies; concentrated corruption removes the transition because it fails to
    destroy leaf information. Returns ``{overlap_floor_uniform,
    overlap_floor_structured, floor_lift, theta_star_uniform, susc_peak_at_edge}``."""
    C0, U = make_lowrank_corruption(v, r, beta=0.0, seed=seed, features=features)
    C1, _ = make_lowrank_corruption(v, r, beta=beta, seed=seed, features=U)
    r0 = rhm_transition_scan(v, s, m, depth, corruption=C0, n_trees=n_trees,
                             seed=seed, grammar_seed=grammar_seed, n_reps=n_reps)
    r1 = rhm_transition_scan(v, s, m, depth, corruption=C1, n_trees=n_trees,
                             seed=seed, grammar_seed=grammar_seed, n_reps=n_reps)
    peak = int(np.argmax(r1["susceptibility"]))
    return {"overlap_floor_uniform": float(r0["overlap"][0]),
            "overlap_floor_structured": float(r1["overlap"][0]),
            "floor_lift": float(r1["overlap"][0] - r0["overlap"][0]),
            "theta_star_uniform": r0["theta_star"],
            "susc_peak_at_edge": bool(peak in (0, len(r1["susceptibility"]) - 1))}
