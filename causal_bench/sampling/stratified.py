"""Stratified (bulk/tail) ESS and a Euclidean positivity region R.

Global Kish ESS is a *false pass* for rare-event SMC: a cloud that is healthy in
the bulk (ESS > N/2) can have a tail stratum where one particle carries all the
mass (tail ESS ~ 1). The global number never sees it — the exact failure the §7
seam of the diffuse_directly design flagged ("resample on global Kish ESS < N/2
hides tail collapse, the exp29 region-R analog"). This module reports ESS_bulk
and ESS_tail *separately* and floors the tail, so the resample trigger fires on
tail degeneracy the global rule would wave through.

The tail is defined by a **positivity region R**: the neighbourhood of a set of
target points (the rare region the sampler must cover — real rare embeddings, or
the localization diagnostic's rare set). Coverage of R (do particles reach it at
all?) and mass in R (how much weight actually lands there?) are distinct
failures — a cloud can reach the tail yet carry vanishing weight there, which is
precisely tail-mass collapse. This is the FLAT correspondent of the geodesic
positivity R (#159): the neighbourhood here is Euclidean; #159 swaps in geodesic
distance on the estimated manifold. All CPU/numpy — no GPU (validated by
synthetic controls, same as the geometry detectors).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


# ─── stratified ESS ────────────────────────────────────────────────────────────

def _normalize(log_w: np.ndarray) -> np.ndarray:
    """Globally-normalized weights; a max-shift so nothing under/overflows.
    Returns all-zeros only if every particle is out of support (all -inf) — the
    caller (a diagnostic) should see that as total collapse, not crash."""
    m = np.max(log_w)
    if not np.isfinite(m):
        return np.zeros_like(log_w, dtype=float)
    w = np.exp(log_w - m)
    s = w.sum()
    return w / s if s > 0 else np.zeros_like(log_w, dtype=float)


def _within_ess(log_w_sub: np.ndarray) -> float:
    """Kish ESS on weights renormalized *within* a stratum. In [1, n] for a
    non-degenerate stratum; 0.0 for an empty stratum or one fully out of
    support. This measures degeneracy AMONG the stratum's particles — the number
    global ESS cannot express."""
    if log_w_sub.size == 0:
        return 0.0
    m = np.max(log_w_sub)
    if not np.isfinite(m):
        return 0.0
    w = np.exp(log_w_sub - m)
    s = w.sum()
    if s <= 0:
        return 0.0
    wn = w / s
    return float(1.0 / np.sum(wn ** 2))


@dataclass
class StratifiedESS:
    """Per-stratum ESS report. `labels[k]` is the k-th stratum's label; the
    other arrays are aligned to it. `global_ess` is the ordinary Kish ESS (the
    number that false-passes). `mass[k]` is stratum k's share of the *global*
    normalized weight; `ess[k]` its within-stratum Kish ESS; `ess_ratio[k] =
    ess[k]/counts[k]` its internal non-degeneracy in [0,1]."""
    n: int
    global_ess: float
    labels: np.ndarray
    counts: np.ndarray
    ess: np.ndarray
    ess_ratio: np.ndarray
    mass: np.ndarray

    def _idx(self, label) -> int:
        hits = np.nonzero(self.labels == label)[0]
        if hits.size == 0:
            raise KeyError(f"no stratum with label {label!r}")
        return int(hits[0])

    def stratum(self, label) -> dict:
        i = self._idx(label)
        return {"count": int(self.counts[i]), "ess": float(self.ess[i]),
                "ess_ratio": float(self.ess_ratio[i]), "mass": float(self.mass[i])}


def stratified_ess(log_w, strata) -> StratifiedESS:
    """ESS within each stratum plus the global ESS. `strata` is a per-particle
    label array — integer labels, or a boolean mask (treated as two strata:
    False=bulk, True=tail). Global ESS alone false-passes on tail collapse; the
    per-stratum `ess`/`ess_ratio` expose it."""
    log_w = np.asarray(log_w, dtype=float)
    strata = np.asarray(strata)
    if strata.shape[0] != log_w.shape[0]:
        raise ValueError(
            f"strata length {strata.shape[0]} != n particles {log_w.shape[0]}")
    if strata.dtype == bool:
        strata = strata.astype(np.int64)  # False->0 (bulk), True->1 (tail)

    wg = _normalize(log_w)
    global_ess = float(1.0 / np.sum(wg ** 2)) if np.any(wg > 0) else 0.0

    labels = np.unique(strata)
    counts = np.empty(labels.size, dtype=np.int64)
    ess = np.empty(labels.size, dtype=float)
    mass = np.empty(labels.size, dtype=float)
    for k, lab in enumerate(labels):
        m = strata == lab
        counts[k] = int(m.sum())
        ess[k] = _within_ess(log_w[m])
        mass[k] = float(wg[m].sum())
    ratio = np.divide(ess, counts, out=np.zeros_like(ess), where=counts > 0)
    return StratifiedESS(n=log_w.shape[0], global_ess=global_ess, labels=labels,
                         counts=counts, ess=ess, ess_ratio=ratio, mass=mass)


def stratified_resample_needed(report: StratifiedESS, *, global_frac: float = 0.5,
                               tail_label=1, tail_frac: float = 0.5,
                               tail_mass_floor: float = 0.0) -> tuple[bool, str]:
    """Resample trigger that floors the tail. Fires if the global rule fires
    (global_ess < global_frac * N) OR the tail stratum is internally degenerate
    (tail ess_ratio < tail_frac) OR the tail carries less than `tail_mass_floor`
    of the global weight. The second clause is the whole point: it catches
    collapse the global rule waves through. Returns (fire, reason)."""
    reasons = []
    if report.global_ess < global_frac * report.n:
        reasons.append(
            f"global ESS {report.global_ess:.1f} < {global_frac}*N={global_frac * report.n:.1f}")
    if tail_label in report.labels:
        t = report.stratum(tail_label)
        if t["ess_ratio"] < tail_frac:
            reasons.append(
                f"tail ESS_ratio {t['ess_ratio']:.3f} < {tail_frac} "
                f"(tail ESS {t['ess']:.1f}/{t['count']}) — global ESS "
                f"{report.global_ess:.1f} would MISS this")
        if t["mass"] < tail_mass_floor:
            reasons.append(f"tail mass {t['mass']:.3g} < floor {tail_mass_floor:g}")
    return (len(reasons) > 0, "; ".join(reasons) if reasons else "healthy")


# ─── Euclidean positivity region R ─────────────────────────────────────────────

@dataclass
class PositivityR:
    """Euclidean positivity-region diagnostic. `coverage` is the fraction of
    target points with at least one particle within `radius` (do we REACH R?).
    `uncovered` flags the targets no particle reaches (structural support gaps —
    no reweighting recovers them; the honest fix is upstream). `in_region` marks
    particles inside R; `mass_in_region` is their share of the global weight (how
    much weight actually LANDS in R)."""
    radius: float
    coverage: float
    uncovered: np.ndarray        # bool mask over targets
    in_region: np.ndarray        # bool mask over particles
    mass_in_region: float


def positivity_overlap(particles, targets, *, radius: float, log_w=None) -> PositivityR:
    """Coverage of the rare region R (targets) by the particle cloud, plus the
    weight mass landing in R. Coverage and mass are distinct failures: reaching
    R with negligible weight is tail-mass collapse, not positivity. FLAT (#159
    swaps Euclidean neighbourhood for geodesic)."""
    P = np.asarray(particles, dtype=float)
    T = np.asarray(targets, dtype=float)
    if P.ndim != 2 or T.ndim != 2 or P.shape[1] != T.shape[1]:
        raise ValueError(
            f"particles {P.shape} and targets {T.shape} must be (n, d) with equal d")

    ptree = cKDTree(P)
    # a target is covered if >=1 particle sits within radius of it
    covered = np.array([len(ptree.query_ball_point(t, radius)) > 0 for t in T])
    coverage = float(covered.mean()) if T.shape[0] else 1.0

    ttree = cKDTree(T)
    # a particle is in R if within radius of any target
    d_to_target, _ = ttree.query(P, k=1)
    in_region = d_to_target <= radius

    if log_w is None:
        mass = float(in_region.mean()) if P.shape[0] else 0.0  # unweighted share
    else:
        wg = _normalize(np.asarray(log_w, dtype=float))
        mass = float(wg[in_region].sum())
    return PositivityR(radius=radius, coverage=coverage, uncovered=~covered,
                       in_region=in_region, mass_in_region=mass)


def strata_from_region(particles, targets, *, radius: float) -> np.ndarray:
    """Boolean tail mask: True where a particle lies in the positivity region R
    (within `radius` of a target). Ties the bulk/tail split to R itself, so
    `stratified_ess(log_w, strata_from_region(...))` reports ESS_tail on exactly
    the rare region the positivity check cares about."""
    P = np.asarray(particles, dtype=float)
    T = np.asarray(targets, dtype=float)
    d, _ = cKDTree(T).query(P, k=1)
    return d <= radius
