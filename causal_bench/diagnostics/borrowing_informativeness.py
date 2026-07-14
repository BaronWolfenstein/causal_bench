"""Per-level identifiability report to inform hierarchical borrowing MANUALLY (#137
follow-up). Turns the SFW/RHM insight into an analyst-facing diagnostic.

The reconciliation (memory `sfw-borrowing-diagnostic`): SFW/RHM does NOT set shrinkage
coefficients and is NOT the discrete-diffusion generator. Its live, off-box role is as
a **qualitative identifiability diagnostic** via the embedding channel — coarse/root
coordinates survive noise more than fine/leaf ones (`t_coarse* > t_fine*`). This module
measures that per hierarchy level and reports it so an analyst can set pooling by hand:

- **high `t*`** (survives lots of VP-SDE noise) → well-identified, high-SNR → *borrow
  little*, the level stands alone;
- **low `t*`** (collapses early) → information-starved → *borrow heavily* from the
  parent level;
- **`t*` of adjacent levels ≈ equal** → the split isn't real in the representation →
  *don't pool across it as if it were a distinct level*.

It does NOT produce the shrinkage — that stays the hierarchical fit's job
(`three_level_bhm`). It tells you which levels are trustworthy to pool across, and
whether the hierarchy you assumed is actually separated in the frozen embedding.
Builds on `theta_time_map.embedding_transition_scan`. numpy only; needs labeled
embeddings (synthetic now; real embeddings gated on-box).
"""
from __future__ import annotations

import numpy as np

from causal_bench.generative.vpsde import Schedule
from causal_bench.diagnostics.theta_time_map import embedding_transition_scan


def level_identifiability(X: np.ndarray, level_labels, *, level_names=None,
                          sch: Schedule | None = None, n_grid: int = 25, n_reps: int = 4,
                          rng: np.random.Generator | None = None) -> list[dict]:
    """For each hierarchy level (``level_labels`` a coarse→fine list of integer label
    arrays), compute class means from the *clean* embeddings and locate the VP-SDE
    identifiability transition ``t_star`` — the noise level at which that level's
    identity collapses. The overlap curve is **averaged over ``n_reps`` independent
    noise realizations** before locating the susceptibility peak; a single scan's
    ``t_star`` is too noisy (±~0.15) to base borrowing guidance on. Trivial single-class
    levels are skipped. Returns a per-level list of ``{name, index, n_classes, t_star,
    t_frac, overlap}`` in the given (coarse→fine) order."""
    sch = sch or Schedule(n_steps=200)
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    levels = []
    for i, labels in enumerate(level_labels):
        labels = np.asarray(labels)
        classes = np.unique(labels)
        if len(classes) < 2:
            continue                                            # trivial: no transition
        remap = {c: j for j, c in enumerate(classes)}
        lab = np.array([remap[c] for c in labels])
        means = np.array([X[labels == c].mean(0) for c in classes])
        scans = [embedding_transition_scan(X, lab, means, sch=sch, n_grid=n_grid, rng=rng)
                 for _ in range(n_reps)]
        t_frac = scans[0]["t_frac"]
        overlap = np.mean([s["overlap"] for s in scans], axis=0)
        susc = -np.diff(overlap) / np.diff(t_frac)              # overlap falls as t grows
        mid = 0.5 * (t_frac[:-1] + t_frac[1:])
        levels.append({
            "name": level_names[i] if level_names else f"L{i}",
            "index": i, "n_classes": int(len(classes)),
            "t_star": float(mid[int(np.argmax(susc))]),
            "t_frac": t_frac, "overlap": overlap,
        })
    return levels


def borrowing_report(levels: list[dict], *, sep_tol: float = 0.05) -> dict:
    """Turn per-level ``t_star`` into manual-borrowing guidance. Coarser (earlier)
    levels should have HIGHER ``t_star`` (survive more noise). Reports, for each
    adjacent coarse→fine pair, the **separation gap** ``t*_coarse − t*_fine`` (positive
    ⇒ genuinely separated; ≈0 ⇒ the finer split is not resolved in the representation →
    don't pool across it as a distinct level), a ``well_separated`` flag, and a
    qualitative per-level pooling recommendation keyed to ``t_star`` relative to the
    most-identifiable level. Guidance only — the shrinkage itself comes from the
    hierarchical fit."""
    ts = [L["t_star"] for L in levels]
    gaps = [ts[i] - ts[i + 1] for i in range(len(ts) - 1)]     # coarse − fine
    well_separated = bool(gaps) and all(g > sep_tol for g in gaps)
    tmax = max(ts) if ts else 1.0
    out = []
    for L in levels:
        r = L["t_star"] / (tmax + 1e-12)
        if r >= 0.85:
            rec = "well-identified — borrow little (level stands on its own)"
        elif r >= 0.5:
            rec = "moderate SNR — partial pooling toward parent"
        else:
            rec = "information-starved — borrow heavily from parent"
        out.append({"name": L["name"], "n_classes": L["n_classes"],
                    "t_star": L["t_star"], "recommendation": rec})
    unresolved = [(levels[i]["name"], levels[i + 1]["name"])
                  for i, g in enumerate(gaps) if g <= sep_tol]
    return {"levels": out, "separation_gaps": gaps,
            "well_separated": well_separated, "unresolved_splits": unresolved}
