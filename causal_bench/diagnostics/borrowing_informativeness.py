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


def suggest_tau_prior(t_star: float, *, tau_sd_min: float = 0.05,
                      tau_sd_max: float = 1.0) -> float:
    """Map a level's identifiability ``t_star`` (VP-SDE schedule fraction in [0,1]) to a
    suggested between-group SD prior scale ``tau_sd``. Monotone increasing: a
    well-identified level (high ``t_star``) → **larger** ``tau_sd`` (weak pooling, let
    subgroups differ); an information-starved level (low ``t_star``) → **smaller**
    ``tau_sd`` (strong pooling, borrow heavily toward the parent)."""
    r = float(np.clip(t_star, 0.0, 1.0))
    return tau_sd_min + r * (tau_sd_max - tau_sd_min)


def recommend_tau_priors(levels: list[dict], *, tau_sd_min: float = 0.05,
                         tau_sd_max: float = 1.0, sep_tol: float = 0.05) -> dict:
    """Wire the per-level identifiability report to the hierarchical fit's borrowing
    knob. For each level, suggest a ``tau_sd`` (the HalfNormal scale on the
    between-subgroup SD τ) for ``estimators.three_level_bhm.fit_three_level_meta`` /
    ``fit_three_level_bhm``: high ``t_star`` (robust) → larger ``tau_sd`` (weak pooling);
    low ``t_star`` (info-starved) → smaller ``tau_sd`` (strong pooling).

    This **informs** the borrowing decision — it does NOT set the shrinkage (that stays
    the hierarchical fit's job) and is not an automatic override. It also surfaces
    ``unresolved_splits``: adjacent levels whose ``t_star`` are ≈equal are not separated
    in the representation and should not be fit as distinct levels — pool them.

    Usage::

        levels = level_identifiability(X, level_labels)
        rec = recommend_tau_priors(levels)
        # analyst reviews rec, then (in the pymc .venv312 stack):
        fit_three_level_meta(theta_hat, se, tau_sd=rec["per_level"]["L3"]["tau_sd"])

    Returns ``{per_level: {name: {t_star, n_classes, tau_sd, recommendation}},
    unresolved_splits, well_separated}``."""
    rep = borrowing_report(levels, sep_tol=sep_tol)
    rec_by_name = {L["name"]: L["recommendation"] for L in rep["levels"]}
    per_level = {
        L["name"]: {
            "t_star": L["t_star"], "n_classes": L["n_classes"],
            "tau_sd": suggest_tau_prior(L["t_star"], tau_sd_min=tau_sd_min,
                                        tau_sd_max=tau_sd_max),
            "recommendation": rec_by_name[L["name"]],
        }
        for L in levels
    }
    return {"per_level": per_level, "unresolved_splits": rep["unresolved_splits"],
            "well_separated": rep["well_separated"]}


def canonical_tau_prior(decode_acc: float, n_classes: int, *, tau_sd_min: float = 0.05,
                        tau_sd_max: float = 1.0) -> float:
    """Correctly-signed canonical map for the BP-decoded-labels pipeline (#144).

    Takes the per-level **decode accuracy** at the working θ₀ (the operating-point
    learnability scalar from ``dgp.joint_hierarchy.decode_cohort_labels``), NOT the
    embedding ``t_star`` and NOT ``theta_c``. Chance-adjusted:
    ``r = clip((acc − 1/K)/(1 − 1/K), 0, 1)``; ``tau_sd = tau_sd_min + r·(tau_sd_max −
    tau_sd_min)``.

    Monotone **increasing** in accuracy: a well-decoded level (``r → 1``) → large
    ``tau_sd`` (weak pooling — its between-group differences are trustworthy); a level
    decoded near chance (``r → 0``) → ``tau_sd_min`` (strong pooling — the differences
    are mostly misclassification noise). Grounded in the misclassification-attenuation
    argument, not the ``suggest_tau_prior`` convention.

    SIGN WARNING: do NOT feed ``theta_c`` (the reconstruction threshold, where *lower* =
    more robust) into this or ``suggest_tau_prior`` — both expect a higher-is-more-robust
    scalar. Decode accuracy is the correct, higher-is-robust operating-point input; a
    ``theta_c`` plug-in would set priors backwards."""
    chance = 1.0 / n_classes
    r = float(np.clip((decode_acc - chance) / (1.0 - chance), 0.0, 1.0))
    return tau_sd_min + r * (tau_sd_max - tau_sd_min)


def recommend_tau_priors_from_decode(decode_result: dict, g: int, b_size: int, *,
                                     tau_sd_min: float = 0.05, tau_sd_max: float = 1.0) -> dict:
    """Per-level ``tau_sd`` from a ``decode_cohort_labels`` result via
    ``canonical_tau_prior``. The coarse (group) level, decoded more accurately, gets a
    larger ``tau_sd`` (weak pooling) than the fine (member) level. Returns
    ``{group: {decode_acc, n_classes, tau_sd}, member: {...}}`` — the analyst feeds each
    ``tau_sd`` to ``fit_three_level_meta`` for that level. Informs, does not set."""
    return {
        "group": {"decode_acc": decode_result["group_decode_acc"], "n_classes": g,
                  "tau_sd": canonical_tau_prior(decode_result["group_decode_acc"], g,
                                                tau_sd_min=tau_sd_min, tau_sd_max=tau_sd_max)},
        "member": {"decode_acc": decode_result["member_decode_acc"], "n_classes": b_size,
                   "tau_sd": canonical_tau_prior(decode_result["member_decode_acc"], b_size,
                                                 tau_sd_min=tau_sd_min, tau_sd_max=tau_sd_max)},
    }


def canonical_tau_discount(decode_acc: float, n_classes: int) -> float:
    """Learnability **discount** in [0, 1] from a level's decode accuracy at θ₀ —
    the *correctly-structured* identifiability input to a borrowing prior (#144, exp41).

    Chance-adjusted: ``r = clip((acc − 1/K)/(1 − 1/K), 0, 1)``. The prior is then
    ``tau_sd = tau_base · discount``, where ``tau_base`` is the analyst's
    effect-heterogeneity scale prior. Identifiability **discounts** that base scale for
    imperfect subgroup resolution (perfect decode → 1 → use the full base scale; near
    chance → 0 → pool), it does NOT set the scale — because identifiability ≠ τ magnitude
    (a well-resolved partition can still have tiny effects).

    This fixes the exp41 power loss: the absolute ``canonical_tau_prior`` mapped high
    decode accuracy straight to a large ``tau_sd`` (≈ ``tau_sd_max``) regardless of the
    true τ, over-widening the μ posterior and killing power at well-decoded levels. As a
    *discount* on a base scale, a well-decoded level recovers the base prior (≈ flat)
    while a poorly-decoded level still pools harder — the honest learnability role.

    **Caveats (independent review, #144).** This discount **equals the symmetric-channel
    misclassification attenuation** ``λ = (a − 1/K)/(1 − 1/K)`` — so ``tau_sd = tau_base·λ``
    tracks the decoded-frame heterogeneity ``τ_obs = λ·τ_true`` *by construction*. That
    makes the exp41 validation partially circular and its Type-I safety **contingent on
    near-symmetric misclassification** (won't hold for structured confusion). It takes a
    **decode accuracy** (higher = robust); do NOT feed ``theta_c`` (lower = robust) or an
    embedding ``t_star`` — different objects, different channel, one already caused a
    sign-inversion bug. **This licenses nothing about frozen-encoder embeddings**: there
    is no exact BP, ``decode_acc`` needs true labels, and if subgroups are defined by
    observed covariates (not decoded from the representation) the whole learnability
    argument is vacuous. Only the structural guidance ([[borrowing-informativeness]]
    ``unresolved_splits``, discount-not-setter) transfers to real embeddings."""
    chance = 1.0 / n_classes
    return float(np.clip((decode_acc - chance) / (1.0 - chance), 0.0, 1.0))
