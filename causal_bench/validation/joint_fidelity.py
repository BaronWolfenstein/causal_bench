"""Borrowing-calibration fidelity engine on the joint DGP — the reusable core of
exp41 (#144 step C). Runs the BP-decoded-labels pipeline end to end and reports the
frequentist operating characteristics of an identifiability-informed borrowing prior.

Pipeline per replicate: ``sample_joint_cohort`` → ``decode_cohort_labels(θ₀)`` →
per-**decoded**-subgroup ``(theta_hat, se)`` → ``fit_three_level_meta(theta_hat, se,
tau_sd=policy)`` → decision on the population effect μ. The estimator only ever sees
BP-decoded labels; the true labels are used solely to compute the oracle τ and the
known μ for scoring.

``tau_policy``:
- ``flat``    — a fixed ``tau_sd`` (the naive baseline);
- ``oracle``  — the true between-subgroup effect SD at the level (best case);
- ``empirical`` — the FIXED van Zwet CDSR LogNormal τ prior (SMD→raw scale-bridged), the
  reference-class baseline the identifiability-aware policy must beat
  (``empirical_tau_prior``); ignores decode accuracy by construction;
- ``canonical`` — the SAME LogNormal, shifted in log-location by
  ``log canonical_tau_discount(decode_acc)`` — i.e. "the van Zwet prior, discounted by
  identifiability" (#144 item 3b). Holding the FAMILY fixed is deliberate: canonical was
  previously a HalfNormal scale, so ``canonical vs empirical`` confounded the
  identifiability discount with the prior family — the one thing the comparison is
  meant to price.

``use_true_labels=True`` is the DECONTAMINATED control (#144 item 1): pool over the true
labels instead of the decoded ones. A truly-null *decoded* subgroup is polluted by units
from non-null siblings before any borrowing occurs, which saturated the partial-null
per-subgroup Type-I; removing that channel leaves inflation attributable to borrowing
alone.

This is the ENGINE (a library function); the exp41 experiment script sweeps regimes ×
θ₀ × grammar configs × policies and compares reject/coverage curves. Requires the 3.12
``[bayes]`` stack (PyMC/NumPyro) via ``fit_three_level_meta``.
"""
from __future__ import annotations

import numpy as np

from causal_bench.dgp.joint_hierarchy import (
    make_joint_hierarchy, sample_joint_cohort, decode_cohort_labels, true_tau_by_level,
)
from causal_bench.diagnostics.borrowing_informativeness import canonical_tau_discount


# The estimator forms its interval as effect +/- 1.96*se (three_level_bhm._decision),
# i.e. a nominal 95% interval. The interval score's penalty leverage is 2/alpha, so
# this constant must track that z -- do not set one without the other.
CI_ALPHA = 0.05

# ── Monte-Carlo error on the OCs (#144 fix item 4) ───────────────────────────
# The v2 run read coverage 0.96-1.00 everywhere and concluded "degenerate". That is only
# a legitimate conclusion with an error bar: at n_reps=100 the MC SE on a coverage near
# 0.95 is ~0.022, so 0.96 and 1.00 sit about one SE apart. Plain binomial SE is the wrong
# tool at the boundary -- it is exactly 0 when the observed proportion is 1, implying
# infinite precision when you have merely not yet seen a failure. Wilson score intervals
# stay finite there, which is the case the K-grid run turns on.
def binom_se(p: float, n: int) -> float:
    """Binomial MC standard error of a proportion. Note this is 0 at p in {0, 1} --
    prefer `wilson_ci` for statements about boundary cells."""
    if n <= 0 or not np.isfinite(p):
        return float("nan")
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for k successes in n trials. Stays informative at k=0 and
    k=n, where the normal-approximation interval degenerates to a point."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    lo, hi = centre - half, centre + half
    # The exact Wilson interval always contains p; at k=0 / k=n round-off can put an
    # endpoint an ulp on the wrong side, so clamp to guarantee containment.
    return (float(min(max(0.0, lo), p)), float(max(min(1.0, hi), p)))


def binom_ci_from_rate(rate: float, n: int, z: float = 1.96) -> tuple:
    """Wilson CI from an ALREADY-AGGREGATED rate — lets runs that predate this reporting
    (e.g. exp41 v3, launched earlier) be interpreted from `(rate, n_used)` without a
    re-run, since a binomial CI needs nothing else."""
    if n <= 0 or not np.isfinite(rate):
        return (float("nan"), float("nan"))
    return wilson_ci(int(round(rate * n)), n, z=z)


def population_effect(spec: dict) -> float:
    """The true population-average treatment effect μ implied by the spec's effect
    tables (subgroups uniform): ``w_group·mean(group_effect) + w_member·mean(member_
    effect)``. Used as ``true_effect`` for coverage and as the null target when the
    effect tables are centered."""
    return float(spec["w_group"] * spec["group_effect"].mean()
                 + spec["w_member"] * spec["member_effect"].mean())


def _subgroup_estimates(Y, A, sub, n_sub, *, min_per_arm=3):
    """Per-decoded-subgroup treatment-effect estimates (mean-difference), SEs, and the
    surviving subgroup indices. Subgroups without at least ``min_per_arm`` units in each
    arm are dropped — ``kept`` maps rows of ``(theta_hat, se)`` back to subgroup ids (so a
    partial-null test can locate the null subgroup after drops)."""
    th, se, kept = [], [], []
    for k in range(n_sub):
        m = sub == k
        y1, y0 = Y[m & (A == 1)], Y[m & (A == 0)]
        if len(y1) < min_per_arm or len(y0) < min_per_arm:
            continue
        th.append(y1.mean() - y0.mean())
        se.append(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
        kept.append(k)
    return np.asarray(th), np.asarray(se), np.asarray(kept, int)


def _subgroup_purity(true_lab, dec_lab, kept):
    """Per-DECODED-subgroup purity: fraction of a decoded subgroup's units whose TRUE
    label is the modal one. Low purity = contaminated by units from other true
    subgroups = the estimate is attenuated/unreliable. This is the per-subgroup
    reliability the `canonical_ps` policy uses to inflate se (regression calibration /
    #182), so the hierarchical fit shrinks impure subgroups more -- the per-subgroup
    reliability weighting a level-wide scalar discount (canonical) structurally cannot do."""
    true_lab = np.asarray(true_lab); dec_lab = np.asarray(dec_lab)
    out = []
    for k in kept:
        tl = true_lab[dec_lab == k]
        if tl.size == 0:
            out.append(1.0); continue
        _, cnts = np.unique(tl, return_counts=True)
        out.append(float(cnts.max() / cnts.sum()))
    return np.asarray(out)


# van Zwet-Więcek-Gelman 2025 empirical CDSR prior on between-study heterogeneity τ,
# SMD/probit scale: log τ ~ N(-1.82, 0.90) (median τ ≈ 0.16 SMD). See the
# reference-class caveat below and memory reference_vanzwet_single_trial_prior.
_VZ_TAU_MU_LOG, _VZ_TAU_SIGMA_LOG = -1.82, 0.90


def empirical_tau_prior(sigma_y: float, *, mu_log: float = _VZ_TAU_MU_LOG,
                        sigma_log: float = _VZ_TAU_SIGMA_LOG) -> tuple:
    """The fixed van Zwet empirical τ prior, bridged from the SMD scale to this DGP's raw
    effect scale. Subgroup effects here are raw mean-differences with outcome SD ``sigma_y``,
    so τ_raw = τ_SMD · σ ⇒ on the log scale the location shifts by ``log σ`` and the spread
    is unchanged. Returns ``("lognormal", (mu_log + log σ, sigma_log))``.

    REFERENCE-CLASS CAVEAT: van Zwet's prior is pooled over ~1600 Cochrane meta-analyses;
    it is a generic "how heterogeneous are effects" prior, not ENCIRCLE-specific, and the
    SMD denominator is approximated here by the residual outcome SD ``sigma_y`` (not a
    separately-estimated total SD). This is the FIXED baseline the identifiability-aware
    policies must beat, not a bespoke elicitation."""
    return ("lognormal", (mu_log + float(np.log(sigma_y)), sigma_log))


def _policy_tau_prior(policy, level, spec, decoded, *, flat_tau_sd, tau_base, tau_sd_min,
                      sigma):
    """Map a policy name to a τ prior ``(family, params)`` for ``fit_three_level_meta``.
    flat/oracle/canonical are HalfNormal scales (the identifiability-aware family); the
    ``empirical`` policy is the fixed van Zwet LogNormal baseline (ignores decode accuracy
    and level by construction — it is the reference-class prior to beat)."""
    if policy == "flat":
        return ("halfnormal", (flat_tau_sd,))
    if policy == "oracle":
        key = "tau_group" if level == "group" else "tau_member"
        return ("halfnormal", (max(true_tau_by_level(spec)[key], 1e-3),))  # scale must be > 0
    if policy == "canonical_ps":
        # per-subgroup reliability policy: the tau prior is the FIXED empirical (van
        # Zwet); the reliability weighting happens on the SE side (se inflated by
        # per-subgroup purity in the rep loop), not the tau scale. So the only change
        # vs empirical is the heteroskedastic, reliability-inflated se -- the test of
        # whether per-subgroup reliability (not a level-wide discount) makes an
        # identifiability-aware policy actually beat empirical.
        return empirical_tau_prior(sigma)
    if policy == "canonical":
        acc = decoded["group_decode_acc" if level == "group" else "member_decode_acc"]
        k = spec["g"] if level == "group" else spec["b_size"]
        # #144 item 3b: canonical is the SAME LogNormal as `empirical`, shifted in
        # log-location by the learnability discount -- i.e. "the van Zwet prior,
        # discounted by identifiability". Previously canonical was a HalfNormal scale,
        # so `canonical vs empirical` confounded the identifiability discount with the
        # PRIOR FAMILY; holding the family fixed isolates the discount, which is the
        # only thing this experiment is trying to price. A well-decoded level recovers
        # the empirical prior; a poorly-decoded one shifts down and pools harder.
        _, (mu_log, sigma_log) = empirical_tau_prior(sigma)
        disc = float(np.clip(canonical_tau_discount(acc, k), 1e-3, 1.0))
        return ("lognormal", (mu_log + float(np.log(disc)), sigma_log))
    if policy == "empirical":
        return empirical_tau_prior(sigma)
    raise ValueError(f"unknown policy {policy!r}")


def _prior_scale(tau_prior: tuple) -> float:
    """A single representative τ scale for reporting (``mean_tau_sd``): the HalfNormal
    scale, or the LogNormal median exp(mu_log)."""
    family, params = tau_prior
    return float(params[0]) if family == "halfnormal" else float(np.exp(params[0]))


def joint_fidelity(spec: dict, *, level: str = "group", policy: str = "canonical",
                   theta0: float = 0.7, n_reps: int = 20, n_units: int = 4000,
                   depth: int = 7, sigma: float = 0.5, flat_tau_sd: float = 0.5,
                   tau_base: float = 0.5, tau_sd_min: float = 0.05, draws: int = 500,
                   tune: int = 500, chains: int = 2, seed: int = 0,
                   chain_method: str = "sequential", fast: bool = False,
                   tail_ess_threshold: float = 100.0, null_subgroup: int | None = None,
                   use_true_labels: bool = False, max_escalations: int = 2,
                   resample_effects: bool = False) -> dict:
    """Operating characteristics of the borrowing prior at one (level, policy, θ₀, spec)
    cell. ``reject_rate`` is the population-μ decision (Type-I under a null spec, power
    under an alt). When ``null_subgroup`` is set (a partial-null spec, e.g.
    ``make_partial_null_spec``), also reports ``subgroup_reject_rate`` — the per-subgroup
    Type-I of that truly-null subgroup, the estimand where strong borrowing inflates size
    by dragging it toward non-null siblings.

    Reports (per the #144 review): ``mean_ci_width`` (μ CI width — the primary honest OC,
    since reject≈0 at small K is a size-≈0 test, not "nominal"), ``coverage``, and BOTH
    the tail-ESS-gated ``reject_rate`` and the flagged-included ``reject_rate_uncond``
    (dropping flagged fits is selection-on-data). Subgroups = the DECODED labels at
    ``level``; the prior is set by ``policy``."""
    from causal_bench.estimators.three_level_bhm import (
        fit_three_level_meta, fit_three_level_meta_fast, tail_ess_ok)

    mu_true = population_effect(spec)
    tau_true = true_tau_by_level(spec)["tau_group" if level == "group" else "tau_member"]
    # v5 fix: make_scenario_spec pins the level's effect table to mean EXACTLY mu (SD
    # exactly tau), and the rep loop reuses the SAME spec, so the population mean has zero
    # sampling variability over the heterogeneity the interval is sized for -> coverage
    # saturates at 1.0 and cannot rank policies. resample_effects re-draws the table
    # GENUINELY, theta_g ~ N(mu, tau) (mean NOT pinned), fresh per replicate, so coverage
    # of the hyper-mean mu is a real frequentist quantity. mu_true stays mu (the target).
    _eff_key = "group_effect" if level == "group" else "member_effect"
    _eff_w = "w_group" if level == "group" else "w_member"
    _n_eff = spec["g"] if level == "group" else spec["b_size"]
    rejects, covers, taus, widths, sub_rejects = [], [], [], [], []
    iscores, pens, sq_errs, sub_risks = [], [], [], []         # sq_errs: mu-MSE; sub_risks: subgroup MSE
    rejects_all, n_flagged, n_used, n_escalated = [], 0, 0, 0
    accs: list = []                                            # decode accuracy per replicate
    for r in range(n_reps):
        if resample_effects:
            _rng = np.random.default_rng(seed + 5000 + r)
            spec[_eff_key] = mu_true + tau_true * _rng.standard_normal(_n_eff)
            spec[_eff_w] = 1.0
        coh = sample_joint_cohort(spec, n_units, depth, sigma=sigma, seed=seed + r)
        dec = decode_cohort_labels(spec, coh, theta0=theta0, seed=seed + 1000 + r)
        # decode accuracy at the level is the canonical policy's INPUT; record it for every
        # replicate (it is a property of the cohort, not of the fit) so a K sweep can be
        # read honestly — a larger K also enlarges the grammar alphabet and can depress
        # decode accuracy, confounding "more subgroups" with "harder decode".
        if use_true_labels:
            # DECONTAMINATED control (#144 fix 1): pool over the TRUE labels. A
            # truly-null DECODED subgroup is polluted by units from non-null siblings
            # before any borrowing occurs, which saturated the partial-null
            # per-subgroup Type-I. Removing that channel leaves inflation
            # attributable to borrowing alone. Equivalent to perfect decode (θ₀=1).
            sub = coh["group" if level == "group" else "member"]
            accs.append(1.0)
        else:
            accs.append(dec["group_decode_acc" if level == "group"
                            else "member_decode_acc"])
            sub = dec["group_decoded" if level == "group" else "member_decoded"]
        n_sub = spec["g"] if level == "group" else spec["b_size"]
        th, se, kept = _subgroup_estimates(coh["Y"], coh["A"], sub, n_sub)
        if len(th) < 2:
            continue
        se_fit = se
        if policy == "canonical_ps":
            # reliability-inflate se: impure (contaminated) decoded subgroups get a
            # larger effective se, so the hierarchical fit shrinks them more. Cap the
            # inflation at 3x (purity floored at 1/3, the chance level for K>=3).
            true_lab = coh["group" if level == "group" else "member"]
            purity = _subgroup_purity(true_lab, sub, kept)
            se_fit = se / np.clip(purity, 1.0 / 3.0, 1.0)
        tau_prior = _policy_tau_prior(policy, level, spec, dec, flat_tau_sd=flat_tau_sd,
                                      tau_base=tau_base, tau_sd_min=tau_sd_min, sigma=sigma)
        def _fit(d, t, sd_seed):
            if fast and null_subgroup is None:                  # compile-once path
                return fit_three_level_meta_fast(
                    th, se_fit, tau_prior=tau_prior, true_effect=mu_true, draws=d, tune=t,
                    chains=chains, seed=sd_seed, chain_method=chain_method, n_pad=n_sub,
                    return_theta=True)                          # for the subgroup-risk metric
            return fit_three_level_meta(
                th, se_fit, tau_prior=tau_prior, true_effect=mu_true, draws=d, tune=t,
                chains=chains, seed=sd_seed, chain_method=chain_method,
                return_theta=True)

        fit = _fit(draws, tune, seed + r)
        # ESCALATE rather than drop (#144). A low tail-ESS is a COMPUTATIONAL failure,
        # not a property of the replicate, so discarding it is selection-on-data: it
        # biases the OCs, makes n_used differ systematically across policies (a diffuse
        # prior samples worse, so `flat` lost far more fits than `empirical`), and
        # thins the very sample the coverage CI is computed from. Re-running with more
        # draws is what oc_simulation_pipeline.mermaid always specified.
        att = 0
        while att < max_escalations and not tail_ess_ok(fit, threshold=tail_ess_threshold):
            att += 1
            fit = _fit(draws * 2 ** att, tune * 2 ** att, seed + r + 7919 * att)
        n_escalated += (att > 0)
        rejects_all.append(fit["rejects_null"])                 # flagged-included sensitivity
        if not tail_ess_ok(fit, threshold=tail_ess_threshold):
            n_flagged += 1
            continue
        rejects.append(fit["rejects_null"])
        covers.append(fit["covers_truth"])
        sq_errs.append(float((fit["effect"] - mu_true) ** 2))  # point-estimate risk (mu-hat; prior-insensitive)
        # SUBGROUP-level risk: MSE of the shrunk theta_g estimates vs the TRUE subgroup
        # effects. This is where the tau prior actually bites (shrinkage of groups toward
        # mu), so unlike mu-MSE it separates policies: a fixed empirical prior over-shrinks
        # when true tau is large, oracle shrinks correctly. Truth = the drawn effect table.
        if "theta_g_mean" in fit and len(kept):
            tg_true = np.asarray(spec[_eff_key], float)[kept]
            tg_hat = np.asarray(fit["theta_g_mean"], float)[:len(kept)]
            sub_risks.append(float(np.mean((tg_hat - tg_true) ** 2)))
        taus.append(_prior_scale(tau_prior))
        widths.append(fit["ci_hi"] - fit["ci_lo"])
        # Interval score (Gneiting & Raftery), the proper scoring rule for an interval
        # forecast. Coverage saturates at 1 and width just reads the prior back, so
        # neither ranks policies on its own; IS penalises width and miscoverage jointly.
        #   IS = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}
        # Kept DECOMPOSED as well as summed: the penalty carries a 2/alpha = 40x leverage
        # at alpha=0.05, so a rare small miss can outweigh a width difference. Reporting
        # only the total would hide whether IS is adding anything over mean_ci_width.
        _l, _u = fit["ci_lo"], fit["ci_hi"]
        _pen = (max(_l - mu_true, 0.0) + max(mu_true - _u, 0.0)) * (2.0 / CI_ALPHA)
        pens.append(float(_pen))
        iscores.append(float((_u - _l) + _pen))
        if null_subgroup is not None:
            pos = np.where(kept == null_subgroup)[0]
            if len(pos):                                        # null subgroup survived drops
                sub_rejects.append(bool(fit["theta_g_rejects"][pos[0]]))
        n_used += 1
    return {
        "reject_rate": float(np.mean(rejects)) if rejects else float("nan"),
        "reject_rate_uncond": float(np.mean(rejects_all)) if rejects_all else float("nan"),
        "subgroup_reject_rate": float(np.mean(sub_rejects)) if sub_rejects else float("nan"),
        "coverage": float(np.mean(covers)) if covers else float("nan"),
        "coverage_se": binom_se(float(np.mean(covers)), len(covers)) if covers else float("nan"),
        "coverage_lo": wilson_ci(int(np.sum(covers)), len(covers))[0] if covers else float("nan"),
        "coverage_hi": wilson_ci(int(np.sum(covers)), len(covers))[1] if covers else float("nan"),
        "reject_rate_se": binom_se(float(np.mean(rejects)), len(rejects)) if rejects else float("nan"),
        "mean_ci_width": float(np.mean(widths)) if widths else float("nan"),
        "mean_ci_width_se": (float(np.std(widths, ddof=1) / np.sqrt(len(widths)))
                             if len(widths) > 1 else float("nan")),
        # Proper scoring rule: lower is better. `mean_penalty` is the miscoverage part
        # ALONE -- if it is ~0 the score has collapsed to the width and adds nothing,
        # which is the check that decides whether IS rescues this design.
        "mean_interval_score": float(np.mean(iscores)) if iscores else float("nan"),
        "mean_interval_score_se": (float(np.std(iscores, ddof=1) / np.sqrt(len(iscores)))
                                   if len(iscores) > 1 else float("nan")),
        "mean_penalty": float(np.mean(pens)) if pens else float("nan"),
        "mean_tau_sd": float(np.mean(taus)) if taus else float("nan"),
        "mean_decode_acc": float(np.mean(accs)) if accs else float("nan"),
        "mu_true": mu_true, "tau_true": float(tau_true),
        # Point-estimate risk (MSE of mu-hat): the DISCRIMINATING metric v4 lacked.
        # Unlike coverage (saturates) and width (reads the prior), MSE separates policies
        # by how well each shrinks -- empirical over-pools when tau_true is large, oracle
        # is best, canonical should sit between. Reported with its MC SE.
        "mse": float(np.mean(sq_errs)) if sq_errs else float("nan"),
        "mse_se": (float(np.std(sq_errs, ddof=1) / np.sqrt(len(sq_errs)))
                   if len(sq_errs) > 1 else float("nan")),
        # subgroup-level risk -- the DISCRIMINATING metric (mu-MSE is prior-insensitive)
        "subgroup_risk": float(np.mean(sub_risks)) if sub_risks else float("nan"),
        "subgroup_risk_se": (float(np.std(sub_risks, ddof=1) / np.sqrt(len(sub_risks)))
                             if len(sub_risks) > 1 else float("nan")),
        "n_flagged": n_flagged, "n_used": n_used, "n_escalated": int(n_escalated),
    }


def make_scenario_spec(g, b_size, s, m, *, level: str, mu: float = 0.0, tau: float = 0.0,
                       seed: int = 0) -> dict:
    """A spec with a KNOWN population effect μ = ``mu`` and between-subgroup SD τ =
    ``tau`` at ``level`` (the other level carries no effect). The level's effect table is
    standardized (mean 0, unit SD) then set to mean ``mu`` and SD ``tau``, weight 1.
    Scenarios: global null ``(mu=0, tau=0)``; heterogeneous null ``(mu=0, tau>0)`` — the
    case where borrowing threatens Type I; alternative ``(mu≠0, ...)``."""
    spec = make_joint_hierarchy(g, b_size, s, m, w_group=0.0, w_member=0.0, seed=seed)
    key, w = ("group_effect", "w_group") if level == "group" else ("member_effect", "w_member")
    e = spec[key] - spec[key].mean()
    sd = e.std()
    z = (e / sd) if sd > 1e-9 else e                            # mean 0, unit SD
    spec[key] = mu + tau * z                                    # mean μ, SD τ
    spec[w] = 1.0
    return spec


def make_null_spec(g, b_size, s, m, *, level: str, tau_scale: float, seed: int = 0) -> dict:
    """Backward-compatible wrapper: a null spec (μ = 0) with between-subgroup SD
    ``tau_scale``. See ``make_scenario_spec``."""
    return make_scenario_spec(g, b_size, s, m, level=level, mu=0.0, tau=tau_scale, seed=seed)


def make_partial_null_spec(g, b_size, s, m, *, level: str, sibling_effect: float,
                           null_idx: int = 0, seed: int = 0) -> dict:
    """A **partial null**: subgroup ``null_idx`` is truly null (θ = 0) while every sibling
    has the SAME-sign effect ``sibling_effect`` (same sign maximizes the population mean μ,
    hence the borrowing drag on the null subgroup — the adversarial case for per-subgroup
    Type-I). Pass ``null_subgroup=null_idx`` to ``joint_fidelity`` to measure whether
    borrowing makes the null subgroup's posterior reject θ = 0."""
    spec = make_joint_hierarchy(g, b_size, s, m, w_group=0.0, w_member=0.0, seed=seed)
    key, w = ("group_effect", "w_group") if level == "group" else ("member_effect", "w_member")
    k = g if level == "group" else b_size
    e = np.full(k, float(sibling_effect))
    e[null_idx] = 0.0                                           # the truly-null subgroup
    spec[key] = e
    spec[w] = 1.0
    return spec
