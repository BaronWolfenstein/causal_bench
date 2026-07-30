"""MMRM under MNAR dropout — a falsification harness (#183 / exp43).

Sibling of exp42: take a conventional method whose assumption this benchmark's DGP
already violates, and *demonstrate* the failure against something that survives, rather
than asserting the estimand choice.

MMRM is the regulatory-standard analysis for continuous longitudinal endpoints and is
valid under **MAR** — it uses every observed visit through the likelihood rather than
imputing. `LatentConfounderCensoringConfig` already encodes **MNAR** dropout via a latent
U, and it is the ENCIRCLE-calibrated mechanism, so MMRM is biased under exactly the
mechanism we simulate.

Mechanism. Dropout here is the textbook MNAR: the probability of dropping at visit t
depends on ``Y_it`` *itself* — the value you never get to record, precisely because you
dropped out. Since ``Y_it`` differs by arm (there is a treatment effect), the selection
is **differential**, which is what biases the estimated effect. Dropout that depended
only on U would shift both arms equally and leave the difference alone; that is why the
MNAR channel is specified on the outcome rather than on U.

Arms, mirroring exp42's control-plus-oracle structure so the result cannot be oversold:

- **MAR control** (``gamma_mnar = 0``, dropout on the *previous observed* outcome) —
  MMRM must be unbiased here. Proves any later bias is the mechanism, not the fit.
- **MNAR sweep** — expect monotone bias in ``gamma_mnar``.
- **IPCW-oracle** — weights built from the TRUE dropout model (which sees ``Y_it``).
  Recovering the effect proves the bias is exactly the unobserved-dropout channel.
- **IPCW-observed** — weights from observed history only. Also biased: *nothing* recovers
  MNAR from observables. This arm is what keeps the demonstration honest; the claim is
  emphatically NOT "IPCW fixes MNAR".
"""
from __future__ import annotations

import numpy as np

from causal_bench.estimators.mmrm import fit_mmrm, mmrm_design, treatment_effect_at


def _expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def simulate_longitudinal_dropout(n: int = 500, T: int = 3, *, effect=(0.0, 0.4, 0.9),
                                  means=(1.0, 1.2, 1.5), sd_u: float = 0.8,
                                  sd_e: float = 1.0, gamma0: float = -1.6,
                                  gamma_obs: float = 0.3, gamma_mnar: float = 0.0,
                                  seed: int = 0) -> dict:
    """Randomised two-arm longitudinal outcome with monotone dropout.

    ``Y_it = means[t] + effect[t]*A_i + U_i + e_it`` with U a subject-level latent shift.
    At each visit t >= 1 a still-enrolled subject drops (all later visits missing) with

        logit P(drop at t) = gamma0 + gamma_obs * Yc_{i,t-1} + gamma_mnar * Yc_{i,t}

    where ``Yc`` is centred. ``gamma_obs`` is a MAR channel (previous *observed* outcome);
    ``gamma_mnar`` is the MNAR channel (the current outcome, never recorded if you drop).
    Returns long-format observed rows plus the complete matrix and the true retention
    probabilities, so oracle arms can be constructed."""
    rng = np.random.default_rng(seed)
    effect = np.asarray(effect, float)
    means = np.asarray(means, float)
    A = rng.integers(0, 2, n)
    U = sd_u * rng.normal(size=n)
    E = sd_e * rng.normal(size=(n, T))
    Y = means[None, :] + A[:, None] * effect[None, :] + U[:, None] + E

    grand = float(Y.mean())
    last = np.full(n, T - 1)                       # last observed visit
    p_stay = np.ones((n, T))                       # P(observed at t | true model)
    for t in range(1, T):
        lin = gamma0 + gamma_obs * (Y[:, t - 1] - grand) + gamma_mnar * (Y[:, t] - grand)
        p_drop = _expit(lin)
        p_stay[:, t] = p_stay[:, t - 1] * (1.0 - p_drop)
        drops = (rng.random(n) < p_drop) & (last >= t)
        last = np.where(drops & (last >= t), t - 1, last)

    subj, vis, yy, aa = [], [], [], []
    for i in range(n):
        for t in range(last[i] + 1):
            subj.append(i); vis.append(t); yy.append(Y[i, t]); aa.append(A[i])
    return {"y": np.array(yy), "subject": np.array(subj), "visit": np.array(vis, int),
            "A_long": np.array(aa), "A": A, "U": U, "Y_complete": Y, "last": last,
            "p_stay": p_stay, "T": T, "effect": effect, "n": n}


def mmrm_effect(d: dict, t: int | None = None) -> tuple:
    """MMRM (REML, unstructured) treatment effect at visit ``t`` (default: final)."""
    T = d["T"]
    t = T - 1 if t is None else t
    X = mmrm_design(d["A_long"], d["visit"], T)
    fit = fit_mmrm(d["y"], d["subject"], d["visit"], X, n_visits=T)
    return treatment_effect_at(fit, t, T)


def _weighted_diff(Y, A, w):
    """Weighted mean difference — the same marginal estimand the MMRM visit-t effect targets."""
    w1, w0 = w[A == 1], w[A == 0]
    if w1.sum() <= 0 or w0.sum() <= 0:
        return float("nan")
    return float(np.average(Y[A == 1], weights=w1) - np.average(Y[A == 0], weights=w0))


def ipcw_effect(d: dict, *, oracle: bool, t: int | None = None) -> float:
    """IPCW mean difference among subjects still observed at visit ``t``.

    ``oracle=True`` uses the TRUE retention probability (which depends on the unrecorded
    ``Y_it``) — available only in simulation, and the arm that proves the bias is the
    unobserved-dropout channel. ``oracle=False`` fits the retention model on observed
    history only, which is what an analyst can actually do — and which does NOT fix MNAR."""
    T = d["T"]
    t = T - 1 if t is None else t
    obs = d["last"] >= t
    Y, A = d["Y_complete"][obs, t], d["A"][obs]
    if oracle:
        w = 1.0 / np.clip(d["p_stay"][obs, t], 1e-3, None)
    else:
        # analyst's model: logistic retention on baseline outcome and arm (observed only)
        y0 = d["Y_complete"][:, 0]
        Xd = np.column_stack([np.ones(d["n"]), y0 - y0.mean(), d["A"]])
        r = (d["last"] >= t).astype(float)
        beta = np.zeros(3)
        # NOTE: elementwise-then-sum rather than `Xd @ beta` — the macOS Accelerate
        # BLAS matmul bug (#113 class) yields spurious invalid-value warnings and can
        # return corrupt results here; this is the workaround already used elsewhere.
        lin = lambda M, b: (M * b[None, :]).sum(axis=1)
        for _ in range(60):                        # Newton IRLS
            p = _expit(lin(Xd, beta))
            W = np.clip(p * (1 - p), 1e-6, None)
            beta = beta + np.linalg.solve(Xd.T @ (Xd * W[:, None]) + 1e-8 * np.eye(3),
                                          Xd.T @ (r - p))
        w = 1.0 / np.clip(_expit(lin(Xd, beta))[obs], 1e-3, None)
    return _weighted_diff(Y, A, w)


def complete_data_effect(d: dict, t: int | None = None) -> float:
    """Benchmark: the effect from the COMPLETE outcome matrix (no dropout at all)."""
    t = (d["T"] - 1) if t is None else t
    Y, A = d["Y_complete"][:, t], d["A"]
    return float(Y[A == 1].mean() - Y[A == 0].mean())


def mnar_dropout_report(*, gamma_mnar: float, gamma_obs: float = 0.3, n: int = 500,
                        T: int = 3, n_reps: int = 10, seed: int = 0) -> dict:
    """Run replicates at one MNAR strength and report each arm's bias vs the known truth."""
    mm, ipo, ipn, cd, retained = [], [], [], [], []
    truth = None
    for r in range(n_reps):
        d = simulate_longitudinal_dropout(n=n, T=T, gamma_obs=gamma_obs,
                                          gamma_mnar=gamma_mnar, seed=seed + r)
        truth = float(d["effect"][T - 1])
        mm.append(mmrm_effect(d)[0])
        ipo.append(ipcw_effect(d, oracle=True))
        ipn.append(ipcw_effect(d, oracle=False))
        cd.append(complete_data_effect(d))
        retained.append(float(np.mean(d["last"] >= T - 1)))
    f = lambda v: float(np.nanmean(v))
    mm, ipo, ipn, cd = map(np.asarray, (mm, ipo, ipn, cd))

    def excess(v):
        """Bias RELATIVE to the complete-data benchmark, paired within replicate.
        The complete outcome matrix does not depend on gamma_mnar (dropout only hides
        values), so on shared seeds the finite-sample error is IDENTICAL across arms
        and cancels in the pairing. What survives is the dropout-induced bias alone --
        and pairing also gives an honest SE, so a small excess can be told from noise."""
        d = v - cd
        se = float(np.nanstd(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        return float(np.nanmean(d)), se

    mm_ex, mm_se = excess(mm)
    ipo_ex, ipo_se = excess(ipo)
    ipn_ex, ipn_se = excess(ipn)
    return {"gamma_mnar": gamma_mnar, "truth": truth,
            "mmrm": f(mm), "mmrm_bias": f(mm) - truth,
            "mmrm_excess": mm_ex, "mmrm_excess_se": mm_se,
            "ipcw_oracle": f(ipo), "ipcw_oracle_bias": f(ipo) - truth,
            "ipcw_oracle_excess": ipo_ex, "ipcw_oracle_excess_se": ipo_se,
            "ipcw_observed": f(ipn), "ipcw_observed_bias": f(ipn) - truth,
            "ipcw_observed_excess": ipn_ex, "ipcw_observed_excess_se": ipn_se,
            "complete_data": f(cd), "complete_data_bias": f(cd) - truth,
            "retained_final": f(retained), "n_reps": n_reps}


def coverage_report(*, n: int = 40, gamma_mnar: float = 0.0, gamma_obs: float = 0.3,
                    T: int = 3, n_reps: int = 300, seed: int = 0, alpha: float = 0.05,
                    effect=None, means=None):
    """Coverage of the final-visit treatment effect: naive df vs Kenward-Roger.

    With an unstructured Sigma the naive variance ``(X'V^-1X)^-1`` ignores that Sigma is
    ESTIMATED, so it is biased down and intervals under-cover — badly at the small n where
    MMRM is actually used. Kenward-Roger inflates the variance and supplies a
    Satterthwaite-type df for it.

    Reporting coverage with naive df would measure OUR shortcut rather than MMRM's
    behaviour, which is why the coverage arm and submission-faithful inference are the
    same task, not alternatives. At ``gamma_mnar=0`` (MAR) KR should reach nominal; under
    MNAR NEITHER can, because the point estimate is biased — an interval cannot rescue an
    estimand that is not identified.
    """
    from scipy import stats
    from causal_bench.estimators.mmrm import fit_mmrm_kr
    from causal_bench.validation.joint_fidelity import wilson_ci

    # KR only bites once q = T(T+1)/2 grows relative to n, so this arm must be runnable
    # at T > 3. simulate_longitudinal_dropout's defaults are length-3 tuples; synthesise
    # a matching profile for any T. (mnar_dropout_report is untouched, so exp43's main
    # result keeps its original (0, 0.4, 0.9) profile exactly.)
    eff = np.linspace(0.0, 0.9, T) if effect is None else np.asarray(effect, float)
    mus = np.linspace(1.0, 1.5, T) if means is None else np.asarray(means, float)

    l = np.zeros(2 * T)
    l[T + (T - 1)] = 1.0                     # treatment effect at the final visit
    z = stats.norm.ppf(1 - alpha / 2)
    cov_naive, cov_kr, widths_n, widths_k, dfs, infl = [], [], [], [], [], []
    n_degen = 0
    for r in range(n_reps):
        d = simulate_longitudinal_dropout(n=n, T=T, gamma_obs=gamma_obs, effect=eff,
                                          means=mus,
                                          gamma_mnar=gamma_mnar, seed=seed + r)
        truth = float(d["effect"][T - 1])
        try:
            f = fit_mmrm_kr(d["y"], d["subject"], d["visit"],
                            mmrm_design(d["A_long"], d["visit"], T),
                            contrast=l, n_visits=T)
        except Exception:
            continue
        est = f["estimate"]
        lo_n, hi_n = est - z * f["se_naive"], est + z * f["se_naive"]
        tq = stats.t.ppf(1 - alpha / 2, f["df"])
        lo_k, hi_k = est - tq * f["se_kr"], est + tq * f["se_kr"]
        cov_naive.append(lo_n <= truth <= hi_n)
        cov_kr.append(lo_k <= truth <= hi_k)
        widths_n.append(hi_n - lo_n)
        widths_k.append(hi_k - lo_k)
        dfs.append(f["df"])
        infl.append(f["var_kr"] / max(f["var_naive"], 1e-12))
        n_degen += int(f.get("kr_degenerate", False))
    m = len(cov_naive)
    # Wilson, not the normal SE: coverage lives near 1 exactly where the normal SE
    # collapses to zero, which would report a spuriously decisive interval.
    ci = lambda v: wilson_ci(int(sum(v)), m) if m else (float("nan"), float("nan"))
    return {"n": n, "T": T, "q": T * (T + 1) // 2, "gamma_mnar": gamma_mnar, "n_fits": m,
            "coverage_naive": float(np.mean(cov_naive)), "coverage_naive_ci": ci(cov_naive),
            "coverage_kr": float(np.mean(cov_kr)), "coverage_kr_ci": ci(cov_kr),
            "width_naive": float(np.mean(widths_n)), "width_kr": float(np.mean(widths_k)),
            # median over FINITE df only: one non-finite df poisons a mean, and the
            # ill-conditioned fits are counted separately in n_kr_degenerate anyway.
            "median_df": float(np.median([d for d in dfs if np.isfinite(d)]))
                          if any(np.isfinite(d) for d in dfs) else float("nan"),
            "mean_var_inflation": float(np.mean(infl)),
            "n_kr_degenerate": n_degen}
