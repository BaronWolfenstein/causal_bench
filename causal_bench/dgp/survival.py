"""Survival DGP for causal_bench.

Implements an AFT model with Weibull-distributed survival times (Gumbel noise),
informative/non-informative censoring, unmeasured confounding, and a negative
control outcome.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from causal_bench.dgp.config import (
    DGPConfig,
    CensoringConfig,
    IndependentCensoringConfig,
    CovariateDependentCensoringConfig,
    InformativeCensoringConfig,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


@lru_cache(maxsize=256)
def _calibrate_censoring_scale(
    censoring_rate: float,
    horizon: float,
    censoring: CensoringConfig,
) -> float:
    """Scale factor so achieved censoring_rate matches target under given mechanism."""
    if censoring_rate <= 0:
        return 1e10
    rng = np.random.default_rng(0)
    n = 5000
    U  = rng.standard_normal(n)
    W1 = rng.standard_normal(n)
    W3 = rng.standard_normal(n)
    A  = rng.binomial(1, 0.5, n).astype(float)
    log_T  = 0.0 + 0.4 * W1 + 0.3 * U + rng.gumbel(0, 1, n)
    T_true = np.exp(log_T)
    gumbel_c = rng.gumbel(0, 1, n)

    if isinstance(censoring, IndependentCensoringConfig):
        log_C_base = 1.5 + gumbel_c
    elif isinstance(censoring, InformativeCensoringConfig):
        log_C_base = 1.5 + censoring.beta_T * T_true + gumbel_c
    else:  # CovariateDependentCensoringConfig
        log_C_base = (1.5 - 0.2 * W1 + 0.1 * W3 - 0.1 * A
                      + 0.4 * U * censoring.informativeness
                      + gumbel_c)
        mnar_weight = max(0.0, censoring.informativeness - 0.5) * 2.0
        if mnar_weight > 0:
            log_C_base -= mnar_weight * (T_true < np.median(T_true)).astype(float)

    C_base = np.exp(np.clip(log_C_base, -700, 700))  # avoid 0/inf overflow at extreme beta_T * T_true
    lo, hi = 0.01, 100.0
    for _ in range(40):
        mid = (lo + hi) / 2
        C = C_base * mid
        censor_rate = np.mean((C < T_true) & (C < horizon))
        if censor_rate > censoring_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _stratified_block_randomize(
    strata_arrays: list,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permuted-block randomization within covariate-defined strata.

    Each array in ``strata_arrays`` is binarised at its median to form a bit
    of the stratum integer (so k arrays → 2^k strata). Within each stratum
    patients are shuffled, then assigned in alternating blocks of size
    ``block_size`` (half treated, half control). A partial final block is
    filled with Bernoulli(0.5) draws.
    """
    n = len(strata_arrays[0])
    strata_id = np.zeros(n, dtype=int)
    for bit, arr in enumerate(strata_arrays):
        strata_id |= (arr >= np.median(arr)).astype(int) << bit

    A = np.empty(n, dtype=float)
    half = block_size // 2
    for s in np.unique(strata_id):
        idx = np.where(strata_id == s)[0]
        perm = rng.permutation(len(idx))
        shuffled = idx[perm]
        n_s = len(shuffled)
        assignments = np.empty(n_s, dtype=float)
        pos = 0
        while pos + block_size <= n_s:
            block = np.concatenate([np.ones(half), np.zeros(block_size - half)])
            rng.shuffle(block)
            assignments[pos:pos + block_size] = block
            pos += block_size
        if pos < n_s:
            assignments[pos:] = rng.binomial(1, 0.5, n_s - pos).astype(float)
        A[shuffled] = assignments
    return A


def generate_data(
    config: DGPConfig,
    rng: np.random.Generator | None = None,
    U: np.ndarray | None = None,
    W1: np.ndarray | None = None,
    W2: np.ndarray | None = None,
    W3: np.ndarray | None = None,
    W4: np.ndarray | None = None,
    return_latents: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Generate one simulated clinical trial dataset.

    Parameters
    ----------
    config:
        DGP configuration dataclass.
    rng:
        Optional numpy random Generator.  If None a fresh generator is seeded
        from ``config.seed``.
    U, W1, W2, W3, W4:
        Optional pre-specified latent/covariate arrays (each length config.n).
        When omitted (the default for every existing caller), each is drawn
        fresh from `rng` exactly as before — this parameter exists solely so
        causal_bench.dgp.augmentation can splice in parent-correlated latents
        for provenance-linked synthetic units without duplicating the rest of
        this function's structural equations.
    return_latents:
        If True, also return the U array actually used (drawn or supplied) so
        a caller can correlate a child unit's latents with this unit's.

    Returns
    -------
    pd.DataFrame with columns: T_obs, Delta, A, W1, W2, W3, W4, compliance,
    enrollment_time, Y_neg. (U itself is never included as a column.)
    If return_latents=True, returns (df, U) instead.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    n = config.n

    # --- Latent + observed covariates ---
    if U is None:
        U = rng.standard_normal(n)
    if W1 is None:
        W1 = rng.standard_normal(n)
    if W2 is None:
        W2 = rng.binomial(1, 0.5, n).astype(float)
    if W3 is None:
        W3 = rng.standard_normal(n)
    if W4 is None:
        W4 = rng.binomial(1, 0.3, n).astype(float)
    enrollment_time = rng.uniform(0, config.enrollment_period, n)

    # --- Treatment assignment ---
    _col_map = {"W1": W1, "W2": W2, "W3": W3, "W4": W4}
    if config.strata_cols:
        A = _stratified_block_randomize(
            [_col_map[c] for c in config.strata_cols],
            block_size=config.strata_block_size,
            rng=rng,
        )
    else:
        p = np.clip(config.treatment_prevalence, 1e-6, 1 - 1e-6)
        logit_A = (
            np.log(p / (1 - p))
            + 0.3 * W1
            + 0.2 * W2
            - 0.2 * W3
            + 0.1 * W4
            + 0.5 * U * config.unmeasured_confounding_strength
            + 0.8 * W1 * W3 * config.positivity_severity
        )
        A = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    # --- Survival time (AFT model with Gumbel noise) ---
    gumbel_noise = rng.gumbel(0, 1, n)
    # Intercept 0.0 (not 1.0) so that median T ≈ 1.0 and ~25-40% events occur within horizon=1.0
    log_T = (
        0.0
        + 0.4 * W1
        - 0.3 * W2
        + 0.2 * W3
        - 0.2 * W4
        + 0.3 * U
        + config.true_tau * A
        + config.enrollment_drift * enrollment_time
        + config.outcome_nonlinearity * (W1 ** 2 - 1)
        + config.effect_heterogeneity * A * W1
        + gumbel_noise
    )
    T_true = np.exp(log_T)

    # --- Compliance covariate (correlated with U, observed) ---
    rho = np.sqrt(config.compliance_censoring_r2)
    compliance_raw = rho * U + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n)
    compliance = _sigmoid(compliance_raw)

    # --- Censoring ---
    scale_factor = _calibrate_censoring_scale(
        config.censoring_rate, config.horizon, config.censoring
    )

    gumbel_c = rng.gumbel(0, 1, n)
    if isinstance(config.censoring, IndependentCensoringConfig):
        # Pure random dropout: C doesn't depend on covariates, treatment, or T_true.
        log_C_base = 1.5 + gumbel_c
    elif isinstance(config.censoring, InformativeCensoringConfig):
        # MNAR: censoring time directly depends on the (unobservable) event time.
        # IPCW conditional only on W, A cannot correct this — it requires T_true.
        log_C_base = 1.5 + config.censoring.beta_T * T_true + gumbel_c
    else:
        # CovariateDependentCensoringConfig — MAR conditional on W, A; optional MNAR-via-U component
        log_C_base = (
            1.5
            - 0.2 * W1
            + 0.1 * W3
            - 0.1 * A
            + 0.4 * U * config.censoring.informativeness
            + gumbel_c
        )
        # MNAR component: early events are more likely to be censored
        mnar_weight = max(0.0, config.censoring.informativeness - 0.5) * 2
        if mnar_weight > 0:
            median_T = np.median(T_true)
            log_C_base -= mnar_weight * (T_true < median_T).astype(float)

    C = np.exp(np.clip(log_C_base, -700, 700)) * scale_factor  # avoid 0/inf overflow at extreme beta_T * T_true

    # --- L1: post-treatment time-varying confounder ---
    # Observed only if the patient is still in the study at t_L1: alive (T_true > t_L1)
    # AND not yet censored (C > t_L1).
    L1_raw = (0.5 * A + 0.4 * W3 + 0.3 * U * config.collider_strength
              + rng.standard_normal(n) * config.sigma_L)
    alive_at_L1 = (T_true > config.t_L1) & (C > config.t_L1)
    L1_obs = np.where(alive_at_L1, L1_raw, np.nan)

    # --- Crossover (treatment switching) ---
    # RPSFTM: control patients who switch at t_cross have residual time scaled by exp(true_tau)
    t_crossover_col = np.full(n, np.nan)
    if config.crossover_rate > 0.0:
        u_who = rng.uniform(0, 1, n)
        crosses = (A == 0) & (u_who < config.crossover_rate)
        # Crossover time: t_cross = T_true * u^(1 + informativeness)
        # informativeness=0 → uniform fraction; informativeness>0 → concentrated near 0
        # (smaller T_true patients also get smaller absolute t_cross → sicker cross sooner)
        u_when = rng.uniform(0, 1, n)
        t_cross = T_true * (u_when ** (1.0 + config.crossover_informativeness))
        t_cross = np.clip(t_cross, 0.0, T_true - 1e-9)
        # RPSFTM: T(1) = T(0)*exp(true_tau), so residual time after switching = (T_true - t_cross)*exp(true_tau)
        T_crossover = t_cross + (T_true - t_cross) * np.exp(config.true_tau)
        T_true = np.where(crosses, T_crossover, T_true)
        t_crossover_col = np.where(crosses, t_cross, np.nan)

    # --- Competing risks (optional) ---
    # Two causes race against each other and against censoring.
    # cause1 = primary event (treatment effect = true_tau)
    # cause2 = competing event (e.g. death from other cause, no treatment on cause2)
    if config.competing_risks:
        # Cause-2 AFT: different baseline hazard, no treatment effect on cause 2 by default.
        # hfh_death_escalation couples cause-1 and cause-2 propensity via shared frailty:
        # patients with low T1 baseline (high HFH risk) are also at higher death risk.
        # The coupling term uses the cause-1 log-time baseline (no noise, no treatment)
        # so treatment effects on the two causes remain orthogonally specified.
        cause1_log_T_baseline = 0.4 * W1 - 0.3 * W2 + 0.2 * W3 - 0.2 * W4
        log_T2 = (
            0.3                                          # higher baseline → later cause-2
            + 0.2 * W1
            - 0.1 * W3
            + 0.2 * U
            + config.cause2_treatment_effect * A
            + config.hfh_death_escalation * cause1_log_T_baseline
            + rng.gumbel(0, 1, n)
        )
        T2_true = np.exp(log_T2)
        # First event wins
        T_first = np.minimum(T_true, T2_true)
        T_obs = np.minimum(T_first, np.minimum(C, config.horizon))
        cause1_event = (T_true <= T2_true) & (T_true <= C) & (T_true <= config.horizon)
        cause2_event = (T2_true < T_true) & (T2_true <= C) & (T2_true <= config.horizon)
        # event_type: 0=censored, 1=cause-1, 2=cause-2
        event_type = np.where(cause1_event, 1, np.where(cause2_event, 2, 0)).astype(int)
        Delta = cause1_event.astype(float)
    else:
        T_obs = np.minimum(T_true, np.minimum(C, config.horizon))
        Delta = ((T_true <= C) & (T_true <= config.horizon)).astype(float)
        event_type = Delta.astype(int)

    # --- Negative control outcome (no treatment effect) ---
    Y_neg = 0.5 * W1 - 0.3 * W3 + 0.4 * U + rng.normal(0, 0.5, n)

    df = pd.DataFrame({
        "T_obs": T_obs,
        "Delta": Delta,
        "event_type": event_type,
        "A": A,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
        "compliance": compliance,
        "enrollment_time": enrollment_time,
        "Y_neg": Y_neg,
        "L1": L1_obs,
        "t_crossover": t_crossover_col,
    })
    # Propagate strata info so concrete bridge can pass Strata argument
    if config.strata_cols:
        df.attrs["strata_cols"] = list(config.strata_cols)
    if return_latents:
        return df, U
    return df


def compute_true_effects(config: DGPConfig, n_ref: int = 50_000) -> dict:
    """Estimate true ATE and ATT via a large reference population.

    Uses shared covariates and shared Gumbel noise so that only treatment
    assignment varies between potential-outcome arms.

    Parameters
    ----------
    config:
        DGP configuration.
    n_ref:
        Size of the reference population (default 50 000).

    Returns
    -------
    dict with keys "ATE" and "ATT" (floats).
    """
    rng = np.random.default_rng(config.seed ^ 0xDEADBEEF)

    # Shared covariates
    U = rng.standard_normal(n_ref)
    W1 = rng.standard_normal(n_ref)
    W2 = rng.binomial(1, 0.5, n_ref).astype(float)
    W3 = rng.standard_normal(n_ref)
    W4 = rng.binomial(1, 0.3, n_ref).astype(float)
    enrollment_time = rng.uniform(0, config.enrollment_period, n_ref)

    # Observed treatment (for ATT)
    p = np.clip(config.treatment_prevalence, 1e-6, 1 - 1e-6)
    logit_A = (
        np.log(p / (1 - p))
        + 0.3 * W1
        + 0.2 * W2
        - 0.2 * W3
        + 0.1 * W4
        + 0.5 * U * config.unmeasured_confounding_strength
        + 0.8 * W1 * W3 * config.positivity_severity
    )
    A_obs = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    # Shared Gumbel noise for potential outcomes
    gumbel_noise = rng.gumbel(0, 1, n_ref)

    def _log_T(a_val: float) -> np.ndarray:
        return (
            0.0
            + 0.4 * W1
            - 0.3 * W2
            + 0.2 * W3
            - 0.2 * W4
            + 0.3 * U
            + config.true_tau * a_val
            + config.enrollment_drift * enrollment_time
            + config.outcome_nonlinearity * (W1 ** 2 - 1)
            + config.effect_heterogeneity * a_val * W1
            + gumbel_noise
        )

    T1 = np.exp(_log_T(1.0))
    T0 = np.exp(_log_T(0.0))

    # Binary event indicator within horizon
    Y1 = (T1 <= config.horizon).astype(float)
    Y0 = (T0 <= config.horizon).astype(float)

    diff = Y1 - Y0
    ATE = float(np.mean(diff))
    ATT = float(np.mean(diff[A_obs == 1]))

    return {"ATE": ATE, "ATT": ATT}


def compute_true_rmst(config: DGPConfig, n_ref: int = 50_000) -> dict:
    """Estimate true RMST difference via g-computation on a large reference population.

    RMST(a) = E[min(T_a, horizon)] = integral_0^horizon P(T_a > t) dt.
    Uses the same shared covariates and Gumbel noise as compute_true_effects()
    so the two benchmarks are directly comparable.

    Parameters
    ----------
    config:
        DGP configuration.
    n_ref:
        Size of the reference population (default 50 000).

    Returns
    -------
    dict with keys:
        "ATE"           — RMST(A=1) − RMST(A=0) marginalised over full population
        "ATT"           — RMST difference for the treated subgroup
        "rmst_treated"  — RMST under A=1 (years lived before horizon)
        "rmst_control"  — RMST under A=0
    """
    rng = np.random.default_rng(config.seed ^ 0xDEADBEEF)

    U  = rng.standard_normal(n_ref)
    W1 = rng.standard_normal(n_ref)
    W2 = rng.binomial(1, 0.5, n_ref).astype(float)
    W3 = rng.standard_normal(n_ref)
    W4 = rng.binomial(1, 0.3, n_ref).astype(float)
    enrollment_time = rng.uniform(0, config.enrollment_period, n_ref)

    p = np.clip(config.treatment_prevalence, 1e-6, 1 - 1e-6)
    logit_A = (
        np.log(p / (1 - p))
        + 0.3 * W1 + 0.2 * W2 - 0.2 * W3 + 0.1 * W4
        + 0.5 * U * config.unmeasured_confounding_strength
        + 0.8 * W1 * W3 * config.positivity_severity
    )
    A_obs = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    gumbel_noise = rng.gumbel(0, 1, n_ref)

    def _log_T(a_val: float) -> np.ndarray:
        return (
            0.0
            + 0.4 * W1 - 0.3 * W2 + 0.2 * W3 - 0.2 * W4
            + 0.3 * U
            + config.true_tau * a_val
            + config.enrollment_drift * enrollment_time
            + config.outcome_nonlinearity * (W1 ** 2 - 1)
            + config.effect_heterogeneity * a_val * W1
            + gumbel_noise
        )

    T1 = np.exp(_log_T(1.0))
    T0 = np.exp(_log_T(0.0))

    rmst1 = float(np.mean(np.minimum(T1, config.horizon)))
    rmst0 = float(np.mean(np.minimum(T0, config.horizon)))
    rmst1_att = float(np.mean(np.minimum(T1[A_obs == 1], config.horizon)))
    rmst0_att = float(np.mean(np.minimum(T0[A_obs == 1], config.horizon)))

    return {
        "ATE":           rmst1 - rmst0,
        "ATT":           rmst1_att - rmst0_att,
        "rmst_treated":  rmst1,
        "rmst_control":  rmst0,
    }


def compute_true_win_ratio(config: DGPConfig, n_ref: int = 50_000) -> dict:
    """Estimate true win ratio via U-statistic on potential outcomes.

    Win ratio = P(T1_i > T0_j) / P(T1_i < T0_j) for independent draws i, j
    from the treated and control potential-outcome distributions.  Computed
    exactly in O(n log n) via sorted arrays + searchsorted (no pairwise loop).

    Uses the same shared covariates and Gumbel noise as compute_true_effects()
    so the reference population is consistent across benchmarks.

    Parameters
    ----------
    config : DGPConfig
    n_ref  : size of the reference population (default 50 000).

    Returns
    -------
    dict with keys:
        "ATE"        — win ratio (marginalised over full population); > 1 means
                       treated win more often, < 1 means treated lose more often.
        "ATT"        — win ratio for the treated subgroup vs full control dist.
        "p_win"      — P(T1 > T0), marginalised
        "p_loss"     — P(T1 < T0), marginalised
        "net_benefit" — p_win − p_loss
    """
    rng = np.random.default_rng(config.seed ^ 0xDEADBEEF)

    U  = rng.standard_normal(n_ref)
    W1 = rng.standard_normal(n_ref)
    W2 = rng.binomial(1, 0.5, n_ref).astype(float)
    W3 = rng.standard_normal(n_ref)
    W4 = rng.binomial(1, 0.3, n_ref).astype(float)
    enrollment_time = rng.uniform(0, config.enrollment_period, n_ref)

    p = np.clip(config.treatment_prevalence, 1e-6, 1 - 1e-6)
    logit_A = (
        np.log(p / (1 - p))
        + 0.3 * W1 + 0.2 * W2 - 0.2 * W3 + 0.1 * W4
        + 0.5 * U * config.unmeasured_confounding_strength
        + 0.8 * W1 * W3 * config.positivity_severity
    )
    A_obs = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    gumbel_noise = rng.gumbel(0, 1, n_ref)

    def _log_T(a_val: float) -> np.ndarray:
        return (
            0.0
            + 0.4 * W1 - 0.3 * W2 + 0.2 * W3 - 0.2 * W4
            + 0.3 * U
            + config.true_tau * a_val
            + config.enrollment_drift * enrollment_time
            + config.outcome_nonlinearity * (W1 ** 2 - 1)
            + config.effect_heterogeneity * a_val * W1
            + gumbel_noise
        )

    T1 = np.exp(_log_T(1.0))
    T0 = np.exp(_log_T(0.0))

    # U-statistic via searchsorted: O(n log n), exact for continuous distributions
    T0_sorted = np.sort(T0)
    p_win  = float(np.searchsorted(T0_sorted, T1, side="left").mean())  / n_ref
    p_loss = float((n_ref - np.searchsorted(T0_sorted, T1, side="right")).mean()) / n_ref
    win_ratio = p_win / p_loss if p_loss > 1e-12 else float("inf")

    # ATT: restrict treated arm to observed treated subjects
    T1_att = T1[A_obs == 1]
    p_win_att  = float(np.searchsorted(T0_sorted, T1_att, side="left").mean())  / n_ref
    p_loss_att = float((n_ref - np.searchsorted(T0_sorted, T1_att, side="right")).mean()) / n_ref
    win_ratio_att = p_win_att / p_loss_att if p_loss_att > 1e-12 else float("inf")

    return {
        "ATE":         win_ratio,
        "ATT":         win_ratio_att,
        "p_win":       p_win,
        "p_loss":      p_loss,
        "net_benefit": p_win - p_loss,
    }
