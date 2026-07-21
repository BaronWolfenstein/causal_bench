"""Built-in selection bias of the hazard ratio — a falsification harness.

The hazard at time t is defined only among individuals still event-free at t, so the HR
is intrinsically conditioned on prior survival. Survival is a **collider** between
treatment and any unmeasured frailty U (``A -> S(t) <- U -> event``), so conditioning on
it opens a non-causal path and biases the HR *even under randomization and with no
baseline confounding*. This is the "depletion of susceptibles" effect: the arm with the
lower hazard accumulates a frailer surviving population, and the contrast among survivors
drifts toward the null.

This module makes that concrete and self-validating. With a Gamma(k, k) frailty (mean 1,
variance 1/k) and conditional hazard ``lambda(t | A, U) = rate * U * exp(log_hr * A)``,
the marginal survival is closed form::

    S_a(t) = (1 + c_a * t / k) ** (-k),        c_a = rate * exp(log_hr * a)

so the MARGINAL hazard ratio is also closed form::

    HR(t) = (c_1 / c_0) * (1 + c_0 t / k) / (1 + c_1 t / k)

which equals the conditional HR at t = 0 and provably attenuates toward 1 as t grows.
The conditional HR is constant by construction, so every bit of that drift is the
built-in selection bias, not a real change in the treatment effect.

Cumulative-risk estimands do not condition on survival and are therefore immune: the
risk difference at a horizon and the RMST difference remain unbiased for their marginal
truths on the *same* replicates. That contrast is the point of the harness — it is the
evidence behind preferring cumulative-incidence/RMST estimands (as ENCIRCLE's primary
KM 1-year rate and the cloglog borrowing scale already do) over a hazard ratio.

Estimand note: even setting selection bias aside, the HR is **non-collapsible**, so a
hazard-scale contrast is not directly comparable to a risk-difference or RMST contrast
in the same table regardless of bias.
"""
from __future__ import annotations

import numpy as np


# ── analytic truth (Gamma(k, k) frailty) ──────────────────────────────────────
def _rate(a: int, *, log_hr: float, baseline_rate: float) -> float:
    """Conditional hazard multiplier for arm ``a`` (the ``c_a`` above)."""
    return baseline_rate * float(np.exp(log_hr * a))


def marginal_survival(t, a: int, *, log_hr: float, frailty_shape: float,
                      baseline_rate: float = 1.0):
    """Marginal (population) survival ``S_a(t)`` after integrating out Gamma(k,k) frailty."""
    c = _rate(a, log_hr=log_hr, baseline_rate=baseline_rate)
    k = frailty_shape
    return (1.0 + c * np.asarray(t, float) / k) ** (-k)


def marginal_hazard_ratio(t, *, log_hr: float, frailty_shape: float,
                          baseline_rate: float = 1.0):
    """The MARGINAL hazard ratio at time ``t``. Equals the conditional HR at t=0 and
    attenuates monotonically toward 1 — the built-in selection bias, in closed form."""
    c0 = _rate(0, log_hr=log_hr, baseline_rate=baseline_rate)
    c1 = _rate(1, log_hr=log_hr, baseline_rate=baseline_rate)
    k = frailty_shape
    t = np.asarray(t, float)
    return (c1 / c0) * (1.0 + c0 * t / k) / (1.0 + c1 * t / k)


def _rmst(horizon: float, a: int, *, log_hr, frailty_shape, baseline_rate=1.0,
          n_grid: int = 20001) -> float:
    """RMST_a(horizon) = ∫_0^horizon S_a(t) dt, by fine-grid trapezoid on the analytic S."""
    ts = np.linspace(0.0, horizon, n_grid)
    s = marginal_survival(ts, a, log_hr=log_hr, frailty_shape=frailty_shape,
                          baseline_rate=baseline_rate)
    return float(np.trapezoid(s, ts)) if hasattr(np, "trapezoid") else float(np.trapz(s, ts))


def true_risk_difference(horizon: float, *, log_hr, frailty_shape,
                         baseline_rate: float = 1.0) -> float:
    """Marginal causal risk difference F_1(τ) − F_0(τ) — the estimand a KM analysis targets."""
    s1 = float(marginal_survival(horizon, 1, log_hr=log_hr, frailty_shape=frailty_shape,
                                 baseline_rate=baseline_rate))
    s0 = float(marginal_survival(horizon, 0, log_hr=log_hr, frailty_shape=frailty_shape,
                                 baseline_rate=baseline_rate))
    return (1.0 - s1) - (1.0 - s0)


def true_rmst_difference(horizon: float, *, log_hr, frailty_shape,
                         baseline_rate: float = 1.0) -> float:
    """Marginal causal RMST difference RMST_1(τ) − RMST_0(τ)."""
    kw = dict(log_hr=log_hr, frailty_shape=frailty_shape, baseline_rate=baseline_rate)
    return _rmst(horizon, 1, **kw) - _rmst(horizon, 0, **kw)


# ── DGP ───────────────────────────────────────────────────────────────────────
def simulate_frailty_survival(n: int, *, log_hr: float = -0.7, frailty_shape: float = 1.0,
                              baseline_rate: float = 1.0, horizon: float = 2.0,
                              seed: int = 0) -> dict:
    """Randomized two-arm survival with unobserved Gamma(k,k) frailty.

    ``A ~ Bernoulli(0.5)`` independent of ``U`` — there is NO confounding. The conditional
    hazard ``rate * U * exp(log_hr * A)`` gives a CONSTANT conditional hazard ratio
    ``exp(log_hr)``. Administrative censoring at ``horizon``. Returns ``A``, the latent
    ``U`` (never given to an estimator), the uncensored ``T_event``, and the observed
    ``(T_obs, Delta)``."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, n)
    U = rng.gamma(shape=frailty_shape, scale=1.0 / frailty_shape, size=n)   # mean 1
    lam = baseline_rate * U * np.exp(log_hr * A)
    T_event = rng.exponential(1.0 / lam)
    T_obs = np.minimum(T_event, horizon)
    Delta = (T_event <= horizon).astype(int)
    return {"A": A, "U": U, "T_event": T_event, "T_obs": T_obs, "Delta": Delta,
            "horizon": horizon, "log_hr": log_hr, "frailty_shape": frailty_shape}


# ── estimators ────────────────────────────────────────────────────────────────
def cox_hazard_ratio(A, T_obs, Delta) -> float:
    """Cox partial-likelihood HR for ``A`` — the estimand that inherits the built-in bias."""
    import pandas as pd
    from lifelines import CoxPHFitter
    df = pd.DataFrame({"A": np.asarray(A, float), "T": np.asarray(T_obs, float),
                       "E": np.asarray(Delta, int)})
    cph = CoxPHFitter().fit(df, duration_col="T", event_col="E")
    return float(np.exp(cph.params_["A"]))


def _km(T_obs, Delta, grid):
    """Kaplan-Meier survival evaluated on ``grid`` (step function, right-continuous)."""
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter().fit(np.asarray(T_obs, float), np.asarray(Delta, int))
    return np.asarray(kmf.predict(grid, interpolate=False)).ravel()


def km_risk_difference(A, T_obs, Delta, *, horizon: float) -> float:
    """KM cumulative-incidence difference at ``horizon`` — does not condition on survival."""
    A = np.asarray(A)
    s1 = _km(np.asarray(T_obs)[A == 1], np.asarray(Delta)[A == 1], [horizon])[0]
    s0 = _km(np.asarray(T_obs)[A == 0], np.asarray(Delta)[A == 0], [horizon])[0]
    return float((1.0 - s1) - (1.0 - s0))


def km_rmst_difference(A, T_obs, Delta, *, horizon: float, n_grid: int = 2001) -> float:
    """KM RMST difference to ``horizon`` — also immune to the built-in selection bias."""
    A = np.asarray(A)
    grid = np.linspace(0.0, horizon, n_grid)
    s1 = _km(np.asarray(T_obs)[A == 1], np.asarray(Delta)[A == 1], grid)
    s0 = _km(np.asarray(T_obs)[A == 0], np.asarray(Delta)[A == 0], grid)
    area = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(area(s1, grid) - area(s0, grid))


def windowed_hazard_ratios(A, T_obs, Delta, *, t_split: float) -> tuple[float, float]:
    """Cox HR on an EARLY window (censored at ``t_split``) and a LATE window (restricted to
    survivors past ``t_split``, clock restarted). The gap between them is the drift the
    built-in selection bias produces — the conditional HR is constant, so a real effect
    change cannot explain it."""
    A, T_obs, Delta = np.asarray(A), np.asarray(T_obs, float), np.asarray(Delta, int)
    early = cox_hazard_ratio(A, np.minimum(T_obs, t_split),
                             np.where(T_obs <= t_split, Delta, 0))
    surv = T_obs > t_split                                   # the depleted risk set
    late = cox_hazard_ratio(A[surv], T_obs[surv] - t_split, Delta[surv])
    return early, late


def hazard_selection_report(*, log_hr: float = -0.7, frailty_shape: float = 1.0,
                            n: int = 20000, horizon: float = 2.0, eval_at: float = 1.0,
                            t_split: float = 1.0, n_reps: int = 20,
                            seed: int = 0) -> dict:
    """Run replicates and summarise the contrast: the Cox HR's bias vs its own conditional
    truth, against the (un)biasedness of the cumulative-risk estimands on the same data."""
    kw = dict(log_hr=log_hr, frailty_shape=frailty_shape)
    hr_cond = float(np.exp(log_hr))
    rd_true = true_risk_difference(eval_at, **kw)
    rmst_true = true_rmst_difference(eval_at, **kw)
    hrs, earlys, lates, rds, rmsts = [], [], [], [], []
    for r in range(n_reps):
        d = simulate_frailty_survival(n, horizon=horizon, seed=seed + r, **kw)
        hrs.append(cox_hazard_ratio(d["A"], d["T_obs"], d["Delta"]))
        e, l = windowed_hazard_ratios(d["A"], d["T_obs"], d["Delta"], t_split=t_split)
        earlys.append(e); lates.append(l)
        rds.append(km_risk_difference(d["A"], d["T_obs"], d["Delta"], horizon=eval_at))
        rmsts.append(km_rmst_difference(d["A"], d["T_obs"], d["Delta"], horizon=eval_at))
    return {
        "hr_conditional_truth": hr_cond,
        "hr_marginal_truth_at_eval": float(marginal_hazard_ratio(eval_at, **kw)),
        "cox_hr": float(np.mean(hrs)), "cox_hr_bias_vs_conditional": float(np.mean(hrs)) - hr_cond,
        "hr_early": float(np.mean(earlys)), "hr_late": float(np.mean(lates)),
        "hr_drift": float(np.mean(lates)) - float(np.mean(earlys)),
        "risk_difference": float(np.mean(rds)), "risk_difference_truth": rd_true,
        "risk_difference_bias": float(np.mean(rds)) - rd_true,
        "rmst_difference": float(np.mean(rmsts)), "rmst_difference_truth": rmst_true,
        "rmst_difference_bias": float(np.mean(rmsts)) - rmst_true,
        "n_reps": n_reps,
    }
