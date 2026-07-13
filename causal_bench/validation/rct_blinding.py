"""RCT-blinding validation for the OC-sim / synthetic comparator (#139).

StandardModel's "second half of AI in biomedicine" names the deepest problem with
any patient-world-model / observational-control (OC) simulation: the counterfactual
(untreated) branch is fundamentally unidentifiable against observational ground truth
("no patient is ever treated and left untreated on the same day"). Their proposed
litmus test — and ours, since for single-arm ENCIRCLE the OC sim *is* the entire
comparator arm — is to **blind the model to a held-out RCT and check whether its
simulated control branch recovers the RCT's (unconfounded) treatment effect and
survival curves**.

This harness is **generator-agnostic and needs no discrete MEDS-token diffusion**: it
validates the OC branch at the level of estimated RMST / survival curves. Its core
(`fit_oc_sim`, `oc_sim_rmst`, `oc_sim_survival`, `rct_blinding_recovery`) takes plain
covariate/treatment/time arrays, so it can be pointed at the canonical DGP, an
embedding-space generator, or a real cohort later.

For a self-contained *demonstration that the guard discriminates*, it ships a
tunable-confounding lognormal-AFT DGP (`make_confounded_cohorts`) — the canonical
`dgp.survival` model deliberately carries only mild confounding, too weak to show the
guard failing a bad comparator. With strong confounding:

- a **naive** OC-sim (crude observational arm difference) is confounded → fails to
  recover the RCT effect → the harness flags it (`recovered=False`);
- an **adjusted** OC-sim (g-computation over measured covariates) recovers it;
- add **unmeasured** confounding and even the adjusted OC-sim fails → flagged — exactly
  the case a synthetic comparator must not silently pass.

Metric-hacking guard: recovery is *always* scored against the held-out RCT, never the
observational data the OC-sim was fit on. The full ``E_eval ≠ E_gen`` encoder
decoupling (#87/#88) attaches when real embeddings enter (box/data-gated); this
synthetic protocol is the precursor. numpy only.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_confounded_cohorts(*, n_obs: int = 6000, n_rct: int = 6000, d: int = 5,
                            tau: float = 0.6, confound: float = 2.5, mu0: float = 0.3,
                            sigma: float = 0.6, horizon: float = 2.0,
                            unmeasured: float = 0.0, seed: int = 0) -> dict:
    """A lognormal-AFT DGP with a tunable **measured**-confounding knob. Covariates
    ``X ~ N(0, I_d)`` give a prognostic index ``g = X·beta`` (``beta`` unit-norm, so
    ``g ~ N(0,1)``). Survival ``log T = mu0 + g + tau·A + unmeasured·U + sigma·eps``
    (``U`` an unmeasured confounder, ``eps ~ N(0,1)``).

    - **Observational** arm: ``A ~ Bernoulli(sigmoid(confound·g + unmeasured·U))`` —
      treatment selected on prognosis (confounding by indication); larger ``confound``
      ⇒ stronger measured confounding ⇒ a naive comparator is more biased.
    - **RCT** arm: ``A ~ Bernoulli(0.5)`` — randomized, treatment ⊥ prognosis.

    Both arms share the same structural outcome equation, so the true effect is
    identical. Returns ``{obs, rct, true_ate, horizon}`` where ``obs``/``rct`` are
    ``{X, A, T}`` dicts and ``true_ate`` is the RMST difference up to ``horizon``
    (large-sample Monte Carlo)."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=d)
    beta /= np.linalg.norm(beta)

    def _arm(n: int, randomized: bool) -> dict:
        X = rng.normal(size=(n, d))
        g = (X * beta).sum(1)                                   # avoid Accelerate gemm
        U = rng.normal(size=n)
        if randomized:
            A = (rng.random(n) < 0.5).astype(float)
        else:
            A = (rng.random(n) < _sigmoid(confound * g + unmeasured * U)).astype(float)
        logT = mu0 + g + tau * A + unmeasured * U + sigma * rng.normal(size=n)
        return {"X": X, "A": A, "T": np.exp(logT)}

    obs, rct = _arm(n_obs, False), _arm(n_rct, True)

    nb = 200_000                                               # true RMST via g-computation
    Xb = rng.normal(size=(nb, d))
    gb = (Xb * beta).sum(1)
    Ub = rng.normal(size=nb)
    t1 = np.exp(mu0 + gb + tau + unmeasured * Ub + sigma * rng.normal(size=nb))
    t0 = np.exp(mu0 + gb + unmeasured * Ub + sigma * rng.normal(size=nb))
    true_ate = float(np.minimum(t1, horizon).mean() - np.minimum(t0, horizon).mean())
    return {"obs": obs, "rct": rct, "true_ate": true_ate, "horizon": horizon, "beta": beta}


def fit_oc_sim(X: np.ndarray, A: np.ndarray, T: np.ndarray, *, adjust: bool) -> dict:
    """Fit the OC-sim control-branch model: an AFT (log-time) least-squares fit on the
    observational cohort. ``adjust=True`` regresses ``log T ~ 1 + X + A`` (g-computation
    adjusts for measured confounders); ``adjust=False`` regresses ``log T ~ 1 + A``
    (naive — confounded when A is selected on X). Returns fitted ``coef``, empirical
    ``resid``, and ``adjust``."""
    y = np.log(np.asarray(T, float))
    A = np.asarray(A, float)
    n = len(y)
    design = [np.ones(n), np.asarray(X, float), A] if adjust else [np.ones(n), A]
    M = np.column_stack(design)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - (M * coef).sum(1)                              # elementwise: avoid gemm
    return {"coef": coef, "resid": resid, "adjust": adjust}


def _linpred(model: dict, X: np.ndarray, a: float) -> np.ndarray:
    """Counterfactual linear predictor ``log T | X, do(A=a)`` for each unit."""
    coef, X = model["coef"], np.asarray(X, float)
    n = len(X)
    if model["adjust"]:
        return coef[0] + (X * coef[1:-1]).sum(1) + coef[-1] * a
    return coef[0] + coef[1] * a + np.zeros(n)                 # keep array-shaped


def oc_sim_rmst(model: dict, X_ref: np.ndarray, a: float, horizon: float, *,
                n_resid: int = 400, seed: int = 0) -> float:
    """g-computation RMST under ``do(A=a)``: marginalize ``min(exp(linpred + residual),
    horizon)`` over reference covariates and the empirical residual distribution."""
    rng = np.random.default_rng(seed)
    lp = _linpred(model, X_ref, a)
    e = model["resid"]
    if len(e) > n_resid:
        e = rng.choice(e, n_resid, replace=False)
    return float(np.minimum(np.exp(lp[:, None] + e[None, :]), horizon).mean())


def oc_sim_survival(model: dict, X_ref: np.ndarray, times: np.ndarray, a: float) -> np.ndarray:
    """g-computation survival curve ``S_a(t)`` under ``do(A=a)``."""
    lp = _linpred(model, X_ref, a)
    T = np.exp(lp[:, None] + model["resid"][None, :]).ravel()
    return np.array([(T > t).mean() for t in np.asarray(times)])


def empirical_rmst(T: np.ndarray, horizon: float) -> float:
    """Crude RMST ``E[min(T, horizon)]`` — unbiased on the randomized RCT arm."""
    return float(np.minimum(np.asarray(T, float), horizon).mean())


def empirical_survival(T: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Empirical survival ``P(T > t)`` (no censoring in the harness DGP)."""
    T = np.asarray(T, float)
    return np.array([(T > t).mean() for t in np.asarray(times)])


def rct_blinding_recovery(*, n_obs: int = 6000, n_rct: int = 6000, confound: float = 2.5,
                          adjust: bool = True, unmeasured: float = 0.0, d: int = 5,
                          tau: float = 0.6, sigma: float = 0.6, horizon: float = 2.0,
                          n_times: int = 50, tol: float = 0.03, seed: int = 0) -> dict:
    """Run the full RCT-blinding check. Fit the OC-sim on the **observational** cohort
    only (blinded to the RCT), then score its counterfactual control branch and
    treatment effect against the **held-out randomized RCT** (and the analytic truth).
    ``recovered`` = the OC-sim's RMST ATE matches the RCT's within ``tol``.

    Returns ATE (OC-sim / RCT / true), biases, the control-branch RMST, the
    control-branch survival-curve L1 distance to the RCT, and ``recovered``."""
    coh = make_confounded_cohorts(n_obs=n_obs, n_rct=n_rct, d=d, tau=tau, confound=confound,
                                  sigma=sigma, horizon=horizon, unmeasured=unmeasured, seed=seed)
    obs, rct = coh["obs"], coh["rct"]
    model = fit_oc_sim(obs["X"], obs["A"], obs["T"], adjust=adjust)

    rmst0_oc = oc_sim_rmst(model, obs["X"], 0.0, horizon, seed=seed)
    ate_oc = oc_sim_rmst(model, obs["X"], 1.0, horizon, seed=seed) - rmst0_oc

    ctrl = rct["T"][rct["A"] == 0]
    trt = rct["T"][rct["A"] == 1]
    rmst0_rct = empirical_rmst(ctrl, horizon)
    ate_rct = empirical_rmst(trt, horizon) - rmst0_rct

    times = np.linspace(0.0, horizon, n_times)
    s0_oc = oc_sim_survival(model, obs["X"], times, 0.0)
    s0_rct = empirical_survival(ctrl, times)
    d_abs = np.abs(s0_oc - s0_rct)                             # trapezoid (np.trapz deprecated)
    control_curve_l1 = float((0.5 * (d_abs[:-1] + d_abs[1:]) * np.diff(times)).sum())

    return {
        "ate_ocsim": ate_oc, "ate_rct": ate_rct, "ate_true": coh["true_ate"],
        "ate_bias_vs_rct": ate_oc - ate_rct, "ate_bias_vs_true": ate_oc - coh["true_ate"],
        "rmst_control_ocsim": rmst0_oc, "rmst_control_rct": rmst0_rct,
        "control_curve_l1": control_curve_l1,
        "recovered": bool(abs(ate_oc - ate_rct) < tol),
    }
