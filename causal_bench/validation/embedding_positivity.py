"""High-dim learned-embedding covariates as a fraught causal adjustment set (exp50).

Controlled synthetic reproduction of the SMB-demo finding (payoff_v7): a low-dim
true confounder ``Ustar`` drives treatment ``A`` and outcome ``Y``; the observed
covariate ``W`` is a high-fidelity high-dim encoding of ``Ustar``. As the
confounding strength rises, ``A`` becomes near-deterministic given the confounder
(severe positivity), and a FLEXIBLE outcome model (HistGradientBoosting) then
**attenuates the treatment effect toward null** -- a doubly-robust AIPW does not
rescue it, because the failure is the adjustment SET / support, not the nuisances.

Key mechanism (refined from the linear scaffold, see issue #206):
  * A *linear* outcome model does NOT show this -- it recovers the effect.
  * A *flexible* outcome model + severe positivity attenuates even the ORACLE
    (adjusting for the true low-dim Ustar), so it is the flexible-Q x positivity
    interaction, not high-dimensionality per se. The high-dim embedding's role in
    the real case is to CREATE the severe positivity (near-perfect propensity).

Positivity-robust / reduction responses (constant effect => every subpopulation ATE
== tau, so these all target tau):
  * dr_ato     -- tier-1: augmented (doubly-robust) overlap weighting; ~halves the bias
    but does not fully recover (ATO != ATE).
  * prognostic -- tier-2: adjust for the 1-D PROGNOSTIC score alone (control-outcome
    surface, Hansen 2008). Unlike the propensity score it is not treatment-degenerate,
    so it escapes the positivity trap and LARGELY RECOVERS the effect.

Self-validating: (i) at conf=0 (no confounding) every method is unbiased; (ii) the
attenuation grows monotonically with confounding strength.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

_CLIP = (0.02, 0.98)


def _make_reg(flex):
    if flex:
        return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                             learning_rate=0.1, random_state=0)
    return Ridge(alpha=1.0)


_EM_DIR = np.array([0.6, 0.8])       # confounder loading `d` of the effect modifier (unit norm)


def simulate(n, d=50, fidelity=6.0, conf=3.0, tau=1.0, gamma=0.0, seed=0):
    """Low-dim confounder ``Ustar`` -> high-dim encoding ``W``; ``conf`` sets positivity
    severity. ``gamma`` sets EFFECT MODIFICATION: the per-unit effect is
    ``tau_i = tau * (1 + gamma * (Ustar @ d))`` (gamma=0 -> constant effect, the exp50
    default). Under effect modification the ATE is the mean of ``tau_i`` and a single
    prognostic score no longer suffices -- the reason to reach for the double-score / SDR
    reduction. ``true_ate`` is returned because with gamma!=0 it is no longer just ``tau``."""
    rng = np.random.default_rng(seed)
    Ustar = rng.normal(size=(n, 2))
    e = 1.0 / (1.0 + np.exp(-conf * (Ustar @ np.array([1.3, -1.0]))))
    A = (rng.random(n) < e).astype(float)
    tau_i = tau * (1.0 + gamma * (Ustar @ _EM_DIR))
    Y = tau_i * A + Ustar @ np.array([1.5, 1.0]) + rng.normal(size=n)
    B = rng.normal(size=(2, d)) / np.sqrt(2)
    W = Ustar @ B + (1.0 / fidelity) * rng.normal(size=(n, d))
    return dict(W=W, Ustar=Ustar, A=A, Y=Y, e_true=e, tau=float(tau),
                tau_i=tau_i, true_ate=float(np.mean(tau_i)))


def _propensity(W, A):
    return np.clip(LogisticRegression(max_iter=2000, C=1.0).fit(W, A).predict_proba(W)[:, 1], *_CLIP)


def _q_predict(W, A, Y, flex):
    n = len(A)
    XA = np.column_stack([A, W])
    if flex:
        m = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                          learning_rate=0.1, random_state=0).fit(XA, Y)
    else:
        m = Ridge(alpha=1.0).fit(XA, Y)
    Q1 = m.predict(np.column_stack([np.ones(n), W]))
    Q0 = m.predict(np.column_stack([np.zeros(n), W]))
    return Q1, Q0


def _aipw(W, A, Y, flex=True, e=None):
    e = _propensity(W, A) if e is None else e
    Q1, Q0 = _q_predict(W, A, Y, flex)
    QA = A * Q1 + (1 - A) * Q0
    H = A / e - (1 - A) / (1 - e)
    return float(np.mean(Q1 - Q0 + H * (Y - QA)))


def _dr_ato(W, A, Y, flex=True):
    """Augmented (doubly-robust) overlap-weighted ATO -- the tier-1 positivity-robust
    response. Bounded overlap weights h=e(1-e); ~halves the attenuation but does not
    fully recover (ATO != ATE)."""
    e = _propensity(W, A)
    Q1, Q0 = _q_predict(W, A, Y, flex)
    h = e * (1 - e)
    plug = np.sum(h * (Q1 - Q0)) / np.sum(h)
    c1 = np.sum(A * (1 - e) * (Y - Q1)) / np.sum(h)
    c0 = np.sum((1 - A) * e * (Y - Q0)) / np.sum(h)
    return float(plug + c1 - c0)


def _prognostic_score(W, A, Y, flex):
    """Prognostic score E[Y | A=0, W], fit on CONTROLS only (Hansen 2008)."""
    Wc, Yc = W[A == 0], Y[A == 0]
    if flex:
        m = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                          learning_rate=0.1, random_state=0).fit(Wc, Yc)
    else:
        m = Ridge(alpha=1.0).fit(Wc, Yc)
    return m.predict(W)


def _prognostic_aipw(W, A, Y, flex=True):
    """Tier-2 candidate: adjust for the 1-D PROGNOSTIC score alone. Unlike the
    propensity score (which IS the positivity direction), the prognostic score is
    not treatment-degenerate, so it controls confounding while escaping the
    positivity trap -- and largely RECOVERS the effect where full-W AIPW fails."""
    prog = _prognostic_score(W, A, Y, flex)
    return _aipw(prog[:, None], A, Y, flex=flex)


# ---------------------------------------------------------------------------
# Effect-modification reductions + interval coverage (the SDR-spec frontier).
# These return (point, se) so the 2-D sweep can report coverage. The 1-D conf
# path above (floats) is unchanged.
# ---------------------------------------------------------------------------

def _aipw_ci(X, A, Y, flex=True, e=None):
    """AIPW point estimate + influence-function SE (for interval coverage)."""
    e = _propensity(X, A) if e is None else e
    Q1, Q0 = _q_predict(X, A, Y, flex)
    QA = A * Q1 + (1 - A) * Q0
    H = A / e - (1 - A) / (1 - e)
    ic = Q1 - Q0 + H * (Y - QA)
    psi = float(np.mean(ic))
    se = float(np.std(ic - psi, ddof=1) / np.sqrt(len(A)))
    return psi, se


def _prognostic_ci(W, A, Y, flex=True):
    return _aipw_ci(_prognostic_score(W, A, Y, flex)[:, None], A, Y, flex=flex)


def _double_score(W, A, Y, flex):
    """The two potential-outcome surfaces (E[Y|A=0,W], E[Y|A=1,W]) as a 2-D balancing
    score. Captures the CATE -- hence effect modification -- that the 1-D prognostic
    (control surface only) cannot. Each arm fit on its own subjects, predicted for all."""
    def fit(Wa, Ya):
        if flex:
            return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                                 learning_rate=0.1, random_state=0).fit(Wa, Ya)
        return Ridge(alpha=1.0).fit(Wa, Ya)
    m0 = fit(W[A == 0], Y[A == 0])
    m1 = fit(W[A == 1], Y[A == 1])
    return np.column_stack([m0.predict(W), m1.predict(W)])


def _double_score_aipw(W, A, Y, flex=True):
    return _aipw_ci(_double_score(W, A, Y, flex), A, Y, flex=flex)


# ---- Sufficient-dimension reduction (SIR + SAVE) of the OUTCOME subspace ----
# Target the subspace relevant to Y (prognostic + effect-modification directions), NOT
# the propensity: the propensity direction IS the positivity direction, so an A-targeted
# reduction would re-import the trap. This is the causal-vs-predictive-sufficiency
# discipline -- reduce toward OUTCOME-sufficiency, then adjust.

def _whiten(W):
    Wc = W - W.mean(0)
    Sigma = np.cov(Wc, rowvar=False) + 1e-6 * np.eye(W.shape[1])
    vals, vecs = np.linalg.eigh(Sigma)
    inv_half = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    return Wc @ inv_half


def _sir_dirs(Z, resp, H=10):
    n, p = Z.shape
    M = np.zeros((p, p))
    for b in np.array_split(np.argsort(resp), H):
        zbar = Z[b].mean(0)
        M += (len(b) / n) * np.outer(zbar, zbar)
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1]                       # columns = directions, desc eigenvalue


def _save_dirs(Z, resp, H=10):
    n, p = Z.shape
    M = np.zeros((p, p))
    for b in np.array_split(np.argsort(resp), H):
        C = np.cov(Z[b], rowvar=False) if len(b) > 1 else np.eye(p)
        D = np.eye(p) - C
        M += (len(b) / n) * (D @ D)
    _, vecs = np.linalg.eigh(M)
    return vecs[:, ::-1]


def _sdr_reduce(W, A, Y, k=2, use_save=True):
    """Arm-stratified central-subspace reduction of the outcome surfaces. Fitting SIR/SAVE
    on the POOLED Y contaminates the subspace with the treatment signal (Y contains the
    tau*A term, and A is confounded), which re-imports the propensity direction and biases
    the adjustment. Instead we estimate the central subspace WITHIN each arm and union them:
    the A=0 fit is the prognostic subspace (Hansen-style balancing score), the A=1 fit adds
    the effect-modification directions -- the linear-reduction analog of the double-score.
    SIR gives first-moment (prognostic) directions; SAVE adds a second-moment direction.
    Returns phi(W) = whitened W projected onto the combined orthonormal subspace."""
    Z = _whiten(W)
    dirs = []
    for a in (0.0, 1.0):
        m = A == a
        dirs.extend(_sir_dirs(Z[m], Y[m]).T[:k])
        if use_save:
            dirs.append(_save_dirs(Z[m], Y[m]).T[0])
    D, _ = np.linalg.qr(np.column_stack(dirs))   # orthonormalize the unioned subspace
    return Z @ D


def _sdr_aipw(W, A, Y, flex=True, k=2):
    return _aipw_ci(_sdr_reduce(W, A, Y, k=k), A, Y, flex=flex)


def _sdr_ato(W, A, Y, flex=True, k=2):
    """Positivity RESPONSE on the reduced set: overlap-weighted (ATO) adjustment on the
    SDR subspace. Where the outcome-relevant subspace still contains the positivity
    direction (severe conf), the ATE on phi is not identified; overlap weighting targets
    the identified ATO instead (== ATE under a constant effect). Combine the reduction
    with this, reporting ATO != ATE, per the spec's positivity handling."""
    return _dr_ato(_sdr_reduce(W, A, Y, k=k), A, Y, flex=flex)


# ---------------------------------------------------------------------------
# Cross-fit (DML) versions for VALID interval coverage. The reduction map is a
# nuisance too, so it is fit out-of-fold alongside the propensity and outcome
# models; the influence function is then evaluated only on held-out folds.
# Each fitter takes training data and returns an apply-map W -> phi.
# ---------------------------------------------------------------------------

def _fit_identity(Wtr, Atr, Ytr, flex):
    return lambda W: W


def _fit_prognostic_map(Wtr, Atr, Ytr, flex):
    m = _make_reg(flex).fit(Wtr[Atr == 0], Ytr[Atr == 0])
    return lambda W: m.predict(W)[:, None]


def _fit_double_score_map(Wtr, Atr, Ytr, flex):
    m0 = _make_reg(flex).fit(Wtr[Atr == 0], Ytr[Atr == 0])
    m1 = _make_reg(flex).fit(Wtr[Atr == 1], Ytr[Atr == 1])
    return lambda W: np.column_stack([m0.predict(W), m1.predict(W)])


def _fit_sdr_map(Wtr, Atr, Ytr, flex, k=2, use_save=True):
    """Fit the arm-stratified SIR+SAVE reduction on training data; return the apply-map
    that whitens with the TRAIN moments and projects onto the TRAIN directions."""
    mu = Wtr.mean(0)
    Sigma = np.cov(Wtr - mu, rowvar=False) + 1e-6 * np.eye(Wtr.shape[1])
    vals, vecs = np.linalg.eigh(Sigma)
    inv_half = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
    Ztr = (Wtr - mu) @ inv_half
    dirs = []
    for a in (0.0, 1.0):
        m = Atr == a
        dirs.extend(_sir_dirs(Ztr[m], Ytr[m]).T[:k])
        if use_save:
            dirs.append(_save_dirs(Ztr[m], Ytr[m]).T[0])
    D, _ = np.linalg.qr(np.column_stack(dirs))
    return lambda W: ((W - mu) @ inv_half) @ D


def _crossfit_ic(W, A, Y, reduce_fit, flex=True, n_folds=5, seed=0, ato=False):
    """Cross-fit AIPW on a fitted reduction: nuisances and the reduction map are trained
    out-of-fold, the influence function evaluated on the held-out fold. Returns (point, se)
    with a DML-valid SE. ato=True targets the overlap-weighted ATO (a ratio estimand)."""
    n = len(A)
    ic = np.zeros(n)
    num = np.zeros(n)
    den = np.zeros(n)
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(W):
        phi = reduce_fit(W[tr], A[tr], Y[tr], flex)
        Ptr, Pte = phi(W[tr]), phi(W[te])
        Ptr = Ptr if Ptr.ndim > 1 else Ptr[:, None]
        Pte = Pte if Pte.ndim > 1 else Pte[:, None]
        e = np.clip(LogisticRegression(max_iter=2000).fit(Ptr, A[tr]).predict_proba(Pte)[:, 1], *_CLIP)
        Qm = _make_reg(flex).fit(np.column_stack([A[tr], Ptr]), Y[tr])
        m = len(te)
        Q1 = Qm.predict(np.column_stack([np.ones(m), Pte]))
        Q0 = Qm.predict(np.column_stack([np.zeros(m), Pte]))
        QA = A[te] * Q1 + (1 - A[te]) * Q0
        if ato:
            h = e * (1 - e)
            num[te] = h * (Q1 - Q0) + A[te] * (1 - e) * (Y[te] - Q1) - (1 - A[te]) * e * (Y[te] - Q0)
            den[te] = h
        else:
            H = A[te] / e - (1 - A[te]) / (1 - e)
            ic[te] = Q1 - Q0 + H * (Y[te] - QA)
    if ato:
        psi = float(num.sum() / den.sum())
        ic = (num - psi * den) / den.mean()          # ratio-estimator IF
        return psi, float(np.std(ic, ddof=1) / np.sqrt(n))
    psi = float(np.mean(ic))
    return psi, float(np.std(ic - psi, ddof=1) / np.sqrt(n))


def _frac_extreme_on(X, A, clip_lo=0.05):
    """Positivity diagnostic on a (reduced) adjustment set: fraction with estimated
    propensity outside [clip_lo, 1-clip_lo]. Small on a positivity-escaping reduction."""
    e = _propensity(X if X.ndim > 1 else X[:, None], A)
    return float(np.mean((e < clip_lo) | (e > 1 - clip_lo)))


def one_rep(n, d, fidelity, conf, tau, seed, flex=True):
    s = simulate(n, d=d, fidelity=fidelity, conf=conf, tau=tau, seed=seed)
    W, A, Y, U = s["W"], s["A"], s["Y"], s["Ustar"]
    return {
        "oracle_Ustar": _aipw(U, A, Y, flex=flex),        # adjust for the true confounder
        "naive_fullW": _aipw(W, A, Y, flex=flex),         # the pathology
        "dr_ato_W": _dr_ato(W, A, Y, flex=flex),          # tier-1: positivity-robust, ~halves bias
        "prog_score_W": _prognostic_aipw(W, A, Y, flex=flex),  # tier-2: largely recovers
        "frac_extreme": float(np.mean((s["e_true"] < 0.05) | (s["e_true"] > 0.95))),
    }


def report_rows(n=2500, d=50, fidelity=6.0, tau=1.0, n_reps=12,
                confs=(0.0, 1.0, 2.0, 3.0, 5.0), flex=True, seed=0):
    methods = ["oracle_Ustar", "naive_fullW", "dr_ato_W", "prog_score_W"]
    rows = []
    for conf in confs:
        acc = {m: [] for m in methods}; fe = []
        for r in range(n_reps):
            res = one_rep(n, d, fidelity, conf, tau, seed + r, flex=flex)
            for m in methods:
                acc[m].append(res[m] - tau)          # bias vs true ATE (== every-subpop ATE)
            fe.append(res["frac_extreme"])
        rows.append({"conf": conf, "frac_extreme": float(np.mean(fe)),
                     **{m: float(np.nanmean(acc[m])) for m in methods}})
    return rows


def _bias_cov(points, ses, truth):
    p, s = np.asarray(points), np.asarray(ses)
    bias = float(np.mean(p - truth))
    cov = float(np.mean((p - 1.96 * s <= truth) & (truth <= p + 1.96 * s)))
    return bias, cov


def report_rows_2d(n=2000, d=50, fidelity=6.0, tau=1.0, n_reps=15,
                   gammas=(0.0, 2.0, 4.0), confs=(1.0, 3.0), flex=True, seed=0,
                   crossfit=True, n_folds=5):
    """The SDR-spec 2-D frontier: effect-modification (gamma) x positivity-severity (conf).
    Reports bias AND 95%-interval coverage for {naive, prognostic, double-score, SDR, and the
    ATO-on-phi response}, plus positivity diagnostics (fraction of near-deterministic
    propensities on the full embedding vs the SDR-reduced set). Estimand = tau (the effect
    modifier is mean-zero, so the population ATE stays tau for every gamma).

    crossfit=True (default) fits the nuisances AND the reduction map out-of-fold (DML), so the
    interval coverage is valid; crossfit=False uses the faster in-sample IF (optimistic SE)."""
    maps = {"naive_fullW": _fit_identity, "prog_score": _fit_prognostic_map,
            "double_score": _fit_double_score_map, "sdr": _fit_sdr_map}

    def estimate(s, fitter, ato=False):
        if crossfit:
            return _crossfit_ic(s["W"], s["A"], s["Y"], fitter, flex=flex,
                                n_folds=n_folds, seed=seed, ato=ato)
        phi = fitter(s["W"], s["A"], s["Y"], flex)(s["W"])
        if ato:
            return _dr_ato(phi, s["A"], s["Y"], flex=flex), float("nan")
        return _aipw_ci(phi, s["A"], s["Y"], flex=flex)

    rows = []
    for gamma in gammas:
        for conf in confs:
            acc = {m: ([], []) for m in maps}
            ato_pt, ato_se, feW, feS = [], [], [], []
            for r in range(n_reps):
                s = simulate(n, d=d, fidelity=fidelity, conf=conf, tau=tau, gamma=gamma, seed=seed + r)
                for m, fitter in maps.items():
                    p, se = estimate(s, fitter)
                    acc[m][0].append(p); acc[m][1].append(se)
                p, se = estimate(s, _fit_sdr_map, ato=True)           # positivity response on phi
                ato_pt.append(p); ato_se.append(se)
                feW.append(_frac_extreme_on(s["W"], s["A"]))
                feS.append(_frac_extreme_on(_sdr_reduce(s["W"], s["A"], s["Y"]), s["A"]))
            b_ato, c_ato = _bias_cov(ato_pt, ato_se, tau)
            row = {"gamma": gamma, "conf": conf,
                   "posv_full": float(np.mean(feW)), "posv_sdr": float(np.mean(feS)),
                   "sdr_ato_bias": b_ato, "sdr_ato_cov": c_ato}       # ATO estimand (==ATE at gamma=0)
            for m in maps:
                b, c = _bias_cov(acc[m][0], acc[m][1], tau)
                row[f"{m}_bias"], row[f"{m}_cov"] = b, c
            rows.append(row)
    return rows
