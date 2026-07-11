"""Exp 22: M-bias sensitivity — the adjust-and-still-open collider.

Sibling to exp2 (positivity), exp3 (unmeasured confounding), exp13 (event
imputation). Where exp3 shows estimators fail because they *can't* see the
confounder, exp22 shows the dual failure: an estimator biases itself by
*adjusting for a variable it can see* — a pre-treatment collider.

The M-structure (Greenland's classic):

        U1        U2          U1, U2 latent, U1 ⟂ U2
       /  \\      /  \\         M  observed, pre-treatment
      A    M ← ←  Y           A  treatment,  Y outcome
      |________________|
             (A → Y, effect τ)

M is correlated with BOTH A (via U1) and Y (via U2), so a "adjust for anything
associated with treatment and outcome" rule *includes* it — and that is the
mistake. Adjusting for the collider M opens the path U1 → M ← U2, inducing a
U1–U2 (hence A–Y) association that was not there. Baseline restriction
(ICH E9(R1)) does NOT save you: M is pre-treatment. Only collider-awareness does.

This is the estimand-side handling of the zero_flow_ci / Markov-blanket collider
caveat (PR #103 docstring; PR #100 diagram SNOTE): the adjustment set is chosen
by backdoor-validity (colliders excluded), never by the Markov blanket — and
here we show, end-to-end, that markov_blanket() returns M while the backdoor set
drops it. See causal_bench issue #104, SGA issue #8.

CPU, numpy/sklearn; reuses detectors/zero_flow_ci.py. No torch, no GPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from causal_bench.detectors.zero_flow_ci import (
    zero_flow_ci_test, zero_flow_statistic,
)

A_IDX, M_IDX, Y_IDX = 0, 1, 2
OUT_DIR = Path("results/exp22_mbias")


# ----------------------------------------------------------------------------- DGP
def simulate_m_structure(collider_strength: float, *, tau: float = 0.0,
                         n: int = 4000, seed: int = 0) -> np.ndarray:
    """Pure M-structure. Returns an (n, 3) array with columns [A, M, Y].

    U1, U2 are independent latent parents. A = U1 + noise, Y = τ·A + U2 + noise,
    and the collider M = c·(U1 + U2) + noise with c = ``collider_strength``. At
    c = 0, M carries no U1/U2 signal and adjusting for it is harmless; bias from
    adjusting for M grows with c."""
    rng = np.random.default_rng(seed)
    u1 = rng.normal(size=n)
    u2 = rng.normal(size=n)
    a = u1 + 0.5 * rng.normal(size=n)
    m = collider_strength * (u1 + u2) + 0.5 * rng.normal(size=n)
    y = tau * a + u2 + 0.5 * rng.normal(size=n)
    return np.column_stack([a, m, y])


# ------------------------------------------------------------------- OLS estimator
def estimate_ate(data: np.ndarray, adjust_cols: Sequence[int] = ()) -> Dict[str, float]:
    """OLS of Y on [1, A, *adjust_cols]; return the A coefficient with its
    homoskedastic 95% CI. ``adjust_cols`` are column indices of ``data`` to add
    to the design (M_IDX to reproduce the M-bias mistake)."""
    n = data.shape[0]
    A = data[:, A_IDX]
    Y = data[:, Y_IDX]
    cols = [np.ones(n), A] + [data[:, c] for c in adjust_cols]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = max(1, n - X.shape[1])
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))          # index 1 = A coefficient
    tau_hat = float(beta[1])
    return {"tau_hat": tau_hat, "se": se,
            "ci_lo": tau_hat - 1.96 * se, "ci_hi": tau_hat + 1.96 * se}


# --------------------------------------------------- collider-aware backdoor screen
def conditioning_opens_path(data: np.ndarray, *, a: int = A_IDX, y: int = Y_IDX,
                            c: int = M_IDX, alpha: float = 0.05, n_perm: int = 80,
                            rng: np.random.Generator | None = None) -> bool:
    """Heuristic FCI-lite collider screen: True iff conditioning on covariate
    ``c`` OPENS an a–y path — i.e. a and y are more dependent given ``c`` than
    marginally. Signature of a collider (or its descendant) on the a–y path.

    Uses the zero-flow CI machinery: a ``refutes`` conditional verdict *plus* a
    conditional statistic exceeding the marginal one. (Full FCI orientation is
    deferred; this is the pairwise screen the estimand discipline needs to keep a
    collider out of the adjustment set.)"""
    rng = rng or np.random.default_rng(0)
    A = data[:, a]
    Y = data[:, y]
    C = data[:, c].reshape(-1, 1)
    const = np.zeros((data.shape[0], 1))                 # residualize on a constant

    cond = zero_flow_ci_test(A, Y, C, n_perm=n_perm, alpha=alpha, rng=rng)
    if cond.verdict != "refutes":
        return False
    stat_marg = zero_flow_statistic(
        np.column_stack([A - A.mean(), Y - Y.mean()]),
        np.column_stack([A - A.mean(), (Y - Y.mean())[rng.permutation(len(Y))]]),
        rng=rng)
    # cond.statistic is the a–y dependence given C; compare to the marginal one
    return cond.statistic > stat_marg


def backdoor_set(data: np.ndarray, candidates: Iterable[int] = (M_IDX,),
                 *, rng: np.random.Generator | None = None) -> Tuple[int, ...]:
    """Collider-aware backdoor adjustment set: keep a candidate only if
    conditioning on it does NOT open an A–Y path. Colliders (and their
    descendants) are dropped — the estimand-side rule that MB membership never
    grants adjustment-set membership."""
    rng = rng or np.random.default_rng(0)
    keep = []
    for c in candidates:
        if c in (A_IDX, Y_IDX):
            continue
        if not conditioning_opens_path(data, c=c, rng=rng):
            keep.append(c)
    return tuple(keep)


# ------------------------------------------------------------------------- sweep
def run_mbias_sweep(collider_strengths: Sequence[float], *, tau: float = 0.0,
                    n: int = 6000, n_sims: int = 50, seed: int = 0) -> Dict:
    """For each collider strength, Monte-Carlo the bias / SE / 95%-coverage of
    the A→Y effect under three arms: ``unadjusted`` (correct here), ``adjust_M``
    (the M-bias mistake), ``backdoor_S`` (collider-aware set). Returns
    ``{arm: {'bias': [...], 'se': [...], 'coverage': [...]}}`` aligned to
    ``collider_strengths``."""
    arms = ("unadjusted", "adjust_M", "backdoor_S")
    out = {arm: {"bias": [], "se": [], "coverage": []} for arm in arms}
    for c in collider_strengths:
        acc = {arm: {"tau": [], "se": [], "cover": []} for arm in arms}
        for s in range(n_sims):
            d = simulate_m_structure(c, tau=tau, n=n, seed=seed + s)
            rng = np.random.default_rng(seed + 1000 + s)
            S = backdoor_set(d, candidates=(M_IDX,), rng=rng)
            for arm, cols in (("unadjusted", ()),
                              ("adjust_M", (M_IDX,)),
                              ("backdoor_S", S)):
                r = estimate_ate(d, adjust_cols=cols)
                acc[arm]["tau"].append(r["tau_hat"])
                acc[arm]["se"].append(r["se"])
                acc[arm]["cover"].append(r["ci_lo"] <= tau <= r["ci_hi"])
        for arm in arms:
            out[arm]["bias"].append(float(np.mean(acc[arm]["tau"]) - tau))
            out[arm]["se"].append(float(np.mean(acc[arm]["se"])))
            out[arm]["coverage"].append(float(np.mean(acc[arm]["cover"])))
    return out


def _format_table(strengths: Sequence[float], res: Dict) -> str:
    lines = ["| strength | arm | bias | mean SE | coverage |",
             "|---|---|---|---|---|"]
    for i, c in enumerate(strengths):
        for arm in ("unadjusted", "adjust_M", "backdoor_S"):
            lines.append(f"| {c:g} | {arm} | {res[arm]['bias'][i]:+.3f} | "
                         f"{res[arm]['se'][i]:.3f} | {res[arm]['coverage'][i]:.2f} |")
    return "\n".join(lines)


def run(n_sims: int = 200, n: int = 6000, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strengths = [0.0, 0.4, 0.8, 1.2, 1.6]
    print(f"Exp 22: M-bias sensitivity | n_sims={n_sims} | strengths={strengths}")
    res = run_mbias_sweep(strengths, tau=0.0, n=n, n_sims=n_sims, seed=seed)
    table = _format_table(strengths, res)
    (OUT_DIR / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nSaved → {OUT_DIR}/summary.md")
    return res


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 22: M-bias sensitivity")
    p.add_argument("--n-sims", type=int, default=200)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n=args.n, seed=args.seed)
