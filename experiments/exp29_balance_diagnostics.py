"""Exp 29: baseline balance table + region-R overlap diagnostics for a weighted SCA.

Synthetic single-arm design: Target Group (n=299) vs an external Baseline Cohort
(n=2000), five covariates, with the severity covariate X5's upper tail defining
the sparse region R (the engineered positivity problem). Odds-of-propensity
weights reweight the Baseline Cohort toward the Target Group (ATT-style).

Three augmentation panels expose failure mode 1 of the weighting review
("extreme weights in R even after augmentation"):

- none:      real Baseline only — the canonical FALSE PASS (global SMDs clear
             0.1 while region-R SMDs fail and R-ESS is thin).
- interior:  synthetic records matched to the Target Group's R-conditional
             distribution — genuine overlap restoration.
- edge:      synthetic records piled just inside R's boundary — the failure
             mode: headline R-ESS rises mechanically (more records) while the
             DEEP interior of R stays unsupported. Exposed by the
             region-resolved ESS map (deep-R column), not the R scalar.

The load-bearing diagnostic is the post-augmentation ESS *map* (global / R /
deep-R), recomputed — never assumed improved.

This is a template harness on a synthetic DGP. It is NOT evidence about any
real cohort: for a real evidence package the same table must be recomputed on
the actual external-registry records (real covariates, the real positivity map
defining R, the production propensity learner).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

OUT_DIR = Path("results/exp29_balance_diagnostics")

COVS = ["X1", "X2", "X3", "X4", "X5"]
R_CUT = 2.0        # region R: X5 >= R_CUT
DEEP_CUT = 2.4     # deep interior of R — where edge-fill leaves no support


def draw_cohort(rng, n, shift, sd, sev_loc, sev_tail):
    tail = rng.random(n) < sev_tail
    return pd.DataFrame({
        "X1": rng.normal(shift[0], sd[0], n),
        "X2": rng.normal(shift[1], sd[1], n),
        "X3": rng.lognormal(shift[2], sd[2], n),
        "X4": rng.binomial(1, shift[3], n).astype(float),
        "X5": np.where(tail, rng.normal(sev_loc + 2.2, 0.45, n),
                       rng.normal(sev_loc, 0.8, n)),
    })


def draw_cohorts(seed: int, n_target: int = 299, n_baseline: int = 2000):
    """Target Group (more severe, fatter X5 tail) and Baseline Cohort (thin tail)."""
    rng = np.random.default_rng(seed)
    target = draw_cohort(rng, n_target, (0.35, -0.25, 0.15, 0.55), (1.0, 1.1, 0.55),
                         sev_loc=0.4, sev_tail=0.18)
    baseline = draw_cohort(rng, n_baseline, (0.0, 0.0, 0.0, 0.42), (1.0, 1.0, 0.5),
                           sev_loc=0.0, sev_tail=0.02)
    return target, baseline


def augment(target: pd.DataFrame, mode: str, n_aug: int, seed: int) -> pd.DataFrame:
    """Synthetic Baseline records in region R.

    interior: jittered resamples of the Target Group's R-conditional rows —
              fills R where the Target Group actually lives.
    edge:     records piled just inside R's boundary (X5 in [R_CUT, R_CUT+0.15])
              with bulk-like other covariates — restores R *membership counts*
              without supporting R's deep interior.
    """
    rng = np.random.default_rng(seed + 1)
    if mode == "interior":
        tr = target[target.X5 >= R_CUT]
        aug = tr.iloc[rng.integers(0, len(tr), n_aug)].copy()
        aug = aug + rng.normal(0, 0.15, (n_aug, len(COVS))) * tr.std().values
        aug["X4"] = (aug["X4"] > 0.5).astype(float)
        aug["X5"] = np.clip(aug["X5"], R_CUT + 0.05, None)
    elif mode == "edge":
        aug = pd.DataFrame({
            "X1": rng.normal(0.0, 1.0, n_aug),
            "X2": rng.normal(0.0, 1.0, n_aug),
            "X3": rng.lognormal(0.0, 0.5, n_aug),
            "X4": rng.binomial(1, 0.42, n_aug).astype(float),
            "X5": rng.uniform(R_CUT, R_CUT + 0.15, n_aug),
        })
    else:
        raise ValueError(f"unknown augmentation mode: {mode}")
    return aug.reset_index(drop=True)


def fit_odds_weights(target: pd.DataFrame, baseline: pd.DataFrame) -> np.ndarray:
    """Normalized odds-of-propensity weights on the Baseline Cohort (ATT-style)."""
    X = pd.concat([target, baseline], ignore_index=True)
    y = np.r_[np.ones(len(target)), np.zeros(len(baseline))]
    Xs = (X - X.mean()) / X.std()
    p = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y).predict_proba(Xs)[:, 1]
    pb = p[len(target):]
    w = pb / (1 - pb)
    return w * len(w) / w.sum()


def _wmean_var(x, w):
    m = np.average(x, weights=w)
    return m, np.average((x - m) ** 2, weights=w)


def smd(t: pd.Series, b: pd.Series, wb=None) -> float:
    wb = np.ones(len(b)) if wb is None else wb
    mb, vb = _wmean_var(b.to_numpy(), wb)
    pooled = np.sqrt((t.var(ddof=1) + vb) / 2)
    return float((t.mean() - mb) / pooled) if pooled > 0 else 0.0


def variance_ratio(t: pd.Series, b: pd.Series, wb=None) -> float:
    wb = np.ones(len(b)) if wb is None else wb
    _, vb = _wmean_var(b.to_numpy(), wb)
    return float(t.var(ddof=1) / vb) if vb > 0 else float("inf")


def kish_ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def balance_panel(target, baseline, w, panel: str) -> tuple[pd.DataFrame, dict]:
    """Per-covariate balance + the region-resolved ESS map (global / R / deep-R)."""
    in_r_t = target.X5 >= R_CUT
    in_r_b = (baseline.X5 >= R_CUT).to_numpy()
    deep_b = (baseline.X5 >= DEEP_CUT).to_numpy()
    rows = []
    for c in COVS:
        rows.append({
            "panel": panel, "covariate": c,
            "smd_pre": smd(target[c], baseline[c]),
            "smd_post": smd(target[c], baseline[c], w),
            "vr_post": variance_ratio(target[c], baseline[c], w),
            "smd_post_R": smd(target.loc[in_r_t, c], baseline.loc[in_r_b, c], w[in_r_b])
                          if in_r_b.sum() >= 5 and in_r_t.sum() >= 5 else float("nan"),
        })
    meta = {
        "panel": panel,
        "n_baseline": len(baseline), "n_baseline_R": int(in_r_b.sum()),
        "n_baseline_deepR": int(deep_b.sum()), "n_target_R": int(in_r_t.sum()),
        "ess_global": kish_ess(w), "ess_R": kish_ess(w[in_r_b]),
        "ess_deepR": kish_ess(w[deep_b]),
        "max_w": float(w.max()), "pct_w_gt10": float((w > 10).mean() * 100),
    }
    return pd.DataFrame(rows), meta


def run_panels(seed: int = 20260702, n_aug: int = 120):
    """Balance table + ESS map for {none, interior, edge} augmentation."""
    target, baseline = draw_cohorts(seed)
    tables, metas = [], []
    for mode in ["none", "interior", "edge"]:
        b = baseline if mode == "none" else pd.concat(
            [baseline, augment(target, mode, n_aug, seed)], ignore_index=True)
        w = fit_odds_weights(target, b)
        t, m = balance_panel(target, b, w, mode)
        tables.append(t)
        metas.append(m)
    return pd.concat(tables, ignore_index=True), pd.DataFrame(metas)


def love_frame(balance: pd.DataFrame, panel: str) -> pd.DataFrame:
    """Long-format |SMD| frame for a panel: global vs region-R post-weighting.

    Feeds ``causal_bench.diagnostics.love_plot``. The 'region R' series is the
    load-bearing view — global balance can pass while region R fails (the false
    pass), and edge-fill augmentation makes region-R imbalance *worse* than none.
    """
    p = balance[balance.panel == panel]
    rows = []
    for _, r in p.iterrows():
        rows.append({"covariate": r.covariate, "series": "global",
                     "abs_smd": abs(r.smd_post)})
        if pd.notna(r.smd_post_R):
            rows.append({"covariate": r.covariate, "series": "region R",
                         "abs_smd": abs(r.smd_post_R)})
    return pd.DataFrame(rows)


def plot_love_regions(balance: pd.DataFrame, panel: str = "edge",
                      save_path=None):
    """Region-split Love plot for one augmentation panel (reuses the shared
    ``diagnostics.love_plot`` renderer)."""
    from causal_bench.diagnostics import love_plot

    return love_plot(love_frame(balance, panel),
                     title=f"Covariate balance — {panel} augmentation (global vs region R)",
                     save_path=save_path)


def run(seed: int = 20260702):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    balance, ess_map = run_panels(seed)
    balance.to_parquet(OUT_DIR / "balance_table.parquet", index=False)
    ess_map.to_parquet(OUT_DIR / "ess_map.parquet", index=False)
    for panel in ["interior", "edge"]:
        plot_love_regions(balance, panel, save_path=str(OUT_DIR / f"love_{panel}.png"))
    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    print(balance.to_string(index=False))
    print(ess_map.to_string(index=False))
    return balance, ess_map


if __name__ == "__main__":
    run()
