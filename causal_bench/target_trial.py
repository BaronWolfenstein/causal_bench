"""Grace-period clone-censor data adapter (#178, ENCIRCLE production CCW).

The immortal-time / grace-period fix is clone-censor-weight, but the weighting +
outcome model already exist (`estimators/ltmle.py`; the `concrete_*` rpy2 bridges,
continuous-time TMLE with `CensoringTV` + competing risks). The *only* new piece
is the target-trial (Hernán) **data construction**, built here.

`clone_censor_expand` turns patient-level rows into long-format cloned person-time
whose columns are the CONCRETE bridge contract — `A` (strategy), `T_obs` (time),
`event_type` (0=censored, 1=event; 2 reserved for a competing risk) — so it feeds
CONCRETE/LTMLE directly for the doubly-robust estimate. `clone_ipcw_risk_difference`
is the no-R fallback (reuses `sampling/ipcw.py`) that also validates the clones
recover the null on the immortal-time DGP.

Construction (strategy = 'implant by grace G'):
  * both clones start at time-zero (removes immortal time by design);
  * DEVICE clone censored at G if it reaches G without implanting; else followed;
  * CONTROL clone censored at implant time w if it implants within G; else followed;
  * a death during the grace window before any deviation is an EVENT for BOTH clones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clone_censor_expand(df: pd.DataFrame, *, grace: float, horizon: float,
                        id_col: str | None = None, event_time_col: str = "T",
                        implant_time_col: str = "w", covariate_cols=("X",)) -> pd.DataFrame:
    """Expand `df` into device/control clones. Returns long-format rows with
    columns: orig_id, clone_id, strategy, A, T_obs, event_type, + covariate_cols —
    the CONCRETE/LTMLE input contract."""
    n = len(df)
    ids = df[id_col].to_numpy() if id_col else np.arange(n)
    T = df[event_time_col].to_numpy(float)
    w = df[implant_time_col].to_numpy(float)
    G, H = float(grace), float(horizon)

    implanted = (w <= G) & (w < T)                 # implant in grace (survived to w)
    died_pre = T <= np.minimum(w, G)               # died in grace before implanting

    blocks = []
    # DEVICE strategy (A=1): implanted → follow to H; died-in-grace → event; else censor at G
    dev_T = np.where(implanted, np.minimum(T, H), np.where(died_pre, T, G))
    dev_e = np.where(implanted, (T <= H).astype(int), np.where(died_pre, 1, 0))
    # CONTROL strategy (A=0): implanted → censor at implant w; else follow to H
    ctl_T = np.where(implanted, w, np.minimum(T, H))
    ctl_e = np.where(implanted, 0, (T <= H).astype(int))

    for strat, A, Tobs, ev in (("device", 1, dev_T, dev_e), ("control", 0, ctl_T, ctl_e)):
        b = pd.DataFrame({"orig_id": ids,
                          "clone_id": [f"{i}_{strat}" for i in ids],
                          "strategy": strat, "A": A,
                          "T_obs": Tobs.astype(float), "event_type": ev.astype(int)})
        for c in covariate_cols:
            b[c] = df[c].to_numpy()
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


def clone_ipcw_risk_difference(cloned: pd.DataFrame, *, horizon: float,
                               covariate_cols=("X",)) -> float:
    """No-R fallback: IPCW risk difference on the cloned data (the production
    estimate is CONCRETE/LTMLE on the same rows). Deviation = censored before the
    horizon; weight uncensored clones by 1/P(not deviated | X). Reuses
    `sampling.ipcw`."""
    from sklearn.linear_model import LogisticRegression
    from causal_bench.sampling.ipcw import ipcw_weights, positivity_floor
    H = float(horizon)
    risks = {}
    for A in (1, 0):
        arm = cloned[cloned["A"] == A]
        Tobs = arm["T_obs"].to_numpy(); ev = arm["event_type"].to_numpy()
        X = arm[list(covariate_cols)].to_numpy()
        deviated = (ev == 0) & (Tobs < H)                      # artificial deviation-censoring
        uncensored = ~deviated
        if deviated.any() and (~deviated).any():
            p_unc = LogisticRegression().fit(X, uncensored.astype(float)).predict_proba(X)[:, 1]
        else:
            p_unc = np.ones(len(arm))
        p_unc, _ = positivity_floor(p_unc, 0.02)
        wt = ipcw_weights(p_unc)[uncensored]
        risks[A] = float(np.average((ev[uncensored] == 1).astype(float), weights=wt))
    return risks[1] - risks[0]
