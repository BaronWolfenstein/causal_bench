"""Grace-period clone-censor adapter (#178). Verifies the clone construction is
structurally correct (CONCRETE contract, censor-at-deviation) and that analyzing
the cloned data recovers the null on the immortal-time DGP — the production CCW
path (feed these rows to CONCRETE/LTMLE)."""
import numpy as np
import pandas as pd

from causal_bench.dgp.immortal_time import (
    draw_immortal_time, grace_period_naive_rd, ImmortalTimeConfig)
from causal_bench.target_trial import clone_censor_expand, clone_ipcw_risk_difference


def test_expansion_shape_and_concrete_contract():
    df = draw_immortal_time(500, 0, ImmortalTimeConfig())
    cl = clone_censor_expand(df, grace=1.0, horizon=3.0, covariate_cols=["X"])
    assert len(cl) == 2 * len(df)                          # two clones per patient
    assert {"A", "T_obs", "event_type"} <= set(cl.columns)  # CONCRETE bridge contract
    assert set(cl["A"].unique()) == {0, 1}
    assert set(cl["event_type"].unique()) <= {0, 1, 2}
    assert cl.groupby("orig_id").size().eq(2).all()        # exactly one device + one control


def test_censor_at_deviation_logic():
    # one patient implants in grace (w<G<T), one never implants (w>G, T>G)
    df = pd.DataFrame({"X": [0.0, 0.0], "w": [0.5, 2.0], "T": [5.0, 5.0]})
    cl = clone_censor_expand(df, grace=1.0, horizon=3.0, covariate_cols=["X"])
    dev = cl[cl["strategy"] == "device"].reset_index(drop=True)
    ctl = cl[cl["strategy"] == "control"].reset_index(drop=True)
    # patient 0 implants at 0.5: device compliant (followed to horizon), control censored at w=0.5
    assert dev.loc[0, "T_obs"] == 3.0 and dev.loc[0, "event_type"] == 0   # survived to horizon
    assert ctl.loc[0, "T_obs"] == 0.5 and ctl.loc[0, "event_type"] == 0   # censored at implant
    # patient 1 never implants by G=1: device censored at G=1, control followed to horizon
    assert dev.loc[1, "T_obs"] == 1.0 and dev.loc[1, "event_type"] == 0   # device deviated at G
    assert ctl.loc[1, "T_obs"] == 3.0 and ctl.loc[1, "event_type"] == 0


def test_early_grace_death_counts_for_both_clones():
    # patient dies at T=0.4 before implanting (w=0.8), grace G=1 → event for BOTH clones
    df = pd.DataFrame({"X": [0.0], "w": [0.8], "T": [0.4]})
    cl = clone_censor_expand(df, grace=1.0, horizon=3.0, covariate_cols=["X"])
    assert (cl["event_type"] == 1).all()                   # both clones: event
    assert (cl["T_obs"] == 0.4).all()


def test_adapter_recovers_null():
    cfg = ImmortalTimeConfig(grace=1.5)
    gnaive, adapt = [], []
    for s in range(150):
        d = draw_immortal_time(3000, s, cfg)
        gnaive.append(grace_period_naive_rd(d, cfg))
        cl = clone_censor_expand(d, grace=cfg.grace, horizon=cfg.horizon, covariate_cols=["X"])
        adapt.append(clone_ipcw_risk_difference(cl, horizon=cfg.horizon, covariate_cols=["X"]))
    gnaive, adapt = float(np.mean(gnaive)), float(np.mean(adapt))
    assert gnaive < -0.08                                  # grace per-protocol immortal-biased
    assert abs(adapt) < 0.06                               # cloned analysis ≈ null
    assert abs(adapt) < 0.5 * abs(gnaive)                  # removes the bulk of the bias
