import numpy as np
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator


class TMLEIPCWCVEstimator(TMLEIPCWEstimator):
    """CV-TMLE: cross-fitted censoring model for calibrated SE at finite n.

    The base TMLEIPCWEstimator already cross-fits g (propensity, via SuperLearner
    OOF) and Q (outcome, via KFold). The remaining source of SE undercoverage is
    that G (censoring survival model) is fit on the full dataset and evaluated on
    the same data.  This subclass overrides _fit_G to produce out-of-fold G
    predictions, making all three nuisance models cross-fitted (CV-TMLE,
    Zheng & van der Laan 2011).

    Empirical note: cross-fitting G alone does not close the se_ratio gap observed
    in exp6/exp9 (se_ratio ≈ 0.81-0.85).  The undercoverage is driven by the
    doubly-robust remainder (interaction of g/Q estimation error) and by
    Super Learner model-selection variance across datasets — neither of which
    the IC formula captures regardless of G cross-fitting.  CV-TMLE is the
    theoretically correct choice but the finite-sample improvement is minimal
    at n ≤ 2000.  For calibrated coverage at practical n use tmle_ipcw_boot.
    """

    @property
    def name(self) -> str:
        return "TMLE+IPCW+CV+Comply" if self.use_compliance else "TMLE+IPCW+CV"

    def _fit_G(self, censor_df, censor_feature_cols, T_obs, n):
        """K-fold cross-fitted censoring model; each G_i uses a model not trained on i."""
        G_oof = np.ones(n)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        fit_cols = censor_feature_cols + ["T_obs", "C_indicator"]

        for tr_idx, val_idx in kf.split(range(n)):
            train_df = censor_df.iloc[tr_idx]
            val_df   = censor_df.iloc[val_idx]
            T_obs_val = T_obs[val_idx]
            try:
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(train_df[fit_cols], duration_col="T_obs",
                        event_col="C_indicator", fit_options={"max_steps": 50})
                G_fold = self._predict_G_sf(cph, val_df[censor_feature_cols],
                                            T_obs_val, len(val_idx))
                G_oof[val_idx] = G_fold
            except Exception:
                pass  # leave G_oof[val_idx] = 1.0

        return np.clip(G_oof, 0.05, 1.0)
