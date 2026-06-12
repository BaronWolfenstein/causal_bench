from causal_bench.estimators.naive import NaiveEstimator
from causal_bench.estimators.kaplan_meier import KaplanMeierEstimator
from causal_bench.estimators.cox import CoxEstimator
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
from causal_bench.estimators.ltmle import LTMLEEstimator
from causal_bench.estimators.ipw import IPWEstimator
from causal_bench.estimators.overlap import OverlapEstimator
from causal_bench.estimators.aipw import AIPWEstimator
from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
from causal_bench.estimators.concrete_win_ratio import ConcreteWinRatioEstimator
from causal_bench.estimators.pointwise_rmst import PointwiseRMSTEstimator

ESTIMATOR_REGISTRY: dict = {
    "naive":              NaiveEstimator(),
    "km":                 KaplanMeierEstimator(),
    "cox":                CoxEstimator(),
    "tmle_ipcw":          TMLEIPCWEstimator(use_compliance=False),
    "tmle_ipcw_comply":   TMLEIPCWEstimator(use_compliance=True),
    "cox_l1":             CoxEstimator(include_L1=True),
    "ltmle":              LTMLEEstimator(),
    "ipw":                IPWEstimator(),
    "overlap":            OverlapEstimator(),
    "aipw":               AIPWEstimator(),
    "concrete_RMST":        ConcreteRMSTEstimator(),
    "concrete_RMST_strata": ConcreteRMSTEstimator(strata_cols=["W2", "W4"]),
    "concrete_WR_direct":  ConcreteWinRatioEstimator(method="direct"),
    "concrete_WR_plugin":  ConcreteWinRatioEstimator(method="plugin"),
    "rmst_k2":            PointwiseRMSTEstimator(n_grid=2),
    "rmst_k5":            PointwiseRMSTEstimator(n_grid=5),
    "rmst_k10":           PointwiseRMSTEstimator(n_grid=10),
    "rmst_k20":           PointwiseRMSTEstimator(n_grid=20),
}

MVP_ESTIMATORS = ["naive", "km", "cox", "tmle_ipcw", "tmle_ipcw_comply"]


def get_estimator(name: str):
    if name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown estimator '{name}'. Known: {list(ESTIMATOR_REGISTRY)}")
    return ESTIMATOR_REGISTRY[name]
