from causal_bench.estimators.naive import NaiveEstimator
from causal_bench.estimators.kaplan_meier import KaplanMeierEstimator
from causal_bench.estimators.cox import CoxEstimator
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
from causal_bench.estimators.ltmle import LTMLEEstimator
from causal_bench.estimators.ipw import IPWEstimator
from causal_bench.estimators.overlap import OverlapEstimator

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
}

MVP_ESTIMATORS = ["naive", "km", "cox", "tmle_ipcw", "tmle_ipcw_comply"]


def get_estimator(name: str):
    if name not in ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown estimator '{name}'. Known: {list(ESTIMATOR_REGISTRY)}")
    return ESTIMATOR_REGISTRY[name]
