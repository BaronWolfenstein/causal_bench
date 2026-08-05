"""Real-8B validation of the SDR-spec reductions (#206) on the ACTUAL SMB Qwen3-8B embeddings.

Extends payoff_v8 (prognostic-only) to the full method suite on the CONTINUOUS heterogeneous
outcome (tau_cont, stronger in the rare subgroup; rare are ~76% treated vs ~36% common -- a real
positivity trap), cross-fit (DML). Estimators come from causal_bench.validation.embedding_positivity.

RESULT (n=1600, true ATE -1.222, bias vs true):
    naive  +0.967   oracle +0.842      <- pathology: the full embedding attenuates ~70-80%
    prog   -0.011   double +0.024      <- the REGRESSION reductions recover
    sdr    +0.701   sdr_ato +0.606     <- the SIR/SAVE SDR FAILS on the nonlinear embedding

This INVERTS the synthetic-frontier result (where the SDR+ATO composition won on the linear DGP):
linear moment methods cannot find the confounding subspace of a nonlinear encoding. RKS-kernel and
learned-bottleneck salvages are explored in sibling scripts. Recommendation (real data): the
flexible regression-based double-score.

Run (box, thread-capped): OMP_NUM_THREADS=4 ~/venv/bin/python payoff_v9.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.expanduser("~/causal_bench"))
import numpy as np
from causal_bench.generative.whiten import zca_fit
from causal_bench.validation import embedding_positivity as ep
import gen_cohort, encode

QWEN = "/media/nvme_fast/models/SMB-v1_Qwen3-8b_multi-objective"
K = 64


def pca_fit(X, k):
    mean = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, int((s > 1e-8).sum()))
    comps = Vt[:k]
    return lambda Z: (Z - mean) @ comps.T


def suite(Z, A, Y, U, true, tag):
    m = {
        "naive":   ep._crossfit_ic(Z, A, Y, ep._fit_identity, n_folds=5)[0] - true,
        "oracle":  ep._crossfit_ic(np.column_stack([Z, U]), A, Y, ep._fit_identity, n_folds=5)[0] - true,
        "prog":    ep._crossfit_ic(Z, A, Y, ep._fit_prognostic_map, n_folds=5)[0] - true,
        "double":  ep._crossfit_ic(Z, A, Y, ep._fit_double_score_map, n_folds=5)[0] - true,
        "sdr":     ep._crossfit_ic(Z, A, Y, ep._fit_sdr_map, n_folds=5)[0] - true,
        "sdr_ato": ep._crossfit_ic(Z, A, Y, ep._fit_sdr_map, n_folds=5, ato=True)[0] - true,
    }
    print(f"[{tag}] bias vs true={true:+.3f}:  " + "  ".join(f"{k}={v:+.3f}" for k, v in m.items()), flush=True)
    return m


def main():
    meds, lab = gen_cohort.make_cohort(1600, 0.12, 1, u_conf=1.0)
    X, _, sids, _ = encode.encode(QWEN, meds, lab, gpu=0, batch_size=8, end_time_col="baseline_time")
    lab = lab.set_index("subject_id").loc[sids]
    A = lab.A.to_numpy(float)
    Y = lab.y_cont.to_numpy(float)              # continuous heterogeneous outcome
    U = lab.frailty.to_numpy(float)             # unmeasured confounder (not emitted -> not in embedding)
    true = float(lab.tau_cont.mean())
    tf = pca_fit(X, K)
    Z = zca_fit(tf(X)).transform(tf(X))
    print(f"n={len(A)} A_rate={A.mean():.2f} rare={lab.is_rare.mean():.2f} "
          f"rare_A={lab[lab.is_rare == 1].A.mean():.2f} common_A={lab[lab.is_rare == 0].A.mean():.2f} "
          f"true_ATE={true:+.3f}  (real 8B, continuous, cross-fit DML)", flush=True)
    print("\n=== SDR-spec method suite on REAL 8B embeddings (bias vs true) ===", flush=True)
    suite(Z, A, Y, U, true, "real-8B")
    print("  -> regression reductions (prog/double) recover; the linear SIR/SAVE SDR fails on the "
          "nonlinear embedding. The double-score is the real-data recommendation.", flush=True)


if __name__ == "__main__":
    main()
