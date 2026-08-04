"""Quick win (#18 step 1): use causal_bench's REAL estimator machinery -- cross-fit
AIPW/TMLE with SuperLearner nuisances (Ridge + random-kitchen-sink; R-free) -- on the
embedding, instead of the hand-rolled OLS/RF that blew up in v1-v6.

Binary progression outcome (point.py / SuperLearner are built for bounded outcomes).
Test: does proper machinery recover the ORACLE truth on the embedding?
  naive  = adjust for embedding only (U omitted)  -> should stay biased by U
  oracle = adjust for embedding + U               -> SHOULD recover true ATE if the
           machinery handles the high-dim embedding adjustment set.
If oracle recovers truth, the estimator was the problem (clean payoff possible). If not,
the embedding-as-adjustment-set is fraught even with proper machinery -> the genuine gap.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.expanduser("~/causal_bench"))
import numpy as np
from causal_bench.generative.whiten import zca_fit
from causal_bench.super_learner import SuperLearner
import gen_cohort, encode

QWEN = "/media/nvme_fast/models/SMB-v1_Qwen3-8b_multi-objective"
K = 64
GCLIP = (0.05, 0.95)


def pca_fit(X, k):
    mean = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, (s > 1e-8).sum())
    comps = Vt[:k]
    return lambda Y: (Y - mean) @ comps.T


def sl_dr(W, A, Y, seed=0):
    """Cross-fit AIPW + TMLE via SuperLearner nuisances. Returns (aipw, tmle, se)."""
    from scipy.special import expit, logit
    n = len(A)
    Xaw = np.column_stack([A, W]); X1 = np.column_stack([np.ones(n), W]); X0 = np.column_stack([np.zeros(n), W])
    Qsl = SuperLearner(task="regression", n_folds=5, random_state=seed); Qsl.fit(Xaw, Y)
    Q1 = np.clip(Qsl.predict(X1), 1e-3, 1 - 1e-3); Q0 = np.clip(Qsl.predict(X0), 1e-3, 1 - 1e-3)
    gsl = SuperLearner(task="classification", n_folds=5, random_state=seed); gsl.fit(W, A)
    g = getattr(gsl, "oof_predictions_", None)
    g = np.clip(g if g is not None else gsl.predict_proba(W), *GCLIP)
    QA = A * Q1 + (1 - A) * Q0; H = A / g - (1 - A) / (1 - g)
    eif = Q1 - Q0 + H * (Y - QA)
    aipw = float(eif.mean()); se = float(eif.std(ddof=1) / np.sqrt(n))
    # one-step TMLE targeting
    denom = float(np.mean(H ** 2)); eps = np.clip(np.mean(H * (Y - QA)) / max(denom, 1e-10), -2, 2)
    Q1s = expit(logit(Q1) + eps / g); Q0s = expit(logit(Q0) - eps / (1 - g))
    tmle = float(np.mean(Q1s - Q0s))
    return aipw, tmle, se


def main():
    meds, lab = gen_cohort.make_cohort(1600, 0.12, 1, u_conf=1.0)
    X, _, sids, _ = encode.encode(QWEN, meds, lab, gpu=0, batch_size=8, end_time_col="baseline_time")
    lab = lab.set_index("subject_id").loc[sids]
    A = lab.A.to_numpy(float); Y = lab.Y.to_numpy(float); U = lab.frailty.to_numpy(float)
    true = float((lab.p_prog_treated - lab.p_prog_untreated).mean())
    tf = pca_fit(X, K); Z = zca_fit(tf(X)).transform(tf(X))
    print(f"n={len(A)} A_rate={A.mean():.2f} U_rate={U.mean():.2f} true_ATE={true:+.3f}  (binary progression)")

    na, nt, nse = sl_dr(Z, A, Y)
    oa, ot, ose = sl_dr(np.column_stack([Z, U]), A, Y)
    print("\n=== REAL machinery: cross-fit SuperLearner AIPW/TMLE on the embedding ===")
    print(f"  true ATE                        : {true:+.3f}")
    print(f"  naive  (embedding, U omitted)   : AIPW {na:+.3f} (SE {nse:.3f})   TMLE {nt:+.3f}   bias {na-true:+.3f}")
    print(f"  oracle (embedding + U measured) : AIPW {oa:+.3f} (SE {ose:.3f})   TMLE {ot:+.3f}   bias {oa-true:+.3f}")
    print("\n  -> if oracle bias ~0: machinery handles the embedding adjustment (clean payoff possible)")
    print("  -> if oracle still biased: embedding-as-adjustment-set is the genuine gap (positivity/SDR, task #18 step2)")


if __name__ == "__main__":
    main()
