"""Real-8B causal payoff v6 -- continuous outcome, LINEAR estimators (well-specified).

y_cont is linear in confounders + HETEROGENEOUS treatment (much stronger in rare) + U,
so linear estimators are unbiased and the two mechanisms separate cleanly:

Panel A (u_conf=0): only bias = SUPPORT. Sparse rare -> linear T-learner extrapolates
  the common slope into the rare region (misses the -3 rare effect) -> biased ATE.
  FE-style structural augmentation (validated rare, TRUE outcomes) -> the model learns
  the rare slope -> bias drops. tau_cont is the individual causal effect => true ATE known.

Panel B (u_conf=1): only bias = CONFOUNDING (latent U). Linear OLS makes the omitted-
  variable-bias closed form EXACT, so QBA recovers the ORACLE (= linear fit with U),
  which here EQUALS the true ATE (linear DGP). Misspecified priors do not. n-independent.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.expanduser("~/causal_bench"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from causal_bench.generative.whiten import zca_fit
import gen_cohort, encode

QWEN = "/media/nvme_fast/models/SMB-v1_Qwen3-8b_multi-objective"
K = 64
CLIP = (0.05, 0.95)


def pca_fit(X, k):
    mean = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, (s > 1e-8).sum())
    comps = Vt[:k]
    return lambda Y: (Y - mean) @ comps.T


def encode_latent(meds, lab, ref_tf=None, ref_X=None):
    X, _, sids, _ = encode.encode(QWEN, meds, lab, gpu=0, batch_size=8, end_time_col="baseline_time")
    lab = lab.set_index("subject_id").loc[sids]
    tf = ref_tf if ref_tf is not None else pca_fit(X, K)
    zca = zca_fit(ref_tf(ref_X) if ref_tf is not None else tf(X))
    return zca.transform(tf(X)), lab, tf, X


def lin_tlearner_aipw(Zfit, Afit, Yfit, Zev, Aev, Yev):
    e = LogisticRegression(max_iter=3000, C=1.0).fit(Zfit, Afit)
    ee = np.clip(e.predict_proba(Zev)[:, 1], *CLIP)
    m1 = LinearRegression().fit(Zfit[Afit == 1], Yfit[Afit == 1])
    m0 = LinearRegression().fit(Zfit[Afit == 0], Yfit[Afit == 0])
    mu1, mu0 = m1.predict(Zev), m0.predict(Zev)
    psi = mu1 - mu0 + Aev * (Yev - mu1) / ee - (1 - Aev) * (Yev - mu0) / (1 - ee)
    return float(psi.mean())


def _ols(X, y):
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    s2 = (resid @ resid) / (len(y) - X.shape[1])
    return beta, np.sqrt(np.diag(XtXi) * s2)


def panel_A():
    print("\n===================== PANEL A: augmentation (support bias; u_conf=0, continuous) =====================")
    meds, lab = gen_cohort.make_cohort(1400, 0.12, 0, u_conf=0.0)
    Z, lab, tf, X = encode_latent(meds, lab)
    A = lab.A.to_numpy(float); Y = lab.y_cont.to_numpy(float); rare = lab.is_rare.to_numpy() == 1
    tau = lab.tau_cont.to_numpy(float)
    keep = ~rare.copy(); keep[np.where(rare)[0][:15]] = True
    Zk, Ak, Yk = Z[keep], A[keep], Y[keep]
    true = float(tau[keep].mean())
    base = lin_tlearner_aipw(Zk, Ak, Yk, Zk, Ak, Yk)
    aug_meds, aug_lab = gen_cohort.make_augmentation_cohort(400, seed=100, u_conf=0.0)
    Za, alab, _, _ = encode_latent(aug_meds, aug_lab, ref_tf=tf, ref_X=X)
    Aa, Ya = alab.A.to_numpy(float), alab.y_cont.to_numpy(float)
    aug = lin_tlearner_aipw(np.vstack([Zk, Za]), np.concatenate([Ak, Aa]),
                            np.concatenate([Yk, Ya]), Zk, Ak, Yk)
    print(f"  true ATE (sparse obs) : {true:+.3f}   (rare effect -3 vs common -1)")
    print(f"  sparse baseline       : {base:+.3f}   bias {base-true:+.3f}")
    print(f"  + FE-struct augment   : {aug:+.3f}   bias {aug-true:+.3f}")
    print(f"  |bias| before/after   : {abs(base-true):.3f} -> {abs(aug-true):.3f}")


def panel_B():
    print("\n===================== PANEL B: QBA (confounding bias; u_conf=1, linear) =====================")
    meds, lab = gen_cohort.make_cohort(1400, 0.12, 1, u_conf=1.0)
    Z, lab, tf, X = encode_latent(meds, lab)
    A = lab.A.to_numpy(float); Y = lab.y_cont.to_numpy(float); U = lab.frailty.to_numpy(float)
    true_ate = float(lab.tau_cont.mean())
    one = np.ones((len(A), 1))
    Xn = np.hstack([one, A[:, None], Z]); Xo = np.hstack([one, A[:, None], Z, U[:, None]])
    bn, sen = _ols(Xn, Y); bo, _ = _ols(Xo, Y)
    naive, se_naive, oracle, beta_U = bn[1], sen[1], bo[1], bo[-1]
    gamma = _ols(Xn, U)[0][1]
    rng = np.random.default_rng(0); S = 100_000

    def qba(bU_c, g_c, bU_sp=0.3, g_sd=0.02):
        bU = rng.triangular(bU_c - bU_sp, bU_c, bU_c + bU_sp, S)
        g = rng.normal(g_c, g_sd, S)
        syst = naive - bU * g
        return syst, syst - rng.normal(0, se_naive, S)
    syst, total = qba(beta_U, gamma)
    syst_mis, _ = qba(0.0, gamma, bU_sp=0.1)
    q = lambda a, p: float(np.percentile(a, p))
    print(f"  DGP true ATE          : {true_ate:+.3f}")
    print(f"  oracle (U measured)   : {oracle:+.3f}   <- QBA target (== true for linear DGP)")
    print(f"  naive (U omitted)     : {naive:+.3f}   bias {naive-true_ate:+.3f}  (SE {se_naive:.3f})")
    print(f"  QBA correct  syst med [2.5,97.5]: {q(syst,50):+.3f} [{q(syst,2.5):+.3f},{q(syst,97.5):+.3f}]")
    print(f"  QBA correct  total med [2.5,97.5]: {q(total,50):+.3f} [{q(total,2.5):+.3f},{q(total,97.5):+.3f}]")
    print(f"  QBA misspec (U harmless): {q(syst_mis,50):+.3f}  -> stays biased (motivates exp49)")
    print(f"  bias params: beta_U={beta_U:+.3f} gamma={gamma:+.3f}")


if __name__ == "__main__":
    panel_A()
    panel_B()
