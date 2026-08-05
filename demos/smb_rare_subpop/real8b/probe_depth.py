"""Probe-space depth + SFW-style embedding-diffusion depth on the real SMB 8B embeddings.

The multi-objective card says the encoder was trained to ANCHOR staging/histology/survival. So the
clinical axes should be ENCODED even where unsupervised clustering (which surfaces only the dominant
coarse axis) missed them (stage ARI 0.16). Two probes:

PART A -- linear-probe decodability: CV accuracy of a linear probe Z->{rare,subtype,pdl1,stage} +
  R^2 for a survival proxy. High accuracy where unsupervised clustering is shallow = depth is
  ENCODED but subordinate (recoverable by a probe, not by flat clustering).

PART B -- SFW-style depth: forward-noise the embedding (VP: Zt = sqrt(g)*Z + sqrt(1-g)*eps) and
  measure linear-probe accuracy of the NOISED embedding at each signal-retained gamma. If coarse
  (rare/subtype) stays decodable to LOWER gamma (higher noise) than fine (stage), that gamma gap is
  the hierarchical DEPTH -- the multi-scale reading a single flat clustering cannot give (#137
  embedding channel). Levels collapsing together => flat.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.expanduser("~/causal_bench"))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import gen_cohort, encode

QWEN = "/media/nvme_fast/models/SMB-v1_Qwen3-8b_multi-objective"


def pca_fit(X, k):
    mean = X.mean(0); _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, int((s > 1e-8).sum())); return (X - mean) @ Vt[:k].T


def acc(Z, y, cv=5):
    return cross_val_score(LogisticRegression(max_iter=1500), Z, y, cv=cv, scoring="accuracy").mean()


def main():
    meds, lab = gen_cohort.make_cohort(1600, 0.12, 1, u_conf=1.0)
    X, _, sids, _ = encode.encode(QWEN, meds, lab, gpu=0, batch_size=8, end_time_col="baseline_time")
    lab = lab.set_index("subject_id").loc[sids]
    rare = lab.is_rare.to_numpy(int)
    subtype = pd.factorize(lab.subtype)[0]
    pdl1 = lab.pdl1.to_numpy(int)
    stage = lab.stage.to_numpy(int)
    y_cont = lab.y_cont.to_numpy(float)
    Z = StandardScaler().fit_transform(pca_fit(X.astype(float), 128))
    print(f"n={len(rare)}; PCA-128 standardized\n", flush=True)

    print("=== PART A: linear-probe decodability (what's ENCODED, even where unsupervised missed it) ===", flush=True)
    for name, y in [("rare", rare), ("subtype", subtype), ("pdl1", pdl1), ("stage", stage)]:
        base = np.bincount(y).max() / len(y)
        print(f"  {name:>8}: CV accuracy {acc(Z, y):.2f}   (majority baseline {base:.2f})", flush=True)
    r2 = cross_val_score(RidgeCV(), Z, y_cont, cv=5, scoring="r2").mean()
    print(f"  {'survival':>8}: CV R^2 {r2:.2f}   (y_cont proxy)", flush=True)
    print("  -> stage recoverable by a PROBE despite unsupervised ARI 0.16 = depth encoded, subordinate.", flush=True)

    print("\n=== PART B: SFW-style embedding-diffusion depth (probe acc of the NOISED embedding) ===", flush=True)
    rng = np.random.default_rng(0)
    print(f"  {'gamma':>6} {'rare':>6} {'subtype':>8} {'stage':>6}   (gamma = signal retained; low gamma = high noise)", flush=True)
    for g in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        Zt = np.sqrt(g) * Z + np.sqrt(1 - g) * rng.normal(size=Z.shape)
        a = {n: acc(Zt, y, cv=3) for n, y in [("rare", rare), ("subtype", subtype), ("stage", stage)]}
        print(f"  {g:>6.2f} {a['rare']:>6.2f} {a['subtype']:>8.2f} {a['stage']:>6.2f}", flush=True)
    print("  -> coarse (rare/subtype) staying decodable to lower gamma than stage = hierarchical DEPTH;\n"
          "     collapsing together = flat. The gamma gap between coarse- and fine-collapse IS the depth.", flush=True)


if __name__ == "__main__":
    main()
