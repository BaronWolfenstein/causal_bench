"""Cohorting + subgroup-DISCOVERY capability on the real SMB 8B embeddings.

(1) hierarchy_check: does the embedding recover the KNOWN generative hierarchy (rare -> subtype
    -> stage)? cophenetic alignment + spectral decay + per-level silhouette.
(2) de-novo discovery: does UNSUPERVISED Ward clustering recover the clinical subgroups WITHOUT
    labels? AMI/ARI vs each label level + rare-cluster enrichment -- the "we discovered the
    subgroups, not just separated known ones" evidence.

RESULT (n=1600, real 8B baseline embeddings -> PCA-128):
  hierarchy: cophenetic Spearman +0.55 (flat baseline +0.53 -> SHALLOW, not a deep tree);
             effective rank 128/128 (spread); silhouette rare/subtype +0.20, stage -0.01.
  discovery: unsupervised Ward -- NO labels -- recovers rare ARI 1.00, subtype ARI 1.00; isolates
             a 100%-pure rare cohort (11% base rate) at k=8/16; STAGE degrades to ARI 0.16.
  => de-novo subgroup discovery works at COARSE-to-mid granularity, degrading at the finest level
     (the coarse-robust / fine-fragile gradient). Discovered cohorts feed the double-score causal
     stack (payoff_v9). Caveat: synthetic MEDS code rare/subtype distinctly by construction, so
     ARI 1.00 partly reflects an easy cohort; real-data subtlety is the open question.

Run (box, thread-capped): OMP_NUM_THREADS=4 ~/venv/bin/python cohort_discovery.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.expanduser("~/causal_bench"))
import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import gen_cohort, encode
import hierarchy_check as hc

QWEN = "/media/nvme_fast/models/SMB-v1_Qwen3-8b_multi-objective"
K = 128


def pca_fit(X, k):
    mean = X.mean(0); _, s, Vt = np.linalg.svd(X - mean, full_matrices=False)
    k = min(k, int((s > 1e-8).sum())); comps = Vt[:k]
    return (X - mean) @ comps.T


def main():
    meds, lab = gen_cohort.make_cohort(1600, 0.12, 1, u_conf=1.0)
    X, _, sids, _ = encode.encode(QWEN, meds, lab, gpu=0, batch_size=8, end_time_col="baseline_time")
    lab = lab.set_index("subject_id").loc[sids]
    rare = lab.is_rare.to_numpy(int)
    subtype = pd.factorize(lab.subtype)[0]
    stage = lab.stage.to_numpy(int)
    labels = np.column_stack([rare, subtype, stage]); names = ["rare", "subtype", "stage"]
    Z = pca_fit(X.astype(float), K)
    print(f"n={len(rare)} embed_dim={X.shape[1]} -> PCA-{Z.shape[1]}; rare={rare.mean():.2f}\n", flush=True)

    print("=== (1) HIERARCHY: does the embedding recover the known clinical hierarchy? ===", flush=True)
    print(hc.report(Z, labels, names, whiten=True), flush=True)

    print("\n=== (2) DE-NOVO DISCOVERY: unsupervised Ward clustering vs known labels ===", flush=True)
    from causal_bench.generative.whiten import zca_fit
    W = zca_fit(Z).transform(Z)
    print(f"{'level':>8} {'k':>3} {'ARI':>6} {'AMI':>6}  (no labels used to cluster)", flush=True)
    for i, name in enumerate(names):
        truth = labels[:, i]; k = len(set(truth))
        cl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(W)
        print(f"{name:>8} {k:>3} {adjusted_rand_score(truth, cl):>6.2f} {adjusted_mutual_info_score(truth, cl):>6.2f}", flush=True)
    for k in (8, 16):
        cl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(W)
        enr = max((rare[cl == c].mean() for c in set(cl) if (cl == c).sum() >= 10), default=0.0)
        print(f"  k={k}: best cluster rare-purity = {enr:.2f}  (base rate {rare.mean():.2f})", flush=True)


if __name__ == "__main__":
    main()
