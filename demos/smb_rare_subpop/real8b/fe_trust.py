"""Settle FE trust on REAL 8B oncology embeddings (the arch-2 gate).

Does the Flow Expander, trained on the frozen encoder's real embedding space and
steered to the rare region, produce TRUSTWORTHY samples? Three checks:

  1. RARE-TARGETING  -- guided samples land in the rare region (probe P(rare)),
                        and clearly more than unguided (steering actually works).
  2. TYPICALITY      -- guided samples are on-manifold: their kNN distance to real
                        embeddings is comparable to real-rare's own kNN distance
                        (not off in empty space).
  3. MEMORIZE-vs-INTERPOLATE -- guided samples are NOT copies: min distance to any
                        real embedding is comparable to real-real nearest-neighbor
                        spacing (interpolated), not ~0 (memorized). This is the
                        go/no-go for the whole embedding-space bet.

High-dim handling: 4096-dim with ~1200 samples is rank-deficient for ZCA, so we
PCA-reduce to k dims first (latent-sample-complexity supports a compact latent),
whiten there, diffuse there, and evaluate there.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.expanduser("~/causal_bench"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from causal_bench.generative.whiten import zca_fit
from causal_bench.generative.vpsde import Schedule
from causal_bench.generative.score_net import (
    ScoreMLP, make_optimizer, train_score, make_torch_score_fn)
from causal_bench.generative.flow_expander import flow_expand

OUT = os.path.expanduser("~/smb_ablation/out")
K = 128            # PCA latent dim
SEED = 0


def load(n=1200, s=0, model="qwen_mo"):
    z = np.load(f"{OUT}/emb_{model}_n{n}_s{s}.npz", allow_pickle=True)
    lab = pd.read_parquet(f"{OUT}/labels_n{n}_s{s}.parquet").set_index("subject_id")
    sids = list(z["subject_id"])
    y = lab.loc[sids, "is_rare"].to_numpy()
    return z["mean"].astype(float), y


def pca_reduce(X, k):
    mean = X.mean(0, keepdims=True)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, (s > 1e-8).sum())
    return U[:, :k] * s[:k]         # PCA scores [n,k]


def knn_dist(query, ref, k=5, exclude_self=False):
    kk = k + (1 if exclude_self else 0)
    nn = NearestNeighbors(n_neighbors=kk).fit(ref)
    d, _ = nn.kneighbors(query)
    return d[:, (1 if exclude_self else 0):].mean(1)   # mean over k neighbors


def main():
    import torch
    X, y = load()
    print(f"real embeddings: {X.shape}  rare={y.mean():.3f} ({int(y.sum())} rare)")
    S = pca_reduce(X, K)
    zca = zca_fit(S)
    Sw = zca.transform(S)                       # whitened latent [n,K]
    dim = Sw.shape[1]

    # score net on the real whitened latent
    sch = Schedule(n_steps=100)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = ScoreMLP(dim, hidden=256)
    train_score(model, Sw, sch, opt=make_optimizer(model, lr=1e-3),
                epochs=300, rng=rng, device="cuda")
    score_fn = make_torch_score_fn(model, sch, device="cuda")

    # guardrail: did PCA reduction PRESERVE the rare signal? (plan sec 0c)
    from sklearn.model_selection import cross_val_score
    auc = cross_val_score(LogisticRegression(max_iter=3000), Sw, y, cv=5,
                          scoring="roc_auc").mean()
    print(f"rare separability in PCA-{K} whitened latent: AUC={auc:.3f}  "
          f"(PCA preserved rare signal if high)")

    # rareness verifier: grad log P(rare|x) = (1 - sigmoid(w.x+b)) * w  -> pushes to rare
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Sw, y)
    w = clf.coef_[0]; b = float(clf.intercept_[0])

    def grad_log_v(x, t):
        x = np.asarray(x, float)
        p = 1.0 / (1.0 + np.exp(-(x @ w + b)))
        return ((1.0 - p)[:, None]) * w[None, :]

    n_gen = 300
    unguided = flow_expand(score_fn, None, sch, np.random.default_rng(1),
                           n=n_gen, dim=dim, alpha=0.0, lam=0.0)
    guided = flow_expand(score_fn, grad_log_v, sch, np.random.default_rng(1),
                         n=n_gen, dim=dim, alpha=0.3, lam=6.0, norm_cap=3.0)

    # ---- evaluate in the whitened latent ----
    real_rare = Sw[y == 1]
    print("\n=== 1. RARE-TARGETING (probe P(rare)) ===")
    for tag, G in (("unguided", unguided), ("guided", guided)):
        p = clf.predict_proba(G)[:, 1]
        print(f"  {tag:8s}: mean P(rare)={p.mean():.3f}  frac>0.5={np.mean(p>0.5):.2f}")

    print("\n=== 2. TYPICALITY (kNN-5 distance to real; on-manifold if ~ real-rare's own) ===")
    real_rare_knn = knn_dist(real_rare, Sw, k=5, exclude_self=True)
    print(f"  real-rare      : {real_rare_knn.mean():.2f} +/- {real_rare_knn.std():.2f}")
    for tag, G in (("unguided", unguided), ("guided", guided)):
        gk = knn_dist(G, Sw, k=5)
        print(f"  {tag:8s} gen   : {gk.mean():.2f} +/- {gk.std():.2f}   "
              f"(ratio to real-rare = {gk.mean()/real_rare_knn.mean():.2f})")

    print("\n=== 3. MEMORIZE-vs-INTERPOLATE (min dist to any real; copy if ~0) ===")
    real_nn = knn_dist(Sw, Sw, k=1, exclude_self=True)          # real-real NN spacing
    print(f"  real-real NN spacing : {real_nn.mean():.2f} +/- {real_nn.std():.2f}")
    for tag, G in (("unguided", unguided), ("guided", guided)):
        gmin = knn_dist(G, Sw, k=1)
        ratio = gmin.mean() / real_nn.mean()
        verdict = "MEMORIZED" if ratio < 0.3 else "interpolated"
        print(f"  {tag:8s} gen min: {gmin.mean():.2f} +/- {gmin.std():.2f}   "
              f"(ratio={ratio:.2f} -> {verdict})")

    print("\n=== VERDICT ===")
    pg = clf.predict_proba(guided)[:, 1]
    gk = knn_dist(guided, Sw, k=5); gmin = knn_dist(guided, Sw, k=1)
    targets = pg.mean() > 0.7
    onman = gk.mean() / real_rare_knn.mean() < 2.0
    interp = gmin.mean() / real_nn.mean() > 0.3
    print(f"  targets rare : {targets}  ({pg.mean():.2f})")
    print(f"  on-manifold  : {onman}  (typicality ratio {gk.mean()/real_rare_knn.mean():.2f})")
    print(f"  interpolated : {interp}  (memorize ratio {gmin.mean()/real_nn.mean():.2f})")
    print(f"  --> FE {'TRUSTWORTHY' if (targets and onman and interp) else 'NOT trustworthy (see above)'}")


if __name__ == "__main__":
    main()
