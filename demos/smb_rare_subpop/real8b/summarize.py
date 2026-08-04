"""Summarize the ablation: multi-objective vs curriculum across n, both poolings.

Reads out/results.jsonl (one record per n,seed) and prints a decision table +
writes out/ablation_summary.png. Aggregates the two backbones of each regime for
per-model metrics; regime-level AGL is already cross-backbone.
"""
from __future__ import annotations
import json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REGIME_MODELS = {"multi_objective": ("qwen_mo", "llama_mo"),
                 "curriculum": ("qwen_c", "llama_c")}


def load(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return recs


def agg(recs, pool):
    """-> data[regime][metric][n] = list over seeds."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        n = r["n"]
        for regime, (ka, kb) in REGIME_MODELS.items():
            if ka not in r["models"] or kb not in r["models"]:
                continue
            ma, mb = r["models"][ka][pool], r["models"][kb][pool]
            avg = lambda key: np.mean([ma[key], mb[key]])
            data[regime]["fewshot_fast_k10"][n].append(avg("fewshot_fast_k10"))
            data[regime]["fewshot_fast_k25"][n].append(avg("fewshot_fast_k25"))
            data[regime]["fullshot_fast"][n].append(avg("fullshot_fast_auc"))
            data[regime]["separability"][n].append(avg("separability_auc"))
            data[regime]["sigreg_T_rare"][n].append(avg("sigreg_T_rare"))
            if regime in r.get("regimes", {}):
                data[regime]["agl_R2_xbb"][n].append(r["regimes"][regime][pool]["agl_R2_crossbackbone"])
    return data


def table(recs, pool):
    data = agg(recs, pool)
    ns = sorted({r["n"] for r in recs})
    metrics = ["fewshot_fast_k10", "fewshot_fast_k25", "fullshot_fast",
               "separability", "agl_R2_xbb", "sigreg_T_rare"]
    higher_better = {"fewshot_fast_k10": True, "fewshot_fast_k25": True,
                     "fullshot_fast": True, "separability": True,
                     "agl_R2_xbb": True, "sigreg_T_rare": False}
    print(f"\n=== pooling={pool} ===")
    hdr = f"{'metric':<22}{'n':>6}  {'multi_obj':>12}  {'curriculum':>12}  winner"
    print(hdr); print("-" * len(hdr))
    flips = {}
    for m in metrics:
        prev = None
        for n in ns:
            mo = data["multi_objective"][m].get(n, [])
            cu = data["curriculum"][m].get(n, [])
            if not mo or not cu:
                continue
            mo_m, cu_m = np.nanmean(mo), np.nanmean(cu)
            if np.isnan(mo_m) or np.isnan(cu_m):
                win = "n/a"
            else:
                win = "multi_obj" if (mo_m > cu_m) == higher_better[m] else "curriculum"
                if prev and prev != win:
                    flips.setdefault(m, []).append((prev, win, n))
                prev = win
            print(f"{m:<22}{n:>6}  {mo_m:>12.3f}  {cu_m:>12.3f}  {win}")
        print()
    if flips:
        print("!! RANKING FLIPS WITH n:")
        for m, fs in flips.items():
            for a, b, n in fs:
                print(f"   {m}: {a} -> {b} at n={n}")
    else:
        print("No ranking flips across n (ranking stable).")
    return data, ns


def plot(recs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("plot skipped:", e); return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [("fewshot_fast_k10", "fast_prog few-shot AUC k=10 (dynamics, DISCRIMINATOR)", True),
              ("fewshot_fast_k25", "fast_prog few-shot AUC k=25 (dynamics)", True),
              ("agl_R2_xbb", "cross-backbone AGL R^2 (guard health)", True),
              ("sigreg_T_rare", "Epps-Pulley T, rare mode (lower=Gaussian)", False)]
    colors = {"multi_objective": "C0", "curriculum": "C1"}
    for ax, (metric, title, _) in zip(axes.flat, panels):
        for pool, ls in (("mean", "-"), ("last", "--")):
            data = agg(recs, pool)
            for regime in REGIME_MODELS:
                d = data[regime][metric]
                if not d:
                    continue
                ns = sorted(d)
                ys = [np.nanmean(d[n]) for n in ns]
                ax.plot(ns, ys, ls, color=colors[regime], marker="o",
                        label=f"{regime.split('_')[0]}/{pool}")
        ax.set_title(title, fontsize=10); ax.set_xlabel("n patients"); ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("SMB encoder ablation: multi-objective vs curriculum (cross-backbone pairs)")
    fig.tight_layout()
    out = os.path.join(HERE, "out", "ablation_summary.png")
    fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "results.jsonl")
    recs = load(path)
    print(f"{len(recs)} records; n values:", sorted({r['n'] for r in recs}))
    for pool in ("mean", "last"):
        table(recs, pool)
    plot(recs)
