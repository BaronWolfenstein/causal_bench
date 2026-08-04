"""Drive the SMB encoder ablation for one cohort size n.

Steps: generate cohort -> encode through the selected encoders (parallel, one GPU
each) -> compute per-model metrics (both poolings) + regime-level cross-backbone
AGL. Appends a JSON record per (n, seed) to <outdir>/results.jsonl.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np
import pandas as pd
import gen_cohort, metrics

MODELS = {
    "qwen_mo":  ("/media/nvme_fast/models/SMB-v1_Qwen3-8b_multi-objective", 0),
    "llama_mo": ("/media/nvme_fast/models/SMB-v1_Llama3-8b_multi-objective", 1),
    "qwen_c":   ("/media/nvme_fast/models/SMB-v1_Qwen3-8b_curriculum", 2),
    "llama_c":  ("/media/nvme_fast/models/SMB-v1_Llama3-8b_curriculum", 3),
}
REGIMES = {"multi_objective": ("qwen_mo", "llama_mo"), "curriculum": ("qwen_c", "llama_c")}
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--rare_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of model keys; default all")
    ap.add_argument("--outdir", default=os.path.join(HERE, "out"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--ambiguity", type=float, default=0.0)
    ap.add_argument("--results", default="results.jsonl")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    keys = [k for k in (a.only.split(",") if a.only else MODELS) if k in MODELS]

    # ---- generate cohort ----
    meds, labels = gen_cohort.make_cohort(a.n, a.rare_frac, a.seed, structured=True,
                                          ambiguity=a.ambiguity)
    mp = os.path.join(a.outdir, f"meds_n{a.n}_s{a.seed}.parquet")
    lp = os.path.join(a.outdir, f"labels_n{a.n}_s{a.seed}.parquet")
    meds.to_parquet(mp); labels.to_parquet(lp)
    print(f"[gen] n={a.n} rare={labels.is_rare.mean():.3f} "
          f"A rare/common={labels[labels.is_rare==1].A.mean():.2f}/{labels[labels.is_rare==0].A.mean():.2f} "
          f"codes={meds.code.nunique()}", flush=True)

    # ---- encode (parallel, one GPU per model) ----
    procs = {}
    for k in keys:
        path, gpu = MODELS[k]
        outpx = os.path.join(a.outdir, f"emb_{k}_n{a.n}_s{a.seed}")
        cmd = [sys.executable, os.path.join(HERE, "encode.py"),
               "--model_path", path, "--gpu", str(gpu), "--meds", mp, "--labels", lp,
               "--out_prefix", outpx, "--batch_size", str(a.batch_size)]
        log = open(outpx + ".log", "w")
        procs[k] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), outpx, log)
        print(f"[encode] launched {k} on gpu{gpu}", flush=True)
    embs = {}
    for k, (p, outpx, log) in procs.items():
        rc = p.wait(); log.close()
        if rc != 0:
            print(f"[encode] {k} FAILED rc={rc}; tail:", flush=True)
            print(open(outpx + ".log").read()[-1500:]); sys.exit(1)
        embs[k] = np.load(outpx + ".npz", allow_pickle=True)
        print(f"[encode] {k} done {embs[k]['mean'].shape}", flush=True)

    # order labels to match encoder subject_id order
    sid_order = list(embs[keys[0]]["subject_id"])
    labels = labels.set_index("subject_id").loc[sid_order].reset_index()

    # ---- metrics (both poolings) ----
    rec = {"n": a.n, "seed": a.seed, "ambiguity": a.ambiguity,
           "rare_frac": float(labels.is_rare.mean()),
           "true_ate": float((labels.p_prog_treated - labels.p_prog_untreated).mean()),
           "models": {}, "regimes": {}}
    for pool in ("mean", "last"):
        for k in keys:
            m = metrics.per_model_metrics(embs[k][pool], labels, seed=a.seed)
            rec["models"].setdefault(k, {})[pool] = m
            print(f"[metric] {k}/{pool}: sep={m['separability_auc']:.3f} "
                  f"fast_k10={m['fewshot_fast_k10']:.3f} fast_k25={m['fewshot_fast_k25']:.3f} "
                  f"fast_full={m['fullshot_fast_auc']:.3f} Trare={m['sigreg_T_rare']:.2f}", flush=True)
        for rname, (ka, kb) in REGIMES.items():
            if ka in embs and kb in embs:
                g = metrics.regime_agl(embs[ka][pool], embs[kb][pool], labels, seed=a.seed)
                rec["regimes"].setdefault(rname, {})[pool] = g
                print(f"[agl] {rname}/{pool}: R2={g['agl_R2']:.3f} "
                      f"R2_xbb={g['agl_R2_crossbackbone']:.3f} slope={g['agl_slope']:.2f}", flush=True)

    with open(os.path.join(a.outdir, a.results), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[done] n={a.n} seed={a.seed} appended to results.jsonl", flush=True)


if __name__ == "__main__":
    main()
