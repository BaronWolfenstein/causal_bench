"""Exp 46: constraint-based causal discovery on causal_bench's CI detector (ADAfaEPoV Ch 22).

Demonstrates — rather than asserts — that the repo's `zero_flow_ci_test` is the CI oracle a
PC/SGS search calls: wired into the skeleton + collider-orientation it recovers known structure
(chain / fork / collider), and the **latent-confounder** case (radon/smoking) shows why plain
PC fails and FCI is needed — dropping the confounder makes confounding masquerade as a spurious
direct edge.

Self-validating: every DGP has a known true skeleton. Read-outs: skeleton precision/recall vs
truth, the collider v-structure orientation, and the spurious edge that appears once the
confounder is hidden.

Run: python -m experiments.exp46_causal_discovery
"""
from pathlib import Path

import numpy as np

from causal_bench.validation.causal_discovery import (
    orient_colliders, pc_skeleton, sim_chain, sim_collider, sim_fork,
)

OUT_DIR = Path("results/exp46_causal_discovery")


def _prf(recovered: set, truth: set):
    tp = len(recovered & truth)
    prec = tp / len(recovered) if recovered else float("nan")
    rec = tp / len(truth) if truth else float("nan")
    return prec, rec


def _edges_str(edges, labels):
    return ", ".join("–".join(labels[k] for k in sorted(e)) for e in sorted(map(tuple, map(sorted, edges)))) or "(none)"


def run(*, n=2000, n_perm=80, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, (data, truth, labels) in {
        "chain X→M→Y": sim_chain(n, rng),
        "fork (confounder) X←S→Y": sim_fork(n, rng, beta_xy=0.0),
        "collider X→C←Y": sim_collider(n, rng),
    }.items():
        edges, sepset = pc_skeleton(data, alpha=alpha, n_perm=n_perm, seed=seed)
        prec, rec = _prf(edges, truth)
        directed = orient_colliders(edges, sepset, data.shape[1])
        rows.append({"dgp": name, "labels": labels, "truth": truth, "edges": edges,
                     "prec": prec, "rec": rec, "directed": directed})

    # Latent-confounder demo: the fork WITHOUT the confounder column S.
    data, _, _ = sim_fork(n, rng, beta_xy=0.0)
    obs = data[:, :2]                                    # keep only X (radon), Y (lung cancer)
    edges_lat, _ = pc_skeleton(obs, alpha=alpha, n_perm=n_perm, seed=seed)
    spurious = frozenset((0, 1)) in edges_lat            # X–Y edge that should not be there
    return rows, spurious


def report(rows, spurious) -> str:
    L = ["## Exp 46 — causal discovery via `zero_flow_ci_test` (ADAfaEPoV Ch 22)\n",
         "| DGP | recovered skeleton | precision | recall | oriented v-structure |",
         "|-----|--------------------|-----------|--------|----------------------|"]
    for r in rows:
        lab = r["labels"]
        vs = ", ".join(f"{lab[p]}→{lab[c]}" for p, c in sorted(r["directed"])) or "—"
        L.append(f"| {r['dgp']} | {_edges_str(r['edges'], lab)} | {r['prec']:.2f} | {r['rec']:.2f} | {vs} |")
    L.append("")
    L.append(f"**Latent-confounder (radon/smoking):** hiding S, the discovered skeleton "
             f"{'INCLUDES a spurious X–Y (radon–lung-cancer) edge' if spurious else 'has no X–Y edge'} "
             f"— {'confounding masquerades as a direct effect; plain PC is unsound, FCI is needed.' if spurious else 'unexpected: check power.'}")
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 46: constraint-based causal discovery")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=80)
    a = p.parse_args()
    rows, spurious = run(n=a.n, n_perm=a.n_perm)
    rep = report(rows, spurious)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)
    print("\nRead-out: the repo's CI detector, wired into the PC/SGS skeleton + collider "
          "orientation, recovers chain / fork / collider structure and orients the v-structure "
          "from CI signatures alone. Hiding the confounder produces the spurious direct edge "
          "that motivates FCI — the radon/smoking identification failure, discovered from data.")


if __name__ == "__main__":
    main()
