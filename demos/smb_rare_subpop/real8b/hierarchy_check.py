"""MEDS-hierarchy diagnostics on patient embeddings — a direct, SFW-free alternative.

The SFW/RHM route infers an *unknown latent* tree via tree-reconstruction-vs-noise; that is the
right tool when the hierarchy is unknown (synthetic RHM/PCFG). For MEDS the hierarchy is *known*
— either the generative structure of a synthetic cohort (rare → subtype → stage) or, on real
data, a clinical ontology (LOINC/SNOMED). So we check the embedding directly against a supplied
ground-truth tree instead of reconstructing one. Three checks:

  1. cophenetic alignment — does the embedding's Ward dendrogram recover the ground-truth
     hierarchical distance (patients sharing more coarse levels are closer)? Spearman/Pearson
     of the two cophenetic distance vectors. This is the headline hierarchy score.
  2. spectral decay — eigen-spectrum of the embedding covariance: effective rank (participation
     ratio) and top-eigenvalue share. Deep hierarchical structure → heavy-tailed spectrum / low
     effective rank.
  3. per-level separability — silhouette of the embedding under each coarse label, i.e. does the
     embedding actually encode each hierarchy level.

Reuse: scipy.cluster.hierarchy (linkage/cophenet), sklearn silhouette, and — matching the FE /
payoff pipeline — causal_bench.generative.whiten.zca_fit for embedding prep. The ground-truth
tree is pluggable: `hierarchy_distance` takes a (n, L) coarse→fine label matrix here; on real data
pass a distance derived from a clinical ontology (e.g. scientific-graph-agent's kg/ontology once
the LOINC/SNOMED bolt-on lands) with the same contract.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score


def hierarchy_distance(labels: np.ndarray) -> np.ndarray:
    """Condensed pairwise ground-truth tree distance from a (n, L) coarse→fine NESTED label
    matrix. Distance(i, j) = L − (length of shared leading prefix): two patients that already
    differ at the coarsest level are maximally far; sharing more nested levels brings them
    closer. Returns the condensed (pdist-style) vector so it aligns with cophenet output."""
    labels = np.asarray(labels)
    n, L = labels.shape
    eq = labels[:, None, :] == labels[None, :, :]            # (n, n, L)
    # shared prefix length = number of leading levels that match (stop at first mismatch)
    prefix = np.cumprod(eq, axis=2).sum(axis=2)              # (n, n)
    dist = (L - prefix).astype(float)
    iu = np.triu_indices(n, k=1)
    return dist[iu]


def cophenetic_alignment(Z: np.ndarray, labels: np.ndarray, method: str = "ward"):
    """Correlation between the embedding's Ward-dendrogram cophenetic distances and the
    ground-truth hierarchy distances. High → the embedding recovers the known hierarchy."""
    link = linkage(Z, method=method)
    _, coph = cophenet(link, pdist(Z))                       # condensed cophenetic distances
    d_true = hierarchy_distance(labels)
    pear = float(np.corrcoef(coph, d_true)[0, 1])
    spear = float(spearmanr(coph, d_true).correlation)
    # baseline: raw embedding-distance vs the tree (how much the dendrogram adds over flat geometry)
    flat = float(spearmanr(pdist(Z), d_true).correlation)
    return {"cophenetic_pearson": pear, "cophenetic_spearman": spear, "flat_spearman": flat}


def spectral_decay(Z: np.ndarray):
    """Eigen-spectrum of the embedding covariance: participation-ratio effective rank and the
    top-eigenvalue share. Low effective rank / high top share → concentrated hierarchical
    structure rather than an isotropic blob."""
    C = np.cov(Z - Z.mean(0), rowvar=False)
    vals = np.sort(np.linalg.eigvalsh(C))[::-1]
    vals = vals[vals > 1e-12]
    eff_rank = float((vals.sum() ** 2) / (vals ** 2).sum())  # participation ratio
    return {"effective_rank": eff_rank, "ambient_dim": int(Z.shape[1]),
            "top1_share": float(vals[0] / vals.sum()),
            "top5_share": float(vals[:5].sum() / vals.sum())}


def per_level_separability(Z: np.ndarray, labels: np.ndarray, names):
    """Silhouette of the embedding under each coarse label — does it encode each level?"""
    out = {}
    for i, name in enumerate(names):
        lab = labels[:, i]
        out[name] = float(silhouette_score(Z, lab)) if len(np.unique(lab)) > 1 else float("nan")
    return out


def report(Z, labels, names, whiten: bool = True) -> str:
    """Run all three checks and format a short report. `labels` is (n, L) coarse→fine, `names`
    are the level names. `whiten` applies ZCA (matching the payoff/FE embedding prep)."""
    if whiten:
        from causal_bench.generative.whiten import zca_fit
        Z = zca_fit(Z).transform(Z)
    coph = cophenetic_alignment(Z, labels)
    spec = spectral_decay(Z)
    sep = per_level_separability(Z, labels, names)
    L = ["MEDS-hierarchy diagnostics (embedding vs known generative hierarchy)",
         "",
         "  cophenetic alignment (embedding dendrogram vs ground-truth tree):",
         f"    Spearman {coph['cophenetic_spearman']:+.3f}  Pearson {coph['cophenetic_pearson']:+.3f}"
         f"   (flat-geometry baseline {coph['flat_spearman']:+.3f})",
         "  spectral decay:",
         f"    effective rank {spec['effective_rank']:.1f} / {spec['ambient_dim']}"
         f"   top-1 share {spec['top1_share']:.2f}  top-5 share {spec['top5_share']:.2f}",
         "  per-level separability (silhouette):",
         "    " + "  ".join(f"{k}={v:+.2f}" for k, v in sep.items()),
         "",
         "  Read: high cophenetic Spearman (above the flat baseline) = the embedding recovers the",
         "  KNOWN hierarchy; low effective rank + level separability confirm exploitable coarse",
         "  structure. This is the direct check the SFW transition is overkill for when the tree is",
         "  known; SFW's niche is the unknown-latent-hierarchy (synthetic RHM/PCFG) setting."]
    return "\n".join(L)
