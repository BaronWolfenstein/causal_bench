"""Subgroup-level hierarchical borrowing for CED cross-registry analysis.

Implements the middle tier of the three-level borrowing scheme:
  discover discrete physiological subgroups from the main cohort (DR-Learner
  CATE + embedding clustering), assign rare-cohort patients to those subgroups
  via embedding proximity, then run a robust MAP prior within each subgroup
  across registries.

This is the CMS communication layer: subgroups are named physiological
categories (e.g. rapid-deteriorators, stable-decliners), each with an
interpretable ESS and a credible interval. Patient-level borrowing (continuous
similarity weights) is the internal precision engine; subgroup-level is what
gets reported.

Subgroup assignment uses KNN on embeddings by default. KNN directly respects
the embedding geometry without assuming linearity in the boundary — preferred
when subgroups may be non-convex in embedding space. Logistic regression
(multiclass one-vs-rest) is provided as an option for settings where linear
separability holds and interpretable coefficients are wanted.

For the OC simulation (exp19), CATE estimation is bypassed when true per-patient
CATEs are available in the DataFrame ("cate" column). For production on real
registry data, a DR-Learner is fitted on the main cohort.

References:
  Kennedy (2023). Towards optimal doubly-robust estimation of heterogeneous
    causal effects. EJS.
  Schmidli et al. (2014). Robust meta-analytic-predictive priors. Biometrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from causal_bench.estimators.hierarchical import (
    BorrowingResult,
    RegistrySummary,
    compute_ess,
    robust_map_posterior,
    summarise_registry,
)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SubgroupModel:
    """Fitted subgroup scheme derived from the main cohort.

    Contains everything needed to assign patients from any registry to the
    main-cohort-derived subgroups.
    """
    n_subgroups: int
    cluster_centers: np.ndarray        # (n_subgroups × n_dims)
    subgroup_labels: np.ndarray        # integer labels 0..n_subgroups-1 for main cohort
    subgroup_names: list[str]          # human-readable label per subgroup
    cate_by_subgroup: np.ndarray       # mean CATE per subgroup (for interpretation)
    cate_sd_by_subgroup: np.ndarray    # SD of CATE per subgroup
    n_by_subgroup: np.ndarray          # main cohort count per subgroup
    classifier_type: str               # "knn" or "logistic"
    # stored classifier for out-of-sample assignment
    _classifier: object = field(repr=False, default=None)


@dataclass
class SubgroupBorrowingResult:
    """Borrowing result for one subgroup in one target registry."""
    subgroup_idx: int
    subgroup_name: str
    borrowing: BorrowingResult
    n_target_in_subgroup: int
    cate_main_mean: float      # mean CATE in this subgroup (main cohort)
    cate_main_sd: float


# ─── CATE estimation ─────────────────────────────────────────────────────────

def estimate_cates(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_folds: int = 5,
) -> np.ndarray:
    """Estimate per-patient CATEs via cross-fit DR-Learner.

    If the DataFrame has a "cate" column (OC simulation with known truth),
    returns those directly — no fitting needed.

    Otherwise fits a cross-fit DR-Learner:
      1. Cross-fit propensity π̂(W) and outcome μ̂(a, W) using embeddings as W.
      2. Compute pseudo-outcomes:
           Ỹ = μ̂(1,W) − μ̂(0,W)
             + A·(Y − μ̂(1,W)) / π̂(W)
             − (1−A)·(Y − μ̂(0,W)) / (1 − π̂(W))
      3. Regress Ỹ on embeddings (Ridge) to get smooth CATE surface.

    Parameters
    ----------
    df : DataFrame with columns Y (outcome), A (treatment).
    embeddings : (n × d) embedding matrix aligned with df rows.
    n_folds : cross-fitting folds.

    Returns
    -------
    cate_hat : (n,) array of per-patient CATE estimates.
    """
    if "cate" in df.columns:
        return df["cate"].values.copy()

    Y = df["Y"].values.astype(float)
    A = df["A"].values.astype(float)
    n = len(df)

    pseudo = np.full(n, np.nan)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    for train_idx, val_idx in folds.split(embeddings, A.astype(int)):
        X_tr, X_va = embeddings[train_idx], embeddings[val_idx]
        Y_tr, Y_va = Y[train_idx], Y[val_idx]
        A_tr, A_va = A[train_idx], A[val_idx]

        # Propensity
        pi_model = LogisticRegression(max_iter=500, random_state=0)
        pi_model.fit(X_tr, A_tr)
        pi_hat = np.clip(pi_model.predict_proba(X_va)[:, 1], 0.05, 0.95)

        # Outcome under each arm (separate models for positivity)
        mu1_model = LogisticRegression(max_iter=500, random_state=0)
        mu0_model = LogisticRegression(max_iter=500, random_state=0)
        treated_tr = A_tr == 1
        control_tr = A_tr == 0

        if treated_tr.sum() < 5 or control_tr.sum() < 5:
            # Degenerate fold: fall back to pooled model
            mu_pool = LogisticRegression(max_iter=500, random_state=0)
            Xa_tr = np.hstack([X_tr, A_tr[:, None]])
            Xa_va1 = np.hstack([X_va, np.ones((len(X_va), 1))])
            Xa_va0 = np.hstack([X_va, np.zeros((len(X_va), 1))])
            mu_pool.fit(Xa_tr, Y_tr)
            mu1_hat = mu_pool.predict_proba(Xa_va1)[:, 1]
            mu0_hat = mu_pool.predict_proba(Xa_va0)[:, 1]
        else:
            mu1_model.fit(X_tr[treated_tr], Y_tr[treated_tr])
            mu0_model.fit(X_tr[control_tr], Y_tr[control_tr])
            mu1_hat = mu1_model.predict_proba(X_va)[:, 1]
            mu0_hat = mu0_model.predict_proba(X_va)[:, 1]

        # DR pseudo-outcome
        pseudo[val_idx] = (
            mu1_hat - mu0_hat
            + A_va * (Y_va - mu1_hat) / pi_hat
            - (1 - A_va) * (Y_va - mu0_hat) / (1 - pi_hat)
        )

    # Smooth pseudo-outcomes via ridge regression on embeddings
    ridge = Ridge(alpha=1.0)
    mask = ~np.isnan(pseudo)
    ridge.fit(embeddings[mask], pseudo[mask])
    cate_hat = ridge.predict(embeddings)

    return cate_hat


# ─── Subgroup discovery ───────────────────────────────────────────────────────

def discover_subgroups(
    main_df: pd.DataFrame,
    main_emb: np.ndarray,
    n_subgroups: int = 4,
    classifier: Literal["knn", "logistic"] = "knn",
    knn_k: int = 10,
    random_state: int = 0,
) -> SubgroupModel:
    """Discover physiological subgroups from the main cohort.

    Clusters patients in embedding space using K-means initialised from
    CATE-stratified quantile seeds (ensures subgroups differ in effect size,
    not just in covariate distribution).

    Parameters
    ----------
    main_df : main cohort DataFrame.
    main_emb : (n_main × d) embedding matrix.
    n_subgroups : number of subgroups to discover.
    classifier : "knn" (default) or "logistic". Used for out-of-sample
        assignment of rare-cohort patients to these subgroups.
        KNN is preferred: it directly respects embedding geometry without
        a linearity assumption. Logistic regression is faster and gives
        interpretable coefficients but assumes linear boundaries.
    knn_k : neighbours for KNN classifier (ignored if classifier="logistic").
    random_state : for K-means reproducibility.

    Returns
    -------
    SubgroupModel ready for cross-registry assignment.
    """
    cate_hat = estimate_cates(main_df, main_emb)

    # CATE-stratified K-means init: seed centroids at CATE quantile means
    quantile_edges = np.quantile(cate_hat, np.linspace(0, 1, n_subgroups + 1))
    init_centers = []
    for i in range(n_subgroups):
        mask = (cate_hat >= quantile_edges[i]) & (cate_hat <= quantile_edges[i + 1])
        if mask.sum() > 0:
            init_centers.append(main_emb[mask].mean(axis=0))
        else:
            init_centers.append(main_emb[i % len(main_emb)])
    init_centers = np.vstack(init_centers)

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")   # suppress degenerate-cluster ConvergenceWarning
        km = KMeans(n_clusters=n_subgroups, init=init_centers, n_init=1,
                    random_state=random_state)
        labels = km.fit_predict(main_emb)

    # If K-means found fewer distinct clusters than requested (e.g. nearly 1D
    # embedding space at high φ), remap to the actual distinct cluster count.
    n_found = len(np.unique(labels))
    if n_found < n_subgroups:
        n_subgroups = n_found
        # Re-run with correct k
        km = KMeans(n_clusters=n_subgroups, n_init="auto", random_state=random_state)
        labels = km.fit_predict(main_emb)

    # Subgroup summary statistics
    unique_g = sorted(np.unique(labels))
    n_subgroups = len(unique_g)   # may have been reduced above
    cate_means = np.array([cate_hat[labels == g].mean() if (labels == g).any() else 0.0
                           for g in unique_g])
    cate_sds   = np.array([cate_hat[labels == g].std()  if (labels == g).sum() > 1 else 0.0
                           for g in unique_g])
    counts     = np.array([(labels == g).sum() for g in unique_g])

    # Order subgroups by CATE (most beneficial first)
    order = np.argsort(cate_means)
    label_map = {old: new for new, old in enumerate(unique_g)}
    labels = np.array([label_map[l] for l in labels])
    cate_means = cate_means[order]
    cate_sds   = cate_sds[order]
    counts     = counts[order]
    cluster_centers = km.cluster_centers_[order]

    # Human-readable names keyed by CATE rank
    _RANK_NAMES = [
        "strong-responders", "moderate-responders",
        "weak-responders",   "non-responders",
        "sub1", "sub2", "sub3", "sub4",
    ]
    subgroup_names = [_RANK_NAMES[i] if i < len(_RANK_NAMES) else f"subgroup-{i}"
                      for i in range(n_subgroups)]

    # Fit classifier for out-of-sample assignment
    clf = _fit_classifier(main_emb, labels, classifier, knn_k)

    return SubgroupModel(
        n_subgroups=n_subgroups,
        cluster_centers=cluster_centers,
        subgroup_labels=labels,
        subgroup_names=subgroup_names,
        cate_by_subgroup=cate_means,
        cate_sd_by_subgroup=cate_sds,
        n_by_subgroup=counts,
        classifier_type=classifier,
        _classifier=clf,
    )


def _fit_classifier(
    emb: np.ndarray,
    labels: np.ndarray,
    classifier: str,
    knn_k: int,
) -> object:
    """Fit the subgroup classifier on main-cohort embeddings."""
    if classifier == "knn":
        # Embeddings are L2-normalised so euclidean distance is monotonically
        # equivalent to cosine distance (||a-b||² = 2(1 - cos(a,b))) — avoids
        # the matmul overflow that sklearn's cosine metric triggers on some arrays.
        clf = KNeighborsClassifier(
            n_neighbors=knn_k,
            metric="euclidean",
        )
    elif classifier == "logistic":
        clf = LogisticRegression(
            max_iter=500,
            random_state=0,
        )
    else:
        raise ValueError(f"classifier must be 'knn' or 'logistic', got {classifier!r}")
    clf.fit(emb, labels)
    return clf


# ─── Cross-registry subgroup assignment ───────────────────────────────────────

def assign_subgroups(
    df: pd.DataFrame,
    emb: np.ndarray,
    model: SubgroupModel,
) -> np.ndarray:
    """Assign patients to main-cohort-derived subgroups via the fitted classifier.

    Parameters
    ----------
    df : patient DataFrame (unused directly; here for API symmetry).
    emb : (n × d) embedding matrix for these patients.
    model : fitted SubgroupModel from discover_subgroups().

    Returns
    -------
    labels : (n,) integer array of subgroup indices 0..n_subgroups-1.
    """
    return model._classifier.predict(emb).astype(int)


# ─── Subgroup-level borrowing ─────────────────────────────────────────────────

def subgroup_level_borrow(
    main_df: pd.DataFrame,
    target_df: pd.DataFrame,
    main_emb: np.ndarray,
    target_emb: np.ndarray,
    model: SubgroupModel,
    target_true_ate: float,
    tau_prior_sd: float = 0.10,
    robust_weight: float = 0.10,
    vague_sd: float = 0.50,
    min_subgroup_n: int = 5,
    alpha: float = 0.05,
) -> list[SubgroupBorrowingResult]:
    """Run robust MAP prior borrowing within each subgroup.

    For each subgroup g:
      - Collect main cohort patients in subgroup g → main_g_summary
      - Collect target patients assigned to subgroup g → target_g_summary
      - Run robust_map_posterior(donors=[main_g_summary], target=target_g_summary)
      - Return BorrowingResult for subgroup g in target registry

    Subgroups with fewer than min_subgroup_n patients in either the main or
    target cohort fall back to population-level borrowing (no subgroup
    stratification) to avoid degenerate estimates.

    Parameters
    ----------
    main_df, target_df : patient DataFrames (must have Y, A columns).
    main_emb, target_emb : aligned embedding matrices.
    model : SubgroupModel from discover_subgroups() on the main cohort.
    target_true_ate : ground truth (DGP-known; for OC evaluation only).
    min_subgroup_n : minimum patients per arm per subgroup to attempt MAP prior.

    Returns
    -------
    List of SubgroupBorrowingResult, one per subgroup.
    """
    target_labels = assign_subgroups(target_df, target_emb, model)
    main_labels   = model.subgroup_labels
    target_registry = target_df["registry"].iloc[0]

    results = []
    z = norm.ppf(1.0 - alpha / 2.0)

    for g in range(model.n_subgroups):
        main_mask   = main_labels   == g
        target_mask = target_labels == g

        main_g   = main_df[main_mask]
        target_g = target_df[target_mask]

        n_main_g   = int(main_mask.sum())
        n_target_g = int(target_mask.sum())

        n_main_t   = int((main_g["A"]   == 1).sum())
        n_main_c   = int((main_g["A"]   == 0).sum())
        n_target_t = int((target_g["A"] == 1).sum())
        n_target_c = int((target_g["A"] == 0).sum())

        too_small = (
            n_main_g   < min_subgroup_n
            or n_target_g < min_subgroup_n
            or n_main_t   < 2 or n_main_c   < 2
            or n_target_t < 2 or n_target_c < 2
        )
        if too_small:
            # Degenerate subgroup: fall back to pooled estimate for this subgroup
            # Use target-only difference-in-means with wide vague prior
            treated = target_g[target_g["A"] == 1]["Y"].values if n_target_g > 0 else np.array([])
            control = target_g[target_g["A"] == 0]["Y"].values if n_target_g > 0 else np.array([])
            if len(treated) < 2 or len(control) < 2:
                continue

            ate_hat = float(treated.mean() - control.mean())
            se_hat  = float(np.sqrt(treated.var(ddof=1) / len(treated)
                                    + control.var(ddof=1) / len(control)))
            borrowing = BorrowingResult(
                level="subgroup",
                target_registry=target_registry,
                ate_posterior=ate_hat,
                se_posterior=se_hat,
                ci_lower=ate_hat - z * se_hat,
                ci_upper=ate_hat + z * se_hat,
                ess_prior=0.0,
                ess_data=float(n_target_g),
                ess_total=float(n_target_g),
                map_weight=0.0,
                rejects_null=bool(abs(ate_hat / max(se_hat, 1e-12)) > z),
                covers_truth=bool(
                    (ate_hat - z * se_hat) <= target_true_ate
                    <= (ate_hat + z * se_hat)
                ),
                true_ate=target_true_ate,
            )
        else:
            main_sum   = summarise_registry(main_g,   target_true_ate, "main")
            target_sum = summarise_registry(target_g, target_true_ate, target_registry)

            post_mean, post_sd, map_w = robust_map_posterior(
                donor_summaries=[main_sum],
                target_summary=target_sum,
                tau_prior_sd=tau_prior_sd,
                robust_weight=robust_weight,
                vague_sd=vague_sd,
            )

            ci_lo = post_mean - z * post_sd
            ci_hi = post_mean + z * post_sd

            ess_prior, ess_data, ess_total = compute_ess(
                prior_sd=float(np.sqrt(main_sum.se_hat ** 2 + tau_prior_sd ** 2)),
                likelihood_sd=target_sum.se_hat,
                posterior_sd=post_sd,
                target_n=target_sum.n,
            )

            borrowing = BorrowingResult(
                level="subgroup",
                target_registry=target_registry,
                ate_posterior=post_mean,
                se_posterior=post_sd,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                ess_prior=ess_prior,
                ess_data=ess_data,
                ess_total=ess_total,
                map_weight=float(map_w),
                rejects_null=bool(abs(post_mean / max(post_sd, 1e-12)) > z),
                covers_truth=bool(ci_lo <= target_true_ate <= ci_hi),
                true_ate=target_true_ate,
            )

        results.append(SubgroupBorrowingResult(
            subgroup_idx=g,
            subgroup_name=model.subgroup_names[g],
            borrowing=borrowing,
            n_target_in_subgroup=n_target_g,
            cate_main_mean=float(model.cate_by_subgroup[g]),
            cate_main_sd=float(model.cate_sd_by_subgroup[g]),
        ))

    return results


# ─── ESS reconciliation ───────────────────────────────────────────────────────

def reconcile_ess(
    subgroup_results: list[SubgroupBorrowingResult],
    population_result: BorrowingResult,
    tol: float = 0.10,
) -> dict:
    """Check that subgroup ESS sums to ≤ population ESS (nesting consistency).

    Subgroup-level ESS should aggregate to at most the population-level ESS.
    Note: the variance-ratio ESS approximation does not have a natural summing
    property across nested levels — subgroup ESS priors can sum to more than the
    population ESS prior even when borrowing is consistent. Treat this check as
    a diagnostic flag rather than a hard constraint; ratios well above 1.0
    indicate the formula is being pushed beyond its valid regime (typically very
    small subgroup sizes).

    Returns a summary dict with the ESS values and a boolean flag.
    """
    subgroup_ess_prior = sum(r.borrowing.ess_prior for r in subgroup_results)
    subgroup_ess_data  = sum(r.borrowing.ess_data  for r in subgroup_results)

    pop_ess_prior = population_result.ess_prior
    pop_ess_data  = population_result.ess_data

    consistent = (
        subgroup_ess_prior <= pop_ess_prior * (1.0 + tol)
        and abs(subgroup_ess_data - pop_ess_data) / max(pop_ess_data, 1) <= tol
    )

    return {
        "subgroup_ess_prior_sum": subgroup_ess_prior,
        "subgroup_ess_data_sum":  subgroup_ess_data,
        "population_ess_prior":   pop_ess_prior,
        "population_ess_data":    pop_ess_data,
        "consistent":             consistent,
        "ess_prior_ratio":        subgroup_ess_prior / max(pop_ess_prior, 1e-8) if pop_ess_prior > 0 else float("nan"),
    }
