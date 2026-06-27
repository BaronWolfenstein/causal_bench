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

Clustering uses Gaussian Mixture Models (GMM) or K-means (default).
GMM is preferred when available: it uses soft assignments, provides per-subgroup
covariance matrices (Σ_k) useful as importance weights for tail-aware diffusion
training, and naturally handles ellipsoidal clusters. K-means is the fallback
when GMM degenerates (singular covariance).

Subgroup assignment uses the GMM's own predict() when clustering="gmm", or KNN
on embeddings when clustering="kmeans". KNN directly respects the embedding
geometry without assuming linearity in the boundary. Logistic regression
(multiclass) is provided as a third option for settings where linear separability
holds and interpretable coefficients are wanted.

For the OC simulation (exp19), CATE estimation is bypassed when true per-patient
CATEs are available in the DataFrame ("cate" column). For production on real
registry data, a DR-Learner is fitted on the main cohort.

References:
  Kennedy (2023). Towards optimal doubly-robust estimation of heterogeneous
    causal effects. EJS.
  Schmidli et al. (2014). Robust meta-analytic-predictive priors. Biometrics.
  Reynolds & Rose (1995). Robust text-independent speaker identification using
    GMM. IEEE TSAP. (GMM density estimation for importance weighting.)
"""
from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from causal_bench.estimators.hierarchical import (
    BorrowingResult,
    RegistrySummary,
    _conjugacy_diagnostic,
    compute_ess,
    robust_map_posterior,
    summarise_registry,
)


# ─── _RemappedGMM ─────────────────────────────────────────────────────────────

class _RemappedGMM:
    """Wraps a fitted GaussianMixture so predict/predict_proba return CATE-ranked labels.

    GaussianMixture.predict() returns component indices in fit order (0..k-1),
    which is not CATE order.  This wrapper applies the permutation so all
    outputs align with SubgroupModel.cate_by_subgroup, .component_covariances,
    and .subgroup_names — which are stored in CATE rank order.
    """

    def __init__(self, gmm: GaussianMixture, order: np.ndarray) -> None:
        # order[cate_rank] = gmm_component_index
        self._gmm   = gmm
        self._order = order
        # inv_order[gmm_component_index] = cate_rank
        self._inv_order = np.empty(len(order), dtype=int)
        for cate_rank, gmm_idx in enumerate(order):
            self._inv_order[gmm_idx] = cate_rank

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._inv_order[self._gmm.predict(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Permute columns so column cate_rank = old column order[cate_rank]
        return self._gmm.predict_proba(X)[:, self._order]

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Full mixture log-density — unaffected by component ordering."""
        return self._gmm.score_samples(X)


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
    classifier_type: str               # "knn", "logistic", or "gmm"
    # GMM-only fields: per-subgroup covariance and mixing weight
    component_covariances: Optional[np.ndarray] = None  # (k × d × d), GMM only
    component_weights: Optional[np.ndarray] = None      # (k,) mixing weights π_k
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
    clustering: Literal["kmeans", "gmm"] = "kmeans",
    classifier: Literal["knn", "logistic"] = "knn",
    knn_k: int = 10,
    random_state: int = 0,
) -> SubgroupModel:
    """Discover physiological subgroups from the main cohort.

    Clusters patients in embedding space using K-means or GMM, initialised from
    CATE-stratified quantile seeds (ensures subgroups differ in effect size,
    not just in covariate distribution).

    When clustering="gmm" (Gaussian Mixture Model):
      - Soft EM clustering with full covariance matrices (per-subgroup Σ_k).
      - The GMM itself serves as the out-of-sample classifier; the `classifier`
        parameter is ignored.
      - component_covariances and component_weights are populated on the returned
        SubgroupModel — use them as importance weights for tail-aware diffusion
        training (GMM log-likelihood is a natural density estimator for the
        rare-mode reweighting needed in Test B′).
      - Falls back to K-means + KNN if the GMM fit fails (singular covariance).

    When clustering="kmeans" (default):
      - Hard K-means with CATE-stratified initialisation.
      - Out-of-sample assignment via the `classifier` parameter (KNN or logistic).

    Parameters
    ----------
    main_df : main cohort DataFrame.
    main_emb : (n_main × d) embedding matrix.
    n_subgroups : number of subgroups to discover.
    clustering : "kmeans" (default) or "gmm".
    classifier : "knn" (default) or "logistic". Only used when clustering="kmeans".
        KNN is preferred: it directly respects embedding geometry without a
        linearity assumption. Logistic regression gives interpretable coefficients
        but assumes linear boundaries.
    knn_k : neighbours for KNN classifier (ignored unless clustering="kmeans" and
        classifier="knn").
    random_state : for reproducibility.

    Returns
    -------
    SubgroupModel ready for cross-registry assignment.
    """
    cate_hat = estimate_cates(main_df, main_emb)

    # CATE-stratified init: seed cluster centers at CATE quantile means
    quantile_edges = np.quantile(cate_hat, np.linspace(0, 1, n_subgroups + 1))
    init_centers = []
    for i in range(n_subgroups):
        mask = (cate_hat >= quantile_edges[i]) & (cate_hat <= quantile_edges[i + 1])
        if mask.sum() > 0:
            init_centers.append(main_emb[mask].mean(axis=0))
        else:
            init_centers.append(main_emb[i % len(main_emb)])
    init_centers = np.vstack(init_centers)

    # ── Clustering ────────────────────────────────────────────────────────────
    component_covariances: Optional[np.ndarray] = None
    component_weights: Optional[np.ndarray] = None
    _gmm: Optional[GaussianMixture] = None
    _active_clustering = clustering

    if clustering == "gmm":
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                _gmm = GaussianMixture(
                    n_components=n_subgroups,
                    covariance_type="full",
                    means_init=init_centers,
                    random_state=random_state,
                    max_iter=300,
                    reg_covar=1e-6,  # regularise to avoid singular covariance
                )
                _gmm.fit(main_emb)
            labels = _gmm.predict(main_emb)
            n_found = len(np.unique(labels))
            if n_found < n_subgroups:
                # Some components degenerated — re-fit with actual k
                with _warnings.catch_warnings():
                    _warnings.simplefilter("ignore")
                    _gmm = GaussianMixture(
                        n_components=n_found,
                        covariance_type="full",
                        random_state=random_state,
                        max_iter=300,
                        reg_covar=1e-6,
                    )
                    _gmm.fit(main_emb)
                labels = _gmm.predict(main_emb)
                n_subgroups = n_found
            component_covariances = _gmm.covariances_.copy()   # (k, d, d)
            component_weights = _gmm.weights_.copy()            # (k,)
        except (ValueError, np.linalg.LinAlgError):
            _warnings.warn(
                "GaussianMixture fit failed (singular covariance); "
                "falling back to K-means + KNN.",
                RuntimeWarning,
                stacklevel=2,
            )
            _active_clustering = "kmeans"
            _gmm = None
            component_covariances = None
            component_weights = None

    if _active_clustering == "kmeans":
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")  # suppress ConvergenceWarning
            km = KMeans(n_clusters=n_subgroups, init=init_centers, n_init=1,
                        random_state=random_state)
            labels = km.fit_predict(main_emb)

        # If K-means found fewer distinct clusters (nearly 1D embedding space at
        # high φ), re-run with the actual k.
        n_found = len(np.unique(labels))
        if n_found < n_subgroups:
            n_subgroups = n_found
            km = KMeans(n_clusters=n_subgroups, n_init="auto",
                        random_state=random_state)
            labels = km.fit_predict(main_emb)

    # ── Subgroup summary statistics ───────────────────────────────────────────
    unique_g = sorted(np.unique(labels))
    n_subgroups = len(unique_g)
    cate_means = np.array([
        cate_hat[labels == g].mean() if (labels == g).any() else 0.0
        for g in unique_g
    ])
    cate_sds = np.array([
        cate_hat[labels == g].std() if (labels == g).sum() > 1 else 0.0
        for g in unique_g
    ])
    counts = np.array([(labels == g).sum() for g in unique_g])

    # Order subgroups by CATE (most beneficial = most negative = first).
    # order[cate_rank] = original_cluster_index
    order = np.argsort(cate_means)
    # Remap patient labels from cluster-discovery order to CATE rank so that
    # subgroup_labels[i] == k means patient i is in CATE-rank-k subgroup —
    # consistent with cate_by_subgroup[k], subgroup_names[k], etc.
    inv_order = np.empty(n_subgroups, dtype=int)
    for cate_rank, cluster_id in enumerate(order):
        inv_order[cluster_id] = cate_rank
    labels = inv_order[labels]
    cate_means = cate_means[order]
    cate_sds   = cate_sds[order]
    counts     = counts[order]

    if _gmm is not None:
        cluster_centers = _gmm.means_[order]
        component_covariances = component_covariances[order]  # type: ignore[index]
        component_weights = component_weights[order]          # type: ignore[index]
    else:
        cluster_centers = km.cluster_centers_[order]

    # Human-readable names keyed by CATE rank
    _RANK_NAMES = [
        "strong-responders", "moderate-responders",
        "weak-responders",   "non-responders",
        "sub1", "sub2", "sub3", "sub4",
    ]
    subgroup_names = [
        _RANK_NAMES[i] if i < len(_RANK_NAMES) else f"subgroup-{i}"
        for i in range(n_subgroups)
    ]

    # ── Classifier for out-of-sample assignment ───────────────────────────────
    if _gmm is not None:
        clf = _RemappedGMM(_gmm, order)
        classifier_type = "gmm"
    else:
        clf = _fit_classifier(main_emb, labels, classifier, knn_k)
        classifier_type = classifier

    return SubgroupModel(
        n_subgroups=n_subgroups,
        cluster_centers=cluster_centers,
        subgroup_labels=labels,
        subgroup_names=subgroup_names,
        cate_by_subgroup=cate_means,
        cate_sd_by_subgroup=cate_sds,
        n_by_subgroup=counts,
        classifier_type=classifier_type,
        component_covariances=component_covariances,
        component_weights=component_weights,
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


def assign_subgroups_soft(
    df: pd.DataFrame,
    emb: np.ndarray,
    model: SubgroupModel,
) -> np.ndarray:
    """Return soft subgroup membership probabilities for each patient.

    Only available when the model was discovered with clustering="gmm".
    The returned matrix can be used as importance weights for tail-aware
    diffusion training (Test B′ in the rare-detail localisation diagnostic):
    a patient's log density under the rare-mode GMM component is
    log p(z_i) = log Σ_k π_k N(z_i; μ_k, Σ_k), obtainable via
    model._classifier.score_samples(emb).

    Parameters
    ----------
    df : patient DataFrame (unused directly; here for API symmetry).
    emb : (n × d) embedding matrix.
    model : SubgroupModel with classifier_type="gmm".

    Returns
    -------
    proba : (n × n_subgroups) array where proba[i, k] = P(subgroup k | z_i).

    Raises
    ------
    ValueError : if the model was not fitted with GMM clustering.
    """
    if model.classifier_type != "gmm":
        raise ValueError(
            f"Soft assignments require classifier_type='gmm', "
            f"got {model.classifier_type!r}. Use assign_subgroups() for hard assignments."
        )
    return model._classifier.predict_proba(emb)


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
                conjugacy_regime="local_approximation",  # degenerate: no MAP prior
                approximation_deviation=float("nan"),
            )
        else:
            main_sum   = summarise_registry(main_g,   target_true_ate, "main")
            target_sum = summarise_registry(target_g, target_true_ate, target_registry)

            post_mean, post_sd, map_w, sigma2_map, *_ = robust_map_posterior(
                donor_summaries=[main_sum],
                target_summary=target_sum,
                tau_prior_sd=tau_prior_sd,
                robust_weight=robust_weight,
                vague_sd=vague_sd,
            )

            ci_lo = post_mean - z * post_sd
            ci_hi = post_mean + z * post_sd

            ess_prior, ess_data, ess_total = compute_ess(
                prior_sd=float(np.sqrt(sigma2_map)),
                likelihood_sd=target_sum.se_hat,
                posterior_sd=post_sd,
                target_n=target_sum.n,
            )

            regime, deviation = _conjugacy_diagnostic(
                post_mean=post_mean,
                map_weight=map_w,
                target_ate=target_sum.ate_hat,
                target_se=target_sum.se_hat,
                vague_sd=vague_sd,
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
                conjugacy_regime=regime,
                approximation_deviation=deviation,
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
