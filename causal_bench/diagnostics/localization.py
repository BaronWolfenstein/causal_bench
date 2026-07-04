"""Rare-detail localisation diagnostic — CPU-side decision procedure.

Implements the three-test decision procedure from the GPU build spec as
encoder-agnostic Python. All test functions take np.ndarray embeddings and
pre-computed reconstruction arrays; diffusion model training and MEDS/encoder
loading are intentionally excluded and belong in the Lambda GPU script.

Decision tree
-------------
                    ┌─ Test A pass ──► Test B ──── pass ──► diffuse_directly
                    │                         └── fail ──► Test B' ─ pass ──► tail_aware
run_diagnostic ─────┤                                             └─ fail ──► Test C ─ pass ──► separate_latent_justified
                    │                                                                  └─ fail ──► escalate
                    └─ Test A fail ──► pretraining_influence? ─ yes ──► spt_recommendation
                                                               └─ no  ──► bound_scope

If reconstructed embeddings for a downstream test are not provided, the
diagnostic returns a `pending_*` terminal and stops — the GPU script
fills in the reconstruction arrays one test at a time.

References
----------
  GPU build spec (causal_bench project notes, 2026-06-22).
  Baek et al. 2026. "The Finetuner's Fallacy." arXiv:2603.16177.
"""
from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class LocalizationResult:
    """Result of one test in the decision procedure."""
    test: str       # "A", "B", "B_prime", "C"
    passed: bool
    metrics: dict   # raw metric values (AUC, L2 ratios, etc.)
    notes: str      # one-line human-readable interpretation


@dataclass
class DiagnosticReport:
    """Full decision-procedure report.

    Terminal values and their meanings
    -----------------------------------
    diffuse_directly           Test B passed AND Test B'' (CFG landing) passed: diffuse on
                               embeddings, no latent needed.
    tail_aware                 Test B' passed AND Test B'' passed: importance-weighted training.
    smc_required               Reconstruction faithful (B or B' passed) but B'' failed — CFG
                               cannot land in the rare region. Twisted-diffusion SMC resampler
                               is the REQUIRED inference-time fix (not optional); diffuse_directly/
                               tail_aware stay unreachable until CFG passes B'' or SMC-guided
                               samples pass the check.
    separate_latent_justified  Tests B+B' failed, C passed: learned latent is warranted.
    spt_recommendation         Test A failed + pretraining influence: fix at encoder via SPT.
    bound_scope                Test A failed, no pretraining influence: encoder is the limit.
    escalate                   Tests B+B'+C all failed: no available fix; bound scope.
    pending_B                  Test A passed but B reconstructions not yet provided.
    pending_B_prime            Test B failed but B' reconstructions not yet provided.
    pending_cfg_landing_check  Reconstruction passed (B or B') but the CFG generative-landing
                               samples for Test B'' not yet provided.
    pending_C                  Tests B+B' failed but C reconstructions not yet provided.
    """
    terminal: str
    tests_run: list  # list[LocalizationResult] in order
    pretraining_influence: bool
    summary: str


# ─── Test A: encoder capacity ─────────────────────────────────────────────────

def test_a(
    rare_emb: np.ndarray,
    common_emb: np.ndarray,
    cv: int = 5,
    mlp_check: bool = True,
    auc_threshold: float = 0.70,
) -> LocalizationResult:
    """Test A — encoder capacity: can we separate rare from common in embedding space?

    Fits logistic regression (and optionally a small MLP) to discriminate rare
    from common patients using only their embeddings. ROC-AUC is
    prevalence-invariant under heavy imbalance; PR-AUC is reported as supplement.

    Parameters
    ----------
    rare_emb   : (n_rare, d) embeddings of the rare subpopulation.
    common_emb : (n_common, d) embeddings of the common subpopulation.
    cv         : cross-validation folds. Automatically reduced if rare class is too small.
    mlp_check  : if True, also fits a small MLP as a nonlinearity capacity check.
    auc_threshold : minimum logistic ROC-AUC to declare Test A passed.

    Returns
    -------
    LocalizationResult with metrics:
        logistic_auc, logistic_pr_auc, logistic_auc_std
        mlp_auc, mlp_pr_auc (if mlp_check=True)
        n_rare, n_common, cv_used
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    n_rare, n_common = len(rare_emb), len(common_emb)

    X = np.vstack([rare_emb, common_emb])
    y = np.concatenate([np.ones(n_rare), np.zeros(n_common)])

    # Reduce cv if rare class is too small
    cv_safe = max(2, min(cv, n_rare // 2)) if n_rare >= 4 else 2
    if cv_safe < cv:
        _warnings.warn(
            f"test_a: rare class n={n_rare} too small for cv={cv}; using cv={cv_safe}.",
            RuntimeWarning,
            stacklevel=2,
        )

    skf = StratifiedKFold(n_splits=cv_safe, shuffle=True, random_state=0)
    scaler = StandardScaler()

    lr_aucs, lr_pr_aucs = [], []
    mlp_aucs, mlp_pr_aucs = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        y_tr, y_va = y[train_idx], y[val_idx]

        if len(np.unique(y_va)) < 2:
            continue

        # Logistic regression
        lr = LogisticRegression(max_iter=500, random_state=0, C=1.0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            lr.fit(X_tr, y_tr)
        lr_prob = lr.predict_proba(X_va)[:, 1]
        lr_aucs.append(roc_auc_score(y_va, lr_prob))
        lr_pr_aucs.append(average_precision_score(y_va, lr_prob))

        # MLP capacity check
        if mlp_check:
            mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                mlp.fit(X_tr, y_tr)
            mlp_prob = mlp.predict_proba(X_va)[:, 1]
            mlp_aucs.append(roc_auc_score(y_va, mlp_prob))
            mlp_pr_aucs.append(average_precision_score(y_va, mlp_prob))

    mean_lr_auc = float(np.mean(lr_aucs)) if lr_aucs else 0.5
    mean_lr_pr  = float(np.mean(lr_pr_aucs)) if lr_pr_aucs else float(n_rare / (n_rare + n_common))
    std_lr_auc  = float(np.std(lr_aucs)) if len(lr_aucs) > 1 else 0.0

    metrics = {
        "logistic_auc":     mean_lr_auc,
        "logistic_pr_auc":  mean_lr_pr,
        "logistic_auc_std": std_lr_auc,
        "n_rare":           n_rare,
        "n_common":         n_common,
        "cv_used":          cv_safe,
    }

    if mlp_check and mlp_aucs:
        metrics["mlp_auc"]    = float(np.mean(mlp_aucs))
        metrics["mlp_pr_auc"] = float(np.mean(mlp_pr_aucs))

    passed = mean_lr_auc >= auc_threshold
    if passed:
        notes = (
            f"Encoder separates rare from common (logistic AUC={mean_lr_auc:.3f} ≥ {auc_threshold}). "
            "Proceed to Test B."
        )
    else:
        notes = (
            f"Encoder does NOT preserve rare-patient signal "
            f"(logistic AUC={mean_lr_auc:.3f} < {auc_threshold}). "
            "Fix lies at the encoder pretraining layer."
        )
        mlp_a = metrics.get("mlp_auc")
        if mlp_a is not None and mlp_a >= auc_threshold:
            notes += (
                f" MLP AUC={mlp_a:.3f} suggests nonlinear separation exists "
                "— linear embeddings underutilise the geometry."
            )

    return LocalizationResult(test="A", passed=passed, metrics=metrics, notes=notes)


# ─── Per-mode reconstruction metrics (shared by B, B', C) ────────────────────

def per_mode_reconstruction_metrics(
    rare_orig: np.ndarray,
    common_orig: np.ndarray,
    rare_recon: np.ndarray,
    common_recon: np.ndarray,
    cv: int = 5,
    auc_threshold: float = 0.70,
) -> dict:
    """Per-mode reconstruction fidelity metrics for Tests B / B' / C.

    Computes per-patient L2 reconstruction error stratified by mode, and
    measures how much the round-trip degrades separation between rare and
    common patients (separation AUC drop).

    The key failure signature is:
      - common_l2 ≈ 0 (bulk mode well-reconstructed)
      - rare_l2 >> common_l2 (tail mode collapsed)
      - separation_auc_recon << separation_auc_orig (rare identity lost)

    Parameters
    ----------
    rare_orig, common_orig   : original embeddings before diffusion round-trip.
    rare_recon, common_recon : reconstructed embeddings after round-trip.
    cv         : folds for AUC estimation.
    auc_threshold : not used here; passed for reference in calling tests.

    Returns
    -------
    dict with keys:
        rare_l2_mean, common_l2_mean, l2_ratio (rare/common),
        separation_auc_orig, separation_auc_recon, auc_drop
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rare_l2   = float(np.mean(np.linalg.norm(rare_orig   - rare_recon,   axis=1)))
    common_l2 = float(np.mean(np.linalg.norm(common_orig - common_recon, axis=1)))
    l2_ratio  = rare_l2 / max(common_l2, 1e-12)

    def _separation_auc(emb_rare: np.ndarray, emb_common: np.ndarray) -> float:
        X = np.vstack([emb_rare, emb_common])
        y = np.concatenate([np.ones(len(emb_rare)), np.zeros(len(emb_common))])
        n_rare = len(emb_rare)
        cv_safe = max(2, min(cv, n_rare // 2)) if n_rare >= 4 else 2
        skf = StratifiedKFold(n_splits=cv_safe, shuffle=True, random_state=0)
        scaler = StandardScaler()
        aucs = []
        for tr, va in skf.split(X, y):
            if len(np.unique(y[va])) < 2:
                continue
            Xtr = scaler.fit_transform(X[tr])
            Xva = scaler.transform(X[va])
            lr = LogisticRegression(max_iter=500, random_state=0)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                lr.fit(Xtr, y[tr])
            aucs.append(roc_auc_score(y[va], lr.predict_proba(Xva)[:, 1]))
        return float(np.mean(aucs)) if aucs else 0.5

    sep_orig  = _separation_auc(rare_orig,  common_orig)
    sep_recon = _separation_auc(rare_recon, common_recon)

    return {
        "rare_l2_mean":          rare_l2,
        "common_l2_mean":        common_l2,
        "l2_ratio":              l2_ratio,
        "separation_auc_orig":   sep_orig,
        "separation_auc_recon":  sep_recon,
        "auc_drop":              sep_orig - sep_recon,
    }


# ─── Test B / B' / C shared logic ─────────────────────────────────────────────

def _reconstruction_test(
    rare_orig: np.ndarray,
    common_orig: np.ndarray,
    rare_recon: np.ndarray,
    common_recon: np.ndarray,
    test_name: str,
    reconstruction_tol: float,
    auc_drop_tol: float,
    cv: int,
) -> LocalizationResult:
    """Internal: run per-mode reconstruction metrics and classify pass/fail."""
    m = per_mode_reconstruction_metrics(
        rare_orig, common_orig, rare_recon, common_recon, cv=cv
    )

    # Pass criteria: rare reconstruction not much worse than common, AND
    # separation AUC not significantly degraded through the round-trip.
    rare_ok = m["l2_ratio"] <= 1.0 + reconstruction_tol
    auc_ok  = m["auc_drop"] <= auc_drop_tol
    passed  = rare_ok and auc_ok

    if passed:
        notes = (
            f"Test {test_name}: rare L2/common L2 = {m['l2_ratio']:.2f} ≤ "
            f"{1 + reconstruction_tol:.2f} and AUC drop = {m['auc_drop']:.3f} ≤ {auc_drop_tol}. "
            "Round-trip is faithful."
        )
    else:
        reasons = []
        if not rare_ok:
            reasons.append(
                f"rare L2/common L2 = {m['l2_ratio']:.2f} > {1 + reconstruction_tol:.2f} (tail collapse)"
            )
        if not auc_ok:
            reasons.append(
                f"AUC drop = {m['auc_drop']:.3f} > {auc_drop_tol} (separation degraded)"
            )
        notes = f"Test {test_name} FAILED: " + "; ".join(reasons) + "."

    return LocalizationResult(test=test_name, passed=passed, metrics=m, notes=notes)


# ─── Test B'' : CFG generative landing ────────────────────────────────────────

def _pairwise_separation_auc(emb_a: np.ndarray, emb_b: np.ndarray, cv: int = 5) -> float:
    """CV logistic ROC-AUC separating emb_a (label 1) from emb_b (label 0).

    Standalone twin of the nested separator in per_mode_reconstruction_metrics;
    kept separate here to avoid touching that tested function. Unify when the
    B'' characterization tests are added.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([emb_a, emb_b])
    y = np.concatenate([np.ones(len(emb_a)), np.zeros(len(emb_b))])
    n_a = len(emb_a)
    cv_safe = max(2, min(cv, n_a // 2)) if n_a >= 4 else 2
    skf = StratifiedKFold(n_splits=cv_safe, shuffle=True, random_state=0)
    scaler = StandardScaler()
    aucs = []
    for tr, va in skf.split(X, y):
        if len(np.unique(y[va])) < 2:
            continue
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        lr = LogisticRegression(max_iter=500, random_state=0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            lr.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[va], lr.predict_proba(Xva)[:, 1]))
    return float(np.mean(aucs)) if aucs else 0.5


def test_b_double_prime(
    rare_guided: np.ndarray,
    real_rare: np.ndarray,
    common_ref: np.ndarray,
    cv: int = 5,
    fidelity_tol: float = 0.65,
    drift_tol: float = 0.70,
) -> LocalizationResult:
    """Test B'' — CFG generative landing (2026-07-02 diagram).

    Reconstruction (Test B) tests denoising near existing points; this tests
    GENERATION from noise under rare-cohort classifier-free guidance (CFG, after
    ELF). `rare_guided` are held-out generated samples (NOT round-tripped). Two
    metrics, never collapsed:

    - Fidelity AUC = separation(rare_guided vs real_rare) — LOWER better. High
      ⟹ generated tail is distinguishable from real ⟹ poor score in the tail.
    - Drift AUC    = separation(rare_guided vs common_ref) — HIGHER better. Low
      ⟹ conditioning too weak, generation collapsed to the bulk.

    Pass = fidelity_auc ≤ fidelity_tol AND drift_auc ≥ drift_tol.
    """
    fidelity_auc = _pairwise_separation_auc(rare_guided, real_rare, cv=cv)
    drift_auc = _pairwise_separation_auc(rare_guided, common_ref, cv=cv)

    fidelity_ok = fidelity_auc <= fidelity_tol
    drift_ok = drift_auc >= drift_tol
    passed = fidelity_ok and drift_ok

    metrics = {"fidelity_auc": fidelity_auc, "drift_auc": drift_auc}
    if passed:
        notes = (
            f"Test B'': fidelity AUC={fidelity_auc:.3f} ≤ {fidelity_tol} (guided ≈ real rare) "
            f"and drift AUC={drift_auc:.3f} ≥ {drift_tol} (guided ≠ common). CFG lands in R."
        )
    else:
        reasons = []
        if not fidelity_ok:
            reasons.append(f"fidelity AUC={fidelity_auc:.3f} > {fidelity_tol} (poor score in tail)")
        if not drift_ok:
            reasons.append(f"drift AUC={drift_auc:.3f} < {drift_tol} (collapsed to bulk)")
        notes = "Test B'' FAILED: " + "; ".join(reasons) + " — CFG cannot land in R."

    return LocalizationResult(test="B_double_prime", passed=passed, metrics=metrics, notes=notes)


def _cfg_landing_gate(cfg_landing, arch_terminal, rare_emb, common_emb, tests_run,
                      pretraining_influence, cv, fidelity_tol, drift_tol, via):
    """Route a reconstruction-passing branch through Test B'' (CFG landing).

    cfg_landing None -> pending_cfg_landing_check (await generated samples). Else
    run Test B'': pass -> the pending production arch (arch_terminal); fail ->
    smc_required (CFG faithful reconstruction but cannot land in R).
    """
    if cfg_landing is None:
        return DiagnosticReport(
            terminal="pending_cfg_landing_check",
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=(
                f"Reconstruction passed (via Test {via}). Proceed to Test B'': generate "
                "held-out samples from noise under rare-cohort CFG and pass "
                "(rare_guided, common_ref) as `cfg_landing`. The production arch is awarded "
                "only after CFG landing clears."
            ),
        )
    result_bpp = test_b_double_prime(
        cfg_landing[0], rare_emb, cfg_landing[1],
        cv=cv, fidelity_tol=fidelity_tol, drift_tol=drift_tol,
    )
    tests_run.append(result_bpp)
    if result_bpp.passed:
        return DiagnosticReport(
            terminal=arch_terminal,
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=(
                f"Reconstruction passed (via Test {via}) and Test B'' passed "
                f"(fidelity AUC={result_bpp.metrics['fidelity_auc']:.3f}, "
                f"drift AUC={result_bpp.metrics['drift_auc']:.3f}). CFG generation lands in "
                f"the rare region. ARCH: {arch_terminal}."
            ),
        )
    return DiagnosticReport(
        terminal="smc_required",
        tests_run=tests_run,
        pretraining_influence=pretraining_influence,
        summary=(
            f"Reconstruction passed (via Test {via}) but Test B'' FAILED ({result_bpp.notes}). "
            "Twisted-diffusion SMC resampler is the REQUIRED inference-time fix "
            f"(asymptotically unbiased, not optional); {arch_terminal} stays unreachable until "
            "CFG passes B'' or SMC-guided samples pass the check."
        ),
    )


# ─── Full decision procedure ──────────────────────────────────────────────────

def run_diagnostic(
    rare_emb: np.ndarray,
    common_emb: np.ndarray,
    pretraining_influence: bool = False,
    recon_b: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_b_prime: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_c: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cfg_landing: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cv: int = 5,
    auc_threshold: float = 0.70,
    reconstruction_tol: float = 0.20,
    auc_drop_tol: float = 0.05,
    fidelity_tol: float = 0.65,
    drift_tol: float = 0.70,
) -> DiagnosticReport:
    """Run the three-test decision procedure.

    GPU-bound steps (diffusion training, MEDS extraction) happen externally.
    Pass the resulting reconstructed embeddings via `recon_b`, `recon_b_prime`,
    and `recon_c`. If a reconstruction tuple is None, the procedure stops at
    `pending_*` and returns a partial report — the caller runs the GPU step and
    re-invokes with the results.

    Parameters
    ----------
    rare_emb, common_emb : embeddings from the frozen encoder.
    pretraining_influence : True if the caller controls the encoder's pretraining
        (e.g. SMB building their own JEPA). Determines SPT vs bound_scope at
        Test A failure.
    recon_b        : (rare_recon, common_recon) from standard-trained diffusion.
    recon_b_prime  : (rare_recon, common_recon) from tail-aware diffusion.
    recon_c        : (rare_recon, common_recon) from separate-latent diffusion.
    cfg_landing    : (rare_guided, common_ref) held-out CFG-generated samples for
        Test B''. When Test B/B' reconstruction passes and this is None, the
        procedure stops at `pending_cfg_landing_check`.
    auc_threshold      : Test A pass threshold on logistic ROC-AUC.
    reconstruction_tol : Test B/B'/C pass threshold: rare_l2 ≤ common_l2 × (1 + tol).
    auc_drop_tol       : Test B/B'/C pass threshold: AUC drop ≤ tol.
    fidelity_tol       : Test B'' pass threshold: fidelity AUC ≤ tol (guided ≈ real rare).
    drift_tol          : Test B'' pass threshold: drift AUC ≥ tol (guided ≠ common).

    Returns
    -------
    DiagnosticReport with terminal, tests_run, and a one-paragraph summary.
    """
    tests_run = []

    # ── Test A ────────────────────────────────────────────────────────────────
    result_a = test_a(
        rare_emb, common_emb,
        cv=cv, mlp_check=True, auc_threshold=auc_threshold,
    )
    tests_run.append(result_a)

    if not result_a.passed:
        if pretraining_influence:
            terminal = "spt_recommendation"
            summary = (
                f"Test A FAILED (logistic AUC={result_a.metrics['logistic_auc']:.3f}). "
                "The frozen encoder does not preserve rare-patient signal. "
                "Recommended fix: Specialized Pretraining (SPT) following Baek et al. 2026 "
                "(arXiv:2603.16177) — include rare-cohort trajectories as a fraction of "
                "pretraining tokens. SPT gains grow precisely when the target domain is "
                "underrepresented, and a 1B SPT model has been shown to outperform a 3B "
                "standard-pretrained model in low-frequency domains. "
                "Note: this result is established for text-domain LLMs; treat as a strong "
                "prior for EHR encoders, not a proven fix."
            )
        else:
            terminal = "bound_scope"
            summary = (
                f"Test A FAILED (logistic AUC={result_a.metrics['logistic_auc']:.3f}). "
                "The encoder does not preserve rare-patient signal, and pretraining "
                "influence is not available (frozen third-party encoder). "
                "Bound scope: the SCA approach cannot be validated for this rare "
                "subpopulation at the current encoder. No downstream fix (diffusion, "
                "latent) can recover information the encoder already destroyed."
            )
        return DiagnosticReport(
            terminal=terminal,
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=summary,
        )

    # ── Test B ────────────────────────────────────────────────────────────────
    if recon_b is None:
        return DiagnosticReport(
            terminal="pending_B",
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=(
                f"Test A PASSED (logistic AUC={result_a.metrics['logistic_auc']:.3f}). "
                "Proceed to Test B: train a score-based diffusion model on ZCA-whitened "
                "embeddings (standard training, no reweighting), run the round-trip, "
                "and pass (rare_recon, common_recon) as `recon_b`."
            ),
        )

    result_b = _reconstruction_test(
        rare_emb, common_emb, recon_b[0], recon_b[1],
        test_name="B",
        reconstruction_tol=reconstruction_tol,
        auc_drop_tol=auc_drop_tol,
        cv=cv,
    )
    tests_run.append(result_b)

    if result_b.passed:
        # Reconstruction faithful — but the arch is awarded only after Test B''
        # (CFG generative landing). Route through the gate.
        return _cfg_landing_gate(
            cfg_landing, "diffuse_directly", rare_emb, common_emb, tests_run,
            pretraining_influence, cv, fidelity_tol, drift_tol, via="B",
        )

    # ── Test B' ───────────────────────────────────────────────────────────────
    if recon_b_prime is None:
        return DiagnosticReport(
            terminal="pending_B_prime",
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=(
                f"Test B FAILED ({result_b.notes}). "
                "Proceed to Test B': retrain the diffusion with tail-aware training "
                "(importance-weight denoising loss by 1/p(z_i) using GMM log-density), "
                "run the round-trip, and pass (rare_recon, common_recon) as `recon_b_prime`."
            ),
        )

    result_bp = _reconstruction_test(
        rare_emb, common_emb, recon_b_prime[0], recon_b_prime[1],
        test_name="B_prime",
        reconstruction_tol=reconstruction_tol,
        auc_drop_tol=auc_drop_tol,
        cv=cv,
    )
    tests_run.append(result_bp)

    if result_bp.passed:
        # Tail-aware reconstruction faithful — arch awarded only after Test B''.
        return _cfg_landing_gate(
            cfg_landing, "tail_aware", rare_emb, common_emb, tests_run,
            pretraining_influence, cv, fidelity_tol, drift_tol, via="B'",
        )

    # ── Test C ────────────────────────────────────────────────────────────────
    if recon_c is None:
        return DiagnosticReport(
            terminal="pending_C",
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=(
                f"Tests B and B' FAILED. "
                "Proceed to Test C: train a learned latent (encode Z→Z', diffuse in Z', "
                "decode Z'→Z), run the round-trip, and pass (rare_recon, common_recon) "
                "as `recon_c`. Also enable the round-trip validator as a permanent gate: "
                "require BOTH low rare-mode L2 AND AUC preservation."
            ),
        )

    result_c = _reconstruction_test(
        rare_emb, common_emb, recon_c[0], recon_c[1],
        test_name="C",
        reconstruction_tol=reconstruction_tol,
        auc_drop_tol=auc_drop_tol,
        cv=cv,
    )
    tests_run.append(result_c)

    if result_c.passed:
        return DiagnosticReport(
            terminal="separate_latent_justified",
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=(
                f"Tests A, B', FAILED. Test C PASSED "
                f"(L2 ratio={result_c.metrics['l2_ratio']:.2f}, "
                f"AUC drop={result_c.metrics['auc_drop']:.3f}). "
                "The embedding geometry is hostile to diffusion at the rare scale; "
                "a separate learned latent recovers fidelity. Carry the round-trip "
                "validator (L2 + AUC gate) into production."
            ),
        )

    return DiagnosticReport(
        terminal="escalate",
        tests_run=tests_run,
        pretraining_influence=pretraining_influence,
        summary=(
            "Tests A PASSED but B, B', and C all FAILED. "
            "The embedding carries rare-patient signal but no diffusion architecture "
            "tested here preserves it. Likely cause: encoder geometry is load-bearing "
            "despite Test A passing (Test A tests linear separability; the rare region "
            "may be geometrically fragile to diffusion noise schedules). Bound scope "
            "for this rare subpopulation, or escalate to encoder-level investigation."
        ),
    )
