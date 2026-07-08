"""Rare-detail localisation diagnostic — CPU-side decision procedure.

Implements the three-test decision procedure from the GPU build spec as
encoder-agnostic Python. All test functions take np.ndarray embeddings and
pre-computed reconstruction arrays; diffusion model training and MEDS/encoder
loading are intentionally excluded and belong in the Lambda GPU script.

Decision tree
-------------
                    ┌ Test A pass ─► Test B ─ recon pass ─► Test B″ ─ land ─► diffuse_directly
                    │                    └ recon fail ─► Test B' ─ pass ─► Test B″ ─ land ─► tail_aware
run_diagnostic ────┤                                          └ fail ─► Test C ─ pass ─► separate_latent_justified
                    │                                                           └ fail ─► escalate
                    └ Test A fail ─► pretraining_influence? ─ yes ─► spt_recommendation
                                                            └ no  ─► bound_scope

Test B″ (CFG generative landing) gates diffuse_directly / tail_aware. A faithful
round-trip (B/B') earns only `pending_cfg_landing_check` until guided-generation
samples (rare_guided, common_ref) are supplied; if CFG-guided generation cannot
land in the rare region R, the terminal is `smc_required` (the twisted-diffusion
SMC reranker is the fix), NOT diffuse_directly.

Metric-hacking guard (Van Assel et al. 2026, arXiv:2606.00514; causal_bench#88):
fidelity verdicts are by default scored in the generation encoder's own space.
Supply `emb_eval` + `recon_*_eval` (the same patients/samples re-encoded by a
DECOUPLED encoder E_eval != E_gen) and pass/fail gates on that space instead,
flagging `metric_hacking` when the generation space passes but E_eval fails.

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
    test: str       # "A", "B", "B_prime", "B_double_prime", "C"
    passed: bool
    metrics: dict   # raw metric values (AUC, L2 ratios, etc.)
    notes: str      # one-line human-readable interpretation


@dataclass
class DiagnosticReport:
    """Full decision-procedure report.

    Terminal values and their meanings
    -----------------------------------
    diffuse_directly           Test B recon pass AND Test B″ CFG landing pass: diffuse on embeddings.
    tail_aware                 Test B fail, B' recon pass AND Test B″ landing pass: tail-aware training.
    separate_latent_justified  Tests B+B' failed, C passed: learned latent is warranted.
    smc_required               Recon faithful but CFG fails B″ landing: twisted-diffusion SMC reranker required.
    spt_recommendation         Test A failed + pretraining influence: fix at encoder via SPT.
    bound_scope                Test A failed, no pretraining influence: encoder is the limit.
    escalate                   Tests B+B'+C all failed: no available fix; bound scope.
    pending_B                  Test A passed but B reconstructions not yet provided.
    pending_B_prime            Test B failed but B' reconstructions not yet provided.
    pending_C                  Tests B+B' failed but C reconstructions not yet provided.
    pending_cfg_landing_check  Reconstruction passed but CFG-landing samples not yet supplied.

    Any reconstruction test's metrics carry `metric_hacking_flag` — True when the
    round-trip passes in the generation encoder's space but fails under a supplied
    decoupled encoder E_eval (Van Assel et al. 2026; causal_bench#88).
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


# ─── Shared: cross-validated pairwise separation AUC ─────────────────────────

def _pairwise_separation_auc(emb_a: np.ndarray, emb_b: np.ndarray, cv: int = 5) -> float:
    """CV logistic ROC-AUC separating `emb_a` (label 1) from `emb_b` (label 0).

    Shared by the reconstruction metrics (rare vs common) and the CFG-landing
    test (guided vs real, guided vs common). Folds are reduced automatically when
    the smaller group is too small.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([emb_a, emb_b])
    y = np.concatenate([np.ones(len(emb_a)), np.zeros(len(emb_b))])
    n_min = min(len(emb_a), len(emb_b))
    cv_safe = max(2, min(cv, n_min // 2)) if n_min >= 4 else 2
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
    rare_l2   = float(np.mean(np.linalg.norm(rare_orig   - rare_recon,   axis=1)))
    common_l2 = float(np.mean(np.linalg.norm(common_orig - common_recon, axis=1)))
    l2_ratio  = rare_l2 / max(common_l2, 1e-12)

    sep_orig  = _pairwise_separation_auc(rare_orig,  common_orig,  cv=cv)
    sep_recon = _pairwise_separation_auc(rare_recon, common_recon, cv=cv)

    return {
        "rare_l2_mean":          rare_l2,
        "common_l2_mean":        common_l2,
        "l2_ratio":              l2_ratio,
        "separation_auc_orig":   sep_orig,
        "separation_auc_recon":  sep_recon,
        "auc_drop":              sep_orig - sep_recon,
    }


# ─── Test B″: CFG generative landing ─────────────────────────────────────────

def cfg_landing_test(
    rare_guided: np.ndarray,
    real_rare: np.ndarray,
    common_ref: np.ndarray,
    cv: int = 5,
    fidelity_tol: float = 0.65,
    drift_threshold: float = 0.70,
) -> LocalizationResult:
    """Test B″ — CFG generative landing: does generation-from-noise land in R?

    Reconstruction (B/B') only tests denoising near existing points. This tests
    whether CFG-guided *generation from noise* lands in the rare region R. Two
    metrics, never collapsed into one:

      fidelity_auc : classifier `rare_guided` vs REAL rare — LOWER is better
                     (near 0.5 ⟹ generated samples indistinguishable from real rare;
                      high ⟹ poor score in the tail).
      drift_auc    : classifier `rare_guided` vs common    — HIGHER is better
                     (high ⟹ CFG held samples in R; low ⟹ conditioning too weak,
                      samples collapsed toward the bulk).

    Pass iff `fidelity_auc <= fidelity_tol` AND `drift_auc >= drift_threshold`.
    """
    fidelity_auc = _pairwise_separation_auc(rare_guided, real_rare,  cv=cv)
    drift_auc    = _pairwise_separation_auc(rare_guided, common_ref, cv=cv)
    fidelity_ok  = fidelity_auc <= fidelity_tol
    drift_ok     = drift_auc >= drift_threshold
    passed       = fidelity_ok and drift_ok

    metrics = {
        "fidelity_auc":    fidelity_auc,
        "drift_auc":       drift_auc,
        "fidelity_tol":    fidelity_tol,
        "drift_threshold": drift_threshold,
    }
    if passed:
        notes = (
            f"Test B″: CFG lands in R (fidelity AUC={fidelity_auc:.3f} ≤ {fidelity_tol} "
            f"AND drift AUC={drift_auc:.3f} ≥ {drift_threshold})."
        )
    else:
        reasons = []
        if not fidelity_ok:
            reasons.append(
                f"fidelity AUC={fidelity_auc:.3f} > {fidelity_tol} "
                "(guided samples distinguishable from real rare — poor tail score)"
            )
        if not drift_ok:
            reasons.append(
                f"drift AUC={drift_auc:.3f} < {drift_threshold} "
                "(conditioning too weak — collapsed toward the bulk)"
            )
        notes = "Test B″ FAILED: " + "; ".join(reasons) + "."

    return LocalizationResult(test="B_double_prime", passed=passed, metrics=metrics, notes=notes)


# ─── Test B / B' / C shared logic ─────────────────────────────────────────────

def _recon_notes(test_name, m, reconstruction_tol, auc_drop_tol, passed, space) -> str:
    """Human-readable pass/fail note for a reconstruction test in a given space."""
    if passed:
        return (
            f"Test {test_name} ({space}): rare L2/common L2 = {m['l2_ratio']:.2f} ≤ "
            f"{1 + reconstruction_tol:.2f} and AUC drop = {m['auc_drop']:.3f} ≤ {auc_drop_tol}. "
            "Round-trip is faithful."
        )
    reasons = []
    if m["l2_ratio"] > 1.0 + reconstruction_tol:
        reasons.append(
            f"rare L2/common L2 = {m['l2_ratio']:.2f} > {1 + reconstruction_tol:.2f} (tail collapse)"
        )
    if m["auc_drop"] > auc_drop_tol:
        reasons.append(
            f"AUC drop = {m['auc_drop']:.3f} > {auc_drop_tol} (separation degraded)"
        )
    return f"Test {test_name} FAILED ({space}): " + "; ".join(reasons) + "."


def _reconstruction_test(
    rare_orig: np.ndarray,
    common_orig: np.ndarray,
    rare_recon: np.ndarray,
    common_recon: np.ndarray,
    test_name: str,
    reconstruction_tol: float,
    auc_drop_tol: float,
    cv: int,
    eval_orig: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    eval_recon: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> LocalizationResult:
    """Run per-mode reconstruction metrics and classify pass/fail.

    When `eval_orig`/`eval_recon` (the same patients re-encoded by a decoupled
    encoder E_eval) are supplied, the verdict gates on the *decoupled* space and
    the generation-space metrics are retained under `gen_*` keys with a
    `metric_hacking_flag` (True ⟺ generation space passes but E_eval fails) —
    the Van Assel et al. 2026 guard (causal_bench#88).
    """
    def _verdict(m):
        return (m["l2_ratio"] <= 1.0 + reconstruction_tol) and (m["auc_drop"] <= auc_drop_tol)

    m_gen = per_mode_reconstruction_metrics(
        rare_orig, common_orig, rare_recon, common_recon, cv=cv
    )
    gen_pass = _verdict(m_gen)

    if eval_orig is not None and eval_recon is not None:
        m_eval = per_mode_reconstruction_metrics(
            eval_orig[0], eval_orig[1], eval_recon[0], eval_recon[1], cv=cv
        )
        eval_pass = _verdict(m_eval)
        passed = eval_pass                       # gate on the DECOUPLED space
        metric_hacking = gen_pass and not eval_pass
        metrics = dict(m_eval)
        metrics.update({f"gen_{k}": v for k, v in m_gen.items()})
        metrics["metric_hacking_flag"] = metric_hacking
        if metric_hacking:
            notes = (
                f"Test {test_name} METRIC-HACKING: passes in the generation space "
                f"(gen l2_ratio={m_gen['l2_ratio']:.2f}, gen auc_drop={m_gen['auc_drop']:.3f}) "
                f"but FAILS under decoupled E_eval "
                f"(l2_ratio={m_eval['l2_ratio']:.2f}, auc_drop={m_eval['auc_drop']:.3f}). "
                "The round-trip is gamed in E_gen's own geometry — treat as FAIL."
            )
        else:
            notes = _recon_notes(
                test_name, m_eval, reconstruction_tol, auc_drop_tol, passed, "decoupled E_eval"
            )
    else:
        passed = gen_pass
        metrics = dict(m_gen)
        metrics["metric_hacking_flag"] = False
        notes = _recon_notes(
            test_name, m_gen, reconstruction_tol, auc_drop_tol, passed, "generation space"
        )

    return LocalizationResult(test=test_name, passed=passed, metrics=metrics, notes=notes)


# ─── Full decision procedure ──────────────────────────────────────────────────

def run_diagnostic(
    rare_emb: np.ndarray,
    common_emb: np.ndarray,
    pretraining_influence: bool = False,
    recon_b: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_b_prime: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_c: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rare_guided: Optional[np.ndarray] = None,
    common_ref: Optional[np.ndarray] = None,
    emb_eval: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_b_eval: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_b_prime_eval: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_c_eval: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rare_guided_eval: Optional[np.ndarray] = None,
    common_ref_eval: Optional[np.ndarray] = None,
    cv: int = 5,
    auc_threshold: float = 0.70,
    reconstruction_tol: float = 0.20,
    auc_drop_tol: float = 0.05,
    fidelity_tol: float = 0.65,
    drift_threshold: float = 0.70,
) -> DiagnosticReport:
    """Run the decision procedure (Tests A → B → B″ / B' → B″ / C).

    GPU-bound steps (diffusion training, MEDS extraction, guided generation)
    happen externally. Pass reconstructed embeddings via `recon_b`,
    `recon_b_prime`, `recon_c`; pass CFG-guided-generation samples via
    `rare_guided` (+ `common_ref`). If a required array is None the procedure
    stops at a `pending_*` terminal and returns a partial report.

    Parameters
    ----------
    rare_emb, common_emb : embeddings from the frozen generation encoder E_gen.
    pretraining_influence : True if the caller controls the encoder's pretraining.
    recon_b / recon_b_prime / recon_c : (rare_recon, common_recon) round-trips
        from standard / tail-aware / separate-latent diffusion.
    rare_guided : samples generated from noise under the rare-cohort CFG condition
        (a held-out generation run, NOT a round-tripped training embedding).
    common_ref  : common-mode reference set for the Test B″ drift metric.
    emb_eval : (rare, common) re-encoded by a DECOUPLED encoder E_eval != E_gen.
    recon_*_eval / rare_guided_eval / common_ref_eval : the corresponding arrays
        in E_eval space. When supplied, fidelity verdicts gate on E_eval (the
        metric-hacking guard, Van Assel et al. 2026; causal_bench#88).
    auc_threshold      : Test A pass threshold on logistic ROC-AUC.
    reconstruction_tol : Test B/B'/C pass: rare_l2 ≤ common_l2 × (1 + tol).
    auc_drop_tol       : Test B/B'/C pass: separation-AUC drop ≤ tol.
    fidelity_tol       : Test B″ pass: fidelity AUC (guided vs real rare) ≤ tol.
    drift_threshold    : Test B″ pass: drift AUC (guided vs common) ≥ threshold.

    Returns
    -------
    DiagnosticReport with terminal, tests_run, and a one-paragraph summary.
    """
    tests_run = []

    def _report(terminal, summary):
        return DiagnosticReport(
            terminal=terminal,
            tests_run=tests_run,
            pretraining_influence=pretraining_influence,
            summary=summary,
        )

    def _eval_kwargs(recon_eval):
        """Build eval_orig/eval_recon kwargs for a reconstruction test when a
        decoupled encoder's embeddings are available."""
        if emb_eval is not None and recon_eval is not None:
            return dict(eval_orig=(emb_eval[0], emb_eval[1]), eval_recon=recon_eval)
        return {}

    def _landing_gate(award_terminal, via):
        """Test B″ CFG landing gate. Award `award_terminal` only if CFG-guided
        generation lands in R; otherwise `smc_required`. `pending_cfg_landing_check`
        if guided samples are not supplied."""
        if rare_guided is None or common_ref is None:
            return _report(
                "pending_cfg_landing_check",
                f"Reconstruction passed (via Test {via}), but this is a denoising test "
                "near existing points — it does not establish that CFG-guided GENERATION "
                "from noise lands in R. Supply `rare_guided` (a held-out generation run "
                "under the rare-cohort condition) and `common_ref` to run Test B″ before "
                f"awarding {award_terminal}.",
            )
        use_eval = (rare_guided_eval is not None and common_ref_eval is not None
                    and emb_eval is not None)
        rg = rare_guided_eval if use_eval else rare_guided
        cr = common_ref_eval if use_eval else common_ref
        real_rare = emb_eval[0] if use_eval else rare_emb
        space = "decoupled E_eval" if use_eval else "generation space"

        result_land = cfg_landing_test(
            rg, real_rare, cr, cv=cv,
            fidelity_tol=fidelity_tol, drift_threshold=drift_threshold,
        )
        tests_run.append(result_land)

        if result_land.passed:
            return _report(
                award_terminal,
                f"Test A PASSED, reconstruction faithful (via Test {via}), Test B″ CFG "
                f"landing PASSED in the {space} (fidelity AUC="
                f"{result_land.metrics['fidelity_auc']:.3f}, drift AUC="
                f"{result_land.metrics['drift_auc']:.3f}). Award {award_terminal} — "
                "diffuse on embeddings, no separate latent needed.",
            )
        return _report(
            "smc_required",
            f"Reconstruction faithful (via Test {via}) but Test B″ FAILED in the {space}: "
            f"{result_land.notes} CFG-guided generation cannot land in R despite faithful "
            "reconstruction — the structural bias of single-trajectory CFG in this "
            "far-from-mass regime. The twisted-diffusion SMC reranker is the required "
            f"inference-time fix; {award_terminal} stays UNREACHABLE until CFG passes B″ "
            "or SMC-guided samples pass the same landing check.",
        )

    # ── Test A ────────────────────────────────────────────────────────────────
    result_a = test_a(
        rare_emb, common_emb,
        cv=cv, mlp_check=True, auc_threshold=auc_threshold,
    )
    tests_run.append(result_a)

    if not result_a.passed:
        if pretraining_influence:
            return _report(
                "spt_recommendation",
                f"Test A FAILED (logistic AUC={result_a.metrics['logistic_auc']:.3f}). "
                "The frozen encoder does not preserve rare-patient signal. "
                "Recommended fix: Specialized Pretraining (SPT) following Baek et al. 2026 "
                "(arXiv:2603.16177) — include rare-cohort trajectories as a fraction of "
                "pretraining tokens. SPT gains grow precisely when the target domain is "
                "underrepresented, and a 1B SPT model has been shown to outperform a 3B "
                "standard-pretrained model in low-frequency domains. "
                "Note: this result is established for text-domain LLMs; treat as a strong "
                "prior for EHR encoders, not a proven fix.",
            )
        return _report(
            "bound_scope",
            f"Test A FAILED (logistic AUC={result_a.metrics['logistic_auc']:.3f}). "
            "The encoder does not preserve rare-patient signal, and pretraining "
            "influence is not available (frozen third-party encoder). "
            "Bound scope: the SCA approach cannot be validated for this rare "
            "subpopulation at the current encoder. No downstream fix (diffusion, "
            "latent) can recover information the encoder already destroyed.",
        )

    # ── Test B ────────────────────────────────────────────────────────────────
    if recon_b is None:
        return _report(
            "pending_B",
            f"Test A PASSED (logistic AUC={result_a.metrics['logistic_auc']:.3f}). "
            "Proceed to Test B: train a score-based diffusion model on ZCA-whitened "
            "embeddings (standard training, no reweighting), run the round-trip, "
            "and pass (rare_recon, common_recon) as `recon_b`.",
        )

    result_b = _reconstruction_test(
        rare_emb, common_emb, recon_b[0], recon_b[1],
        test_name="B",
        reconstruction_tol=reconstruction_tol,
        auc_drop_tol=auc_drop_tol,
        cv=cv,
        **_eval_kwargs(recon_b_eval),
    )
    tests_run.append(result_b)

    if result_b.passed:
        return _landing_gate("diffuse_directly", via="B")

    # ── Test B' ───────────────────────────────────────────────────────────────
    if recon_b_prime is None:
        return _report(
            "pending_B_prime",
            f"Test B FAILED ({result_b.notes}). "
            "Proceed to Test B': retrain the diffusion with tail-aware training "
            "(importance-weight denoising loss by 1/p(z_i) using GMM log-density), "
            "run the round-trip, and pass (rare_recon, common_recon) as `recon_b_prime`.",
        )

    result_bp = _reconstruction_test(
        rare_emb, common_emb, recon_b_prime[0], recon_b_prime[1],
        test_name="B_prime",
        reconstruction_tol=reconstruction_tol,
        auc_drop_tol=auc_drop_tol,
        cv=cv,
        **_eval_kwargs(recon_b_prime_eval),
    )
    tests_run.append(result_bp)

    if result_bp.passed:
        return _landing_gate("tail_aware", via="B'")

    # ── Test C ────────────────────────────────────────────────────────────────
    if recon_c is None:
        return _report(
            "pending_C",
            "Tests B and B' FAILED. "
            "Proceed to Test C: train a learned latent (encode Z→Z', diffuse in Z', "
            "decode Z'→Z), run the round-trip, and pass (rare_recon, common_recon) "
            "as `recon_c`. Also enable the round-trip validator as a permanent gate: "
            "require BOTH low rare-mode L2 AND AUC preservation.",
        )

    result_c = _reconstruction_test(
        rare_emb, common_emb, recon_c[0], recon_c[1],
        test_name="C",
        reconstruction_tol=reconstruction_tol,
        auc_drop_tol=auc_drop_tol,
        cv=cv,
        **_eval_kwargs(recon_c_eval),
    )
    tests_run.append(result_c)

    if result_c.passed:
        return _report(
            "separate_latent_justified",
            f"Tests A, B', FAILED. Test C PASSED "
            f"(L2 ratio={result_c.metrics['l2_ratio']:.2f}, "
            f"AUC drop={result_c.metrics['auc_drop']:.3f}). "
            "The embedding geometry is hostile to diffusion at the rare scale; "
            "a separate learned latent recovers fidelity. Carry the round-trip "
            "validator (L2 + AUC gate) into production.",
        )

    return _report(
        "escalate",
        "Tests A PASSED but B, B', and C all FAILED. "
        "The embedding carries rare-patient signal but no diffusion architecture "
        "tested here preserves it. Likely cause: encoder geometry is load-bearing "
        "despite Test A passing (Test A tests linear separability; the rare region "
        "may be geometrically fragile to diffusion noise schedules). Bound scope "
        "for this rare subpopulation, or escalate to encoder-level investigation.",
    )


# ─── Lineage-collapse (consumes the SMC sampler's ancestor multiplicity) ──────

def lineage_collapse_score(multiplicity) -> float:
    """Normalized Gini of ancestor multiplicity: 0 = every particle survives
    equally, ->1 = a handful of survivors dominate (rare-event degeneracy).

    Consumes `causal_bench.sampling.diagnostics.lineage_multiplicity(result)` —
    the histogram of how many descendants each particle has at the last resample.
    A high score is the sampler-side signature of the same rare-mode collapse the
    reconstruction tests detect on the generative side.
    """
    m = np.sort(np.asarray(multiplicity, dtype=float))
    n = len(m)
    if n == 0 or m.sum() == 0:
        return 0.0
    cum = np.cumsum(m)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
