"""Characterization tests for the rare-detail localisation diagnostic.

localization.py was shipped untested. These pin its current behavior: Test A
(encoder capacity), the per-mode reconstruction metrics, the pass/fail
classifier, and every terminal of run_diagnostic's decision tree. All inputs
are constructed embeddings + reconstruction arrays — no diffusion/GPU.
"""
import numpy as np
import pytest

from causal_bench.diagnostics.localization import (
    per_mode_reconstruction_metrics,
    run_diagnostic,
)
# aliased so pytest doesn't collect the imported `test_*` functions as test cases
from causal_bench.diagnostics.localization import test_a as _test_a
from causal_bench.diagnostics.localization import test_b_double_prime as _test_b_double_prime

D = 5
_SHIFT = 3.0  # rare-cluster shift used throughout; cfg-landing fixtures below
              # must match it to stay consistent with _separable()'s default.


def _separable(seed=0, n_rare=40, n_common=200, shift=_SHIFT):
    """Rare shifted far from common -> linearly separable in embedding space."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((n_common, D))
    rare = rng.standard_normal((n_rare, D)) + shift
    return rare, common


def _faithful_recon(rare, common):
    """Round-trip that barely perturbs either mode -> faithful."""
    rng = np.random.default_rng(1)
    return (rare + 1e-3 * rng.standard_normal(rare.shape),
            common + 1e-3 * rng.standard_normal(common.shape))


def _collapsed_recon(rare, common):
    """Rare reconstructed into the common region (tail collapse): rare_l2 large
    and rare/common no longer separable -> both pass-criteria fail."""
    rng = np.random.default_rng(2)
    rare_recon = rng.standard_normal(rare.shape)          # unshifted ~ common
    common_recon = common + 1e-3 * rng.standard_normal(common.shape)
    return rare_recon, common_recon


def _cfg_landing_pass(rare, common, seed=10):
    """Held-out CFG-guided samples that land in R: drawn like `rare` (same
    shift), so they resemble real rare (low fidelity AUC) and differ from
    common (high drift AUC) -> Test B'' passes."""
    rng = np.random.default_rng(seed)
    rare_guided = rng.standard_normal((len(rare), D)) + _SHIFT
    return rare_guided, common


def _cfg_landing_collapsed(rare, common, seed=11):
    """Held-out CFG-guided samples collapsed to the bulk: drawn unshifted (like
    `common`), so they look like common (low drift AUC) and unlike real rare
    (high fidelity AUC) -> Test B'' fails on both metrics."""
    rng = np.random.default_rng(seed)
    rare_guided = rng.standard_normal((len(rare), D))
    return rare_guided, common


# ── Test A ────────────────────────────────────────────────────────────────────

def test_test_a_separable_passes():
    rare, common = _separable()
    r = _test_a(rare, common, mlp_check=False)
    assert r.test == "A" and r.passed is True
    assert r.metrics["logistic_auc"] >= 0.70
    assert r.metrics["n_rare"] == 40 and r.metrics["n_common"] == 200
    assert "Proceed to Test B" in r.notes


def test_test_a_overlapping_fails():
    rare, common = _separable(shift=0.0)   # same distribution -> AUC ~ 0.5
    r = _test_a(rare, common, mlp_check=False)
    assert r.passed is False
    assert r.metrics["logistic_auc"] < 0.70
    assert "encoder" in r.notes.lower()


def test_test_a_small_rare_reduces_cv_with_warning():
    rare, common = _separable(n_rare=4, n_common=50)
    with pytest.warns(RuntimeWarning, match="too small for cv"):
        r = _test_a(rare, common, cv=5, mlp_check=False)
    assert r.metrics["cv_used"] < 5


# ── Per-mode reconstruction metrics ───────────────────────────────────────────

def test_reconstruction_metrics_faithful_vs_collapse():
    rare, common = _separable()
    # faithful: tiny L2, separation preserved
    rr, cr = _faithful_recon(rare, common)
    m = per_mode_reconstruction_metrics(rare, common, rr, cr)
    assert m["l2_ratio"] < 1.2
    assert abs(m["auc_drop"]) < 0.05
    # collapse: rare L2 >> common L2, separation destroyed
    rr2, cr2 = _collapsed_recon(rare, common)
    m2 = per_mode_reconstruction_metrics(rare, common, rr2, cr2)
    assert m2["l2_ratio"] > 1.2
    assert m2["auc_drop"] > 0.05


# ── Test B'' (CFG generative landing) ─────────────────────────────────────────

def test_test_b_double_prime_lands_in_rare_region():
    rare, common = _separable()
    rare_guided, common_ref = _cfg_landing_pass(rare, common)
    r = _test_b_double_prime(rare_guided, rare, common_ref)
    assert r.test == "B_double_prime" and r.passed is True
    assert r.metrics["fidelity_auc"] <= 0.65
    assert r.metrics["drift_auc"] >= 0.70
    assert "lands in R" in r.notes


def test_test_b_double_prime_collapsed_to_bulk_fails():
    rare, common = _separable()
    rare_guided, common_ref = _cfg_landing_collapsed(rare, common)
    r = _test_b_double_prime(rare_guided, rare, common_ref)
    assert r.passed is False
    assert r.metrics["fidelity_auc"] > 0.65   # guided distinguishable from real rare
    assert r.metrics["drift_auc"] < 0.70      # guided indistinguishable from common
    assert "poor score in tail" in r.notes and "collapsed to bulk" in r.notes


# ── run_diagnostic: the CFG-landing gate on B/B' passes ───────────────────────

def test_run_diagnostic_diffuse_directly_when_b_and_cfg_pass():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_faithful_recon(rare, common),
        cfg_landing=_cfg_landing_pass(rare, common),
    )
    assert rep.terminal == "diffuse_directly"
    assert [t.test for t in rep.tests_run] == ["A", "B", "B_double_prime"]


def test_run_diagnostic_smc_required_when_b_passes_but_cfg_fails():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_faithful_recon(rare, common),
        cfg_landing=_cfg_landing_collapsed(rare, common),
    )
    assert rep.terminal == "smc_required"
    assert [t.test for t in rep.tests_run] == ["A", "B", "B_double_prime"]


def test_run_diagnostic_tail_aware_when_bprime_and_cfg_pass():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_collapsed_recon(rare, common),
        recon_b_prime=_faithful_recon(rare, common),
        cfg_landing=_cfg_landing_pass(rare, common),
    )
    assert rep.terminal == "tail_aware"
    assert [t.test for t in rep.tests_run] == ["A", "B", "B_prime", "B_double_prime"]


def test_run_diagnostic_smc_required_via_bprime_when_cfg_fails():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_collapsed_recon(rare, common),
        recon_b_prime=_faithful_recon(rare, common),
        cfg_landing=_cfg_landing_collapsed(rare, common),
    )
    assert rep.terminal == "smc_required"


# ── run_diagnostic decision tree — every terminal ─────────────────────────────

def test_run_diagnostic_test_a_fail_pretraining_spt():
    rare, common = _separable(shift=0.0)
    rep = run_diagnostic(rare, common, pretraining_influence=True)
    assert rep.terminal == "spt_recommendation"
    assert len(rep.tests_run) == 1 and rep.tests_run[0].passed is False


def test_run_diagnostic_test_a_fail_no_pretraining_bound_scope():
    rare, common = _separable(shift=0.0)
    rep = run_diagnostic(rare, common, pretraining_influence=False)
    assert rep.terminal == "bound_scope"


def test_run_diagnostic_pending_b_when_recon_missing():
    rare, common = _separable()
    rep = run_diagnostic(rare, common)           # recon_b=None
    assert rep.terminal == "pending_B"


def test_run_diagnostic_pending_cfg_landing_check_when_b_faithful_no_landing():
    # Since Test B'' (#79/2026-07-02 diagram), a faithful B reconstruction alone
    # no longer awards diffuse_directly -- it awaits CFG-landing samples.
    rare, common = _separable()
    rep = run_diagnostic(rare, common, recon_b=_faithful_recon(rare, common))
    assert rep.terminal == "pending_cfg_landing_check"
    assert [t.test for t in rep.tests_run] == ["A", "B"]


def test_run_diagnostic_pending_b_prime_when_b_fails_and_bprime_missing():
    rare, common = _separable()
    rep = run_diagnostic(rare, common, recon_b=_collapsed_recon(rare, common))
    assert rep.terminal == "pending_B_prime"


def test_run_diagnostic_pending_cfg_landing_check_when_bprime_faithful_no_landing():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_collapsed_recon(rare, common),
        recon_b_prime=_faithful_recon(rare, common),
    )
    assert rep.terminal == "pending_cfg_landing_check"
    assert [t.test for t in rep.tests_run] == ["A", "B", "B_prime"]


def test_run_diagnostic_pending_c_when_b_and_bprime_fail():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_collapsed_recon(rare, common),
        recon_b_prime=_collapsed_recon(rare, common),
    )
    assert rep.terminal == "pending_C"


def test_run_diagnostic_separate_latent_when_c_faithful():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_collapsed_recon(rare, common),
        recon_b_prime=_collapsed_recon(rare, common),
        recon_c=_faithful_recon(rare, common),
    )
    assert rep.terminal == "separate_latent_justified"


def test_run_diagnostic_escalate_when_all_recon_fail():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_collapsed_recon(rare, common),
        recon_b_prime=_collapsed_recon(rare, common),
        recon_c=_collapsed_recon(rare, common),
    )
    assert rep.terminal == "escalate"
    assert [t.test for t in rep.tests_run] == ["A", "B", "B_prime", "C"]
