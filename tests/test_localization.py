"""Characterization tests for the rare-detail localisation diagnostic.

localization.py was shipped untested. These pin its current behavior: Test A
(encoder capacity), the per-mode reconstruction metrics, the pass/fail
classifier, and every terminal of run_diagnostic's decision tree. All inputs
are constructed embeddings + reconstruction arrays — no diffusion/GPU.
"""
import numpy as np
import pytest

from causal_bench.diagnostics.localization import (
    cfg_landing_test,
    per_mode_reconstruction_metrics,
    run_diagnostic,
)
# aliased so pytest doesn't collect the imported `test_a` as a test case
from causal_bench.diagnostics.localization import test_a as _test_a

D = 5


def _separable(seed=0, n_rare=40, n_common=200, shift=3.0):
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


def _good_landing(seed=3, n_guided=40, shift=3.0):
    """CFG-guided samples that land in R: near REAL rare (fidelity low) and far
    from common (drift high)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_guided, D)) + shift


def _drifted_landing(seed=4, n_guided=40):
    """CFG-guided samples that collapsed toward the bulk: distinguishable from
    real rare (fidelity high) AND indistinguishable from common (drift low)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_guided, D))             # unshifted ~ common


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


def test_run_diagnostic_faithful_b_without_landing_is_pending():
    # Faithful reconstruction alone no longer earns diffuse_directly — it must
    # clear the Test B″ CFG-landing gate first.
    rare, common = _separable()
    rep = run_diagnostic(rare, common, recon_b=_faithful_recon(rare, common))
    assert rep.terminal == "pending_cfg_landing_check"
    assert [t.test for t in rep.tests_run] == ["A", "B"]


def test_run_diagnostic_diffuse_directly_when_b_and_landing_pass():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_faithful_recon(rare, common),
        rare_guided=_good_landing(),
        common_ref=common,
    )
    assert rep.terminal == "diffuse_directly"
    assert [t.test for t in rep.tests_run] == ["A", "B", "B_double_prime"]


def test_run_diagnostic_smc_required_when_landing_fails():
    # Reconstruction faithful, but CFG-guided generation drifts to the bulk.
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_faithful_recon(rare, common),
        rare_guided=_drifted_landing(),
        common_ref=common,
    )
    assert rep.terminal == "smc_required"
    assert rep.tests_run[-1].test == "B_double_prime"
    assert rep.tests_run[-1].passed is False


def test_run_diagnostic_pending_b_prime_when_b_fails_and_bprime_missing():
    rare, common = _separable()
    rep = run_diagnostic(rare, common, recon_b=_collapsed_recon(rare, common))
    assert rep.terminal == "pending_B_prime"


def test_run_diagnostic_tail_aware_when_bprime_faithful_and_landing_passes():
    rare, common = _separable()
    rep = run_diagnostic(
        rare, common,
        recon_b=_collapsed_recon(rare, common),
        recon_b_prime=_faithful_recon(rare, common),
        rare_guided=_good_landing(),
        common_ref=common,
    )
    assert rep.terminal == "tail_aware"
    assert [t.test for t in rep.tests_run] == ["A", "B", "B_prime", "B_double_prime"]


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


# ── Test B″ CFG landing (unit) ────────────────────────────────────────────────

def test_cfg_landing_pass_when_guided_matches_rare_and_avoids_common():
    rare, common = _separable()
    r = cfg_landing_test(_good_landing(), rare, common)
    assert r.test == "B_double_prime" and r.passed is True
    assert r.metrics["fidelity_auc"] <= 0.65     # indistinguishable from real rare
    assert r.metrics["drift_auc"] >= 0.70        # held in R, distinct from common


def test_cfg_landing_fails_when_guided_drifts_to_bulk():
    rare, common = _separable()
    r = cfg_landing_test(_drifted_landing(), rare, common)
    assert r.passed is False
    # drifted samples are distinguishable from real rare (fidelity high) — a fail
    assert r.metrics["fidelity_auc"] > 0.65


# ── Metric-hacking guard (decoupled E_eval, #88) ──────────────────────────────

def test_metric_hacking_flag_when_gen_passes_but_eval_fails():
    # Round-trip looks faithful in the generation space but the rare mode has
    # collapsed under the decoupled encoder E_eval -> Test B gates on E_eval and
    # fails, raising the metric_hacking flag.
    rare, common = _separable(seed=0)
    rare_eval, common_eval = _separable(seed=5)          # same patients in E_eval space
    rep = run_diagnostic(
        rare, common,
        recon_b=_faithful_recon(rare, common),           # gen space: faithful
        emb_eval=(rare_eval, common_eval),
        recon_b_eval=_collapsed_recon(rare_eval, common_eval),   # E_eval: collapsed
    )
    result_b = [t for t in rep.tests_run if t.test == "B"][0]
    assert result_b.passed is False                      # gated on the decoupled space
    assert result_b.metrics["metric_hacking_flag"] is True
    assert "gen_l2_ratio" in result_b.metrics            # gen-space metrics retained
    # gen-space passed but decoupled space did not, so B is treated as a failure:
    assert rep.terminal in ("pending_B_prime", "smc_required", "tail_aware", "separate_latent_justified", "escalate", "pending_C")


def test_no_metric_hacking_flag_without_eval_inputs():
    rare, common = _separable()
    rep = run_diagnostic(rare, common, recon_b=_faithful_recon(rare, common))
    result_b = [t for t in rep.tests_run if t.test == "B"][0]
    assert result_b.metrics["metric_hacking_flag"] is False


# ── lineage-collapse score (SMC ancestor multiplicity → degeneracy signal) ────

def test_lineage_collapse_score_uniform_vs_collapsed():
    from causal_bench.diagnostics.localization import lineage_collapse_score
    assert lineage_collapse_score(np.ones(10)) < 1e-9          # uniform survival -> 0
    collapsed = np.zeros(10); collapsed[0] = 10.0
    assert lineage_collapse_score(collapsed) > 0.8             # near-total collapse -> ~1
    assert lineage_collapse_score([]) == 0.0                   # empty -> 0
    assert lineage_collapse_score(np.zeros(5)) == 0.0          # no survivors -> 0
