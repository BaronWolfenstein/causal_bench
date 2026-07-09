"""Test ELF render->re-encode + the #88 metric-hacking guard.

A reconstruction that is faithful in E_gen but collapses after render->re-encode
through E_eval should raise metric_hacking_flag in run_diagnostic across all gates
(Test B, Test B', and the B'' landing gate).
"""
import numpy as np
import pytest
from causal_bench.generative.encoder import make_encoder_pair
from causal_bench.generative.render import CodebookRenderer, render_and_reencode, eval_space_inputs
from causal_bench.diagnostics.localization import run_diagnostic


def test_render_reencode_surfaces_metric_hacking_test_b():
    """Test B metric-hacking guard: passes in E_gen, fails in E_eval.

    The codebook consists of only common raw features (rare detail is NOT
    representable). When we render a rare embedding, it snaps to a common token.
    In E_gen space, we hand back the original rare embeddings (faithful).
    In E_eval space, render->re-encode collapses the rare mode.
    This should trigger metric_hacking_flag = True on Test B.
    """
    rng = np.random.default_rng(0)
    in_dim, out_dim = 8, 6
    e_gen, e_eval = make_encoder_pair(in_dim, out_dim)

    # Generate rare (far from origin) and common (near origin) raw features
    raw_rare = rng.standard_normal((40, in_dim)) + 3.0
    raw_common = rng.standard_normal((200, in_dim))

    # Encode to embedding space
    rare = e_gen(raw_rare)
    common = e_gen(raw_common)

    # Codebook = common raw features only (rare detail NOT representable)
    renderer = CodebookRenderer(codebook_raw=raw_common)

    # In E_gen space, reconstruction is faithful (we hand back the originals)
    recon_b = (rare.copy(), common.copy())

    # In E_eval space, render->re-encode collapses the rare embeddings
    rare_eval = e_eval(raw_rare)
    common_eval = e_eval(raw_common)
    rare_recon_eval = render_and_reencode(rare, renderer, e_eval)
    common_recon_eval = render_and_reencode(common, renderer, e_eval)

    # Run diagnostic
    rep = run_diagnostic(
        rare, common,
        recon_b=recon_b,
        emb_eval=(rare_eval, common_eval),
        recon_b_eval=(rare_recon_eval, common_recon_eval),
    )

    # Check that Test B result has metric_hacking_flag = True
    result_b = [t for t in rep.tests_run if t.test == "B"][0]
    assert result_b.metrics["metric_hacking_flag"] is True, (
        "Test B should have metric_hacking_flag=True when passes in E_gen "
        "but fails in E_eval"
    )


def test_render_reencode_surfaces_metric_hacking_test_b_prime():
    """Test B' metric-hacking guard: passes in E_gen, fails in E_eval.

    Similar to Test B but entering via tail-aware retraining path.
    """
    rng = np.random.default_rng(1)
    in_dim, out_dim = 8, 6
    e_gen, e_eval = make_encoder_pair(in_dim, out_dim)

    # Generate rare and common raw features
    raw_rare = rng.standard_normal((40, in_dim)) + 3.0
    raw_common = rng.standard_normal((200, in_dim))

    rare = e_gen(raw_rare)
    common = e_gen(raw_common)

    # Codebook = common raw features only
    renderer = CodebookRenderer(codebook_raw=raw_common)

    # Test B fails (recon_b is degraded in E_eval), so we proceed to B'
    # For B' to run, Test B must fail. We provide a degraded recon_b.
    recon_b = (rare.copy() + 0.5, common.copy())  # add noise to fail Test B

    # B' reconstruction is the render->re-encode one (also degraded)
    rare_eval = e_eval(raw_rare)
    common_eval = e_eval(raw_common)
    rare_recon_b_eval = rare.copy() + 0.5  # add same noise to fail Test B in eval
    common_recon_b_eval = common.copy()

    rare_recon_eval = render_and_reencode(rare, renderer, e_eval)
    common_recon_eval = render_and_reencode(common, renderer, e_eval)

    # Run diagnostic — B fails, proceeds to B'
    rep = run_diagnostic(
        rare, common,
        recon_b=recon_b,
        recon_b_prime=(rare.copy(), common.copy()),  # B' reconstruction is faithful in E_gen
        emb_eval=(rare_eval, common_eval),
        recon_b_eval=(rare_recon_b_eval, common_recon_b_eval),
        recon_b_prime_eval=(rare_recon_eval, common_recon_eval),
    )

    # Check that Test B' result has metric_hacking_flag = True
    result_bp = [t for t in rep.tests_run if t.test == "B_prime"][0]
    assert result_bp.metrics["metric_hacking_flag"] is True, (
        "Test B' should have metric_hacking_flag=True when passes in E_gen "
        "but fails in E_eval"
    )


def test_render_reencode_surfaces_metric_hacking_cfg_landing():
    """Test B'' (CFG landing) metric-hacking guard: passes in E_gen, fails in E_eval.

    Reconstruction passes in both E_gen and E_eval, but CFG-guided generation
    landing fails in E_eval when it would have passed in E_gen. This tests the
    guard at the CFG landing stage.
    """
    rng = np.random.default_rng(2)
    in_dim, out_dim = 8, 6
    e_gen, e_eval = make_encoder_pair(in_dim, out_dim)

    # Generate rare and common raw features
    raw_rare = rng.standard_normal((40, in_dim)) + 3.0
    raw_common = rng.standard_normal((200, in_dim))

    rare = e_gen(raw_rare)
    common = e_gen(raw_common)

    # Codebook = common raw features only
    renderer = CodebookRenderer(codebook_raw=raw_common)

    # Faithful reconstruction in E_gen (Test B passes)
    recon_b = (rare.copy(), common.copy())

    # For the reconstruction to also pass in E_eval, use faithful reconstruction
    rare_eval = e_eval(raw_rare)
    common_eval = e_eval(raw_common)
    rare_recon_eval = rare_eval.copy()
    common_recon_eval = common_eval.copy()

    # Generate CFG-guided samples that land in R in E_gen but collapse in E_eval
    # In E_gen space: guided rare samples are close to real rare (good landing)
    rare_guided = rare.copy() + 0.01 * rng.standard_normal(rare.shape)

    # In E_eval space: the same samples collapse toward common via render->re-encode
    # (simulating the metric-hacking at the CFG stage)
    rare_guided_eval = render_and_reencode(rare_guided, renderer, e_eval)
    common_ref = common.copy()
    common_ref_eval = e_eval(raw_common)

    # Run diagnostic with CFG-landing arrays
    rep = run_diagnostic(
        rare, common,
        recon_b=recon_b,
        rare_guided=rare_guided,
        common_ref=common_ref,
        emb_eval=(rare_eval, common_eval),
        recon_b_eval=(rare_recon_eval, common_recon_eval),
        rare_guided_eval=rare_guided_eval,
        common_ref_eval=common_ref_eval,
    )

    # Test B should pass (faithful reconstruction)
    result_b = [t for t in rep.tests_run if t.test == "B"][0]
    assert result_b.passed, (
        "Test B should pass with faithful reconstruction"
    )
    assert result_b.metrics["metric_hacking_flag"] is False, (
        "Test B should not have metric_hacking_flag when faithful in both spaces"
    )

    # The B'' landing test should exist and fail in E_eval due to collapsed guided samples
    result_landing_list = [t for t in rep.tests_run if t.test == "B_double_prime"]
    assert len(result_landing_list) > 0, (
        "B'' landing test should run after B passes"
    )
    result_landing = result_landing_list[0]
    # The landing should fail due to collapsed guided samples in E_eval
    assert not result_landing.passed, (
        "B'' landing should fail in E_eval when guided samples are collapsed via render->re-encode"
    )
    # The B'' landing gate's own metric-hacking guard should fire: guided
    # samples land in R in the generation space but collapse under E_eval.
    assert result_landing.metrics["metric_hacking_flag"] is True, (
        "B'' landing should have metric_hacking_flag=True when CFG lands in "
        "E_gen but collapses under decoupled E_eval"
    )


def test_eval_space_inputs_exercises_render_and_reencode_end_to_end():
    """`eval_space_inputs` must actually render->re-encode the GENERATION-space
    reconstruction/guided arrays (not silently hand back faithful originals),
    and feeding its outputs into `run_diagnostic` must surface the B'' landing
    metric-hacking flag — the guard this helper exists to serve.
    """
    rng = np.random.default_rng(3)
    in_dim, out_dim = 8, 6
    e_gen, e_eval = make_encoder_pair(in_dim, out_dim)

    raw_rare = rng.standard_normal((40, in_dim)) + 3.0
    raw_common = rng.standard_normal((200, in_dim))

    rare = e_gen(raw_rare)
    common = e_gen(raw_common)

    # Codebook = common raw features only (rare detail NOT representable)
    renderer = CodebookRenderer(codebook_raw=raw_common)

    # Faithful reconstruction and CFG landing in E_gen space.
    recon_b = (rare.copy(), common.copy())
    rare_guided = rare.copy() + 0.01 * rng.standard_normal(rare.shape)
    common_ref = common.copy()

    # Exercise eval_space_inputs end-to-end. recon_b is intentionally NOT
    # passed in: this renderer collapses everything toward the common codebook,
    # so rendering recon_b through it would also fail Test B's reconstruction
    # gate in E_eval and the B'' landing gate would never run. Leaving recon_b
    # out (as an unsupplied gen-space array) is itself part of the contract
    # under test: eval_space_inputs must return None for arrays it wasn't given
    # rather than fabricating something, so Test B gates on E_gen alone while
    # the B'' landing gate — fed rare_guided/common_ref — gates on E_eval.
    emb_eval, recon_b_eval, recon_b_prime_eval, recon_c_eval, guided_eval = eval_space_inputs(
        raw_rare, raw_common, renderer, e_eval,
        rare_guided=rare_guided,
        common_ref=common_ref,
    )
    rare_guided_eval, common_ref_eval = guided_eval

    # It must have actually gone through render->re-encode: the eval-space
    # guided array should differ from a plain re-encode of the raw rare
    # features (which is what the old dead-code path returned).
    plain_reencode_rare = e_eval(raw_rare)
    assert not np.allclose(rare_guided_eval, plain_reencode_rare), (
        "rare_guided_eval should be the render->re-encode of the GEN-space "
        "guided samples, not a plain re-encode of the raw rare features"
    )
    # Cross-check against calling render_and_reencode directly.
    assert np.allclose(rare_guided_eval, render_and_reencode(rare_guided, renderer, e_eval))

    # recon_b / recon_b_prime / recon_c were not supplied — must come back None.
    assert recon_b_eval is None
    assert recon_b_prime_eval is None
    assert recon_c_eval is None

    rep = run_diagnostic(
        rare, common,
        recon_b=recon_b,
        rare_guided=rare_guided,
        common_ref=common_ref,
        emb_eval=emb_eval,
        rare_guided_eval=rare_guided_eval,
        common_ref_eval=common_ref_eval,
    )

    result_landing = [t for t in rep.tests_run if t.test == "B_double_prime"][0]
    assert result_landing.metrics["metric_hacking_flag"] is True, (
        "eval_space_inputs' render->re-encode outputs should surface the B'' "
        "metric-hacking guard end-to-end"
    )
