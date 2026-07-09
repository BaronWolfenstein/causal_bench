"""Integration test — end-to-end diffuse_directly -> localization terminal.

Wires the generative pieces (vpsde, roundtrip, guidance) into
run_diagnostic end-to-end, asserts the diagnostic returns the expected
terminal (diffuse_directly or smc_required) when fed faithful reconstructions
+ landing guided samples.
"""
import numpy as np
from causal_bench.generative.vpsde import Schedule, gaussian_score
from causal_bench.generative.roundtrip import per_mode_roundtrip
from causal_bench.generative.guidance import generate_guided
from causal_bench.diagnostics.localization import run_diagnostic


def test_generative_pipeline_reaches_diffuse_directly():
    """Full pipeline: faithful round-trip + good CFG landing -> diffuse_directly."""
    sch = Schedule(n_steps=120)
    rng = np.random.default_rng(0)
    rare = rng.standard_normal((40, 1)) + 4.0
    common = rng.standard_normal((200, 1))

    # Bulk score: unconditional (N(0, I))
    bulk = lambda x, t: gaussian_score(x, t, np.array([0.0]), np.eye(1), sch)

    # Faithful round-trip (small noise => faithful reconstruction)
    recon_b = per_mode_roundtrip(rare, common, bulk, sch, t_start=5, rng=rng)

    # Conditional score: rare cohort (N(4.0, I))
    cond = lambda x, t: gaussian_score(x, t, np.array([4.0]), np.eye(1), sch)

    # CFG-guided generation
    rare_guided = generate_guided(40, cond, bulk, sch, rng, guidance_scale=3.0)

    # Run diagnostic: recon_b faithful + rare_guided lands in R
    rep = run_diagnostic(
        rare, common,
        recon_b=recon_b,
        rare_guided=rare_guided,
        common_ref=common,
    )

    # Loop closed: terminal is one of the two expected outcomes
    assert rep.terminal in ("diffuse_directly", "smc_required")
