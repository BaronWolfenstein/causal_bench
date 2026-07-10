"""CPU demo — end-to-end diffuse_directly pipeline, generating arrays.

Mirrors `experiments/demo_localization.py` but generates reconstruction and
guidance arrays using the generative package (vpsde, roundtrip, guidance),
then runs `run_diagnostic` to close the loop. Demonstrates how the full
pipeline (Train diffusion on embedding space → encode patients → reconstruct
via round-trip → generate CFG-guided samples → diagnostic → terminal)
operates. With guidance_scale=3.0, CFG overshoots, so the expected terminal
is smc_required (the SMC reranker fixes overshoot). This is correct, documented behavior.

    PYTHONPATH=. python experiments/demo_diffuse_directly.py
"""
from __future__ import annotations

import warnings

import numpy as np

# Benign, pre-existing: on this hardware sklearn's Accelerate-BLAS matmul emits
# divide/overflow RuntimeWarnings on the extreme-valued overshoot logits (mean≈8);
# the separation AUC still computes correctly. Silence them so the demo reads clean.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from causal_bench.generative.vpsde import Schedule, gaussian_score
from causal_bench.generative.roundtrip import per_mode_roundtrip
from causal_bench.generative.guidance import generate_guided
from causal_bench.sampling.twist import linear_anneal
from causal_bench.diagnostics.localization import run_diagnostic


def main():
    """Full pipeline: embedding space diffusion with generative round-trip and CFG."""
    rng = np.random.default_rng(0)
    sch = Schedule(n_steps=120)

    # Synthetic embeddings: rare and common patients
    rare = rng.standard_normal((40, 1)) + 4.0
    common = rng.standard_normal((200, 1))

    print("=" * 78)
    print("Diffuse-Directly Pipeline Demo")
    print("-" * 78)
    print(f"Rare embeddings:   mean={rare.mean():.2f}, std={rare.std():.2f}, shape={rare.shape}")
    print(f"Common embeddings: mean={common.mean():.2f}, std={common.std():.2f}, shape={common.shape}")

    # ── Diffusion training: train on combined (or just common as "bulk") ──────
    # Unconditional (bulk) score: N(0, I)
    bulk = lambda x, t: gaussian_score(x, t, np.array([0.0]), np.eye(1), sch)
    print(f"\nTraining diffusion model on {len(common)} common + {len(rare)} rare samples...")

    # ── Round-trip Test B: encode -> forward-noise -> reverse -> decode ───────
    print("Running Test B round-trip (faithful reconstruction check)...")
    recon_b = per_mode_roundtrip(rare, common, bulk, sch, t_start=5, rng=rng)
    rare_recon, common_recon = recon_b
    print(f"  Rare recon:   mean={rare_recon.mean():.2f}, std={rare_recon.std():.2f}")
    print(f"  Common recon: mean={common_recon.mean():.2f}, std={common_recon.std():.2f}")

    # ── CFG-guided generation: sample from rare-cohort condition ─────────────
    # Conditional score: rare cohort at N(4.0, I)
    cond = lambda x, t: gaussian_score(x, t, np.array([4.0]), np.eye(1), sch)
    print("\nGenerating CFG-guided samples (rare-cohort condition, guidance_scale=3.0)...")
    rare_guided = generate_guided(40, cond, bulk, sch, rng, guidance_scale=3.0)
    print(f"  Guided samples: mean={rare_guided.mean():.2f}, std={rare_guided.std():.2f}")

    # ── Annealed β_t: weak guidance while noisy, sharpening as denoising finishes ──
    # linear_anneal(v_at_t0, v_at_tmax, n_steps): the reverse loop runs t = T..0, so
    # v_at_tmax (weak, 0.3) applies at high noise and v_at_t0 (2.0) as x0 sharpens.
    # Softer early steps curb the early overshoot that pushes the constant run to R's
    # exterior (→ smc_required).
    anneal = linear_anneal(2.0, 0.3, sch.n_steps)
    print("\nGenerating with ANNEALED β_t (linear_anneal 2.0→0.3 over the noise schedule)...")
    rare_guided_annealed = generate_guided(40, cond, bulk, sch,
                                           np.random.default_rng(1), guidance_scale=anneal)
    print(f"  Annealed guided samples: mean={rare_guided_annealed.mean():.2f}, "
          f"std={rare_guided_annealed.std():.2f}")
    rep_annealed = run_diagnostic(rare, common, recon_b=recon_b,
                                  rare_guided=rare_guided_annealed, common_ref=common)
    print(f"  Annealed terminal: {rep_annealed.terminal} "
          f"(vs constant-scale below)")

    # ── Diagnostic: Test A (encoder capacity) + Test B + Test B″ (CFG landing) ───
    print("\nRunning diagnostic decision procedure...")
    rep = run_diagnostic(
        rare, common,
        recon_b=recon_b,
        rare_guided=rare_guided,
        common_ref=common,
    )

    # ── Results ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"TERMINAL: {rep.terminal}")
    print(f"TESTS RUN: {[t.test for t in rep.tests_run]}")
    print("-" * 78)
    for t in rep.tests_run:
        flag = t.metrics.get("metric_hacking_flag")
        tag = "  [METRIC-HACKING]" if flag else ""
        print(f"Test {t.test:<15} passed={t.passed!s:<5}{tag}")
    print(f"\nSummary:\n{rep.summary}")
    print("=" * 78)


if __name__ == "__main__":
    main()
