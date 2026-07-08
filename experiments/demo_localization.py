"""CPU demo — rare-detail localization diagnostic, end to end, no GPU.

Runs `run_diagnostic` through four scenarios using two *distinct* stand-in frozen
encoders (E_gen for generation, E_eval for the metric-hacking guard). Everything
is synthetic embeddings + synthesized reconstruction/landing arrays — the real
pipeline swaps in a frozen EHR encoder (MOTOR/CLMBR) and a trained diffusion for
the reconstruction/guided-generation steps; the decision procedure is unchanged.

    python experiments/demo_localization.py

Scenarios
---------
  1. diffuse_directly  — Test A pass, B recon faithful, B″ CFG landing passes.
  2. smc_required      — recon faithful but CFG-guided generation drifts to bulk.
  3. metric-hacking    — recon faithful under E_gen but collapsed under E_eval
                         (the Van Assel et al. 2026 guard; causal_bench#88).
  4. bound_scope       — Test A fails (encoder does not separate rare from common).
"""
from __future__ import annotations

import numpy as np

from causal_bench.diagnostics.localization import run_diagnostic

RNG = np.random.default_rng(0)
D_RAW = 8       # raw patient-feature dimension
D_EMB = 6       # encoder output dimension
N_RARE = 40
N_COMMON = 200


def _random_encoder(seed: int):
    """A fixed random linear map R^{D_RAW} -> R^{D_EMB} standing in for a frozen
    encoder. Two different seeds = two genuinely distinct embedding geometries."""
    W = np.random.default_rng(seed).standard_normal((D_RAW, D_EMB))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return lambda X: X @ W


def _patients(rare_shift: float):
    """Raw patient features. `rare_shift` controls how far rare sits from common;
    small shift => the encoders cannot separate them (Test A fails)."""
    common = RNG.standard_normal((N_COMMON, D_RAW))
    rare = RNG.standard_normal((N_RARE, D_RAW)) + rare_shift
    return rare, common


def _faithful(emb):
    return emb + 1e-3 * RNG.standard_normal(emb.shape)


def _collapsed(rare_emb, common_emb):
    """Rare reconstructed onto the common centroid — tail collapse in this space."""
    centroid = common_emb.mean(axis=0, keepdims=True)
    return centroid + 0.1 * RNG.standard_normal(rare_emb.shape)


def _guided_good(rare_emb):
    """Guided generation that lands in R: near real rare embeddings."""
    idx = RNG.integers(0, len(rare_emb), size=N_RARE)
    return rare_emb[idx] + 0.2 * RNG.standard_normal((N_RARE, rare_emb.shape[1]))


def _guided_drifted(common_emb):
    """Guided generation that collapsed toward the bulk: looks like common."""
    idx = RNG.integers(0, len(common_emb), size=N_RARE)
    return common_emb[idx] + 0.2 * RNG.standard_normal((N_RARE, common_emb.shape[1]))


def _show(title, report):
    print(f"\n{'=' * 78}\n{title}\n{'-' * 78}")
    print(f"TERMINAL: {report.terminal}")
    print(f"tests run: {[t.test for t in report.tests_run]}")
    for t in report.tests_run:
        flag = t.metrics.get("metric_hacking_flag")
        tag = "  [METRIC-HACKING]" if flag else ""
        print(f"  Test {t.test:<15} passed={t.passed!s:<5}{tag}")
    print(f"summary: {report.summary}")


def main():
    E_gen = _random_encoder(seed=11)
    E_eval = _random_encoder(seed=29)   # decoupled evaluation encoder

    # ── 1. diffuse_directly ──────────────────────────────────────────────────
    rare_raw, common_raw = _patients(rare_shift=3.0)
    rare, common = E_gen(rare_raw), E_gen(common_raw)
    _show(
        "1. diffuse_directly — faithful recon + CFG landing passes",
        run_diagnostic(
            rare, common,
            recon_b=(_faithful(rare), _faithful(common)),
            rare_guided=_guided_good(rare),
            common_ref=common,
        ),
    )

    # ── 2. smc_required ──────────────────────────────────────────────────────
    _show(
        "2. smc_required — recon faithful but CFG drifts to the bulk",
        run_diagnostic(
            rare, common,
            recon_b=(_faithful(rare), _faithful(common)),
            rare_guided=_guided_drifted(common),
            common_ref=common,
        ),
    )

    # ── 3. metric-hacking guard (decoupled E_eval) ───────────────────────────
    rare_eval, common_eval = E_eval(rare_raw), E_eval(common_raw)   # same patients
    _show(
        "3. metric-hacking — faithful under E_gen, collapsed under E_eval (#88)",
        run_diagnostic(
            rare, common,
            recon_b=(_faithful(rare), _faithful(common)),          # E_gen: faithful
            emb_eval=(rare_eval, common_eval),
            recon_b_eval=(_collapsed(rare_eval, common_eval),      # E_eval: collapsed
                          _faithful(common_eval)),
        ),
    )

    # ── 4. bound_scope ───────────────────────────────────────────────────────
    rare_raw2, common_raw2 = _patients(rare_shift=0.0)   # rare == common distribution
    _show(
        "4. bound_scope — Test A fails (encoder cannot separate rare from common)",
        run_diagnostic(
            E_gen(rare_raw2), E_gen(common_raw2),
            pretraining_influence=False,
        ),
    )


if __name__ == "__main__":
    main()
