"""CPU demo — STRUCT-S stratification screen (S2-S4) on synthetic embeddings.

Runs the embedding-only stratification detector on three cases and prints the
verdict, showing what S2 (spectral component count), S3 (local-ID heterogeneity),
and S4 (density gap) each contribute:

  1. FLAT      — one smooth blob                -> not a candidate (flat OK)
  2. STRATIFIED — two well-separated sheets      -> candidate (S2 + S4 fire)
  3. MIXED-DIM  — a 1D sheet + a higher-D blob   -> candidate (S3 heterogeneity)

Every case prints `needs_S1_to_confirm=True`: S2-S4 can flag a *candidate* (or
clear it), but only the on-box S1 (displacement bimodality + intercurrent-event
alignment) confirms the event-driven jump structure that licenses a jump-diffusion.

    PYTHONPATH=. python experiments/demo_struct_s.py
"""
from __future__ import annotations

import warnings

import numpy as np

# Benign, pre-existing: sklearn's Accelerate-BLAS matmul emits RuntimeWarnings on
# this hardware; the distances/screen are computed correctly. Silence for a clean run.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from causal_bench.diagnostics.struct_s import struct_s_screen  # noqa: E402


def _report(name: str, Z: np.ndarray) -> None:
    r = struct_s_screen(Z)
    verdict = "CANDIDATE stratified" if r["candidate_stratified"] else "flat OK"
    print(f"\n{name}  (n={len(Z)}, d={Z.shape[1]})")
    print(f"  S2 n_strata        = {r['S2_n_strata']}")
    print(f"  S3 local-ID CV     = {r['S3_local_id_cv']:.2f}")
    print(f"  S4 gap_ratio       = {r['S4_gap_ratio']:.2f}   has_gap={r['S4_has_gap']}")
    print(f"  --> candidate_stratified = {r['candidate_stratified']}  ({verdict})")
    print(f"      needs_S1_to_confirm  = {r['needs_S1_to_confirm']}")


def main() -> None:
    rng = np.random.default_rng(0)
    print("=" * 74)
    print("STRUCT-S stratification screen (S2-S4) — embedding-only, CPU")
    print("-" * 74)

    # 1. FLAT — a single smooth blob
    flat = rng.standard_normal((200, 4))

    # 2. STRATIFIED — two well-separated sheets (a discrete regime split)
    stratified = np.vstack([rng.standard_normal((100, 4)),
                            rng.standard_normal((100, 4)) + 18.0])

    # 3. MIXED-DIM — a ~1D sheet and a ~4D blob (differing intrinsic dimension)
    line = np.zeros((100, 6)); line[:, 0] = rng.standard_normal(100) * 6.0
    blob = rng.standard_normal((100, 6))
    mixed = np.vstack([line, blob])

    _report("1. FLAT (one smooth blob)", flat)
    _report("2. STRATIFIED (two separated sheets)", stratified)
    _report("3. MIXED-DIM (1D sheet + 4D blob)", mixed)

    print("\n" + "=" * 74)
    print("S2-S4 corroborate/rule out; S1 (event-aligned displacement bimodality)")
    print("is the DECISIVE test and is on-box-gated (needs real MEDS trajectories +")
    print("intercurrent-event timestamps). See the manifold-aware design spec.")
    print("=" * 74)


if __name__ == "__main__":
    main()
