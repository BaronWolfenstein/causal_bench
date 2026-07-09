# SMC on-box multi-GPU validation runbook

Prereqs (spec §0): `nvidia-smi` (note idle GPUs), `nvidia-smi topo -m`
(confirm NVLink adjacency — `NV#` good; `PHB`/`PXB`/`SYS` = PCIe), run inside
`tmux`, keep data on the box's local NVMe (Tailscale is control-plane only).

## Ladder (stop at the first failure)
1. **2 ranks — decisive.** Exercises all_reduce + all_gather + (next) all_to_all.
   ```
   CUDA_VISIBLE_DEVICES=<2 free NVLink-adjacent ids> \
   torchrun --nproc_per_node=2 scripts/smc_distributed_validate.py --seed 7
   ```
   Expect: `distributed==oracle OK`.
2. **cuda==cpu parity** (Task 5): `pytest tests/test_smc_cuda_parity.py -v` → PASS.
3. **Throughput sweep** at 2/4/8 ranks: rerun step 1 with `--nproc_per_node` 4 then 8,
   record wall-clock and all-to-all comm cost (the O(N·dim) NVLink transfer — the
   only thing off-box measurement cannot tell us, spec §3).

Do NOT re-derive on-box: resample-trigger rate, O(N) per-particle scaling,
ESS/weight-degeneracy health, distributed==serial index invariant — all
CPU-settled and hardware-independent (spec §3).
