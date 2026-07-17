"""Distributed (multi-GPU / NCCL) SMC resample — the global all-to-all path.

Wires the validated collectives into the resample barrier:
  1. ``all_gather`` local weights -> every rank holds the full w[1..N].
  2. shared-seed systematic indices -> byte-identical ancestor vector on all ranks
     (the invariant validated by ``scripts/smc_distributed_validate.py``).
  3. ``all_to_all`` -> redistribute ONLY the particles whose owner changed
     (minimal movement, spec §1d), reproducing the serial resample exactly.

``island_resample`` (sharded.py) is the low-communication default; this global
path is what the ``distributed == serial`` equivalence check needs and what the
NVLink throughput profile (§1d) measures.  The index math here is deliberately
side-effect-free and unit-testable off-box via ``plan_all_to_all`` /
``simulate_all_to_all`` (pure numpy, no torch.distributed).
"""
from __future__ import annotations

import numpy as np

from .resample import systematic_resample


# ---------------------------------------------------------------------------
# Pure-numpy planning (testable without a GPU / process group)
# ---------------------------------------------------------------------------
def global_indices(full_w: np.ndarray, seed: int) -> np.ndarray:
    """Shared-seed systematic ancestor indices over the full (normalized) w."""
    w = np.asarray(full_w, dtype=float)
    w = w / w.sum()
    return systematic_resample(w, np.random.default_rng(seed))


def plan_all_to_all(idx: np.ndarray, rank: int, world: int, n_local: int):
    """Given the global ancestor vector, compute this rank's all-to-all plan.

    Returns:
      in_split[d]   : # rows this rank SENDS to dest d (ancestors in A_d I own).
      out_split[s]  : # rows this rank RECEIVES from src s (my ancestors owned by s).
      send_local    : local row indices to send, concatenated by dest d (A_d order).
      place_order   : permutation s.t. out[place_order] = recv_buffer, i.e. the
                      received rows (grouped by src, A_me order within src) map
                      back to this rank's output-slice positions.
    """
    A_me = idx[rank * n_local:(rank + 1) * n_local]      # my output ancestors
    owner_me = A_me // n_local                           # src rank per output pos

    in_split, send_local = [], []
    for d in range(world):
        A_d = idx[d * n_local:(d + 1) * n_local]
        sel = A_d[(A_d // n_local) == rank]              # ancestors I own that d wants
        send_local.append(sel - rank * n_local)          # -> my local row indices
        in_split.append(int(sel.size))
    send_local = (np.concatenate(send_local).astype(np.int64)
                  if send_local else np.empty(0, np.int64))

    out_split = [int(np.count_nonzero(owner_me == s)) for s in range(world)]
    # recv buffer is ordered by (src, output-position); sorting output positions
    # by (owner, position) yields that order, so its inverse places recv -> out.
    place_order = np.lexsort((np.arange(n_local), owner_me)).astype(np.int64)
    return in_split, out_split, send_local, place_order


def simulate_all_to_all(full_x: np.ndarray, full_w: np.ndarray, world: int,
                        seed: int) -> np.ndarray:
    """In-process numpy simulation of the distributed all-to-all resample across
    ``world`` ranks — for unit-testing the plan without NCCL.  Must equal the
    serial ``full_x[systematic_resample(full_w, seed)]``."""
    n = len(full_w); n_local = n // world
    assert n % world == 0
    idx = global_indices(full_w, seed)
    out = np.empty_like(full_x)
    for r in range(world):
        in_split, out_split, send_local, place_order = plan_all_to_all(idx, r, world, n_local)
        # gather the rows each source sends to r, in (src, A_r order)
        recv_parts = []
        for s in range(world):
            A_r = idx[r * n_local:(r + 1) * n_local]
            sel = A_r[(A_r // n_local) == s]             # ancestors r wants from s
            recv_parts.append(full_x[sel])               # s owns them; ship rows
        recv = np.concatenate(recv_parts, axis=0)
        out_slice = np.empty((n_local,) + full_x.shape[1:], dtype=full_x.dtype)
        out_slice[place_order] = recv
        out[r * n_local:(r + 1) * n_local] = out_slice
    return out


# ---------------------------------------------------------------------------
# Real distributed resample (torch.distributed / NCCL) — runs on-box
# ---------------------------------------------------------------------------
def distributed_ess(local_log_w, group=None) -> float:
    """Global Kish ESS via the fused (num, den) all_reduce (the only true sync).
    ESS = (Σw)^2 / Σw^2 ; reduce Σw and Σw^2 across ranks."""
    import torch
    import torch.distributed as dist
    lw = local_log_w if isinstance(local_log_w, torch.Tensor) else torch.as_tensor(local_log_w)
    m = lw.max()
    if group is not None or dist.is_initialized():
        dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)   # global max, overflow-safe
    w = torch.exp(lw - m)
    stats = torch.stack([w.sum(), (w * w).sum()])
    if group is not None or dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    sw, sw2 = float(stats[0]), float(stats[1])
    return (sw * sw) / sw2


def all_to_all_resample(local_x, local_w, seed: int, rank: int, world: int, group=None):
    """Distributed global resample. local_x: (n_local, d) torch cuda tensor;
    local_w: (n_local,) local (unnormalized) weights.  Returns this rank's
    resampled output slice (n_local, d), reproducing the serial resample."""
    import torch
    import torch.distributed as dist
    n_local, d = int(local_x.shape[0]), int(local_x.shape[1])

    # 1) all_gather weights -> full w
    wbufs = [torch.empty_like(local_w) for _ in range(world)]
    dist.all_gather(wbufs, local_w, group=group)
    full_w = torch.cat(wbufs).detach().cpu().numpy().astype(float)

    # 2) shared-seed global indices + plan (pure numpy, identical on every rank)
    idx = global_indices(full_w, seed)
    in_split, out_split, send_local, place_order = plan_all_to_all(idx, rank, world, n_local)

    # 3) all_to_all only the moved particles
    send_idx = torch.as_tensor(send_local, device=local_x.device, dtype=torch.long)
    send_buf = local_x.index_select(0, send_idx).contiguous()
    recv_buf = torch.empty((int(sum(out_split)), d), dtype=local_x.dtype, device=local_x.device)
    dist.all_to_all_single(recv_buf, send_buf, out_split, in_split, group=group)

    out = torch.empty_like(local_x)
    out[torch.as_tensor(place_order, device=local_x.device, dtype=torch.long)] = recv_buf
    return out
