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


def _plan_gpu(idx, rank: int, world: int, n_local: int):
    """cupy version of plan_all_to_all — the O(N) index work stays on-GPU (CUB
    bincount/argsort), only the world-length split lists cross to host.  ``idx``
    is a cupy int array.  Returns (in_split list, out_split list, send_local
    cupy, place_order cupy)."""
    import cupy as cp
    owner_all = idx // n_local
    A_me = idx[rank * n_local:(rank + 1) * n_local]
    owner_me = A_me // n_local
    out_split = cp.bincount(owner_me, minlength=world)
    dest_all = cp.arange(idx.size) // n_local          # which dest-slice each pos is in
    mine = owner_all == rank
    sel = dest_all[mine]
    # cupy.bincount raises on an empty input (a rank owning zero survivors, e.g. in
    # extreme weight degeneracy); numpy tolerates it — guard for parity.
    in_split = (cp.bincount(sel, minlength=world) if sel.size
                else cp.zeros(world, dtype=cp.int64))
    send_local = idx[mine] - rank * n_local            # natural order == (dest, pos) order
    place_order = cp.argsort(owner_me, kind="stable")  # recv (src,pos) order -> output pos
    return in_split.get().tolist(), out_split.get().tolist(), send_local, place_order


def all_to_all_resample(local_x, local_w, seed, rank: int, world: int,
                        group=None, plan_on: str = "gpu", dedup: bool = False):
    """Distributed global resample. local_x: (n_local, d) torch cuda tensor;
    local_w: (n_local,) local (unnormalized) weights.  ``seed`` is an int OR a
    numpy Generator (pass a SHARED, advancing Generator from the SMC loop so the
    per-resample uniform matches the serial rng stream).  Returns this rank's
    resampled output slice (n_local, d), reproducing the serial resample.

    plan_on="gpu" keeps the all-gathered weights, systematic resample, and the
    send/recv plan on-GPU (cupy via DLPack, zero-copy from the torch tensors) so
    the only host traffic is the world-length split lists — the NVLink all_to_all
    then dominates (spec §1d).  plan_on="cpu" is the numpy reference path."""
    import torch
    import torch.distributed as dist
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # 1) all_gather local weights -> full w (stays on device)
    wbufs = [torch.empty_like(local_w) for _ in range(world)]
    dist.all_gather(wbufs, local_w, group=group)
    full_w_t = torch.cat(wbufs)
    n_local = int(local_x.shape[0])

    use_gpu = plan_on == "gpu" and local_x.is_cuda
    if use_gpu:
        try:
            import cupy as cp
        except Exception:
            use_gpu = False

    if use_gpu:
        torch.cuda.synchronize()                       # torch stream done before cupy reads
        w_cp = cp.from_dlpack(full_w_t)                # zero-copy torch->cupy
        w_cp = (w_cp / w_cp.sum()).astype(cp.float64)
        idx = systematic_resample(w_cp, rng)           # cupy CUB path (shared host uniform)
    else:
        full_w = full_w_t.detach().cpu().numpy().astype(float)
        idx = systematic_resample(full_w / full_w.sum(), rng)
    if dedup:
        idx_np = idx.get() if hasattr(idx, "get") else np.asarray(idx)
        return _scatter_by_index_dedup(local_x, idx_np, rank, world, n_local, group)
    return _scatter_by_index(local_x, idx, rank, world, n_local, group, use_gpu)


def _scatter_by_index_dedup(local_x, idx, rank, world, n_local, group):
    """Ancestor-index indirection (spec §1d): communicate each UNIQUE surviving
    ancestor exactly once and replicate locally, instead of shipping duplicated
    rows.  Big win in the rare-event degeneracy regime (few distinct survivors
    duplicated massively); ~neutral when weights are near-uniform.  ``idx`` is a
    host numpy vector; reproduces the same output as _scatter_by_index."""
    import torch
    import torch.distributed as dist
    d = int(local_x.shape[1])
    A_me = idx[rank * n_local:(rank + 1) * n_local]
    owner_me = A_me // n_local

    recv_split, uniq_by_src = [], []
    for s in range(world):
        us = np.unique(A_me[owner_me == s])            # sorted unique globals owned by s
        uniq_by_src.append(us); recv_split.append(len(us))
    send_split, send_local = [], []
    for dd in range(world):                            # what each dest needs from me (unique)
        A_d = idx[dd * n_local:(dd + 1) * n_local]
        ud = np.unique(A_d[(A_d // n_local) == rank])   # matches dest dd's recv order (sorted)
        send_local.append(ud - rank * n_local); send_split.append(len(ud))
    send_local = (np.concatenate(send_local).astype(np.int64)
                  if send_local else np.empty(0, np.int64))

    send_buf = local_x.index_select(0, torch.as_tensor(send_local, device=local_x.device,
                                                       dtype=torch.long)).contiguous()
    recv_buf = torch.empty((int(sum(recv_split)), d), dtype=local_x.dtype, device=local_x.device)
    dist.all_to_all_single(recv_buf, send_buf, recv_split, send_split, group=group)

    # local replication: each output pos -> its unique row in recv_buf
    recv_off = np.concatenate([[0], np.cumsum(recv_split)])[:-1]
    place = np.empty(n_local, dtype=np.int64)
    for s in range(world):
        mask = owner_me == s
        place[mask] = recv_off[s] + np.searchsorted(uniq_by_src[s], A_me[mask])
    return recv_buf.index_select(0, torch.as_tensor(place, device=local_x.device, dtype=torch.long))


def _scatter_by_index(local_x, idx, rank, world, n_local, group, use_gpu):
    """Given the global ancestor vector ``idx`` (cupy if use_gpu else numpy),
    all_to_all only the moved particles and return this rank's output slice."""
    import torch
    import torch.distributed as dist
    d = int(local_x.shape[1])
    if use_gpu:
        import cupy as cp
        in_split, out_split, send_local_cp, place_order_cp = _plan_gpu(idx, rank, world, n_local)
        cp.cuda.get_current_stream().synchronize()
        send_idx = torch.from_dlpack(send_local_cp).long()
        place_order = torch.from_dlpack(place_order_cp).long()
    else:
        in_split, out_split, send_local, place_order_np = plan_all_to_all(idx, rank, world, n_local)
        send_idx = torch.as_tensor(send_local, device=local_x.device, dtype=torch.long)
        place_order = torch.as_tensor(place_order_np, device=local_x.device, dtype=torch.long)
    send_buf = local_x.index_select(0, send_idx).contiguous()
    recv_buf = torch.empty((int(sum(out_split)), d), dtype=local_x.dtype, device=local_x.device)
    dist.all_to_all_single(recv_buf, send_buf, out_split, in_split, group=group)
    out = torch.empty_like(local_x)
    out[place_order] = recv_buf
    return out
