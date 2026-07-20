"""Multi-GPU sharder for exp41's --full calibration (#148, ENCIRCLE §8).

Partitions the 54-cell grid across N GPU workers (one process per GPU, pinned via
CUDA_VISIBLE_DEVICES), each running `exp41 --shard i/N --chain-method vectorized`,
then collates the per-worker JSON into one row set. Embarrassingly parallel — no
NCCL, no cross-rank sync (spec §8). Turns the ~16–31 h serial run into ~2–4 h on
8 A100s.

Run on the box (weeden-gpu):
    ~/weeden-gpu/bin/python scripts/exp41_multigpu.py --gpus 0 1 2 3 4 5 6 7 --full
Extra flags after the known ones pass through to exp41 (--full, --n-reps, --draws …).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def worker_command(python: str, shard_i: int, n_workers: int, out: str, passthrough):
    """The exp41 worker argv for shard `shard_i` of `n_workers` (pure → testable)."""
    return [python, "-m", "experiments.exp41_borrowing_calibration",
            "--shard", f"{shard_i}/{n_workers}", "--out", out,
            "--chain-method", "vectorized", *passthrough]


def worker_env(gpu: int) -> dict:
    """Env pinning a worker to one GPU (no XLA preallocation so 8 share the box)."""
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                XLA_PYTHON_CLIENT_PREALLOCATE="false")


def collate(paths) -> list:
    """Merge per-worker JSON row files into one cell-sorted list."""
    rows: list = []
    for p in paths:
        rows.extend(json.loads(Path(p).read_text()))
    return sorted(rows, key=lambda r: r["cell"])


def main() -> None:
    ap = argparse.ArgumentParser(description="exp41 multi-GPU sharder")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--outdir", default="results/exp41_full")
    args, passthrough = ap.parse_known_args()   # passthrough: --full, --n-reps, --draws, …

    n = len(args.gpus)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    procs, outs = [], []
    for wi, gpu in enumerate(args.gpus):
        out = str(outdir / f"worker_{wi}.json")
        outs.append(out)
        cmd = worker_command(args.python, wi, n, out, passthrough)
        print(f"launch gpu{gpu} shard {wi}/{n}: {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=worker_env(gpu)))

    codes = [p.wait() for p in procs]
    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        print(f"WARNING: workers exited nonzero: {failed}", flush=True)

    rows = collate([o for o in outs if Path(o).exists()])
    (outdir / "rows.json").write_text(json.dumps(rows))
    print(f"collated {len(rows)} cells → {outdir}/rows.json", flush=True)


if __name__ == "__main__":
    main()
