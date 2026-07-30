"""Array-namespace backend seam. numpy on CPU (the reference path every test
uses), cupy on CUDA for the A100 box.

The SMC hot loop (`smc_step`'s resample branch, `normalize_log_weights`,
`systematic_resample`) is namespace-generic: each routes through
`get_namespace(...)` and operates on numpy or cupy alike. `run_smc(device=...)`
converts the initial state on-device and returns host numpy via these helpers.
`device="cuda"` numerical parity with `device="cpu"` is asserted by
`tests/test_smc_cuda_parity.py`, which runs on the A100 box (skips off-box where
cupy is absent); the multi-GPU distributed==serial validation runs on the box
per the deployment runbook."""
from __future__ import annotations

import numpy as np


def array_namespace(device: str = "cpu"):
    if device == "cpu":
        return np
    if device.startswith("cuda"):
        import cupy as cp                # lazy: only needed on the GPU box
        return cp
    raise ValueError(f"unknown device: {device!r} (use 'cpu' or 'cuda')")


def asarray(x, device: str = "cpu"):
    return array_namespace(device).asarray(x)


def to_numpy(x):
    """Move any array (numpy or cupy) back to host numpy."""
    if type(x).__module__.startswith("cupy"):
        return x.get()
    return np.asarray(x)


def get_namespace(*arrays):
    """Infer the array namespace (numpy or cupy) from the arrays themselves,
    so hot-loop functions stay device-agnostic without threading a `device`
    string through their signatures. Mirrors `to_numpy`'s module sniff.
    Satisfies spec §1a ('route through xp = array_namespace(device)')."""
    for a in arrays:
        if type(a).__module__.startswith("cupy"):
            import cupy as cp            # lazy: only on the GPU box
            return cp
    return np
