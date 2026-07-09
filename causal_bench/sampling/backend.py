"""Array-namespace backend seam. numpy on CPU (the reference path every test
uses), cupy on CUDA for the A100 box.

Scope (be honest about it): this establishes the seam and the CPU reference path.
`run_smc(device=...)` converts the initial state and returns host numpy via these
helpers, but the SMC *hot loop* (`smc_step`'s resample branch, `normalize_log_weights`,
`systematic_resample`) still calls `np.*` directly, so `device="cuda"` is NOT yet a
validated end-to-end path — porting the hot loop to `xp = array_namespace(device)`
and validating it (distributed == serial, then throughput) belongs to the deferred
multi-GPU plan, on the box. Do not rely on the cuda path until then."""
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
