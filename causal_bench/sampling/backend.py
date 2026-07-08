"""Array-namespace backend seam. numpy on CPU (the reference path every test
uses), cupy on CUDA for the A100 box. The SMC hot path can run on either by
selecting `xp = array_namespace(device)`; GPU is a namespace swap, not a rewrite.
Absolute GPU throughput is validated on the box — correctness is device-agnostic."""
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
