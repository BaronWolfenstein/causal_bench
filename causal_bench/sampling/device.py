"""Device resolution (single-device torch) and multi-GPU gating (which CUDA
ids are visible). Both lazy-import torch so CPU-only installs/CI stay torch-free.
An empty cuda_available_devices() is the signal to fall back to CPU run_smc."""
from __future__ import annotations

import os


def resolve_device(prefer: str = "auto") -> str:
    """Single-device torch resolution: cuda → mps → cpu. A non-'auto' value
    passes through unchanged so callers can pin explicitly."""
    if prefer != "auto":
        return prefer
    import torch                                   # lazy
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def cuda_available_devices() -> list[int]:
    """Ordered visible CUDA device ids, or [] on a CPU box. Honors
    CUDA_VISIBLE_DEVICES (respecting its order); otherwise torch.cuda.device_count().
    Never raises on a CPU box — returns []."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        env = env.strip()
        if env == "":
            return []
        return [int(x) for x in env.split(",") if x.strip() != ""]
    try:
        import torch                               # lazy
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []
