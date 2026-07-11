"""Score-net training checkpoints — save / load / resume / rollback.

A100 deployment spec §6 deliverable (§4 assigns training checkpoints to
diffuse_directly §6): a long GPU training run must be able to **crash-recover**
and **roll back to a selected epoch** for model selection. This is pure
`state_dict` persistence — model weights + optimizer moments + a small metadata
dict — so it is CPU/MPS-validatable and needs no A100; only the training run it
protects is box work.

Lazy torch import (the `[gpu]` extra), so CPU-only installs / CI stay torch-free.

Security note: `load_checkpoint` uses `weights_only=False` because the metadata
dict holds plain Python scalars, not just tensors. Only load checkpoints you
produced/trust — same discipline as any pickle-backed artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

_PathLike = Union[str, Path]


def save_checkpoint(path: _PathLike, model, opt=None, *, meta: Optional[dict] = None) -> None:
    """Persist ``model`` (and optionally the optimizer ``opt``) plus a ``meta``
    dict (e.g. ``{"epoch": n, "dim": d}``) to ``path``. Passing the optimizer is
    what makes a later resume bit-identical — Adam's moments are part of the
    training state."""
    import torch
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict() if opt is not None else None,
            "meta": dict(meta or {}),
        },
        str(path),
    )


def load_checkpoint(path: _PathLike, model, opt=None, *,
                    map_location: str = "cpu") -> dict:
    """Restore ``model`` (and ``opt`` if given and present) in place from
    ``path``; returns the saved ``meta`` dict. ``map_location`` keeps loading
    device-agnostic (default 'cpu' — move the model afterwards if needed)."""
    import torch
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("opt") is not None:
        opt.load_state_dict(ckpt["opt"])
    return ckpt.get("meta", {})
