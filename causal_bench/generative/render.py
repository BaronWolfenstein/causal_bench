"""ELF-style final-step discretization + the render->re-encode bridge for the
#88 metric-hacking guard. A generated embedding is snapped to the nearest
codebook token (shared-weight discretization), decoded to raw features, then
re-encoded by a DECOUPLED encoder E_eval. Runs entirely on stand-in encoders —
no MOTOR. When the codebook cannot represent rare detail, render->re-encode
collapses the rare mode in E_eval space and the diagnostic flags metric-hacking."""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from .encoder import FrozenEncoder


class CodebookRenderer:
    """Snap embeddings to nearest codebook tokens (shared-weight discretization).

    In the real ELF pipeline, the codebook is the embedding matrix of a frozen
    neural codec. Here, we use raw features as the codebook entries for testing
    the stand-in path.
    """

    def __init__(self, codebook_raw: np.ndarray):
        """Initialize with a codebook in raw feature space.

        Parameters
        ----------
        codebook_raw : np.ndarray
            Shape (V, in_dim) where V is vocabulary size.
        """
        self.codebook = np.asarray(codebook_raw, dtype=float)

    def render(self, emb_or_raw: np.ndarray) -> np.ndarray:
        """Nearest-codebook token ids.

        For the stand-in path we snap raw features directly (the real ELF ties
        this to the encoder's embedding matrix).

        Parameters
        ----------
        emb_or_raw : np.ndarray
            Shape (n, d) embeddings or raw features to quantize.

        Returns
        -------
        np.ndarray
            Shape (n,) integer token ids in [0, V).
        """
        d = np.linalg.norm(
            emb_or_raw[:, None, :] - self.codebook[None, :, :], axis=2
        )
        return d.argmin(axis=1)

    def decode(self, ids: np.ndarray) -> np.ndarray:
        """Retrieve codebook entries by token id.

        Parameters
        ----------
        ids : np.ndarray
            Shape (n,) integer token ids.

        Returns
        -------
        np.ndarray
            Shape (n, in_dim) raw features corresponding to the ids.
        """
        return self.codebook[ids]


def render_and_reencode(
    emb_gen_space: np.ndarray,
    renderer: CodebookRenderer,
    e_eval: FrozenEncoder,
) -> np.ndarray:
    """Snap embedding to nearest codebook token, decode, and re-encode in E_eval.

    This is the render->re-encode bridge: a generated embedding (E_gen space)
    is snapped to the nearest codebook entry, decoded to raw features, then
    re-encoded by a DECOUPLED encoder E_eval. When the codebook cannot
    represent rare detail, this step collapses the rare mode in E_eval space.

    Parameters
    ----------
    emb_gen_space : np.ndarray
        Shape (n, out_dim) embeddings in the generation encoder's space.
    renderer : CodebookRenderer
        Codebook with render/decode methods.
    e_eval : FrozenEncoder
        Decoupled evaluation encoder (a callable).

    Returns
    -------
    np.ndarray
        Shape (n, out_dim) re-encoded embeddings in E_eval space.
    """
    q = np.asarray(emb_gen_space, dtype=float)
    cb = renderer.codebook

    # For the stand-in path, we snap using the leading coordinates of the
    # embedding as a query on raw space. The real ELF renders MEDS tokens
    # then re-encodes.
    k = min(q.shape[1], cb.shape[1])
    ids = np.linalg.norm(q[:, None, :k] - cb[None, :, :k], axis=2).argmin(axis=1)
    raw = renderer.decode(ids)
    return e_eval(raw)


def eval_space_inputs(
    rare_raw: np.ndarray,
    common_raw: np.ndarray,
    renderer: CodebookRenderer,
    e_eval: FrozenEncoder,
    recon_b: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_b_prime: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    recon_c: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    rare_guided: Optional[np.ndarray] = None,
    common_ref: Optional[np.ndarray] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],            # emb_eval
    Optional[Tuple[np.ndarray, np.ndarray]],  # recon_b_eval
    Optional[Tuple[np.ndarray, np.ndarray]],  # recon_b_prime_eval
    Optional[Tuple[np.ndarray, np.ndarray]],  # recon_c_eval
    Optional[Tuple[np.ndarray, np.ndarray]],  # (rare_guided_eval, common_ref_eval)
]:
    """Helper to emit the eval-space arrays for run_diagnostic.

    `emb_eval` is the direct re-encode of the true originals — `e_eval(rare_raw)`
    / `e_eval(common_raw)` — since there is no generation step to render for the
    real patients themselves. Every OTHER eval-space array is produced by
    actually rendering the corresponding GENERATION-space array (the diffusion
    model's round-trip / CFG-guided samples, in E_gen space) through
    `render_and_reencode`: snap to the nearest codebook token, decode to raw
    features, re-encode with the decoupled `e_eval`. When the codebook cannot
    represent rare detail, this collapses the rare mode in E_eval space —
    which is the metric-hacking signature Tests B/B'/C/B'' are meant to catch.

    Only the gen-space arrays that are supplied get an eval-space counterpart;
    omitted ones return `None` (matching `run_diagnostic`'s "not yet provided"
    convention for `recon_*_eval` / `rare_guided_eval` / `common_ref_eval`).

    Parameters
    ----------
    rare_raw, common_raw : np.ndarray
        Raw features of the rare and common populations (ground truth).
    renderer : CodebookRenderer
        Codebook for render->re-encode.
    e_eval : FrozenEncoder
        Decoupled evaluation encoder.
    recon_b, recon_b_prime, recon_c : optional (rare_recon, common_recon) pairs
        in E_gen space — the diffusion round-trips for Tests B / B' / C.
    rare_guided, common_ref : optional CFG-guided-generation arrays in E_gen
        space, for the B'' landing gate.

    Returns
    -------
    tuple of (emb_eval, recon_b_eval, recon_b_prime_eval, recon_c_eval,
    (rare_guided_eval, common_ref_eval)) to pass as emb_eval, recon_b_eval,
    recon_b_prime_eval, recon_c_eval, rare_guided_eval, common_ref_eval to
    run_diagnostic. Entries for gen-space arrays that were not supplied are
    None.
    """
    emb_eval = (e_eval(rare_raw), e_eval(common_raw))

    def _render_pair(pair):
        if pair is None:
            return None
        rare_gen, common_gen = pair
        return (
            render_and_reencode(rare_gen, renderer, e_eval),
            render_and_reencode(common_gen, renderer, e_eval),
        )

    recon_b_eval = _render_pair(recon_b)
    recon_b_prime_eval = _render_pair(recon_b_prime)
    recon_c_eval = _render_pair(recon_c)

    if rare_guided is not None and common_ref is not None:
        guided_eval = (
            render_and_reencode(rare_guided, renderer, e_eval),
            render_and_reencode(common_ref, renderer, e_eval),
        )
    else:
        guided_eval = None

    return emb_eval, recon_b_eval, recon_b_prime_eval, recon_c_eval, guided_eval
