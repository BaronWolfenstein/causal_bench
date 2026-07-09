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
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # emb_eval
    Tuple[np.ndarray, np.ndarray],  # recon_b_eval
    Tuple[np.ndarray, np.ndarray],  # recon_b_prime_eval
    Tuple[np.ndarray, np.ndarray],  # recon_c_eval
    Tuple[np.ndarray, np.ndarray],  # rare_guided_eval, common_ref_eval (B'' landing)
]:
    """Helper to emit all five eval-space arrays for run_diagnostic.

    When the codebook renders rare embeddings to common tokens (metric-hacking
    scenario), this produces render->re-encode outputs in E_eval space for
    Tests B, B', C, and the B'' landing gate.

    Parameters
    ----------
    rare_raw, common_raw : np.ndarray
        Raw features of rare and common populations.
    renderer : CodebookRenderer
        Codebook for render->re-encode.
    e_eval : FrozenEncoder
        Decoupled evaluation encoder.

    Returns
    -------
    tuple of five tuples (rare_eval, common_eval), (rare_recon_b_eval, common_recon_b_eval), ...
        to pass as emb_eval, recon_b_eval, recon_b_prime_eval, recon_c_eval,
        rare_guided_eval, common_ref_eval to run_diagnostic.
    """
    # All five eval-space arrays are the same render->re-encode outputs
    # (the codebook determines the collapse; different tests just verify
    # the same metric-hacking across different gates).
    rare_eval = e_eval(rare_raw)
    common_eval = e_eval(common_raw)

    # For stand-in encoders, we don't have actual embeddings; this is a
    # placeholder that the tests will replace with their own logic.
    # In real usage, these would be generated by the diffusion model,
    # the tail-aware diffusion, the separate-latent diffusion, and
    # CFG-guided generation, respectively.
    emb_eval = (rare_eval, common_eval)
    recon_b_eval = (rare_eval, common_eval)
    recon_b_prime_eval = (rare_eval, common_eval)
    recon_c_eval = (rare_eval, common_eval)
    rare_guided_eval = rare_eval
    common_ref_eval = common_eval

    return emb_eval, recon_b_eval, recon_b_prime_eval, recon_c_eval, (
        rare_guided_eval,
        common_ref_eval,
    )
