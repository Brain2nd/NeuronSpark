"""Hugging Face kernel entrypoint of the NeuronSpark operator ``xq_form_t1``.

``run`` is :func:`neuronspark.ops.triton.pool_ring.xq_form_t1`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pool_ring.xq_form_t1_ref` (same parameters).

Parameters: ``u, wb, cos, sin, psi, head_gain, pos``.
Inputs: u (B, 256) bf16, wb (4, 256, 64) bf16, cos / sin (B, 128) bf16 (with pos (B,) int64: the (positions, 128)
table band, read at pos), psi (B, 4) bf16, head_gain (1,) fp32.
Returns: (B, 4, 256) bf16.
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pool_ring import xq_form_t1 as run
from .neuronspark.ops.triton.pool_ring import xq_form_t1_ref as ref

__all__ = ["run", "ref"]
