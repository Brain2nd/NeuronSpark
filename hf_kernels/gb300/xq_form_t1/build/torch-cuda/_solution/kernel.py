"""Hugging Face kernel entrypoint of the NeuronSpark operator ``xq_form_t1``.

``run`` is :func:`neuronspark.ops.triton.pool_ring.xq_form_t1`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pool_ring.xq_form_t1_ref` (same parameters).

Parameters: ``u, wb, cos, sin, psi, head_gain``.
Inputs: u (B, 256), wb (4, 256, 64) bf16, cos / sin, psi (B, 4), head_gain; decode batch b not in {1, 32}.
Returns: (B, 4, 256) fp32.
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pool_ring import xq_form_t1 as run
from .neuronspark.ops.triton.pool_ring import xq_form_t1_ref as ref

__all__ = ["run", "ref"]
