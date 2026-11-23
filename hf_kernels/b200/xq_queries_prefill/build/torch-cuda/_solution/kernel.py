"""Hugging Face kernel entrypoint of the NeuronSpark operator ``xq_queries_prefill``.

``run`` is :func:`neuronspark.ops.triton.xq_query.xq_queries_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.xq_query.xq_queries_prefill_ref` (same parameters).

Parameters: ``qs, psi, head_gain, cos, sin``.
Inputs: qs 4 x (T, B, 256) bf16 contiguous projections, psi (T, B, 4) fp32 clamped gain projection, head_gain one-
element bf16 match gain of the last head, cos / sin (T, B, 128) bf16 last-head RoPE band.
Returns: (B, 4, T, 256) bf16 gained unit extra queries (view of a contiguous (4, T, B, 256) buffer).
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.xq_query import xq_queries_prefill as run
from .neuronspark.ops.triton.xq_query import xq_queries_prefill_ref as ref

__all__ = ["run", "ref"]
