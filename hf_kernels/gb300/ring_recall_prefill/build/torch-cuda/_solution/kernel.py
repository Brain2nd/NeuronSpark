"""Hugging Face kernel entrypoint of the NeuronSpark operator ``ring_recall_prefill``.

``run`` is :func:`neuronspark.ops.triton.pool_ring.ring_recall_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pool_ring.ring_recall_prefill_ref` (same parameters).

Parameters: ``q4, k_ring, gv_ring, cum_ring, ct3, p0, ind4, scale, T, B, G, W, dk, dv, out_dtype, xq, hr``.
Inputs: queries, rings and clocks of the call, p0 first position, ind4 = (match matrices, beta), scale 16.0; xq =
False / hr = 0 for the six heads, xq = True for the extra queries.
Returns: (T, B, G * dv) out_dtype.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pool_ring import ring_recall_prefill as run
from .neuronspark.ops.triton.pool_ring import ring_recall_prefill_ref as ref

__all__ = ["run", "ref"]
