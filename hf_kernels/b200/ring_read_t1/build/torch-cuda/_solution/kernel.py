"""Hugging Face kernel entrypoint of the NeuronSpark operator ``ring_read_t1``.

``run`` is :func:`neuronspark.ops.triton.pool_ring.ring_read_t1`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pool_ring.ring_read_t1_ref` (same parameters).

Parameters: ``q, xq, cum_t, pool_k, pool_gv, pool_cum, pos, ring_ptr, matches, beta, gain``.
Inputs: q (B, 6, 256), xq (B, 4, 256), cum_t this step's clocks, rings, counters, induction matches, beta, gain.
Returns: (B, 1, 1536 + extra-query width) bf16.
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pool_ring import ring_read_t1 as run
from .neuronspark.ops.triton.pool_ring import ring_read_t1_ref as ref

__all__ = ["run", "ref"]
