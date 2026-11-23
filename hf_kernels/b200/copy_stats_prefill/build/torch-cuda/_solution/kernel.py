"""Hugging Face kernel entrypoint of the NeuronSpark operator ``copy_stats_prefill``.

``run`` is :func:`neuronspark.ops.triton.copy_stats.copy_stats_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.copy_stats.copy_stats_prefill_ref` (same parameters).

Parameters: ``ring, ring_ptr, pos, T``.
Inputs: ring (B, W, 96) fp32 unit operands (the call's own written at (ring_ptr + t) % W), ring_ptr (1,) int64 pointer
at the call's start, pos (B,) int64 tokens before the call, T the call's tokens.
Returns: (maximum, lse) each (T, B) fp32 contiguous; -inf where a query has no valid slot.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.copy_stats import copy_stats_prefill as run
from .neuronspark.ops.triton.copy_stats import copy_stats_prefill_ref as ref

__all__ = ["run", "ref"]
