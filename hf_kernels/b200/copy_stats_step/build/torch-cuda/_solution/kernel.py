"""Hugging Face kernel entrypoint of the NeuronSpark operator ``copy_stats_step``.

``run`` is :func:`neuronspark.ops.triton.copy_step.copy_stats_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.copy_step.copy_stats_step_ref` (same parameters).

Parameters: ``z, ring, ring_ptr, pos``.
Inputs: z (B, 96) fp32 order-1 slice of the token's lexical vector, ring (B, W, 96) fp32 unit operands (the token's
written at ring_ptr % W), ring_ptr (1,) int64, pos (B,) int64 tokens before this one.
Returns: (maximum, lse) each (1, B) fp32; -inf where the row has no valid slot.
Regimes: decode.  Written in place: ring.
"""
from __future__ import annotations

from .neuronspark.ops.triton.copy_step import copy_stats_step as run
from .neuronspark.ops.triton.copy_step import copy_stats_step_ref as ref

__all__ = ["run", "ref"]
