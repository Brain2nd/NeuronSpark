"""Hugging Face kernel entrypoint of the NeuronSpark operator ``induction_step``.

``run`` is :func:`neuronspark.ops.triton.induction_step.induction_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.induction_step.induction_step_ref` (same parameters).

Parameters: ``ids, win, keys, ring_ptr, pad``.
Inputs: ids (B,) int64, win (B, 4) int64 the rows' last ids, keys (B, W, 4) int64 ring, ring_ptr (1,) int64, pad
151669.
Returns: (4, B, W) uint8: entry k - 1 the order-k match of every ring slot.
Regimes: decode.  Written in place: win, keys.
"""
from __future__ import annotations

from .neuronspark.ops.triton.induction_step import induction_step as run
from .neuronspark.ops.triton.induction_step import induction_step_ref as ref

__all__ = ["run", "ref"]
