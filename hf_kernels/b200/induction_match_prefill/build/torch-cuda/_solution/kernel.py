"""Hugging Face kernel entrypoint of the NeuronSpark operator ``induction_match_prefill``.

``run`` is :func:`neuronspark.ops.triton.induction_prefill.induction_match_prefill`; ``ref`` is its pure-torch
reference :func:`neuronspark.ops.triton.induction_prefill.induction_match_prefill_ref` (same parameters).

Parameters: ``keys, qwin, ring_ptr, pad``.
Inputs: keys (B, W, 4) int64 ring with the call's keys written, qwin (B, T, 4) int64 the queries' last ids, ring_ptr
(1,) int64 the pointer at the start of the call, pad 151669.
Returns: (4, B, T, W) uint8: entry k - 1 the order-k match of every query against every ring slot.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.induction_prefill import induction_match_prefill as run
from .neuronspark.ops.triton.induction_prefill import induction_match_prefill_ref as ref

__all__ = ["run", "ref"]
