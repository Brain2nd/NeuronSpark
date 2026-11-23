"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_prefill_join``.

``run`` is :func:`neuronspark.ops.triton.tp_bulk.tp_prefill_join`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_bulk.tp_prefill_join_ref` (same parameters).

Parameters: ``parts, h, flag, rank, epoch``.
Inputs: parts N (B T, 1536) fp32 contiguous out-projection partials in (b, t) row order (the other GPUs' read through
their mappings), h (T, B, 1536) fp32 with unit last stride, flag int64 (N,) this GPU's row, rank, epoch int64 (1,).
Returns: (T, B, 1536) fp32 contiguous h + f32(bf16(the N partials summed in rank order)).
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_bulk import tp_prefill_join as run
from .neuronspark.ops.triton.tp_bulk import tp_prefill_join_ref as ref

__all__ = ["run", "ref"]
