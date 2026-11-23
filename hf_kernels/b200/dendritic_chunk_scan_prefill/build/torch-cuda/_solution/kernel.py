"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_chunk_scan_prefill``.

``run`` is :func:`neuronspark.ops.dendritic_prefill.dendritic_chunk_scan_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.dendritic_prefill.dendritic_chunk_scan_prefill_ref` (same parameters).

Parameters: ``x, dt, A, Bm, Cm, D, dt_bias, initial_state, *, chunk_size``.
Inputs: x (B, T, 144, 64) bf16, dt (B, T, 144) bf16, A (144,) fp32, Bm / Cm (B, T, 1, 128) bf16, D (144,), dt_bias
(144,), initial_state (B, 144, 64, 128) fp32; chunk_size = 256.
Returns: (y (B, T, 144, 64) bf16, final_state (B, 144, 64, 128) fp32).
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.dendritic_prefill import dendritic_chunk_scan_prefill as run
from .neuronspark.ops.dendritic_prefill import dendritic_chunk_scan_prefill_ref as ref

__all__ = ["run", "ref"]
