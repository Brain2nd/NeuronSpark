"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_prenorm_rows``.

``run`` is :func:`neuronspark.ops.triton.dendritic_step.dendritic_prenorm_rows`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.dendritic_step.dendritic_prenorm_rows_ref` (same parameters).

Parameters: ``x, gain, eps``.
Inputs: x (M, D) fp32 rows, gain (D,) fp32.
Returns: (M, D) bf16.
Regimes: prefill, decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.dendritic_step import dendritic_prenorm_rows as run
from .neuronspark.ops.triton.dendritic_step import dendritic_prenorm_rows_ref as ref

__all__ = ["run", "ref"]
