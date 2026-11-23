"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_res_add``.

``run`` is :func:`neuronspark.ops.triton.dendritic_residual.dendritic_res_add`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.dendritic_residual.dendritic_res_add_ref` (same parameters).

Parameters: ``h, upd``.
Inputs: h (T, B, 1536) fp32 residual stream, upd (T, B, 1536) bf16 with unit last stride (the transposed view of the
(B, T, 1536) W_out output).
Returns: (T, B, 1536) fp32 contiguous = h + upd.float().
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.dendritic_residual import dendritic_res_add as run
from .neuronspark.ops.triton.dendritic_residual import dendritic_res_add_ref as ref

__all__ = ["run", "ref"]
