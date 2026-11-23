"""Hugging Face kernel entrypoint of the NeuronSpark operator ``row_rmsnorm_tee``.

``run`` is :func:`neuronspark.ops.triton.hippo_norms.row_rmsnorm_tee`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_norms.row_rmsnorm_tee_ref` (same parameters).

Parameters: ``x, weight, eps, fp32, bf16``.
Inputs: x (T, B, C) fp32, weight (C,) fp32 or None (gain 1), fp32: return the fp32 norm, bf16 in (0, 1, 2): the number
of bf16 outputs (one per bf16 consumer).
Returns: (x, y (T, B, C) fp32 or None, bf16 x [(T, B, C) bf16]): the residual stream handed on, the norm, its bf16
roundings.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_norms import row_rmsnorm_tee as run
from .neuronspark.ops.triton.hippo_norms import row_rmsnorm_tee_ref as ref

__all__ = ["run", "ref"]
