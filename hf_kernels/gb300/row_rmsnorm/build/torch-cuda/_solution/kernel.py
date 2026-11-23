"""Hugging Face kernel entrypoint of the NeuronSpark operator ``row_rmsnorm``.

``run`` is :func:`neuronspark.ops.triton.hippo_norms.row_rmsnorm`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_norms.row_rmsnorm_ref` (same parameters).

Parameters: ``x, weight, eps``.
Inputs: x (T, B, C) fp32|bf16, weight (C,) fp32 or None (gain 1).
Returns: (T, B, C) fp32.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_norms import row_rmsnorm as run
from .neuronspark.ops.triton.hippo_norms import row_rmsnorm_ref as ref

__all__ = ["run", "ref"]
