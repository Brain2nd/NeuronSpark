"""Hugging Face kernel entrypoint of the NeuronSpark operator ``zc_center``.

``run`` is :func:`neuronspark.ops.triton.hippo_residual.zc_center`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_residual.zc_center_ref` (same parameters).

Parameters: ``z, zcum``.
Inputs: z (T, B, C) fp32, zcum (T, B, C) fp32 running sum.
Returns: (T, B, C) fp32 contiguous.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_residual import zc_center as run
from .neuronspark.ops.triton.hippo_residual import zc_center_ref as ref

__all__ = ["run", "ref"]
