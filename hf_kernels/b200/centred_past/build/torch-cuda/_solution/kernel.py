"""Hugging Face kernel entrypoint of the NeuronSpark operator ``centred_past``.

``run`` is :func:`neuronspark.ops.triton.cumsum.centred_past`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.cumsum.centred_past_ref` (same parameters).

Parameters: ``x``.
Inputs: x (T, B, C) fp32.
Returns: (T, B, C) bf16 contiguous: x minus the mean of the earlier tokens.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.cumsum import centred_past as run
from .neuronspark.ops.triton.cumsum import centred_past_ref as ref

__all__ = ["run", "ref"]
