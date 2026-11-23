"""Hugging Face kernel entrypoint of the NeuronSpark operator ``short_conv_train``.

``run`` is :func:`neuronspark.ops.triton.short_conv.short_conv_train`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.short_conv.short_conv_train_ref` (same parameters).

Parameters: ``qkv, weight``.
Inputs: qkv (T, B, 4608) bf16, weight (4608, 1, 4) fp32.
Returns: (T, B, 4608) bf16 T-major; backward: dW (4608, 1, 4) fp32 as weight.grad.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.short_conv import short_conv_train as run
from .neuronspark.ops.triton.short_conv import short_conv_train_ref as ref

__all__ = ["run", "ref"]
