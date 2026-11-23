"""Hugging Face kernel entrypoint of the NeuronSpark operator ``linear``.

``run`` is :func:`neuronspark.ops.linear.linear`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.linear.linear_ref` (same parameters).

Parameters: ``x, weight``.
Inputs: x (..., K) fp32|bf16, weight (N, K) fp32; inference: the bf16 weight cache when present.
Returns: (..., N) bf16; backward: dx in x.dtype, dW fp32 as weight.grad.
Regimes: train, prefill, decode.
"""
from __future__ import annotations

from .neuronspark.ops.linear import linear as run
from .neuronspark.ops.linear import linear_ref as ref

__all__ = ["run", "ref"]
