"""Hugging Face kernel entrypoint of the NeuronSpark operator ``linear_add``.

``run`` is :func:`neuronspark.ops.linear.linear_add`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.linear.linear_add_ref` (same parameters).

Parameters: ``x, weight, addend, x_scale``.
Inputs: x (..., K) fp32|bf16, weight (N, K) fp32, addend (..., N) fp32|bf16, x_scale None or bf16 of x's shape (x bf16
then).
Returns: (..., N) in addend's dtype = addend + linear(x [* x_scale], weight), a new tensor.
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.linear import linear_add as run
from .neuronspark.ops.linear import linear_add_ref as ref

__all__ = ["run", "ref"]
