"""Hugging Face kernel entrypoint of the NeuronSpark operator ``rmsnorm_rows``.

``run`` is :func:`neuronspark.ops.triton.hippo_norms.rmsnorm_rows`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_norms.rmsnorm_rows_ref` (same parameters).

Parameters: ``x, eps, weight, zero_centered, out_dtype``.
Inputs: x (M, N) any float, weight (N,) fp32 or None, zero_centered: gain 1 + weight; N % 128 == 0, N <= 4096.
Returns: (M, N) in out_dtype (default x.dtype).
Regimes: prefill, decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_norms import rmsnorm_rows as run
from .neuronspark.ops.triton.hippo_norms import rmsnorm_rows_ref as ref

__all__ = ["run", "ref"]
