"""Hugging Face kernel entrypoint of the NeuronSpark operator ``centre_rows``.

``run`` is :func:`neuronspark.ops.triton.hippo_residual.centre_rows`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_residual.centre_rows_ref` (same parameters).

Parameters: ``r``.
Inputs: r (T, B, D) bf16 output projection of the attention sub-block.
Returns: (T, B, D) bf16 = r - mean_c(r).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_residual import centre_rows as run
from .neuronspark.ops.triton.hippo_residual import centre_rows_ref as ref

__all__ = ["run", "ref"]
