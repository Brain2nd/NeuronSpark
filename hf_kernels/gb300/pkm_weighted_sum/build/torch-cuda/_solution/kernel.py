"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_weighted_sum``.

``run`` is :func:`neuronspark.ops.triton.pkm_read.pkm_weighted_sum`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_read.pkm_weighted_sum_ref` (same parameters).

Parameters: ``w, u, V``.
Inputs: w fp32 slot weights, u int64 slot ids, V (16384, 768) fp32.
Returns: (T, B, 768) fp32.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_read import pkm_weighted_sum as run
from .neuronspark.ops.triton.pkm_read import pkm_weighted_sum_ref as ref

__all__ = ["run", "ref"]
