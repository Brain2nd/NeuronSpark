"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_value_read``.

``run`` is :func:`neuronspark.ops.triton.pkm_read.pkm_value_read`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_read.pkm_value_read_ref` (same parameters).

Parameters: ``w, u, V``.
Inputs: w (T, B, 64) fp32 slot weights, u (T, B, 64) int64 slot ids, V (16384, 768) fp32.
Returns: (T, B, 768) fp32; backward: dw, dV.
Regimes: train, prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_read import pkm_value_read as run
from .neuronspark.ops.triton.pkm_read import pkm_value_read_ref as ref

__all__ = ["run", "ref"]
