"""Hugging Face kernel entrypoint of the NeuronSpark operator ``read_gain_residual``.

``run`` is :func:`neuronspark.ops.triton.hippo_residual.read_gain_residual`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_residual.read_gain_residual_ref` (same parameters).

Parameters: ``h, r, f``.
Inputs: h (T, B, D) fp32, r (T, B, D) bf16, f (T, B) fp32 read gain.
Returns: h' (T, B, D) fp32.
Regimes: train, prefill, decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_residual import read_gain_residual as run
from .neuronspark.ops.triton.hippo_residual import read_gain_residual_ref as ref

__all__ = ["run", "ref"]
