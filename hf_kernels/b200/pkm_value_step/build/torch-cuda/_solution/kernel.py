"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_value_step``.

``run`` is :func:`neuronspark.ops.triton.pkm_step.pkm_value_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_step.pkm_value_step_ref` (same parameters).

Parameters: ``w, u, values, gain, glog, eps``.
Inputs: w (B, 64) fp32, u (B, 64) int64, values (16384, 768) fp32, gain (768,) fp32, glog (B, 768) bf16, eps 1e-6.
Returns: (B, 768) bf16.
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_step import pkm_value_step as run
from .neuronspark.ops.triton.pkm_step import pkm_value_step_ref as ref

__all__ = ["run", "ref"]
