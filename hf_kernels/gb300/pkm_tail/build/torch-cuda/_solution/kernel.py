"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_tail``.

``run`` is :func:`neuronspark.ops.triton.hippo_residual.pkm_tail`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_residual.pkm_tail_ref` (same parameters).

Parameters: ``h, ffn, y, rho``.
Inputs: h (T, B, D) fp32, ffn (T, B, D) bf16, y (T, B, D) bf16, rho (T, B) fp32 (no gradient).
Returns: h_out (T, B, D) fp32.
Regimes: train, prefill, decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_residual import pkm_tail as run
from .neuronspark.ops.triton.hippo_residual import pkm_tail_ref as ref

__all__ = ["run", "ref"]
