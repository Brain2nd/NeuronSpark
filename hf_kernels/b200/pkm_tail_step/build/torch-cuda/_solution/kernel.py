"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_tail_step``.

``run`` is :func:`neuronspark.ops.triton.hippo_step.pkm_tail_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_step.pkm_tail_step_ref` (same parameters).

Parameters: ``h, ffn, y, rho``.
Inputs: h (B, 1536) fp32, ffn / y (B, 1536) bf16, rho (B,) fp32.
Returns: (B, 1536) fp32.
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_step import pkm_tail_step as run
from .neuronspark.ops.triton.hippo_step import pkm_tail_step_ref as ref

__all__ = ["run", "ref"]
