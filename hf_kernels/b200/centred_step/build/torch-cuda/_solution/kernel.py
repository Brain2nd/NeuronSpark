"""Hugging Face kernel entrypoint of the NeuronSpark operator ``centred_step``.

``run`` is :func:`neuronspark.ops.triton.plif_step.centred_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.plif_step.centred_step_ref` (same parameters).

Parameters: ``x, run_sum, count, out_dtype``.
Inputs: x (B, C), run_sum (B, C) fp32 carried sum, count (B,) int64 tokens before this one.
Returns: (B, C) in out_dtype.
Regimes: decode.  Written in place: run_sum.
"""
from __future__ import annotations

from .neuronspark.ops.triton.plif_step import centred_step as run
from .neuronspark.ops.triton.plif_step import centred_step_ref as ref

__all__ = ["run", "ref"]
