"""Hugging Face kernel entrypoint of the NeuronSpark operator ``centred_running``.

``run`` is :func:`neuronspark.ops.triton.cumsum.centred_running`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.cumsum.centred_running_ref` (same parameters).

Parameters: ``x, run_sum, count, out_dtype``.
Inputs: x (T, B, C), run_sum (B, C) fp32 carried sum, count (B,) int64 tokens before this call.
Returns: (T, B, C) in out_dtype.
Regimes: prefill.  Written in place: run_sum.
"""
from __future__ import annotations

from .neuronspark.ops.triton.cumsum import centred_running as run
from .neuronspark.ops.triton.cumsum import centred_running_ref as ref

__all__ = ["run", "ref"]
