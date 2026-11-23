"""Hugging Face kernel entrypoint of the NeuronSpark operator ``short_conv_step``.

``run`` is :func:`neuronspark.ops.triton.short_conv.short_conv_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.short_conv.short_conv_step_ref` (same parameters).

Parameters: ``x, hist, weight``.
Inputs: x (B, 4608) bf16, hist (B, 4608, 3) fp32, weight (4608, 1, 4) fp32.
Returns: (B, 4608) bf16.
Regimes: decode.  Written in place: hist.
"""
from __future__ import annotations

from .neuronspark.ops.triton.short_conv import short_conv_step as run
from .neuronspark.ops.triton.short_conv import short_conv_step_ref as ref

__all__ = ["run", "ref"]
