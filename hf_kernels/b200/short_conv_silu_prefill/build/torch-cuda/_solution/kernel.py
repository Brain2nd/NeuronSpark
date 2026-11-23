"""Hugging Face kernel entrypoint of the NeuronSpark operator ``short_conv_silu_prefill``.

``run`` is :func:`neuronspark.ops.triton.short_conv.short_conv_silu_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.short_conv.short_conv_silu_prefill_ref` (same parameters).

Parameters: ``x, hist, weight``.
Inputs: x (T, B, 4608) bf16 with T >= 3, hist (B, 4608, 3) fp32, weight (4608, 1, 4) fp32.
Returns: (T, B, 4608) bf16 silu(conv(x)).
Regimes: prefill.  Written in place: hist.
"""
from __future__ import annotations

from .neuronspark.ops.triton.short_conv import short_conv_silu_prefill as run
from .neuronspark.ops.triton.short_conv import short_conv_silu_prefill_ref as ref

__all__ = ["run", "ref"]
