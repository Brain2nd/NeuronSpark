"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_conv_silu_prefill``.

``run`` is :func:`neuronspark.ops.triton.dendritic_conv.dendritic_conv_silu_prefill`; ``ref`` is its pure-torch
reference :func:`neuronspark.ops.triton.dendritic_conv.dendritic_conv_silu_prefill_ref` (same parameters).

Parameters: ``xBC, conv_w, conv_b, tail``.
Inputs: xBC (B, T, 9472) bf16 strided view of the projection, conv_w (9472, 4), conv_b (9472,), tail (B, 9472, 3) fp32
= the previous three inputs.
Returns: (xbc (B, T, 9472) bf16 contiguous, tail_new (B, 9472, 3) fp32).
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.dendritic_conv import dendritic_conv_silu_prefill as run
from .neuronspark.ops.triton.dendritic_conv import dendritic_conv_silu_prefill_ref as ref

__all__ = ["run", "ref"]
