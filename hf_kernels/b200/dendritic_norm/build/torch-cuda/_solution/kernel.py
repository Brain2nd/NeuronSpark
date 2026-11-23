"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_norm``.

``run`` is :func:`neuronspark.ops.triton.dendritic_prenorm.dendritic_norm`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.dendritic_prenorm.dendritic_norm_ref` (same parameters).

Parameters: ``h, gain, eps``.
Inputs: h (T, B, D) fp32 with unit last stride, gain (D,) fp32.
Returns: hn bf16, same shape and layout as h.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.dendritic_prenorm import dendritic_norm as run
from .neuronspark.ops.triton.dendritic_prenorm import dendritic_norm_ref as ref

__all__ = ["run", "ref"]
