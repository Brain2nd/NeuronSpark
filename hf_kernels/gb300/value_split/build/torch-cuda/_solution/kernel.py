"""Hugging Face kernel entrypoint of the NeuronSpark operator ``value_split``.

``run`` is :func:`neuronspark.ops.triton.hippo_heads.value_split`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_heads.value_split_ref` (same parameters).

Parameters: ``v, H, dv``.
Inputs: v (T, B, 1536) bf16.
Returns: H tensors (T, B, dv) bf16.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_heads import value_split as run
from .neuronspark.ops.triton.hippo_heads import value_split_ref as ref

__all__ = ["run", "ref"]
