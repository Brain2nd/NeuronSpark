"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pool_heads``.

``run`` is :func:`neuronspark.ops.triton.hippo_heads.pool_heads`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_heads.pool_heads_ref` (same parameters).

Parameters: ``q, k, gate, gain, wbias, H, dk``.
Inputs: q, k (T, B, 1536) bf16 after RoPE, gate (T, B, 1536) fp32, gain (6,) bf16 match gain, wbias (6,) fp32.
Returns: (qn [H x (T, B, dk) fp32], kn [H x (T, B, dk) fp32], g [H x (T, B) fp32]).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_heads import pool_heads as run
from .neuronspark.ops.triton.hippo_heads import pool_heads_ref as ref

__all__ = ["run", "ref"]
