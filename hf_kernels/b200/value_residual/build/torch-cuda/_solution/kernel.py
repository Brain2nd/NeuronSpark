"""Hugging Face kernel entrypoint of the NeuronSpark operator ``value_residual``.

``run`` is :func:`neuronspark.ops.triton.hippo_residual.value_residual`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_residual.value_residual_ref` (same parameters).

Parameters: ``v, lam0, gate, carry, glu_add``.
Inputs: v (T, B, 1536) bf16 before its SiLU, lam0 (1536,) fp32 static mix, gate (T, B, 6) bf16 gate logits, carry (T,
B, 1536) bf16 or None (hippocampal layer 0), glu_add (T, B, 256) bf16.
Returns: (v' (T, B, 1536) bf16, silu(v) (T, B, 1536) bf16 for hippocampal layer 0, else None).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_residual import value_residual as run
from .neuronspark.ops.triton.hippo_residual import value_residual_ref as ref

__all__ = ["run", "ref"]
