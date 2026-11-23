"""Hugging Face kernel entrypoint of the NeuronSpark operator ``value_residual``.

``run`` is :func:`neuronspark.ops.triton.hippo_residual.value_residual`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_residual.value_residual_ref` (same parameters).

Parameters: ``v, lam, carry, glu_add``.
Inputs: v (T, B, 1536) bf16, lam fp32 mix, carry (T, B, 1536) bf16 or None (hippocampal layer 0), glu_add (T, B, 256)
fp32.
Returns: v' (T, B, 1536) bf16.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_residual import value_residual as run
from .neuronspark.ops.triton.hippo_residual import value_residual_ref as ref

__all__ = ["run", "ref"]
