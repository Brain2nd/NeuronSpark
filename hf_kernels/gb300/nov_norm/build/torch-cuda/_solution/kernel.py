"""Hugging Face kernel entrypoint of the NeuronSpark operator ``nov_norm``.

``run`` is :func:`neuronspark.ops.triton.hippo_norms.nov_norm`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_norms.nov_norm_ref` (same parameters).

Parameters: ``r, eps``.
Inputs: r (T, B, 1536) bf16.
Returns: (T, B, 1536) fp32.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_norms import nov_norm as run
from .neuronspark.ops.triton.hippo_norms import nov_norm_ref as ref

__all__ = ["run", "ref"]
