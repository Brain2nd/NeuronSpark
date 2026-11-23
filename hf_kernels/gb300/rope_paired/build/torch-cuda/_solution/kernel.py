"""Hugging Face kernel entrypoint of the NeuronSpark operator ``rope_paired``.

``run`` is :func:`neuronspark.ops.triton.hippo_rope.rope_paired`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_rope.rope_paired_ref` (same parameters).

Parameters: ``x, cos, sin``.
Inputs: x (T, B, C) bf16, cos / sin (T, 768) bf16.
Returns: (T, B, C) bf16.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_rope import rope_paired as run
from .neuronspark.ops.triton.hippo_rope import rope_paired_ref as ref

__all__ = ["run", "ref"]
