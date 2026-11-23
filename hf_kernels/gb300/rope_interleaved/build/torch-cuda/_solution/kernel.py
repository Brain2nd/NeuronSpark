"""Hugging Face kernel entrypoint of the NeuronSpark operator ``rope_interleaved``.

``run`` is :func:`neuronspark.ops.triton.hippo_rope.rope_interleaved`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_rope.rope_interleaved_ref` (same parameters).

Parameters: ``x, cos, sin``.
Inputs: x (T, B, C) bf16, cos / sin broadcastable bf16.
Returns: (T, B, C) bf16.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_rope import rope_interleaved as run
from .neuronspark.ops.triton.hippo_rope import rope_interleaved_ref as ref

__all__ = ["run", "ref"]
