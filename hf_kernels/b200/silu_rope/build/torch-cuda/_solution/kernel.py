"""Hugging Face kernel entrypoint of the NeuronSpark operator ``silu_rope``.

``run`` is :func:`neuronspark.ops.triton.hippo_rope.silu_rope`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_rope.silu_rope_ref` (same parameters).

Parameters: ``x, cos, sin``.
Inputs: x (T, B, C) bf16, cos / sin (T, 768) bf16.
Returns: (T, B, C) bf16 = rope(silu(x)).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_rope import silu_rope as run
from .neuronspark.ops.triton.hippo_rope import silu_rope_ref as ref

__all__ = ["run", "ref"]
