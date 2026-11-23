"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_scores_step``.

``run`` is :func:`neuronspark.ops.triton.pkm_step.pkm_scores_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_step.pkm_scores_step_ref` (same parameters).

Parameters: ``q, kh``.
Inputs: q (B, 512) bf16 query projection, kh (4, 2, 128, 64) fp32 unit sub-keys.
Returns: (B, 4, 2, 128) fp32.
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_step import pkm_scores_step as run
from .neuronspark.ops.triton.pkm_step import pkm_scores_step_ref as ref

__all__ = ["run", "ref"]
