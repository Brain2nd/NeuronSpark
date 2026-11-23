"""Hugging Face kernel entrypoint of the NeuronSpark operator ``vr_mix_prefill``.

``run`` is :func:`neuronspark.ops.triton.vr_mix.vr_mix_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.vr_mix.vr_mix_prefill_ref` (same parameters).

Parameters: ``v, carry, gl, lam0, glu_add``.
Inputs: v (T, B, 1536) bf16, carry (T, B, 1536) bf16 or None (hippocampal layer 0), gl (T, B, 6) bf16 gate projection,
lam0 (1536,) fp32 contiguous static mix, glu_add (T, B, 256) fp32 GLU addend of the last head.
Returns: (T, B, 1536) bf16 contiguous value stream after the value-residual mix and the GLU tail.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.vr_mix import vr_mix_prefill as run
from .neuronspark.ops.triton.vr_mix import vr_mix_prefill_ref as ref

__all__ = ["run", "ref"]
