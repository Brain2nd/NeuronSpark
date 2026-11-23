"""Hugging Face kernel entrypoint of the NeuronSpark operator ``novelty_read_gain``.

``run`` is :func:`neuronspark.ops.triton.novelty_gain.novelty_read_gain`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.novelty_gain.novelty_read_gain_ref` (same parameters).

Parameters: ``h, z, r, pi, w, b, v, kappa, eps``.
Inputs: h, z (T, B, D) fp32, r (T, B, D) bf16, pi (T, B) fp32 copy confidence (no gradient), w / v (D,) fp32, b (1,)
fp32, kappa = 4.0 read gain, eps = 1e-6.
Returns: (h' (T, B, D) fp32 in h's layout, c (T, B) fp32 novelty factor); backward: h, z, r, w, b, v.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.novelty_gain import novelty_read_gain as run
from .neuronspark.ops.triton.novelty_gain import novelty_read_gain_ref as ref

__all__ = ["run", "ref"]
