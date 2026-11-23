"""Hugging Face kernel entrypoint of the NeuronSpark operator ``read_gain_step``.

``run`` is :func:`neuronspark.ops.triton.hippo_step.read_gain_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_step.read_gain_step_ref` (same parameters).

Parameters: ``h, z, r, cmax, clse, sink, strength, nov_w, nov_v, nov_b, norm_weight, kappa, scale, eps_r, eps_n``.
Inputs: h, z (B, 1536) fp32, r (B, 1536) bf16 output projection, cmax / clse (B,) fp32, sink / strength (1,), nov_w /
nov_v (1536,), nov_b (1,), norm_weight (1536,), kappa 4.0, scale 32.0, eps 1e-6 / 1e-6.
Returns: (h' (B, 1536) fp32, x_hat (B, 1536) fp32, x_hat (B, 1536) bf16, rho (B,) fp32).
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_step import read_gain_step as run
from .neuronspark.ops.triton.hippo_step import read_gain_step_ref as ref

__all__ = ["run", "ref"]
