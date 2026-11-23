"""Hugging Face kernel entrypoint of the NeuronSpark operator ``clock_increments``.

``run`` is :func:`neuronspark.ops.triton.pool_clock.clock_increments`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pool_clock.clock_increments_ref` (same parameters).

Parameters: ``logit, rate, gains, base, floor``.
Inputs: logit (T, B, 256) bf16 decay projection, rate (256,) bf16, gains (6,) fp32 eligibility slopes, base the base
decay, floor the decay floor.
Returns: (T, B, 6) fp32 = gains mean_c log(max(exp(-rate softplus(logit)) base, floor)).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pool_clock import clock_increments as run
from .neuronspark.ops.triton.pool_clock import clock_increments_ref as ref

__all__ = ["run", "ref"]
