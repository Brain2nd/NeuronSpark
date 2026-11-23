"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pool_clock_prefill``.

``run`` is :func:`neuronspark.ops.triton.pool_clock_prefill.pool_clock_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pool_clock_prefill.pool_clock_prefill_ref` (same parameters).

Parameters: ``decay_logit, rate, base_decay, floor, gains, run, cum_ring, ptr``.
Inputs: decay_logit (T, B, 256) bf16 decay projection, rate (256,) bf16 decay rate, base_decay / floor the decay's
factor and lower clamp, gains (6,) fp32 eligibility slopes, run (B, 6) fp32 carried clocks, cum_ring (B, 6, W) fp32
clock rings (written at (ptr + t) % W), ptr (1,) int64 ring pointer.
Returns: (B, 6, T) fp32 contiguous clocks of the call's tokens.
Regimes: prefill.  Written in place: run, cum_ring.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pool_clock_prefill import pool_clock_prefill as run
from .neuronspark.ops.triton.pool_clock_prefill import pool_clock_prefill_ref as ref

__all__ = ["run", "ref"]
