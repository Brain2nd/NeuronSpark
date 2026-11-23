"""Hugging Face kernel entrypoint of the NeuronSpark operator ``ring_write_step``.

``run`` is :func:`neuronspark.ops.triton.ring_step.ring_write_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.ring_step.ring_write_step_ref` (same parameters).

Parameters: ``qkv, conv_hist, conv_weight, cos, sin, pos, ring_ptr, gate, gl, lam0, carry, ga, gb, glu_u, dlog, rate,
base_decay, gains, qgain, wbias, sparsity, pool_k, pool_gv, pool_cum, pool_run``.
Inputs: qkv (B, 4608) bf16 projection, conv_hist (B, 4608, 3) fp32, conv_weight (4608, 1, 4), cos / sin (8192, 768)
bf16 tables, pos (B,) int64, ring_ptr (1,) int64, gate (B, 1536) fp32 spikes, gl (B, 6) bf16, lam0 (1536,), carry (B,
1536) bf16 or None, ga / gb (B, 128) bf16, glu_u (256, 128), dlog (B, 256) bf16, rate (256,) bf16, base_decay, gains /
qgain / wbias (6,), sparsity, the rings and the carried clocks.
Returns: (q (B, 6, 256) bf16 gained unit queries, v (B, 1536) bf16 values after the SiLU).
Regimes: decode.  Written in place: conv_hist, pool_k, pool_gv, pool_cum, pool_run.
"""
from __future__ import annotations

from .neuronspark.ops.triton.ring_step import ring_write_step as run
from .neuronspark.ops.triton.ring_step import ring_write_step_ref as ref

__all__ = ["run", "ref"]
