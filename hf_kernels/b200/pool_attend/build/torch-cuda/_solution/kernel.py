"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pool_attend``.

``run`` is :func:`neuronspark.ops.pool_attend.pool_attend`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.pool_attend.pool_attend_ref` (same parameters).

Parameters: ``q, k, gate, v, gain, wbias, sparsity, xq, psi, cos, sin, cum, beta, qtok, ktok, orders, scale``.
Inputs: q, k (T, B, 1536) bf16 after SiLU and RoPE, gate (T, B, 1536) fp32 write-gate spikes, v (T, B, 1536) bf16,
gain (6,) bf16 match gain, wbias (6,) fp32, sparsity 2.0, xq 4 x (T, B, 256) bf16 extra-query projections, psi (T, B,
4) bf16, cos / sin (T, 128) bf16 last-head RoPE band, then the operands of pool_read: the head preparation, the tagged
values, the extra queries and the read.
Returns: (heads (T, B, 1536) fp32, xreads (T, B, 1024) bf16, q_all (T, B, 10, 256) bf16, k_all (T, B, 6, 256) bf16).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.pool_attend import pool_attend as run
from .neuronspark.ops.pool_attend import pool_attend_ref as ref

__all__ = ["run", "ref"]
