"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pool_read``.

``run`` is :func:`neuronspark.ops.pool_read.pool_read`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.pool_read.pool_read_ref` (same parameters).

Parameters: ``q, k, gv, cum, beta, qtok, ktok, orders, scale``.
Inputs: q (T, B, 10, 256) fp32 gained unit queries (6 heads, 4 extra queries), k (T, B, 6, 256) fp32 unit keys, gv (T,
B, 6, 256) bf16 tagged values, cum (T, B, 6) fp32 clocks, beta (3, 6) fp32 = pool_ind_beta, qtok / ktok (B, T, 4)
int32 id windows, orders (2, 3, 4), scale 16.0.
Returns: (T, B, 10, 256) bf16: the 6 heads' reads, then the last head's 4 extra-query reads.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.pool_read import pool_read as run
from .neuronspark.ops.pool_read import pool_read_ref as ref

__all__ = ["run", "ref"]
