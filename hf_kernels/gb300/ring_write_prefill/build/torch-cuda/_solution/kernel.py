"""Hugging Face kernel entrypoint of the NeuronSpark operator ``ring_write_prefill``.

``run`` is :func:`neuronspark.ops.triton.pool_ring.ring_write_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pool_ring.ring_write_prefill_ref` (same parameters).

Parameters: ``k_h, q_h, g_h, v_h, qgain, wbias, sparsity, pool_k, pool_gv, ring_ptr, T, B, H, W``.
Inputs: per-head k / q / gate / v of this call, qgain (H,) fp32 match gain, wbias (H,) fp32, sparsity 2.0, rings (B,
H, W, 256) bf16.
Returns: q fp32: the gained unit queries of the call.
Regimes: prefill.  Written in place: pool_k, pool_gv.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pool_ring import ring_write_prefill as run
from .neuronspark.ops.triton.pool_ring import ring_write_prefill_ref as ref

__all__ = ["run", "ref"]
