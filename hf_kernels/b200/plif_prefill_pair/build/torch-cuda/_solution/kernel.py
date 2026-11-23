"""Hugging Face kernel entrypoint of the NeuronSpark operator ``plif_prefill_pair``.

``run`` is :func:`neuronspark.ops.triton.plif.plif_prefill_pair`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.plif.plif_prefill_pair_ref` (same parameters).

Parameters: ``x0, w0, b0, zp0, th0, v0, x1, w1, b1, zp1, th1, v1, out_dtype``.
Inputs: two populations of one shape: x (T, B, C), w_eta, b_eta, v_th (C,) fp32, zp (T, B, C) bf16, v (B, C) fp32.
Returns: spikes (2, T, B, C) out_dtype.
Regimes: prefill.  Written in place: v0, v1.
"""
from __future__ import annotations

from .neuronspark.ops.triton.plif import plif_prefill_pair as run
from .neuronspark.ops.triton.plif import plif_prefill_pair_ref as ref

__all__ = ["run", "ref"]
