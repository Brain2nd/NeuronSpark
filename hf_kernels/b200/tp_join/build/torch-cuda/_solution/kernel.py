"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_join``.

``run`` is :func:`neuronspark.ops.triton.tp_exchange.tp_join`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_exchange.tp_join_ref` (same parameters).

Parameters: ``partial, h, boxes, rank, gain, eps``.
Inputs: partial (B, 1536) fp32 out-projection partial, h (B, 1536) fp32 residual stream, boxes N regions of N - 1
slots of B x 1536 words, rank, gain (1536,) fp32 the next dendritic layer's pre-norm gain or None, eps 1e-5.
Returns: (h_new (B, 1536) fp32 = h + f32(bf16(the N partials summed in rank order)), hn (B, 1536) bf16 pre-norm of
h_new or None).
Regimes: decode.  Written in place: boxes.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_exchange import tp_join as run
from .neuronspark.ops.triton.tp_exchange import tp_join_ref as ref

__all__ = ["run", "ref"]
