"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_tail_gather``.

``run`` is :func:`neuronspark.ops.triton.tp_exchange.tp_tail_gather`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_exchange.tp_tail_gather_ref` (same parameters).

Parameters: ``h, ffn, y, rho, lo, batch, boxes, rank, gain, eps``.
Inputs: h (n, 1536) fp32, ffn / y (n, 1536) bf16, rho (n,) fp32 (this GPU's rows [lo, lo + n) of the batch), boxes N
regions of batch x 1536 words, rank, gain (1536,) fp32 or None, eps 1e-5.
Returns: (h_out (batch, 1536) fp32 every row: this GPU's the tail of its rows, the others' rows theirs; hn (batch,
1536) bf16 or None).
Regimes: decode.  Written in place: boxes.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_exchange import tp_tail_gather as run
from .neuronspark.ops.triton.tp_exchange import tp_tail_gather_ref as ref

__all__ = ["run", "ref"]
