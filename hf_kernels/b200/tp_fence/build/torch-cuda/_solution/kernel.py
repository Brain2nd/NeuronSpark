"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_fence``.

``run`` is :func:`neuronspark.ops.triton.tp_bulk.tp_fence`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_bulk.tp_fence_ref` (same parameters).

Parameters: ``calls, rank, epoch``.
Inputs: calls N int64 (N,) rows, rank, epoch int64 (1,).
Returns: None: the epoch stored into word rank of every other GPU's call words, then a wait until every other word of
this GPU's holds at least it.
Regimes: prefill.  Written in place: calls.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_bulk import tp_fence as run
from .neuronspark.ops.triton.tp_bulk import tp_fence_ref as ref

__all__ = ["run", "ref"]
