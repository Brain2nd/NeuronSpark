"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_greedy_pick``.

``run`` is :func:`neuronspark.ops.triton.tp_exchange.tp_greedy_pick`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_exchange.tp_greedy_pick_ref` (same parameters).

Parameters: ``value, index, offset, boxes, rank``.
Inputs: value (B,) bf16 this shard's per-row maximum, index (B,) int64 its shard-local id, offset the shard's first
vocabulary row (0 or 75808 at N = 2), boxes N regions of N - 1 slots of 2 B words, rank.
Returns: (B,) int64 greedy tokens across the N vocabulary shards.
Regimes: prefill, decode.  Written in place: boxes.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_exchange import tp_greedy_pick as run
from .neuronspark.ops.triton.tp_exchange import tp_greedy_pick_ref as ref

__all__ = ["run", "ref"]
