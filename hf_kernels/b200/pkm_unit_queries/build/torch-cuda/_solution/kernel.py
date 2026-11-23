"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_unit_queries``.

``run`` is :func:`neuronspark.ops.triton.pkm_query.pkm_unit_queries`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_query.pkm_unit_queries_ref` (same parameters).

Parameters: ``q, heads, half``.
Inputs: q (T, B, 512) bf16 query projection, heads = 4, half = 64.
Returns: (T, B, 4, 2, 64) fp32 unit half-queries; backward: dq bf16.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_query import pkm_unit_queries as run
from .neuronspark.ops.triton.pkm_query import pkm_unit_queries_ref as ref

__all__ = ["run", "ref"]
