"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_topk_prefill``.

``run`` is :func:`neuronspark.ops.triton.pkm_topk.pkm_topk_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_topk.pkm_topk_prefill_ref` (same parameters).

Parameters: ``s1, s2, k``.
Inputs: s1, s2 (T, B, 4, 128) fp32 half-query scores, k = 16.
Returns: (sc, u, ii, jj).
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_topk import pkm_topk_prefill as run
from .neuronspark.ops.triton.pkm_topk import pkm_topk_prefill_ref as ref

__all__ = ["run", "ref"]
