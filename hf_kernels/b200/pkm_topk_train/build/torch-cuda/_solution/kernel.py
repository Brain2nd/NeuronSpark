"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_topk_train``.

``run`` is :func:`neuronspark.ops.triton.pkm_topk.pkm_topk_train`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_topk.pkm_topk_train_ref` (same parameters).

Parameters: ``s, k``.
Inputs: s (T, B, 4, 2, 128) fp32 half-query scores, k = 16.
Returns: (sc fp32, u, ii, jj int64) each (T, B, 4, 16); backward -> ds.
Regimes: train, prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_topk import pkm_topk_train as run
from .neuronspark.ops.triton.pkm_topk import pkm_topk_train_ref as ref

__all__ = ["run", "ref"]
