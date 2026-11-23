"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_topk_softmax_t1``.

``run`` is :func:`neuronspark.ops.triton.pkm_topk.pkm_topk_softmax_t1`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_topk.pkm_topk_softmax_t1_ref` (same parameters).

Parameters: ``s_raw, beta, k``.
Inputs: s_raw (B, 4, 2, 128) fp32, beta (4, 2) fp32 score gains, k = 16.
Returns: (sc, w, u, ii, jj).
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_topk import pkm_topk_softmax_t1 as run
from .neuronspark.ops.triton.pkm_topk import pkm_topk_softmax_t1_ref as ref

__all__ = ["run", "ref"]
