"""Hugging Face kernel entrypoint of the NeuronSpark operator ``lex_read_train``.

``run`` is :func:`neuronspark.ops.triton.lex_read_train.lex_read_train`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.lex_read_train.lex_read_train_ref` (same parameters).

Parameters: ``h, q_raw, k_raw, o, q_gain, k_gain, beta, eps``.
Inputs: h (T, B, 1536) fp32, q_raw / k_raw (T, B, 64) bf16, o (T, B, 1536) bf16, q_gain / k_gain (64,) fp32, beta (1,)
fp32, eps = 1e-6.
Returns: (T, B, 1536) fp32 h + alpha * o in h's layout; backward: every tensor argument.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.lex_read_train import lex_read_train as run
from .neuronspark.ops.triton.lex_read_train import lex_read_train_ref as ref

__all__ = ["run", "ref"]
