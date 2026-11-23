"""Hugging Face kernel entrypoint of the NeuronSpark operator ``lexical_train``.

``run`` is :func:`neuronspark.ops.triton.lex_train.lexical_train`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.lex_train.lexical_train_ref` (same parameters).

Parameters: ``ids, tables, orders, hash_mult, hash_add, hash_wrap, vocab_size, scale``.
Inputs: ids (B, T) int64, tables 5 x (rows, d) fp32, orders (1, 2, 2, 3, 3), hash_mult / hash_add / hash_wrap (4,)
int64, vocab_size 151669, scale 50.0.
Returns: (T, B, 160) fp32 lexical vectors of the whole row; backward: the tables (dense fp32, or added into a buffer
bound with bind_grad_).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.lex_train import lexical_train as run
from .neuronspark.ops.triton.lex_train import lexical_train_ref as ref

__all__ = ["run", "ref"]
