"""Hugging Face kernel entrypoint of the NeuronSpark operator ``lexical_step``.

``run`` is :func:`neuronspark.ops.triton.lex_step.lexical_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.lex_step.lexical_step_ref` (same parameters).

Parameters: ``ids, win, tables, orders, hash_mult, hash_add, hash_wrap, vocab_size, scale``.
Inputs: ids (B,) int64, win (B, 2) int64 the rows' last ids, tables 5 x (rows, d) fp32, orders (1, 2, 2, 3, 3),
hash_mult / hash_add / hash_wrap (4,) int64, vocab_size 151669, scale 50.0.
Returns: (B, 160) fp32 lexical vector of one token.
Regimes: decode.  Written in place: win.
"""
from __future__ import annotations

from .neuronspark.ops.triton.lex_step import lexical_step as run
from .neuronspark.ops.triton.lex_step import lexical_step_ref as ref

__all__ = ["run", "ref"]
