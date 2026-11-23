"""Hugging Face kernel entrypoint of the NeuronSpark operator ``balance_add``.

``run`` is :func:`neuronspark.ops.triton.balance_add.balance_add`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.balance_add.balance_add_ref` (same parameters).

Parameters: ``logits, contribs, widths``.
Inputs: logits (R, 151669) bf16 readout (unit column stride), contribs ((R, 16), (R, 32)) bf16 contiguous, widths (16,
32).
Returns: (R, V) bf16 dense balanced logits.
Regimes: train, prefill, decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.balance_add import balance_add as run
from .neuronspark.ops.triton.balance_add import balance_add_ref as ref

__all__ = ["run", "ref"]
