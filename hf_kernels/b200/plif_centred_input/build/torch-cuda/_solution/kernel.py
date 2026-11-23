"""Hugging Face kernel entrypoint of the NeuronSpark operator ``plif_centred_input``.

``run`` is :func:`neuronspark.ops.triton.plif.plif_centred_input`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.plif.plif_centred_input_ref` (same parameters).

Parameters: ``x, out_dtype``.
Inputs: x (T, B, C) fp32; out_dtype bf16 for the eta projection's operand.
Returns: out_dtype (default x.dtype) (T, B, C): x minus its running mean.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.plif import plif_centred_input as run
from .neuronspark.ops.triton.plif import plif_centred_input_ref as ref

__all__ = ["run", "ref"]
