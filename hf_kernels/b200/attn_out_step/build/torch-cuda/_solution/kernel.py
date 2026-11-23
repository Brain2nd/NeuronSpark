"""Hugging Face kernel entrypoint of the NeuronSpark operator ``attn_out_step``.

``run`` is :func:`neuronspark.ops.triton.attn_out_step.attn_out_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.attn_out_step.attn_out_step_ref` (same parameters).

Parameters: ``red, xq_add, gate, weight, eps``.
Inputs: red (B, 1536 + extra-query width) bf16 ring read, xq_add (B, 256) bf16, gate (B, 1536) bf16 logits, weight
(1536,) fp32 zero-centred norm weight, eps.
Returns: (B, 1536) bf16 gated output norm (the output projection's operand).
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.attn_out_step import attn_out_step as run
from .neuronspark.ops.triton.attn_out_step import attn_out_step_ref as ref

__all__ = ["run", "ref"]
