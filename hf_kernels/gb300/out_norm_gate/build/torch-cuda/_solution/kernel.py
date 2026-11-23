"""Hugging Face kernel entrypoint of the NeuronSpark operator ``out_norm_gate``.

``run`` is :func:`neuronspark.ops.triton.hippo_norms.out_norm_gate`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_norms.out_norm_gate_ref` (same parameters).

Parameters: ``attn, w, gate_logits, eps``.
Inputs: attn (T, B, 1536) fp32, w (1536,) fp32 (zero-centred), gate_logits (T, B, 1536) bf16.
Returns: (T, B, 1536) fp32 = zero-centred RMSNorm(attn) * sigmoid(gate_logits).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_norms import out_norm_gate as run
from .neuronspark.ops.triton.hippo_norms import out_norm_gate_ref as ref

__all__ = ["run", "ref"]
