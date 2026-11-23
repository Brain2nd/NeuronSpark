"""Hugging Face kernel entrypoint of the NeuronSpark operator ``rmsnorm_gate_rows``.

``run`` is :func:`neuronspark.ops.triton.hippo_norms.rmsnorm_gate_rows`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.hippo_norms.rmsnorm_gate_rows_ref` (same parameters).

Parameters: ``x, eps, weight, gate``.
Inputs: x (..., 1536) fp32, weight (1536,) fp32 (zero-centred), gate (..., 1536) bf16 output-gate logits.
Returns: (..., N) bf16 = bf16(sigmoid(gate) * zero-centred RMSNorm(x)).
Regimes: prefill, decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.hippo_norms import rmsnorm_gate_rows as run
from .neuronspark.ops.triton.hippo_norms import rmsnorm_gate_rows_ref as ref

__all__ = ["run", "ref"]
