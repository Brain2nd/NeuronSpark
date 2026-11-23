"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_norm_gate``.

``run`` is :func:`neuronspark.ops.triton.pkm_gate.pkm_norm_gate`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_gate.pkm_norm_gate_ref` (same parameters).

Parameters: ``val, gain, glog, eps``.
Inputs: val (T, B, 768) fp32 value read, gain (768,) fp32 (zero-centred), glog (T, B, 768) bf16 gate logits.
Returns: (T, B, 768) bf16 = bf16(zero-centred RMSNorm(val) * sigmoid(glog)); backward: dval, dgain, dglog.
Regimes: train, prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_gate import pkm_norm_gate as run
from .neuronspark.ops.triton.pkm_gate import pkm_norm_gate_ref as ref

__all__ = ["run", "ref"]
