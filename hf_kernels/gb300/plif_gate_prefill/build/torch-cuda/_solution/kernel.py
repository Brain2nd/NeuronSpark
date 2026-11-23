"""Hugging Face kernel entrypoint of the NeuronSpark operator ``plif_gate_prefill``.

``run`` is :func:`neuronspark.ops.triton.plif.plif_gate_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.plif.plif_gate_prefill_ref` (same parameters).

Parameters: ``x, w_eta, b_eta, zp``.
Inputs: x (T, B, C), w_eta, b_eta (C,) fp32, zp (T, B, C) bf16.
Returns: (a, drive) (T, B, C) fp32.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.plif import plif_gate_prefill as run
from .neuronspark.ops.triton.plif import plif_gate_prefill_ref as ref

__all__ = ["run", "ref"]
