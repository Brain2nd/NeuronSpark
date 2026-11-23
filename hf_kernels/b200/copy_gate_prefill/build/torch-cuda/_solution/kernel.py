"""Hugging Face kernel entrypoint of the NeuronSpark operator ``copy_gate_prefill``.

``run`` is :func:`neuronspark.ops.triton.copy_gate.copy_gate_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.copy_gate.copy_gate_prefill_ref` (same parameters).

Parameters: ``maximum, lse, zw, rv, tau, bias, strength, kappa``.
Inputs: maximum / lse (T, B) fp32 the copy detector's statistics, zw / rv (T, B) fp32 the novelty predictor's
products, tau / bias / strength one fp32 value each, kappa the read gain (4.0).
Returns: (f, rho): the read gain's factor and the PKM copy gate, (T, B) fp32 each.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.copy_gate import copy_gate_prefill as run
from .neuronspark.ops.triton.copy_gate import copy_gate_prefill_ref as ref

__all__ = ["run", "ref"]
