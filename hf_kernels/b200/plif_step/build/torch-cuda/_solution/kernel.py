"""Hugging Face kernel entrypoint of the NeuronSpark operator ``plif_step``.

``run`` is :func:`neuronspark.ops.triton.plif_step.plif_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.plif_step.plif_step_ref` (same parameters).

Parameters: ``x, eta_V, eta_U, w_eta, b_eta, v_th, v, out_dtype, run_sum, count``.
Inputs: x (B, C) fp32|bf16, eta_V (rank, C) / eta_U (C, rank) fp32 eta projection, w_eta / b_eta / v_th (C,) fp32, v
(B, C) fp32 carried membrane; the centred neuron: run_sum (B, C) fp32 carried input sum, count (B,) int64 tokens
before this one.
Returns: spikes S (B, C) out_dtype.
Regimes: decode.  Written in place: v, run_sum.
"""
from __future__ import annotations

from .neuronspark.ops.triton.plif_step import plif_step as run
from .neuronspark.ops.triton.plif_step import plif_step_ref as ref

__all__ = ["run", "ref"]
