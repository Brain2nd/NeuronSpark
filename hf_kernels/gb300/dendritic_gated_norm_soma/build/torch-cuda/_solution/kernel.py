"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_gated_norm_soma``.

``run`` is :func:`neuronspark.ops.triton.dendritic_gated_norm.dendritic_gated_norm_soma`; ``ref`` is its pure-torch
reference :func:`neuronspark.ops.triton.dendritic_gated_norm.dendritic_gated_norm_soma_ref` (same parameters).

Parameters: ``y, z, theta, gamma, eps``.
Inputs: y (M, 9216) bf16, z (M, 9216) bf16 (strided view of the projection), theta / gamma (9216,) fp32.
Returns: x (M, 9216) bf16 = bf16(S(n, theta) * gamma), n the bf16 gated RMSNorm (unit weight).
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.dendritic_gated_norm import dendritic_gated_norm_soma as run
from .neuronspark.ops.triton.dendritic_gated_norm import dendritic_gated_norm_soma_ref as ref

__all__ = ["run", "ref"]
