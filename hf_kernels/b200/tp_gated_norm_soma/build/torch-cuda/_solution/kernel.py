"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_gated_norm_soma``.

``run`` is :func:`neuronspark.ops.triton.tp_dendritic.tp_gated_norm_soma`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_dendritic.tp_gated_norm_soma_ref` (same parameters).

Parameters: ``y, z, theta, gamma, eps, n_all, boxes, rank, ticket, rstd_out, n_out``.
Inputs: y (M, 4608) bf16, z (M, 4608) bf16 view with unit last stride, theta / gamma (4608,) fp32, eps 1e-5, n_all
9216, boxes N regions of N - 1 slots of M words, rank, ticket int32 (1,) zero (the programs take their rows from it;
left zero), rstd_out (M,) fp32 or None, n_out (M, 4608) bf16 or None.
Returns: x (M, 4608) bf16 contiguous soma read-out, normalized over n_all channels.
Regimes: prefill.  Written in place: boxes, ticket, rstd_out, n_out.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_dendritic import tp_gated_norm_soma as run
from .neuronspark.ops.triton.tp_dendritic import tp_gated_norm_soma_ref as ref

__all__ = ["run", "ref"]
