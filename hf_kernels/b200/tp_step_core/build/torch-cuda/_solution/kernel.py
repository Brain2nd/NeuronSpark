"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_step_core``.

``run`` is :func:`neuronspark.ops.triton.tp_dendritic.tp_step_core`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_dendritic.tp_step_core_ref` (same parameters).

Parameters: ``proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps, n_all, boxes, rank``.
Inputs: proj (B, 9544) bf16 local projection [z | x | B | C | dt], conv_state (B, 4864, 3) fp32, ssm_state (B, 72, 64,
128) bf16|fp32, conv_w (4864, 4) / conv_b (4864,) / A / dt_bias / D (72,) / theta / gamma (4608,) fp32, eps 1e-5,
n_all 9216, boxes N regions of N - 1 slots of k2_words(B, 4608) words (one per row and K2 program), rank.
Returns: xs (B, 4608) bf16: the soma read-out of this GPU's heads, normalized over n_all channels.
Regimes: decode.  Written in place: conv_state, ssm_state, boxes.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_dendritic import tp_step_core as run
from .neuronspark.ops.triton.tp_dendritic import tp_step_core_ref as ref

__all__ = ["run", "ref"]
