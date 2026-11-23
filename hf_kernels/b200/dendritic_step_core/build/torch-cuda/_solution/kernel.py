"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_step_core``.

``run`` is :func:`neuronspark.ops.triton.dendritic_step.dendritic_step_core`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.dendritic_step.dendritic_step_core_ref` (same parameters).

Parameters: ``proj, conv_state, ssm_state, conv_w, conv_b, A, dt_bias, D, theta, gamma, eps``.
Inputs: proj (B, 18832) bf16, conv_state (B, 9472, 3) fp32, ssm_state (B, 144, 64, 128) bf16|fp32 (stored in its own
dtype), conv_w (9472, 4), conv_b, A (144,), dt_bias, D, theta (9216,), gamma (9216,).
Returns: xs (B, 9216) bf16.
Regimes: decode.  Written in place: conv_state, ssm_state.
"""
from __future__ import annotations

from .neuronspark.ops.triton.dendritic_step import dendritic_step_core as run
from .neuronspark.ops.triton.dendritic_step import dendritic_step_core_ref as ref

__all__ = ["run", "ref"]
