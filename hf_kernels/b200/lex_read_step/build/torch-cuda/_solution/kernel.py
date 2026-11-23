"""Hugging Face kernel entrypoint of the NeuronSpark operator ``lex_read_step``.

``run`` is :func:`neuronspark.ops.triton.lex_read_step.lex_read_step`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.lex_read_step.lex_read_step_ref` (same parameters).

Parameters: ``h, q_raw, k_raw, o, q_gain, k_gain, beta, eps, eps_z``.
Inputs: h (B, 1536) fp32, q_raw / k_raw (B, 64) bf16, o (B, 1536) bf16, q_gain / k_gain (64,) fp32, beta (1,) fp32,
eps = 1e-6, eps_z the frozen norm's eps.
Returns: (h' (B, 1536) fp32, h' bf16, z (B, 1536) fp32 frozen norm of h', z bf16).
Regimes: decode.
"""
from __future__ import annotations

from .neuronspark.ops.triton.lex_read_step import lex_read_step as run
from .neuronspark.ops.triton.lex_read_step import lex_read_step_ref as ref

__all__ = ["run", "ref"]
