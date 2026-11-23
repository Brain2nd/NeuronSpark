"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_resid_norm``.

``run`` is :func:`neuronspark.ops.triton.dendritic_prenorm.dendritic_resid_norm`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.dendritic_prenorm.dendritic_resid_norm_ref` (same parameters).

Parameters: ``h, upd, gain_next, eps``.
Inputs: h (T, B, D) fp32, upd (T, B, D) bf16, gain_next (D,) fp32: h_new = h + upd.float(), hn_next = norm(h_new).
Returns: (h_new (T, B, D) fp32, hn_next (T, B, D) bf16).
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.dendritic_prenorm import dendritic_resid_norm as run
from .neuronspark.ops.triton.dendritic_prenorm import dendritic_resid_norm_ref as ref

__all__ = ["run", "ref"]
