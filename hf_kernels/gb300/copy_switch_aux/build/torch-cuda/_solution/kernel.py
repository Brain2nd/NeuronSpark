"""Hugging Face kernel entrypoint of the NeuronSpark operator ``copy_switch_aux``.

``run`` is :func:`neuronspark.ops.triton.copy_switch.copy_switch_aux`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.copy_switch.copy_switch_aux_ref` (same parameters).

Parameters: ``ids, targets, feats, weights, lam``.
Inputs: ids, targets (B, T) int64, feats 6 x (z, r, pi) detached, weights 6 x (w, v, b), lam = 0.5.
Returns: scalar fp32.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.copy_switch import copy_switch_aux as run
from .neuronspark.ops.triton.copy_switch import copy_switch_aux_ref as ref

__all__ = ["run", "ref"]
