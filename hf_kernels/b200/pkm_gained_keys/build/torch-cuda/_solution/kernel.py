"""Hugging Face kernel entrypoint of the NeuronSpark operator ``pkm_gained_keys``.

``run`` is :func:`neuronspark.ops.triton.pkm_query.pkm_gained_keys`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.pkm_query.pkm_gained_keys_ref` (same parameters).

Parameters: ``subkeys, log_gain``.
Inputs: subkeys (4, 2, 128, 64) fp32, log_gain (8,) fp32.
Returns: (4, 2, 128, 64) fp32 unit sub-keys times sqrt(64) exp(log_gain); backward: dsubkeys, dlog_gain.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.pkm_query import pkm_gained_keys as run
from .neuronspark.ops.triton.pkm_query import pkm_gained_keys_ref as ref

__all__ = ["run", "ref"]
