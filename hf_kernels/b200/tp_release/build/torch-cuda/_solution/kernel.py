"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_release``.

``run`` is :func:`neuronspark.ops.triton.tp_bulk.tp_release`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_bulk.tp_release_ref` (same parameters).

Parameters: ``flags, rank, epoch``.
Inputs: flags N int64 (N,) rows, rank, epoch int64 (1,).
Returns: None: the epoch stored into word rank of every other GPU's flag row of the site.
Regimes: prefill.  Written in place: flags.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_bulk import tp_release as run
from .neuronspark.ops.triton.tp_bulk import tp_release_ref as ref

__all__ = ["run", "ref"]
