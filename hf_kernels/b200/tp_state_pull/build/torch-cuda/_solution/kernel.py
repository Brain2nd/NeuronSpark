"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_state_pull``.

``run`` is :func:`neuronspark.ops.triton.tp_bulk.tp_state_pull`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_bulk.tp_state_pull_ref` (same parameters).

Parameters: ``dst, src_peer, flag, epoch``.
Inputs: dst / src_peer (18, 4, 589824) bf16 (the 18 dendritic layers' SSM states of 72 heads for 4 rows) or (18, 4,
13824) fp32 (conv taps of 4,608 channels), unit last stride, src_peer read through that GPU's mapping; flag / epoch
int64 (1,).
Returns: None: dst set to another GPU's block src_peer once this GPU's flag word of it holds the epoch.
Regimes: prefill.  Written in place: dst.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_bulk import tp_state_pull as run
from .neuronspark.ops.triton.tp_bulk import tp_state_pull_ref as ref

__all__ = ["run", "ref"]
