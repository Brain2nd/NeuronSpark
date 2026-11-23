"""Hugging Face kernel entrypoint of the NeuronSpark operator ``tp_prefill_gather``.

``run`` is :func:`neuronspark.ops.triton.tp_bulk.tp_prefill_gather`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.tp_bulk.tp_prefill_gather_ref` (same parameters).

Parameters: ``x, parts, flag, rank, epoch, batch``.
Inputs: x (T, n, 1536) fp32 with unit last stride (this GPU's rows), parts N buffers (>= T n_j, 1536) fp32 contiguous,
GPU j's rows in (t, row) order, flag int64 (N,) this GPU's row, rank, epoch int64 (1,).
Returns: (T, batch, 1536) fp32 contiguous every row: this GPU's rows (row_bounds) and every other GPU's.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.tp_bulk import tp_prefill_gather as run
from .neuronspark.ops.triton.tp_bulk import tp_prefill_gather_ref as ref

__all__ = ["run", "ref"]
