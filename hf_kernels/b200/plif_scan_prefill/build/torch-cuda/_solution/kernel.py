"""Hugging Face kernel entrypoint of the NeuronSpark operator ``plif_scan_prefill``.

``run`` is :func:`neuronspark.ops.triton.plif.plif_scan_prefill`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.plif.plif_scan_prefill_ref` (same parameters).

Parameters: ``a, drive, v_th, v0, out_dtype``.
Inputs: a, drive (T, B, C) fp32, v_th (C,) fp32, v0 (B, C) fp32 carried membrane.
Returns: spikes S (T, B, C) out_dtype.
Regimes: prefill.  Written in place: v0.
"""
from __future__ import annotations

from .neuronspark.ops.triton.plif import plif_scan_prefill as run
from .neuronspark.ops.triton.plif import plif_scan_prefill_ref as ref

__all__ = ["run", "ref"]
