"""Hugging Face kernel entrypoint of the NeuronSpark operator ``plif_scan_train``.

``run`` is :func:`neuronspark.ops.triton.plif.plif_scan_train`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.plif.plif_scan_train_ref` (same parameters).

Parameters: ``w_eta, b_eta, zp, x, v_th, *, alpha, out_dtype``.
Inputs: w_eta, b_eta (C,) fp32, zp (T, B, C) bf16 eta projection, x (T, B, C) fp32|bf16, v_th (C,) fp32; alpha = 8.0,
T >= 2.
Returns: spikes S (T, B, C) out_dtype; grads for w_eta, b_eta, zp, x, v_th.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.plif import plif_scan_train as run
from .neuronspark.ops.triton.plif import plif_scan_train_ref as ref

__all__ = ["run", "ref"]
