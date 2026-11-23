"""Hugging Face kernel entrypoint of the NeuronSpark operator ``dendritic_chunk_scan_soma``.

``run`` is :func:`neuronspark.ops.dendritic_combined.dendritic_chunk_scan_soma`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.dendritic_combined.dendritic_chunk_scan_soma_ref` (same parameters).

Parameters: ``zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, *, alpha, chunk_size``.
Inputs: zxbcdt (B, T, 18832) bf16, conv_w (9472, 4) fp32, conv_b (9472,), dt_bias (144,), A (144,) fp32 = -exp(A_log),
D (144,), theta (9216,), gamma (9216,); alpha = 8.0, chunk_size = 256; T % 256 == 0.
Returns: xs (B*T, 9216) bf16; grads for zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.dendritic_combined import dendritic_chunk_scan_soma as run
from .neuronspark.ops.dendritic_combined import dendritic_chunk_scan_soma_ref as ref

__all__ = ["run", "ref"]
