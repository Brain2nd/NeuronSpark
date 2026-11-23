"""Hugging Face kernel entrypoint of the NeuronSpark operator ``shared_operand``.

``run`` is :func:`neuronspark.ops.triton.shared_operand.shared_operand`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.shared_operand.shared_operand_ref` (same parameters).

Parameters: ``x, uses``.
Inputs: x (T, B, 1536) fp32 (any strides), uses the number of projections (1 .. 8); x's gradient is the fp32 sum of
the uses' bf16 gradients.
Returns: uses x (T, B, 1536) bf16 row-major: the bf16 operand of the projections that read x.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.shared_operand import shared_operand as run
from .neuronspark.ops.triton.shared_operand import shared_operand_ref as ref

__all__ = ["run", "ref"]
