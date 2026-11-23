"""Hugging Face kernel entrypoint of the NeuronSpark operator ``cumsum0``.

``run`` is :func:`neuronspark.ops.triton.cumsum.cumsum0`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.cumsum.cumsum0_ref` (same parameters).

Parameters: ``x``.
Inputs: x (T, ...) fp32; the narrow / wide dispatch is internal and fixed.
Returns: inclusive cumsum over dim 0, fp32.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.cumsum import cumsum0 as run
from .neuronspark.ops.triton.cumsum import cumsum0_ref as ref

__all__ = ["run", "ref"]
