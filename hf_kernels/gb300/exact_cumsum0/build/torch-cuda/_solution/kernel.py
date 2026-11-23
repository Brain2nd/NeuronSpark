"""Hugging Face kernel entrypoint of the NeuronSpark operator ``exact_cumsum0``.

``run`` is :func:`neuronspark.ops.triton.cumsum.exact_cumsum0`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.cumsum.exact_cumsum0_ref` (same parameters).

Parameters: ``x``.
Inputs: x (T, N) fp32: sequential RN additions.
Returns: (T, N) fp32.
Regimes: prefill.
"""
from __future__ import annotations

from .neuronspark.ops.triton.cumsum import exact_cumsum0 as run
from .neuronspark.ops.triton.cumsum import exact_cumsum0_ref as ref

__all__ = ["run", "ref"]
