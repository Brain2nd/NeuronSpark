"""Hugging Face kernel entrypoint of the NeuronSpark operator ``balance_ce_per_token``.

``run`` is :func:`neuronspark.ops.triton.lm_tail.balance_ce_per_token`; ``ref`` is its pure-torch reference
:func:`neuronspark.ops.triton.lm_tail.balance_ce_per_token_ref` (same parameters).

Parameters: ``logits, labels, contribs, widths``.
Inputs: logits (R, V) bf16 dense, labels (R,) int64 (-100 ignored), contribs ((R, 16), (R, 32)) bf16, widths (16, 32);
a stack of three or four power-of-two widths up to 512: (R, w) bf16 per width.
Returns: (R,) fp32 per-token cross entropy of the balanced logits.
Regimes: train.
"""
from __future__ import annotations

from .neuronspark.ops.triton.lm_tail import balance_ce_per_token as run
from .neuronspark.ops.triton.lm_tail import balance_ce_per_token_ref as ref

__all__ = ["run", "ref"]
