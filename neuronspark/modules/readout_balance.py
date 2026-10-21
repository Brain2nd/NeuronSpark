"""Readout balance: zero-initialised low-width projections of the readout rows, tiled across the vocabulary.

``ReadoutBalance(D, widths=(16, 32))`` holds ``projs.{k}`` = a bias-free ``Linear(D -> w_k)`` per width (key
``projs.{k}.weight``, zero at launch, so the stack is exactly neutral at step 0).  Each contribution
``c_k = linear(rows, W_k)`` (R, w_k) bf16 is added to every vocabulary column ``v`` as ``c_k[:, v % w_k]``, width after
width in ``widths`` order, each addition rounded to bf16 once (fp32 add, RNE)::

    logits = bf16(logits + c_16[:, v % 16]);   logits = bf16(logits + c_32[:, v % 32])

Two separate roundings: folding the widths into one addition, or adding in fp32 and rounding once, is a different
function.

Regimes:

* inference / scoring (:meth:`ReadoutBalance.balance`): the balanced bf16 (R, V) logits from the readout GEMM's output,
  by the operator ``balance_add`` (one pass over the logits);
* training (:meth:`ReadoutBalance.ce_per_token`): the balanced logits are never materialised; the contributions and
  the labels go to the fused balance + cross-entropy operator ``balance_ce_per_token`` (forward and backward).

The default configuration uses the widths (16, 32).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from neuronspark import ops as _ops
from neuronspark.modules.linear import Linear

__all__ = ["ReadoutBalance", "BALANCE_IMPLS", "BALANCE_INITS"]

BALANCE_IMPLS = ("kernel", "ref")
BALANCE_INITS = ("launch", "empty")


class ReadoutBalance(nn.Module):
    """``ReadoutBalance(D, widths=(16, 32), device=None, dtype=torch.float32, impl="kernel", init="launch")``.

    ``init="launch"`` zeroes the projections (no random draw); ``"empty"`` leaves them uninitialised (a build followed
    by a checkpoint load).  ``impl`` picks the operator namespace (``"kernel"`` / ``"ref"``).
    """

    def __init__(self, D: int, widths: Sequence[int] = (16, 32), device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32, impl: str = "kernel", init: str = "launch") -> None:
        super().__init__()
        if impl not in BALANCE_IMPLS:
            raise ValueError(f"ReadoutBalance: impl must be one of {BALANCE_IMPLS}, got {impl!r}")
        if init not in BALANCE_INITS:
            raise ValueError(f"ReadoutBalance: init must be one of {BALANCE_INITS}, got {init!r}")
        if dtype != torch.float32:
            raise TypeError(f"ReadoutBalance: the projections are fp32 masters, got {dtype}")
        self.D = int(D)
        self.widths = tuple(int(w) for w in widths)
        if not self.widths or any(w <= 0 for w in self.widths):
            raise ValueError(f"ReadoutBalance: widths must be positive, got {widths}")
        self.impl = impl
        self.projs = nn.ModuleList([Linear(self.D, w, bias=False, device=device, dtype=torch.float32, init="empty")
                                    for w in self.widths])
        if init == "launch":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for p in self.projs:
            p.weight.zero_()

    @property
    def _o(self):
        return _ops.namespace(self.impl)

    def linear_weights(self):
        return [p.weight for p in self.projs]

    def extra_repr(self) -> str:
        return f"D={self.D}, widths={self.widths}, impl={self.impl!r}"

    def contributions(self, rows: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """``rows`` (R, D) -> one bf16 (R, w) contribution per width, in order."""
        if rows.dim() != 2 or int(rows.shape[1]) != self.D:
            raise ValueError(f"ReadoutBalance: expected (R, {self.D}) rows, got {tuple(rows.shape)}")
        o = self._o
        return tuple(o.linear(rows, p.weight) for p in self.projs)

    def balance(self, logits: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Inference / scoring: the balanced logits, a new dense bf16 (R, V) tensor, of the bf16 (R, V) ``logits``
        (unit column stride) of ``rows`` (R, D)."""
        if logits.dim() != 2 or logits.dtype != torch.bfloat16 or logits.stride(-1) != 1:
            raise RuntimeError(f"ReadoutBalance.balance: expected bf16 (R, V) logits with unit last stride, got "
                               f"{logits.dtype} {tuple(logits.shape)} strides {tuple(logits.stride())}")
        if int(rows.shape[0]) != int(logits.shape[0]):
            raise ValueError(f"ReadoutBalance.balance: {int(rows.shape[0])} rows for {int(logits.shape[0])} "
                             "logit rows")
        return self._o.balance_add(logits, self.contributions(rows), self.widths)

    def ce_per_token(self, logits: torch.Tensor, rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Training: per-token cross entropy (R,) fp32 of the balanced ``logits`` (R, V) bf16 (dense, row stride V)
        for ``labels`` (R,) int64 (-100 ignored), through the fused operator; differentiable in ``logits`` and the
        projections."""
        if int(rows.shape[0]) != int(logits.shape[0]) or labels.dim() != 1 or int(labels.shape[0]) != int(logits.shape[0]):
            raise ValueError(f"ReadoutBalance.ce_per_token: logits {tuple(logits.shape)}, rows {tuple(rows.shape)}, "
                             f"labels {tuple(labels.shape)}")
        return self._o.balance_ce_per_token(logits, labels, self.contributions(rows), self.widths)
