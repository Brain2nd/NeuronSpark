"""Readout balance of the inference and scoring readout in one kernel.

The readout balance adds, per width ``w`` in order, the bf16 contribution ``c_w`` (R, w) to every vocabulary column
``v`` of the bf16 readout logits (R, V) as ``c_w[:, v % w]``, with one rounding per width::

    logits[r, v] = bf16(fp32(logits[r, v]) + fp32(c_w[r, v % w]))

The torch chain (:func:`balance_add_ref`) reads and writes the whole (R, V) tensor once per width.  Here one program
covers ``BALANCE_ADD_BLOCK`` columns of one row: it reads the logits once, applies the widths in order with the same
fp32 addition and bf16 rounding (round to nearest even) after each, and writes a new (R, V) tensor once.  The output
equals the chain's.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["BALANCE_ADD_BLOCK", "BALANCE_ADD_WARPS", "balance_add", "balance_add_ref"]

# ================================================================================================ fixed constants
#: columns per program, warps; grid (rows, column blocks)
BALANCE_ADD_BLOCK = 1024
BALANCE_ADD_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _balance_add_kernel(L, OUT, C0, C1, V, ld_l, W0: tl.constexpr, W1: tl.constexpr, NW: tl.constexpr,
                        BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    v = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = v < V
    x = tl.load(L + r * ld_l + v, mask=m, other=0.0).to(tl.float32)
    c = tl.load(C0 + r * W0 + (v % W0), mask=m, other=0.0).to(tl.float32)
    x = (x + c).to(tl.bfloat16)
    if NW > 1:
        c = tl.load(C1 + r * W1 + (v % W1), mask=m, other=0.0).to(tl.float32)
        x = (x.to(tl.float32) + c).to(tl.bfloat16)
    tl.store(OUT + r * V + v, x, mask=m)


# ================================================================================================ launcher
def _check(logits: torch.Tensor, contribs: Sequence[torch.Tensor], widths: Sequence[int]):
    require_cuda(logits, "balance_add (logits)")
    if logits.dim() != 2 or logits.dtype != torch.bfloat16 or logits.stride(1) != 1:
        raise RuntimeError(f"neuronspark.balance_add: logits must be bf16 (R, V) with unit column stride, got "
                           f"{logits.dtype} {tuple(logits.shape)} strides {tuple(logits.stride())}")
    R, V = (int(s) for s in logits.shape)
    if not 1 <= len(widths) <= 2 or len(contribs) != len(widths):
        raise RuntimeError(f"neuronspark.balance_add: 1 or 2 widths with one contribution each, got widths "
                           f"{tuple(widths)} and {len(contribs)} contributions")
    for c, w in zip(contribs, widths):
        require_cuda(c, "balance_add (contribution)")
        if tuple(c.shape) != (R, int(w)) or c.dtype != torch.bfloat16 or not c.is_contiguous() or not 0 < int(w) <= V:
            raise RuntimeError(f"neuronspark.balance_add: the contribution of width {w} must be a contiguous bf16 "
                               f"({R}, {w}) tensor, got {c.dtype} {tuple(c.shape)}")
    return R, V


def _balance_add_fwd(logits: torch.Tensor, contribs: List[torch.Tensor], widths: List[int]) -> torch.Tensor:
    R, V = _check(logits, contribs, widths)
    out = torch.empty((R, V), dtype=torch.bfloat16, device=logits.device)
    if R == 0:
        return out
    c1 = contribs[1] if len(contribs) > 1 else contribs[0]
    w1 = int(widths[1]) if len(widths) > 1 else 1
    _balance_add_kernel[(R, triton.cdiv(V, BALANCE_ADD_BLOCK))](
        logits, out, contribs[0], c1, V, int(logits.stride(0)), W0=int(widths[0]), W1=w1, NW=len(widths),
        BLOCK=BALANCE_ADD_BLOCK, num_warps=BALANCE_ADD_WARPS)
    return out


@torch.library.custom_op("neuronspark::balance_add", mutates_args=())
def _balance_add_op(logits: torch.Tensor, contribs: List[torch.Tensor], widths: List[int]) -> torch.Tensor:
    return _balance_add_fwd(logits, contribs, widths)


@_balance_add_op.register_fake
def _balance_add_fake(logits, contribs, widths):
    return logits.new_empty(tuple(logits.shape), dtype=torch.bfloat16)


def balance_add(logits: torch.Tensor, contribs: Sequence[torch.Tensor], widths: Sequence[int]) -> torch.Tensor:
    """The balanced logits, a new dense (R, V) bf16 tensor: ``logits`` (R, V) bf16 with unit column stride (any row
    stride), ``contribs`` one contiguous bf16 (R, w) contribution per width of ``widths`` (1 or 2 widths, applied in
    order).  No autograd (inference and scoring)."""
    return torch.ops.neuronspark.balance_add(logits, list(contribs), [int(w) for w in widths])


# ================================================================================================ reference
def balance_add_ref(logits: torch.Tensor, contribs: Sequence[torch.Tensor], widths: Sequence[int]) -> torch.Tensor:
    """The readout balance as torch ops: per width in order, ``bf16(fp32(logits) + fp32(c_w[:, v % w]))`` (the
    contribution tiled across the vocabulary)."""
    x = logits
    V = int(x.shape[-1])
    for w, c in zip(widths, contribs):
        n = -(-V // int(w))
        tiled = c.repeat(1, n)[:, :V]
        x = (x.float() + tiled.float()).to(torch.bfloat16)
    return x
