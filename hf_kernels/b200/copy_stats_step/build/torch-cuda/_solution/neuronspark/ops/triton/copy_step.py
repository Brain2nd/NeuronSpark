"""Decode-step copy-detector statistics (T = 1): the token's unit operand written into the copy ring and its scores
against the ring reduced, in two kernels.

The step's operand is the order-1 slice ``z`` (B, d) of the token's lexical vector
(:mod:`neuronspark.modules.copy_detector`).  Per row b, with ``p0`` the ring pointer and ``n_b`` the row's tokens
before this one::

    u            = z / max(|z|, 1e-12)                          written at ring[b, p0 % W]
    pos(w)       = p0 - ((p0 - w) mod W)                        the row position held by slot w (< 0: never written)
    valid[w]     = p0 - n_b <= pos(w) <= p0 - 2                 the strict past of the row (events at s <= t - 2)
    s[w]         = <u, ring[b, w]>                              fp32
    maximum[b]   = max_w s[w]                 over valid w      (-inf without a valid slot)
    lse[b]       = logsumexp_w 32 s[w]        over valid w      (-inf without a valid slot)

The first kernel runs one program per (row, tile of ``COPY_STEP_SLOTS`` ring slots): the operand, the tile's scores
and the tile's partial statistics (its max m and the sum of exp(32 (s - m))); the program whose tile holds the step's
slot writes the operand there (the slot's own score is outside the strict past, so what a program loads from that
slot is never used).
The second kernel merges a row's partials.  These are the statistics of the prefill operator
(:func:`neuronspark.ops.triton.copy_stats.copy_stats_prefill`) for a call of one token; the sums run in another order.
Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the torch chain
(``copy_stats_step_ref``).
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....neuronspark.ops.triton import require_cuda
from ....neuronspark.ops.triton.copy_stats import COPY_LAG, COPY_LOGIT_SCALE, copy_stats_prefill_ref

__all__ = ["COPY_STEP_SLOTS", "COPY_STEP_WARPS", "COPY_STEP_REDUCE_WARPS", "copy_stats_step", "copy_stats_step_ref"]

# ================================================================================================ fixed constants
#: ring slots per program of the first kernel, its warps; warps of the merge (one program per row)
COPY_STEP_SLOTS = 64
COPY_STEP_WARPS = 16
COPY_STEP_REDUCE_WARPS = 1


# ================================================================================================ kernels
@triton.jit
def _copy_step_part_kernel(Z, RING, PART, P0, NPOS, s_z, s_rb, s_rw, W, D, NCH,
                           BW: tl.constexpr, DB: tl.constexpr, SCALE: tl.constexpr, LAG: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    p0 = tl.load(P0)                                                # int64: the ring pointer (>= 0)
    slot = p0 % W
    base = RING + b * s_rb
    dd = tl.arange(0, DB)
    dm = dd < D
    z = tl.load(Z + b * s_z + dd, mask=dm, other=0.0)
    u = z / tl.maximum(libdevice.sqrt(tl.sum(z * z, axis=0)), 1e-12)        # the unit operand of the token
    ww = ch * BW + tl.arange(0, BW)
    wm = ww < W
    k = tl.load(base + ww[:, None].to(tl.int64) * s_rw + dd[None, :], mask=wm[:, None] & dm[None, :], other=0.0)
    s = tl.sum(k * u[None, :], axis=1)                              # (BW,) the tile's scores
    r = (p0 - ww) % W
    r = tl.where(r < 0, r + W, r)                                   # the non-negative remainder
    pos = p0 - r                                                    # the position held by slot w
    start = p0 - tl.load(NPOS + b)                                  # the row's first event
    valid = wm & (pos >= start) & (pos <= p0 - LAG)
    s = tl.where(valid, s, float("-inf"))
    m = tl.max(s, axis=0)
    acc = tl.sum(tl.where(valid, tl.exp(SCALE * (s - m)), 0.0), axis=0)
    out = PART + (b * NCH + ch) * 2
    tl.store(out, m)
    tl.store(out + 1, acc)
    # the token's operand into its slot: by the program whose tile holds the slot
    tl.store(base + slot * s_rw + dd, u, mask=dm & (slot // BW == ch))


@triton.jit
def _copy_step_merge_kernel(PART, MAXO, LSEO, NCH, CB: tl.constexpr, SCALE: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    cc = tl.arange(0, CB)
    cm = cc < NCH
    mc = tl.load(PART + (b * NCH + cc) * 2, mask=cm, other=float("-inf"))
    ac = tl.load(PART + (b * NCH + cc) * 2 + 1, mask=cm, other=0.0)
    m = tl.max(mc, axis=0)
    live = mc > float("-inf")
    acc = tl.sum(tl.where(live, ac * tl.exp(SCALE * (mc - m)), 0.0), axis=0)
    tl.store(MAXO + b, m)
    tl.store(LSEO + b, tl.where(m > float("-inf"), SCALE * m + tl.log(acc), float("-inf")))


# ================================================================================================ launcher
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.copy_stats_step: " + msg)


def _copy_stats_step_fwd(z: torch.Tensor, ring: torch.Tensor, ring_ptr: torch.Tensor,
                         pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    require_cuda(z, "copy_stats_step")
    dev = z.device
    _need(ring.dim() == 3 and ring.dtype == torch.float32 and ring.stride(2) == 1 and ring.device == dev,
          f"the ring must be fp32 (B, W, d) with unit inner stride, got {ring.dtype} {tuple(ring.shape)}")
    B, W, D = (int(s) for s in ring.shape)
    _need(tuple(z.shape) == (B, D) and z.dtype == torch.float32 and z.stride(1) == 1,
          f"the operand must be fp32 ({B}, {D}) with unit inner stride, got {z.dtype} {tuple(z.shape)}")
    _need(ring_ptr.numel() == 1 and ring_ptr.dtype == torch.int64 and ring_ptr.device == dev,
          "ring_ptr must be a (1,) int64 tensor")
    _need(tuple(pos.shape) == (B,) and pos.dtype == torch.int64 and pos.is_contiguous() and pos.device == dev,
          f"pos must be a contiguous ({B},) int64 tensor")
    nch = triton.cdiv(W, COPY_STEP_SLOTS)
    part = torch.empty((B, nch, 2), dtype=torch.float32, device=dev)
    maximum = torch.empty((1, B), dtype=torch.float32, device=dev)
    lse = torch.empty((1, B), dtype=torch.float32, device=dev)
    _copy_step_part_kernel[(B, nch)](z, ring, part, ring_ptr, pos, int(z.stride(0)), int(ring.stride(0)),
                                     int(ring.stride(1)), W, D, nch, BW=COPY_STEP_SLOTS,
                                     DB=triton.next_power_of_2(D), SCALE=COPY_LOGIT_SCALE, LAG=COPY_LAG,
                                     num_warps=COPY_STEP_WARPS, enable_fp_fusion=False)
    _copy_step_merge_kernel[(B,)](part, maximum, lse, nch, CB=max(2, triton.next_power_of_2(nch)),
                                  SCALE=COPY_LOGIT_SCALE, num_warps=COPY_STEP_REDUCE_WARPS, enable_fp_fusion=False)
    return maximum, lse


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::copy_stats_step", mutates_args=("ring",))
def _copy_stats_step_op(z: torch.Tensor, ring: torch.Tensor, ring_ptr: torch.Tensor,
                        pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _copy_stats_step_fwd(z, ring, ring_ptr, pos)


@_copy_stats_step_op.register_fake
def _copy_stats_step_fake(z, ring, ring_ptr, pos):
    B = int(ring.shape[0])
    return (ring.new_empty((1, B), dtype=torch.float32), ring.new_empty((1, B), dtype=torch.float32))


# ================================================================================================ public function
def copy_stats_step(z: torch.Tensor, ring: torch.Tensor, ring_ptr: torch.Tensor,
                    pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(maximum, lse)``, each (1, B) fp32, of one token per row against the copy ring (module docstring).

    ``z`` (B, d) fp32 with unit inner stride: the order-1 slice of the token's lexical vector; ``ring`` (B, W, d) fp32
    with unit inner stride: the unit operands of the rows (the token's own written at ``ring_ptr % W``); ``ring_ptr``
    (1,) int64, a count (>= 0); ``pos`` (B,) int64 contiguous: the tokens each row consumed before this one.  ``z``
    shares no memory with the ring and the ring's rows share none with one another.  A row without a valid slot has
    ``maximum = lse = -inf``."""
    return _KERNEL_OPS.copy_stats_step(z, ring, ring_ptr, pos)


# ================================================================================================ pure-torch reference
@torch.no_grad()
def copy_stats_step_ref(z: torch.Tensor, ring: torch.Tensor, ring_ptr: torch.Tensor,
                        pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The torch chain of the session detector for one token: the normalisation, the ring write, and the prefill
    operator's reference for a call of one token.  Same arguments, state update and outputs as
    :func:`copy_stats_step`."""
    W = int(ring.shape[1])
    with torch.autocast(z.device.type, enabled=False):
        u = F.normalize(z.float(), dim=-1)
        ring.index_copy_(1, ring_ptr.reshape(1) % W, u.unsqueeze(1))
    return copy_stats_prefill_ref(ring, ring_ptr, pos, 1)
