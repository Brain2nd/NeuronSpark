"""Prefill induction match of the model-level keys ring (T >= 2): one kernel over (row, query tile, slot tile).

Every ring slot ``w`` holds the ``n`` ids that preceded the event written there (``keys[b, w, i] = x_{s-1-i}``); the
call's keys are already written (slot ``(ptr + tau) % W`` for query ``tau``).  Query ``tau`` of the call (row position
``t = ptr + tau``) carries the window of its last ids ``qwin[b, tau, i] = x_{t-i}``.  Per row, query and slot::

    end1     = ptr + T - 1
    pos(w)   = end1 - ((end1 - w) mod W)                              the row position held by slot w (< 0: never)
    m_k[w]   = [0 <= pos(w) <= t - 1] and for all i < k: keys[w, i] == qwin[tau, i] and qwin[tau, i] != pad

for k = 1 .. n, stored as uint8 (n, B, T, W): entry k - 1 is the order-k match.  Integer comparisons only: the matches
are those of the torch chain (:func:`induction_match_prefill_ref`, the expression of
:func:`neuronspark.modules.induction.induction_session_match`).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["INDUCTION_PREFILL_BT", "INDUCTION_PREFILL_BW", "INDUCTION_PREFILL_WARPS", "induction_match_prefill",
           "induction_match_prefill_ref"]

# ================================================================================================ fixed constants
#: queries and ring slots per program, warps
INDUCTION_PREFILL_BT = 32
INDUCTION_PREFILL_BW = 128
INDUCTION_PREFILL_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _induction_match_prefill_kernel(KEYS, QWIN, PTR, OUT, B, T, W, pad, N: tl.constexpr, BT: tl.constexpr,
                                    BW: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    t0 = tl.program_id(1) * BT
    w0 = tl.program_id(2) * BW
    tl_ = tl.arange(0, BT)
    wl = tl.arange(0, BW)
    tt = t0 + tl_
    ww = w0 + wl
    tm = tt < T
    wm = ww < W
    ptr = tl.load(PTR)
    end1 = ptr + (T - 1)
    r = (end1 - ww) % W
    r = tl.where(r < 0, r + W, r)                                          # the non-negative remainder
    pos = end1 - r
    rel = (pos - ptr).to(tl.int32)                                         # in [T - W, T - 1]: the call's own frame
    acc = (pos >= 0)[None, :] & (rel[None, :] <= tt[:, None] - 1)          # an event of the strict past
    m2 = tm[:, None] & wm[None, :]
    q_row = QWIN + (b * T + t0) * N                                        # int64 bases, int32 offsets in the tile
    k_row = KEYS + (b * W + w0) * N
    tile = tl_[:, None] * W + wl[None, :]
    for i in tl.static_range(N):
        q = tl.load(q_row + tl_ * N + i, mask=tm, other=0)
        key = tl.load(k_row + wl * N + i, mask=wm, other=0)
        acc = acc & (key[None, :] == q[:, None]) & (q != pad)[:, None]
        o_row = OUT + ((i * B + b) * T + t0) * W + w0
        tl.store(o_row + tile, acc.to(tl.uint8), mask=m2)


# ================================================================================================ launcher, op
def _match_prefill(keys: torch.Tensor, qwin: torch.Tensor, ring_ptr: torch.Tensor, pad: int) -> torch.Tensor:
    require_cuda(keys, "induction_match_prefill")
    if keys.dim() != 3 or keys.dtype != torch.int64 or not keys.is_contiguous():
        raise RuntimeError(f"induction_match_prefill: keys must be a contiguous int64 (B, W, n) ring, got "
                           f"{keys.dtype} {tuple(keys.shape)}")
    B, W, n = (int(s) for s in keys.shape)
    if qwin.dim() != 3 or qwin.dtype != torch.int64 or int(qwin.shape[0]) != B or int(qwin.shape[2]) != n \
            or qwin.device != keys.device:
        raise RuntimeError(f"induction_match_prefill: qwin must be int64 ({B}, T, {n}) on {keys.device}, got "
                           f"{qwin.dtype} {tuple(qwin.shape)}")
    if tuple(ring_ptr.shape) != (1,) or ring_ptr.dtype != torch.int64 or ring_ptr.device != keys.device:
        raise RuntimeError("induction_match_prefill: ring_ptr must be a (1,) int64 tensor on the keys' device")
    T = int(qwin.shape[1])
    qc = qwin if qwin.is_contiguous() else qwin.contiguous()
    out = torch.empty((n, B, T, W), dtype=torch.uint8, device=keys.device)
    if B * T * W:
        grid = (B, triton.cdiv(T, INDUCTION_PREFILL_BT), triton.cdiv(W, INDUCTION_PREFILL_BW))
        _induction_match_prefill_kernel[grid](keys, qc, ring_ptr, out, B, T, W, int(pad), N=n,
                                              BT=INDUCTION_PREFILL_BT, BW=INDUCTION_PREFILL_BW,
                                              num_warps=INDUCTION_PREFILL_WARPS)
    return out


@torch.library.custom_op("neuronspark::induction_match_prefill", mutates_args=())
def _induction_match_prefill_op(keys: torch.Tensor, qwin: torch.Tensor, ring_ptr: torch.Tensor,
                                pad: int) -> torch.Tensor:
    return _match_prefill(keys, qwin, ring_ptr, pad)


@_induction_match_prefill_op.register_fake
def _induction_match_prefill_fake(keys, qwin, ring_ptr, pad):
    return keys.new_empty((keys.shape[2], keys.shape[0], qwin.shape[1], keys.shape[1]), dtype=torch.uint8)


def induction_match_prefill(keys: torch.Tensor, qwin: torch.Tensor, ring_ptr: torch.Tensor, pad: int) -> torch.Tensor:
    """(n, B, T, W) uint8 matches of the call's queries against every slot of the keys ring (entry k - 1: order k):
    ``keys`` (B, W, n) int64 ring with this call's keys written, ``qwin`` (B, T, n) int64 query windows
    (``qwin[b, tau, i] = x_{ptr + tau - i}``), ``ring_ptr`` (1,) int64 the ring pointer at the start of the call,
    ``pad`` the pad id (a window entry equal to it matches nothing)."""
    return torch.ops.neuronspark.induction_match_prefill(keys, qwin, ring_ptr, int(pad))


# ================================================================================================ reference
def induction_match_prefill_ref(keys: torch.Tensor, qwin: torch.Tensor, ring_ptr: torch.Tensor,
                                pad: int) -> torch.Tensor:
    """Pure-torch reference of :func:`induction_match_prefill`: the torch chain of the session match."""
    B, W, n = (int(s) for s in keys.shape)
    T = int(qwin.shape[1])
    end1 = ring_ptr + (T - 1)
    w = torch.arange(W, device=keys.device, dtype=torch.int64)
    pos = (end1 - torch.remainder(end1 - w, W)).view(1, W)
    t_abs = ring_ptr + torch.arange(T, device=keys.device, dtype=torch.int64)
    past = (pos >= 0) & (pos <= t_abs.view(T, 1) - 1)                      # (T, W)
    eq = (keys.unsqueeze(1) == qwin.unsqueeze(2)) & (qwin != int(pad)).unsqueeze(2)          # (B, T, W, n)
    acc = past.unsqueeze(0).expand(B, T, W)
    out = []
    for k in range(n):
        acc = acc & eq[..., k]
        out.append(acc)
    return torch.stack(out, dim=0).to(torch.uint8)
