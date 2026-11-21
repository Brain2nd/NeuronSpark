"""Exchanges between the two GPUs of a tensor-parallel inference step, inside the captured step (no host round trip).

A value travels to the other GPU in a mailbox word: one 8-byte word holding the fp32 bits in its low half and a flag
(1) in its high half, written with one system-scope 64-bit store into the peer's mailbox (a symmetric-memory buffer
mapped on both GPUs: :class:`neuronspark.utils.tensor_parallel.TensorParallelComm`).  The receiver polls its own
mailbox with system-scope 64-bit loads until every word it needs carries the flag, then clears those words (after a
barrier of its program, so that no thread clears a word another thread has not seen yet).  Data and flag arrive in
one store: no fence, no counter, no epoch.  Every word has one writing thread: a block whose layout holds an element
in several threads (a program's scalar, a block smaller than the program's threads) would publish the word once per
copy, and a copy arriving after the receiver has cleared the word flags it again with this use's value, which the next
use then reads.  A program's scalar goes through :func:`mailbox_swap1` (one thread publishes, every thread gets the
peer's value).

Reuse of a word.  A mailbox region belongs to one call site of the step (``TensorParallelComm.region``); the same
site's next call is the next use.  The peer can publish the next use only after it has passed an exchange that this
GPU enters after it has finished (and cleared) the present use, because every call site of a step exchanges in the
same order on both GPUs.  So the clear never erases a published word and a published word is never overwritten
before it is read.  A step with a single call site has no such exchange between two uses of it: every step has at
least two (a split dendritic neuron layer has two), or an NCCL collective between its uses (the prefill).

Two properties are assumed, not proved here.  (i) An aligned 64-bit system-scope store reaches the peer's memory as
one access (data and flag together), as NCCL's LL protocol assumes of its 8-byte flag-and-data lines.  (ii) The
programs of an exchanging kernel are dispatched in order of their index on both GPUs, so the lowest unfinished program
index is resident on both and finishes: a program waits only for the peer's program of the same index, never for
another program of its own GPU.

Ops (inference, no autograd; ``send`` is the peer's region, ``recv`` this GPU's, int64 of at least the words named):

* ``tp_all_reduce(x, send, recv) -> y``: ``y = x + x_peer`` for fp32 ``x`` (M, N) contiguous (M * N words).
* ``tp_greedy_pick(value, index, offset, send, recv) -> tokens``: greedy selection across the two vocabulary shards
  of the readout: each GPU's per-row maximum ``value`` (B,) and its shard-local ``index`` (B,); the row's token is
  the global id (``index + offset``) of the larger maximum, the lower id on a tie, a nan larger than every number
  (two nan: the lower id) -- the first index of the whole row, as ``argmax``; (B,) int64 on both GPUs (2 B words).
* ``tp_join(partial, h, send, recv, gain=None, eps=1e-5) -> (h_new, hn)``: the end of a split dendritic neuron
  layer's decode step: ``upd = bf16(partial + partial_peer)`` (the out-projection's two fp32 partial products, summed
  in fp32 and rounded once, as the one-GPU product is), ``h_new = h + f32(upd)`` (fp32 residual stream, (B, D)), and
  with ``gain`` the next dendritic neuron layer's pre-norm ``hn = bf16(h_new * rstd * gain)``, ``rstd =
  1 / sqrt(mean(h_new^2) + eps)`` (else None).  One program per row (B * D words).

With two GPUs ``a + b`` is the same fp32 number on both (addition is commutative), so both GPUs hold the same sums.

Launch constants: ``JOIN_WARPS`` warps per row program, its block at least 32 x ``JOIN_WARPS`` columns;
``ALL_REDUCE_BLOCK`` values per program of ``tp_all_reduce``; the greedy pick's one warp holds a block of at least 32
rows (one word per thread).

References: :func:`tp_all_reduce_ref`, :func:`tp_join_ref` take the peer's values as an argument.  A one-GPU test
passes the same buffer as ``send`` and ``recv`` (the program then reads its own words: the peer's values are its
own).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["JOIN_WARPS", "ALL_REDUCE_BLOCK", "ALL_REDUCE_WARPS", "mailbox_swap", "mailbox_swap1", "tp_all_reduce", "tp_all_reduce_ref",
           "tp_join", "tp_join_ref", "tp_greedy_pick", "tp_greedy_pick_ref"]

JOIN_WARPS = 4
ALL_REDUCE_BLOCK = 1024
ALL_REDUCE_WARPS = 4


# ================================================================================================ mailbox
@triton.jit
def _publish(ptr, word, m):
    """system-scope 64-bit store of ``word`` at ``ptr`` where ``m`` (int32) is non-zero"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; @p st.relaxed.sys.global.b64 [$1], $2; mov.u32 $0, 0; }",
        "=r,l,l,r", [ptr, word, m], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _poll(ptr, m):
    """system-scope 64-bit load at ``ptr`` where ``m`` is non-zero; a flagged zero elsewhere"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b64 $0, 4294967296; @p ld.relaxed.sys.global.b64 $0, [$1]; }",
        "=l,l,r", [ptr, m], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def mailbox_swap(x, send_ptrs, recv_ptrs, m):
    """fp32 ``x`` (a block whose elements each live in one thread) to the peer's words ``send_ptrs``; returns the
    peer's values from this GPU's words ``recv_ptrs`` (cleared after use).  ``m``: the block's mask (bool)."""
    mi = m.to(tl.int32)
    bits = x.to(tl.int32, bitcast=True).to(tl.int64) & 4294967295
    _publish(send_ptrs, bits | 4294967296, mi)
    w = _poll(recv_ptrs, mi)
    while tl.min((w >> 32).to(tl.int32)) == 0:
        w = _poll(recv_ptrs, mi)
    tl.debug_barrier()                       # every thread has its words before any thread clears one
    _publish(recv_ptrs, tl.zeros_like(w), mi)
    return (w & 4294967295).to(tl.int32).to(tl.float32, bitcast=True)


@triton.jit
def mailbox_swap1(x, send_ptr, recv_ptr, LANES: tl.constexpr):
    """A program's fp32 scalar ``x`` to the peer's word ``send_ptr``, published by one thread; returns the peer's value
    from this GPU's word ``recv_ptr`` (cleared after use) in every thread.  ``LANES``: the program's threads (32 x
    num_warps), so that a block of LANES elements has one element per thread and its lane 0 one owner."""
    lane = tl.arange(0, LANES)
    first = lane == 0
    q = mailbox_swap(tl.where(first, x, 0.0), send_ptr + lane * 0, recv_ptr + lane * 0, first)
    return tl.sum(tl.where(first, q, 0.0), axis=0)


# ================================================================================================ kernels
@triton.jit
def _tp_all_reduce_kernel(X, Y, SEND, RECV, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(X + offs, mask=m, other=0.0)
    q = mailbox_swap(x, SEND + offs, RECV + offs, m)
    tl.store(Y + offs, x + q, mask=m)


@triton.jit
def _tp_join_kernel(P, H, G, HO, HN, SEND, RECV, D, eps, s_p, s_h, s_ho, s_hn,
                    BD: tl.constexpr, NORM: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BD)
    m = o < D
    p = tl.load(P + row * s_p + o, mask=m, other=0.0)
    q = mailbox_swap(p, SEND + row * D + o, RECV + row * D + o, m)
    upd = (p + q).to(tl.bfloat16).to(tl.float32)
    h = tl.load(H + row * s_h + o, mask=m, other=0.0) + upd
    tl.store(HO + row * s_ho + o, h, mask=m)
    if NORM:
        var = tl.sum(tl.where(m, h * h, 0.0), axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        g = tl.load(G + o, mask=m, other=0.0)
        tl.store(HN + row * s_hn + o, (h * rstd * g).to(tl.bfloat16), mask=m)


@triton.jit
def _tp_greedy_pick_kernel(VAL, IDX, OUT, SEND, RECV, offset, B, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < B
    v = tl.load(VAL + o, mask=m, other=0.0).to(tl.float32)
    i = (tl.load(IDX + o, mask=m, other=0) + offset).to(tl.int32)
    w = 2 * o[:, None] + tl.arange(0, 2)[None, :]
    mine = tl.join(v, i.to(tl.float32, bitcast=True))
    peer = mailbox_swap(mine, SEND + w, RECV + w, m[:, None] & (w >= 0))
    pv, pf = tl.split(peer)
    pi = pf.to(tl.int32, bitcast=True)
    vn = v != v                              # nan: larger than every number, two nan by the lower id
    pn = pv != pv
    take = tl.where(vn | pn, pn & ((v == v) | (pi < i)), (pv > v) | ((pv == v) & (pi < i)))
    tl.store(OUT + o, tl.where(take, pi, i).to(tl.int64), mask=m)


# ================================================================================================ launchers
def _check_region(send: torch.Tensor, recv: torch.Tensor, words: int, device) -> None:
    for name, t in (("send", send), ("recv", recv)):
        if (t.dtype != torch.int64 or t.dim() != 1 or int(t.numel()) < words or t.device != device
                or not t.is_contiguous()):
            raise RuntimeError(f"neuronspark.tp_exchange: {name} must be a contiguous int64 region of at least {words} "
                               f"words on {device}, got {t.dtype} {tuple(t.shape)} stride {tuple(t.stride())} on "
                               f"{t.device}")


def tp_all_reduce(x: torch.Tensor, send: torch.Tensor, recv: torch.Tensor) -> torch.Tensor:
    """``x + x_peer`` (fp32, contiguous; see the module docstring)."""
    require_cuda(x, "tp_all_reduce")
    if x.dtype != torch.float32 or not x.is_contiguous():
        raise RuntimeError(f"neuronspark.tp_all_reduce: x must be contiguous fp32, got {x.dtype} {tuple(x.stride())}")
    n = int(x.numel())
    _check_region(send, recv, n, x.device)
    y = torch.empty_like(x)
    if n:
        with torch.cuda.device(x.device.index):
            _tp_all_reduce_kernel[(triton.cdiv(n, ALL_REDUCE_BLOCK),)](x, y, send, recv, n, BLOCK=ALL_REDUCE_BLOCK,
                                                                       num_warps=ALL_REDUCE_WARPS)
    return y


def tp_join(partial: torch.Tensor, h: torch.Tensor, send: torch.Tensor, recv: torch.Tensor,
            gain: Optional[torch.Tensor] = None, eps: float = 1e-5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(h + f32(bf16(partial + partial_peer)), bf16 pre-norm or None)`` (see the module docstring)."""
    require_cuda(partial, "tp_join")
    if partial.dim() != 2 or partial.dtype != torch.float32 or partial.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_join: partial must be fp32 (B, D) with unit last stride, got "
                           f"{partial.dtype} {tuple(partial.shape)}")
    B, D = (int(s) for s in partial.shape)
    if tuple(h.shape) != (B, D) or h.dtype != torch.float32 or h.stride(-1) != 1 or h.device != partial.device:
        raise RuntimeError(f"neuronspark.tp_join: h must be fp32 ({B}, {D}) with unit last stride, got {h.dtype} "
                           f"{tuple(h.shape)}")
    if gain is not None and (tuple(gain.shape) != (D,) or gain.dtype != torch.float32 or not gain.is_contiguous()):
        raise RuntimeError(f"neuronspark.tp_join: gain must be contiguous fp32 ({D},)")
    _check_region(send, recv, B * D, partial.device)
    ho = torch.empty((B, D), dtype=torch.float32, device=h.device)
    hn = torch.empty((B, D), dtype=torch.bfloat16, device=h.device) if gain is not None else None
    with torch.cuda.device(h.device.index):
        _tp_join_kernel[(B,)](partial, h, gain if gain is not None else ho, ho, hn if hn is not None else ho,
                              send, recv, D, float(eps), partial.stride(0), h.stride(0), ho.stride(0),
                              hn.stride(0) if hn is not None else 0,
                              BD=max(triton.next_power_of_2(D), 32 * JOIN_WARPS),
                              NORM=gain is not None, num_warps=JOIN_WARPS)
    return ho, hn


def tp_greedy_pick(value: torch.Tensor, index: torch.Tensor, offset: int, send: torch.Tensor,
                   recv: torch.Tensor) -> torch.Tensor:
    """(B,) int64 greedy tokens across the two vocabulary shards (see the module docstring)."""
    require_cuda(value, "tp_greedy_pick")
    if value.dim() != 1 or tuple(index.shape) != tuple(value.shape) or index.dtype != torch.int64:
        raise RuntimeError(f"neuronspark.tp_greedy_pick: value (B,) and int64 index (B,), got {value.dtype} "
                           f"{tuple(value.shape)}, {index.dtype} {tuple(index.shape)}")
    B = int(value.shape[0])
    _check_region(send, recv, 2 * B, value.device)
    out = torch.empty((B,), dtype=torch.int64, device=value.device)
    with torch.cuda.device(value.device.index):
        _tp_greedy_pick_kernel[(1,)](value.contiguous(), index.contiguous(), out, send, recv, int(offset), B,
                                     BLOCK=max(32, triton.next_power_of_2(B)), num_warps=1)
    return out


# ================================================================================================ references
def tp_greedy_pick_ref(values, indices, offsets) -> torch.Tensor:
    """Pure torch: the global id of the larger per-row maximum over the shards (the lower id on a tie; a nan larger
    than every number)."""
    v = torch.stack([x.float() for x in values], 1)
    ids = torch.stack([i + int(o) for i, o in zip(indices, offsets)], 1)
    nan = v.isnan()
    best = torch.where(nan.any(dim=1, keepdim=True), nan, v == v.max(dim=1, keepdim=True).values)
    ids = torch.where(best, ids, torch.full_like(ids, torch.iinfo(torch.int64).max))
    return ids.min(dim=1).values


def tp_all_reduce_ref(x: torch.Tensor, x_peer: torch.Tensor) -> torch.Tensor:
    return x + x_peer


def tp_join_ref(partial: torch.Tensor, partial_peer: torch.Tensor, h: torch.Tensor,
                gain: Optional[torch.Tensor] = None, eps: float = 1e-5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure torch: ``upd = bf16(partial + partial_peer)``, ``h_new = h + f32(upd)``, the pre-norm in fp32 rounded to
    bf16 once."""
    h_new = h + (partial + partial_peer).to(torch.bfloat16).float()
    if gain is None:
        return h_new, None
    rstd = torch.rsqrt(h_new.pow(2).mean(dim=-1, keepdim=True) + eps)
    return h_new, (h_new * rstd * gain).to(torch.bfloat16)
