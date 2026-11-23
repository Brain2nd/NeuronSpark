"""Exchanges between the GPUs of a tensor-parallel inference step, inside the captured step (no host round trip).

N GPUs (2 <= N <= ``MAX_GPUS``; ``NR`` in the kernels), this one of rank ``RANK``.  A value travels to another GPU in
a mailbox word: one 8-byte word holding the fp32 bits in its low half and a flag (1) in its high half, written with one
system-scope 64-bit store into the receiver's mailbox (a symmetric-memory buffer mapped on every GPU:
:class:`neuronspark.utils.tensor_parallel.TensorParallelComm`).  The receiver polls its own mailbox with system-scope
64-bit loads until every word it needs carries the flag, then clears those words (after a barrier of its program, so
that no thread clears a word another thread has not seen yet).  Data and flag arrive in one store: no fence, no
counter, no epoch.  Every word has one writing thread: a block whose layout holds an element in several threads (a
program's scalar, a block smaller than the program's threads) would publish the word once per copy, and a copy arriving
after the receiver has cleared the word flags it again with this use's value, which the next use then reads.  A
program's scalar goes through :func:`mailbox_sum1` (one thread publishes, every thread gets the sum).

Regions.  A call site of the step has a region on every GPU: ``boxes``, GPU i's region mapped on this GPU, this GPU's
own at index ``rank`` (the kernels take them as eight pointer arguments ``B0`` .. ``B7``, the unused ones this GPU's,
and select one by a compile-time index).  A site that sums the GPUs' values (the joins, the norms' sums of squares, the
greedy pick) holds N - 1 slots of the call's words on every GPU: GPU s writes into slot (s - o - 1) mod N of GPU o's
region, so with two GPUs a region is one slot, written by the other GPU.  A site that gathers rows (a hippocampal
layer's tail) holds one slot: a row's words are written by the GPU that holds the row into every other GPU's region,
at the row's place.

Reuse of a word.  A region belongs to one call site of the step (``TensorParallelComm.region``); the same site's next
call is the next use.  Every call site of a step exchanges among all the GPUs (each receives from every other), in the
same order on every GPU.  GPU s can publish the next use of a word of GPU o only after it has passed an exchange that
GPU o enters after it has finished (and cleared) the present use, since s receives from o there.  So the clear never
erases a published word and a published word is never overwritten before it is read.  A step with a single call site
has no such exchange between two uses of it: every decode step has at least two (a split dendritic neuron layer has
two), and a prefill call's mailbox sites have the joins and the call's starting fence of
:mod:`neuronspark.ops.triton.tp_bulk` between their uses.

Three properties are assumed, not proved here.  (i) An aligned 64-bit system-scope store reaches the receiver's memory
as one access (data and flag together), as NCCL's LL protocol assumes of its 8-byte flag-and-data lines.  (ii) A
program waits only for the other GPUs' programs of the same index, never for another program of its own GPU, so a wait
ends once the lowest unfinished index is resident on every GPU.  A decode exchange's grid fits on the GPU at the
model's shapes (at most a few hundred programs), so all its programs are resident when it runs alone on its stream;
beside other kernels the lowest index is resident if programs are dispatched in order of their index, which CUDA does
not specify.  The prefill's gated norm, a program per row, exceeds the GPU, and takes its rows by ticket instead
(:func:`neuronspark.ops.triton.tp_dendritic.tp_gated_norm_soma`): the rows its started programs hold are the lowest on
every GPU whatever the dispatch order.  (iii) A GPU's kernels run in the order of its stream, each after the one before
has finished (no programmatic dependent launch): the stores and polls are relaxed, and the reuse argument above orders
a word's clear (at the end of one kernel) before the GPU's next publication (a later kernel), and a GPU's observation
of a word before its later stores, through that order alone.

Sums.  Every GPU adds the N values in rank order, starting from rank 0's (not from zero: ``0 + (-0)`` is ``+0``): the
same fp32 number on every GPU.  With two GPUs that is ``a + b`` on both, the number each GPU formed as its own value
plus the other's (addition is commutative).

Ops (inference, no autograd; ``boxes`` the site's regions (int64, at least the words named), ``rank`` this GPU's):

* ``tp_greedy_pick(value, index, offset, boxes, rank) -> tokens``: greedy selection across the N vocabulary shards of
  the readout: each GPU's per-row maximum ``value`` (B,) and its shard-local ``index`` (B,); the row's token is the
  global id (``index + offset``) of the largest maximum, the lowest id on a tie, a nan larger than every number (several
  nan: the lowest id) -- the first index of the whole row, as ``argmax``; (B,) int64 on every GPU (N - 1 slots of 2 B
  words; a global id travels as int32, so ``index + offset`` must be below 2^31).
* ``tp_join(partial, h, boxes, rank, gain=None, eps=1e-5) -> (h_new, hn)``: the end of a split dendritic neuron
  layer's decode step: ``upd = bf16(sum of the N partials)`` (the out-projection's fp32 partial products, summed in
  fp32 in rank order and rounded once, as the one-GPU product is), ``h_new = h + f32(upd)`` (fp32 residual stream,
  (B, D)), and with ``gain`` the next dendritic neuron layer's pre-norm ``hn = bf16(h_new * rstd * gain)``, ``rstd =
  1 / sqrt(mean(h_new^2) + eps)`` (else None).  One program per row (N - 1 slots of B * D words).
* ``tp_tail_gather(h, ffn, y, rho, lo, batch, boxes, rank, gain=None, eps=1e-5) -> (h_out, hn)``: the end of a
  hippocampal layer's decode step whose batch is split between the GPUs: this GPU's rows [lo, lo + n) of the
  ``batch`` rows are the layer's last addition, formed here (``pkm_tail_step(h, ffn, y, rho)`` of
  :mod:`neuronspark.ops.triton.hippo_step`, the same arithmetic: ``h`` (n, D) fp32, ``ffn`` / ``y`` (n, D) bf16,
  ``rho`` (n,) fp32), the other GPUs form the others; ``h_out`` (batch, D) fp32 every row on every GPU, and with
  ``gain`` the next dendritic neuron layer's pre-norm of every row (``tp_join``'s).  One program per row: a row of
  this GPU's is formed and sent (its words at row * D of every other GPU's region), another GPU's row received (one
  slot of batch * D words; every word has one writer, the GPU that holds its row).

Launch constants: ``JOIN_WARPS`` warps per row program of ``tp_join``, its block at least 32 x ``JOIN_WARPS``
columns; ``tp_tail_gather`` the warps of the one-GPU tail (``HIPPO_STEP_WARPS``), its block at least 32 x as many
columns (at the model's widths, D >= 512, the tail kernel's own block: the same row sum, the same rows); the greedy
pick's one warp holds a block of at least 32 rows (one word per thread).

References: :func:`tp_join_ref`, :func:`tp_tail_gather_ref` and :func:`tp_greedy_pick_ref` take the kernels'
arguments and make the exchange in torch (:func:`publish_ref` the words this GPU sends, :func:`take_ref` the words it
receives, which must be flagged already, then cleared; :func:`sum_ref` a sum site's).  A one-GPU call either passes
the same buffer as every region of two (the program then reads its own words: the other GPU's values are its own), or
fills its own region with the other GPUs' flagged words before the call (any N).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ....neuronspark.ops.triton import require_cuda, require_no_grad
from ....neuronspark.ops.triton.hippo_step import HIPPO_STEP_WARPS, pkm_tail_row, pkm_tail_step_ref

__all__ = ["JOIN_WARPS", "MAX_GPUS", "mailbox_send", "mailbox_recv", "mailbox_publish", "mailbox_take", "mailbox_sum",
           "mailbox_sum1", "box_args", "slot", "publish_ref", "take_ref", "sum_ref", "tp_join", "tp_join_ref",
           "tp_tail_gather", "tp_tail_gather_ref", "tp_greedy_pick", "tp_greedy_pick_ref"]

JOIN_WARPS = 4
MAX_GPUS = 8


def slot(src: int, dst: int, n: int) -> int:
    """The slot of GPU ``dst``'s region of a sum site that GPU ``src`` writes, of ``n`` GPUs."""
    return (int(src) - int(dst) - 1) % int(n)


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
def _box(i: tl.constexpr, B0, B1, B2, B3, B4, B5, B6, B7):
    """GPU ``i``'s region of the site (``i`` < 8, a compile-time index)"""
    p = B0
    if i == 1:
        p = B1
    if i == 2:
        p = B2
    if i == 3:
        p = B3
    if i == 4:
        p = B4
    if i == 5:
        p = B5
    if i == 6:
        p = B6
    if i == 7:
        p = B7
    return p


@triton.jit
def mailbox_send(x, send_ptrs, m):
    """fp32 ``x`` (a block whose elements each live in one thread) to another GPU's words ``send_ptrs`` where ``m``
    (bool)."""
    bits = x.to(tl.int32, bitcast=True).to(tl.int64) & 4294967295
    _publish(send_ptrs, bits | 4294967296, m.to(tl.int32))


@triton.jit
def mailbox_recv(recv_ptrs, m):
    """The fp32 values another GPU published at this GPU's words ``recv_ptrs`` where ``m`` (bool), once every one is
    flagged; the words cleared after the program's barrier."""
    mi = m.to(tl.int32)
    w = _poll(recv_ptrs, mi)
    while tl.min((w >> 32).to(tl.int32)) == 0:
        w = _poll(recv_ptrs, mi)
    tl.debug_barrier()                       # every thread has its words before any thread clears one
    _publish(recv_ptrs, tl.zeros_like(w), mi)
    return (w & 4294967295).to(tl.int32).to(tl.float32, bitcast=True)


@triton.jit
def _slot_base(k: tl.constexpr, words):
    """the int64 word offset of slot ``k`` of a region whose slots are ``words`` long"""
    return tl.full([], k, tl.int64) * words


@triton.jit
def mailbox_publish(x, off, m, words, RANK: tl.constexpr, NR: tl.constexpr, B0, B1, B2, B3, B4, B5, B6, B7):
    """fp32 ``x`` (a block whose elements each live in one thread) into every other GPU's region of a sum site: the
    words ``off`` (a block of word offsets, ``m`` its mask) of slot (RANK - j - 1) mod NR of GPU j's region, the slots
    ``words`` long."""
    for j in tl.static_range(NR):
        if j != RANK:
            mailbox_send(x, _box(j, B0, B1, B2, B3, B4, B5, B6, B7) + _slot_base((RANK - j - 1 + 2 * NR) % NR, words)
                         + off, m)


@triton.jit
def mailbox_take(src: tl.constexpr, off, m, words, RANK: tl.constexpr, NR: tl.constexpr, B0, B1, B2, B3, B4, B5, B6,
                 B7):
    """GPU ``src``'s fp32 values from this GPU's slot (src - RANK - 1) mod NR of a sum site (the words ``off`` where
    ``m``), once flagged; the words cleared after use."""
    return mailbox_recv(_box(RANK, B0, B1, B2, B3, B4, B5, B6, B7) + _slot_base((src - RANK - 1 + 2 * NR) % NR, words)
                        + off, m)


@triton.jit
def mailbox_sum(x, off, m, words, RANK: tl.constexpr, NR: tl.constexpr, B0, B1, B2, B3, B4, B5, B6, B7):
    """The NR GPUs' fp32 ``x`` summed in rank order from rank 0's value, the same number on every GPU: this GPU's
    published first (:func:`mailbox_publish`), every other GPU's taken from this GPU's slots."""
    mailbox_publish(x, off, m, words, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7)
    if RANK == 0:
        s = x
    else:
        s = mailbox_take(0, off, m, words, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7)
    for src in tl.static_range(1, NR):
        if src == RANK:
            s = s + x
        else:
            s = s + mailbox_take(src, off, m, words, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7)
    return s


@triton.jit
def mailbox_sum1(x, off, words, LANES: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr, B0, B1, B2, B3, B4, B5, B6,
                 B7):
    """A program's fp32 scalar ``x`` summed over the GPUs in rank order (:func:`mailbox_sum` at word ``off`` of the
    slots, published by one thread); the sum in every thread.  ``LANES``: the program's threads (32 x num_warps), so
    that a block of LANES elements has one element per thread and its lane 0 one owner."""
    lane = tl.arange(0, LANES)
    first = lane == 0
    s = mailbox_sum(tl.where(first, x, 0.0), off + lane * 0, first, words, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7)
    return tl.sum(tl.where(first, s, 0.0), axis=0)


# ================================================================================================ kernels
@triton.jit
def _tp_join_kernel(P, H, G, HO, HN, B0, B1, B2, B3, B4, B5, B6, B7, D, words, eps, s_p, s_h, s_ho, s_hn,
                    BD: tl.constexpr, NORM: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BD)
    m = o < D
    p = tl.load(P + row * s_p + o, mask=m, other=0.0)
    tot = mailbox_sum(p, row * D + o, m, words, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7)
    upd = tot.to(tl.bfloat16).to(tl.float32)
    h = tl.load(H + row * s_h + o, mask=m, other=0.0) + upd
    tl.store(HO + row * s_ho + o, h, mask=m)
    if NORM:
        var = tl.sum(tl.where(m, h * h, 0.0), axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        g = tl.load(G + o, mask=m, other=0.0)
        tl.store(HN + row * s_hn + o, (h * rstd * g).to(tl.bfloat16), mask=m)


@triton.jit
def _tp_tail_gather_kernel(H, F, Y, RHO, G, HO, HN, B0, B1, B2, B3, B4, B5, B6, B7, D, lo, hi, eps, s_h, s_f, s_y,
                           s_ho, s_hn, BD: tl.constexpr, NORM: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BD)
    m = o < D
    if (row >= lo) & (row < hi):                                     # this GPU's row: formed, sent to every other GPU
        h = pkm_tail_row(H, F, Y, RHO, (row - lo).to(tl.int64), s_h, s_f, s_y, D, o, m)
        for j in tl.static_range(NR):
            if j != RANK:
                mailbox_send(h, _box(j, B0, B1, B2, B3, B4, B5, B6, B7) + row * D + o, m)
    else:                                                            # another GPU's row: received
        h = mailbox_recv(_box(RANK, B0, B1, B2, B3, B4, B5, B6, B7) + row * D + o, m)
    tl.store(HO + row * s_ho + o, h, mask=m)
    if NORM:
        var = tl.sum(tl.where(m, h * h, 0.0), axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        g = tl.load(G + o, mask=m, other=0.0)
        tl.store(HN + row * s_hn + o, (h * rstd * g).to(tl.bfloat16), mask=m)


@triton.jit
def _better(bv, bi, cv, ci):
    """whether the candidate (cv, ci) beats (bv, bi): the larger value, a nan larger than every number, the lower id on
    a tie (two nan: the lower id)"""
    bn = bv != bv
    cn = cv != cv
    return tl.where(bn | cn, cn & ((bv == bv) | (ci < bi)), (cv > bv) | ((cv == bv) & (ci < bi)))


@triton.jit
def _tp_greedy_pick_kernel(VAL, IDX, OUT, B0, B1, B2, B3, B4, B5, B6, B7, offset, B, BLOCK: tl.constexpr,
                           RANK: tl.constexpr, NR: tl.constexpr):
    o = tl.arange(0, BLOCK)
    m = o < B
    v = tl.load(VAL + o, mask=m, other=0.0).to(tl.float32)
    i = (tl.load(IDX + o, mask=m, other=0) + offset).to(tl.int32)
    w = 2 * o[:, None] + tl.arange(0, 2)[None, :]
    mw = m[:, None] & (w >= 0)
    mailbox_publish(tl.join(v, i.to(tl.float32, bitcast=True)), w, mw, 2 * B, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7)
    if RANK == 0:                                                    # the candidates folded in rank order
        bv = v
        bi = i
    else:
        bv, bf = tl.split(mailbox_take(0, w, mw, 2 * B, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7))
        bi = bf.to(tl.int32, bitcast=True)
    for src in tl.static_range(1, NR):
        if src == RANK:
            cv = v
            ci = i
        else:
            cv, cf = tl.split(mailbox_take(src, w, mw, 2 * B, RANK, NR, B0, B1, B2, B3, B4, B5, B6, B7))
            ci = cf.to(tl.int32, bitcast=True)
        take = _better(bv, bi, cv, ci)
        bv = tl.where(take, cv, bv)
        bi = tl.where(take, ci, bi)
    tl.store(OUT + o, bi.to(tl.int64), mask=m)


# ================================================================================================ launchers
def box_args(boxes: Sequence[torch.Tensor], rank: int, words: int, sum_site: bool, device,
             what: str) -> Tuple[List[torch.Tensor], int]:
    """(the eight region arguments of a kernel: GPU i's region at i, this GPU's in the unused places; N) after checking
    ``boxes``: 2 to MAX_GPUS contiguous int64 regions on ``device``, each of at least ``words`` per slot (N - 1 slots
    for a sum site, one otherwise), and ``rank`` among them."""
    n = len(boxes)
    if not 2 <= n <= MAX_GPUS or not 0 <= int(rank) < n:
        raise RuntimeError(f"neuronspark.{what}: 2 to {MAX_GPUS} regions (one per GPU) and this GPU's rank among them, "
                           f"got {n} regions and rank {rank}")
    need = int(words) * ((n - 1) if sum_site else 1)
    for k, t in enumerate(boxes):
        if (not isinstance(t, torch.Tensor) or t.dtype != torch.int64 or t.dim() != 1 or int(t.numel()) < need
                or t.device != device or not t.is_contiguous()):
            raise RuntimeError(f"neuronspark.{what}: region {k} must be a contiguous int64 region of at least {need} "
                               f"words on {device}, got {getattr(t, 'dtype', type(t))} {tuple(getattr(t, 'shape', ()))}"
                               f" on {getattr(t, 'device', None)}")
    return list(boxes) + [boxes[int(rank)]] * (MAX_GPUS - n), n


def tp_join(partial: torch.Tensor, h: torch.Tensor, boxes: Sequence[torch.Tensor], rank: int,
            gain: Optional[torch.Tensor] = None, eps: float = 1e-5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(h + f32(bf16(sum of the GPUs' partials)), bf16 pre-norm or None)`` (see the module docstring)."""
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
    bx, n = box_args(boxes, rank, B * D, True, partial.device, "tp_join")
    require_no_grad("tp_join", partial, h, gain)
    ho = torch.empty((B, D), dtype=torch.float32, device=h.device)
    hn = torch.empty((B, D), dtype=torch.bfloat16, device=h.device) if gain is not None else None
    with torch.cuda.device(h.device.index):
        _tp_join_kernel[(B,)](partial, h, gain if gain is not None else ho, ho, hn if hn is not None else ho,
                              *bx, D, B * D, float(eps), partial.stride(0), h.stride(0), ho.stride(0),
                              hn.stride(0) if hn is not None else 0,
                              BD=max(triton.next_power_of_2(D), 32 * JOIN_WARPS),
                              NORM=gain is not None, RANK=int(rank), NR=n, num_warps=JOIN_WARPS)
    return ho, hn


def tp_tail_gather(h: torch.Tensor, ffn: torch.Tensor, y: torch.Tensor, rho: torch.Tensor, lo: int, batch: int,
                   boxes: Sequence[torch.Tensor], rank: int, gain: Optional[torch.Tensor] = None,
                   eps: float = 1e-5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(every row (batch, D) fp32, bf16 pre-norm of every row or None)``: this GPU's rows [lo, lo + n) the tail
    ``pkm_tail_step(h, ffn, y, rho)`` of its n rows, the others the other GPUs' (see the module docstring)."""
    require_cuda(h, "tp_tail_gather")
    if h.dim() != 2 or h.dtype != torch.float32 or h.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_tail_gather: h must be fp32 (n, D) with unit last stride, got {h.dtype} "
                           f"{tuple(h.shape)} stride {tuple(h.stride())}")
    n, D = (int(s) for s in h.shape)
    for name, t in (("ffn", ffn), ("y", y)):
        if tuple(t.shape) != (n, D) or t.dtype != torch.bfloat16 or t.stride(-1) != 1 or t.device != h.device:
            raise RuntimeError(f"neuronspark.tp_tail_gather: {name} must be bf16 ({n}, {D}) with unit last stride on "
                               f"{h.device}, got {t.dtype} {tuple(t.shape)} stride {tuple(t.stride())}")
    if tuple(rho.shape) != (n,) or rho.dtype != torch.float32 or not rho.is_contiguous() or rho.device != h.device:
        raise RuntimeError(f"neuronspark.tp_tail_gather: rho must be a contiguous fp32 ({n},) tensor on {h.device}, "
                           f"got {rho.dtype} {tuple(rho.shape)}")
    B, lo = int(batch), int(lo)
    if lo < 0 or lo + n > B or not 1 <= n < B:
        raise RuntimeError(f"neuronspark.tp_tail_gather: this GPU's rows [{lo}, {lo + n}) must lie in the batch of "
                           f"{B} rows, the other GPUs holding the rest")
    if gain is not None and (tuple(gain.shape) != (D,) or gain.dtype != torch.float32 or not gain.is_contiguous()
                             or gain.device != h.device):
        raise RuntimeError(f"neuronspark.tp_tail_gather: gain must be contiguous fp32 ({D},) on {h.device}")
    bx, nr = box_args(boxes, rank, B * D, False, h.device, "tp_tail_gather")
    require_no_grad("tp_tail_gather", h, ffn, y, rho, gain)
    ho = torch.empty((B, D), dtype=torch.float32, device=h.device)
    hn = torch.empty((B, D), dtype=torch.bfloat16, device=h.device) if gain is not None else None
    with torch.cuda.device(h.device.index):
        _tp_tail_gather_kernel[(B,)](h, ffn, y, rho, gain if gain is not None else ho, ho,
                                     hn if hn is not None else ho, *bx, D, lo, lo + n, float(eps), h.stride(0),
                                     ffn.stride(0), y.stride(0), ho.stride(0), hn.stride(0) if hn is not None else 0,
                                     BD=max(triton.next_power_of_2(D), 32 * HIPPO_STEP_WARPS), NORM=gain is not None,
                                     RANK=int(rank), NR=nr, num_warps=HIPPO_STEP_WARPS, enable_fp_fusion=False)
    return ho, hn


def tp_greedy_pick(value: torch.Tensor, index: torch.Tensor, offset: int, boxes: Sequence[torch.Tensor],
                   rank: int) -> torch.Tensor:
    """(B,) int64 greedy tokens across the GPUs' vocabulary shards (see the module docstring)."""
    require_cuda(value, "tp_greedy_pick")
    if value.dim() != 1 or tuple(index.shape) != tuple(value.shape) or index.dtype != torch.int64:
        raise RuntimeError(f"neuronspark.tp_greedy_pick: value (B,) and int64 index (B,), got {value.dtype} "
                           f"{tuple(value.shape)}, {index.dtype} {tuple(index.shape)}")
    B = int(value.shape[0])
    if not 0 <= int(offset) < 2 ** 31:
        raise RuntimeError(f"neuronspark.tp_greedy_pick: the global ids travel as int32 (index + offset < 2^31), got "
                           f"offset {offset}")
    bx, n = box_args(boxes, rank, 2 * B, True, value.device, "tp_greedy_pick")
    out = torch.empty((B,), dtype=torch.int64, device=value.device)
    with torch.cuda.device(value.device.index):
        _tp_greedy_pick_kernel[(1,)](value.contiguous(), index.contiguous(), out, *bx, int(offset), B,
                                     BLOCK=max(32, triton.next_power_of_2(B)), RANK=int(rank), NR=n, num_warps=1)
    return out


# ================================================================================================ references
def publish_ref(region: torch.Tensor, start: int, values: torch.Tensor) -> None:
    """Pure torch: the flagged words of ``values`` (fp32, or int32 bit patterns) stored at ``region[start:]`` in their
    row-major order, as the programs publish them."""
    bits = values.contiguous().view(torch.int32).reshape(-1).to(torch.int64) & 0xFFFFFFFF
    region[start:start + bits.numel()] = bits | (1 << 32)


def take_ref(region: torch.Tensor, start: int, n: int, what: str) -> torch.Tensor:
    """Pure torch: the int32 bit patterns of the ``n`` words at ``region[start:]``, then those words cleared, as the
    programs receive them; every word must be flagged (the kernel waits until it is: a reference cannot)."""
    w = region[start:start + n]
    if not bool(((w >> 32) != 0).all()):
        raise RuntimeError(f"neuronspark.{what}: a word of this GPU's region is not flagged (the kernel would wait for "
                           f"another GPU to publish it)")
    bits = (w & 0xFFFFFFFF).to(torch.int32)
    region[start:start + n] = 0
    return bits


def _check_boxes_ref(boxes: Sequence[torch.Tensor], rank: int, what: str) -> int:
    n = len(boxes)
    if not 2 <= n <= MAX_GPUS or not 0 <= int(rank) < n:
        raise RuntimeError(f"neuronspark.{what}: 2 to {MAX_GPUS} regions (one per GPU) and this GPU's rank among them, "
                           f"got {n} regions and rank {rank}")
    return n


def sum_ref(x: torch.Tensor, boxes: Sequence[torch.Tensor], rank: int, what: str) -> torch.Tensor:
    """Pure torch, the exchange of a sum site: the words of this GPU's ``x`` (fp32, row-major) published into slot
    (rank - j - 1) mod N of every other GPU j's region, every other GPU's values taken from this GPU's slots; the N
    values summed in rank order from rank 0's (fp32)."""
    n = _check_boxes_ref(boxes, rank, what)
    words = int(x.numel())
    for j in range(n):
        if j != int(rank):
            publish_ref(boxes[j], slot(rank, j, n) * words, x)
    s = None
    for src in range(n):
        v = x if src == int(rank) else \
            take_ref(boxes[int(rank)], slot(src, rank, n) * words, words, what).view(torch.float32).view(x.shape)
        s = v.clone() if s is None else s + v
    return s


def tp_greedy_pick_ref(value: torch.Tensor, index: torch.Tensor, offset: int, boxes: Sequence[torch.Tensor],
                       rank: int) -> torch.Tensor:
    """Pure torch, the exchange included: this GPU's maximum and global id of row b published at words 2 b and 2 b + 1
    of its slot of every other GPU's region, the other GPUs' taken from this GPU's slots; the global id of the largest
    maximum, the lowest id on a tie, a nan larger than every number (several nan: the lowest id)."""
    n = _check_boxes_ref(boxes, rank, "tp_greedy_pick_ref")
    B = int(value.shape[0])
    v = value.float().contiguous()
    i = (index + int(offset)).to(torch.int32)
    mine = torch.stack([v.view(torch.int32), i], 1)
    for j in range(n):
        if j != int(rank):
            publish_ref(boxes[j], slot(rank, j, n) * 2 * B, mine)
    cands = []
    for src in range(n):
        if src == int(rank):
            cands.append((v, i))
        else:
            got = take_ref(boxes[int(rank)], slot(src, rank, n) * 2 * B, 2 * B, "tp_greedy_pick_ref").view(B, 2)
            cands.append((got[:, 0].contiguous().view(torch.float32), got[:, 1].contiguous()))
    vals = torch.stack([c[0] for c in cands], 1)
    ids = torch.stack([c[1] for c in cands], 1).to(torch.int64)
    nan = vals.isnan()
    best = torch.where(nan.any(dim=1, keepdim=True), nan, vals == vals.max(dim=1, keepdim=True).values)
    ids = torch.where(best, ids, torch.full_like(ids, torch.iinfo(torch.int64).max))
    return ids.min(dim=1).values


def tp_join_ref(partial: torch.Tensor, h: torch.Tensor, boxes: Sequence[torch.Tensor], rank: int,
                gain: Optional[torch.Tensor] = None, eps: float = 1e-5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure torch, the exchange included (:func:`sum_ref`, word row * D + column of the slots): ``upd = bf16(sum of the
    N partials)``, ``h_new = h + f32(upd)``, the pre-norm in fp32 rounded to bf16 once."""
    tot = sum_ref(partial, boxes, rank, "tp_join_ref")
    h_new = h + tot.to(torch.bfloat16).float()
    return h_new, _prenorm_ref(h_new, gain, eps)


def tp_tail_gather_ref(h: torch.Tensor, ffn: torch.Tensor, y: torch.Tensor, rho: torch.Tensor, lo: int, batch: int,
                       boxes: Sequence[torch.Tensor], rank: int, gain: Optional[torch.Tensor] = None,
                       eps: float = 1e-5) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure torch, the exchange included: rows [lo, lo + n) the tail of this GPU's rows
    (:func:`neuronspark.ops.triton.hippo_step.pkm_tail_step_ref`), published into every other GPU's region (word
    row * D + column); the other rows taken from this GPU's region; the pre-norm in fp32 rounded to bf16 once."""
    nr = _check_boxes_ref(boxes, rank, "tp_tail_gather_ref")
    x = pkm_tail_step_ref(h, ffn, y, rho).float()
    n, D = (int(s) for s in x.shape)
    B, lo = int(batch), int(lo)
    out = torch.empty((B, D), dtype=torch.float32, device=x.device)
    out[lo:lo + n] = x
    for j in range(nr):
        if j != int(rank):
            publish_ref(boxes[j], lo * D, x)
    for a, b in ((0, lo), (lo + n, B)):
        if b > a:
            out[a:b] = take_ref(boxes[int(rank)], a * D, (b - a) * D,
                                "tp_tail_gather_ref").view(torch.float32).view(b - a, D)
    return out, _prenorm_ref(out, gain, eps)


def _prenorm_ref(h: torch.Tensor, gain: Optional[torch.Tensor], eps: float) -> Optional[torch.Tensor]:
    if gain is None:
        return None
    rstd = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (h * rstd * gain).to(torch.bfloat16)
