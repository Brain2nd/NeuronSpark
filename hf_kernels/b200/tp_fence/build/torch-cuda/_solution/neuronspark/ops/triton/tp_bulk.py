"""Bulk exchanges between the GPUs of a tensor-parallel prefill: a whole buffer in symmetric memory, read by the other
GPUs once a flag word says that it is complete.

N GPUs (2 <= N <= ``MAX_GPUS``; ``NR`` in the kernels), this one of rank ``RANK``.  A GPU writes its buffer (the
out-projection's fp32 partial product, or its rows of a stream whose batch is split between the GPUs) into its symmetric
memory; then a kernel of one thread stores the call's epoch into its flag word for the site on every other GPU
(:func:`tp_release`: ``fence.proxy.alias``, then ``st.release.sys``, cumulative over the writes of the kernels before it
in the stream).  A site's flags on a GPU are N words, word s written by GPU s (``flags`` below: GPU j's row of the
site, mapped on this GPU, this GPU's own at index ``rank``).  The reading kernel's programs each wait until every word
of their own row but their own holds at least the epoch (a thread per word polls with ``ld.acquire.sys``, the program's
barrier follows) and then read the other GPUs' buffers through their mappings (volatile loads).  The GPUs take a site's
buffers of the same size (they grow together).

Aliases.  A buffer is written through its owner's mapping and read through another GPU's: two virtual addresses of one
allocation, which PTX orders only through an alias proxy fence on the path from the write to the read (and from a read
to the next write of the same buffer).  Every such path passes a release (the writer's for the site, or the reader's for
a later site or the next call's fence), and every release executes one.  A flag word is likewise stored through one
mapping and polled through another; that a system-scope release and acquire synchronize across the mappings is
assumed, as PyTorch's symmetric-memory signal pads assume it.

Progress.  Every word a program waits for is stored by another GPU from a kernel that stores before it waits for
anything (the release waits for nothing; the fence stores, then waits).  Another GPU's release of site s follows its
own wait of site s - 1, which this GPU's release of site s - 1 (with every other GPU's) ends, and that release precedes
this GPU's wait of site s; the fences of a call follow the waits of the call before, which end the same way.  So every
wait ends, whatever order the programs of any GPU are dispatched in, and a waiting grid may be larger than the GPU.

Reuse.  The epoch is a device counter advanced once at the start of every prefill call (inside a captured call as
well), equal on every GPU, which run the same calls, each GPU's one after another (on one stream, or on streams ordered
by events: the epoch, the flags and the buffers are shared by all of them); a site's flag words are written once per
call and compared with >=, so they are never cleared.  The buffers alternate by site parity (a join's and a gather's
sites alike): within a call a GPU writes a buffer again two sites later, after it has passed the wait of the site
between, which every other GPU releases only after its read of the earlier site has finished.  Across calls the fence
that starts every prefill call orders them (:func:`tp_fence`): a GPU stores the new epoch into its call word on every
other GPU and waits until each of its own holds it, which the other GPUs store only after their previous calls have
finished, their reads of this GPU's buffers among them; so a call may have any number of sites, and an eager call may
follow a captured one.

Rows.  A batch of B >= N rows split between the GPUs gives GPU r the rows [lo_r, lo_(r+1)), lo_r = ceil(r B / N)
(:func:`row_bounds`; at least one row each).

Ops (inference, no autograd; ``rank`` this GPU's, ``flags`` / ``calls`` the site's flag rows / the call words, one
int64 (N,) row per GPU):

* ``tp_fence(calls, rank, epoch)``: the start of a prefill call: the epoch (int64 (1,)) stored into word ``rank`` of
  every other GPU's call words, then a wait until every other word of this GPU's holds at least the epoch; one program.
* ``tp_release(flags, rank, epoch)``: word ``rank`` of every other GPU's flag row of the site set to the epoch.
* ``tp_prefill_join(parts, h, flag, rank, epoch) -> h_new``: the end of a split dendritic neuron layer's prefill: ``upd
  = bf16(parts[0] + ... + parts[N - 1])`` (the GPUs' out-projection fp32 partial products, (B T, D) in (b, t) row order,
  summed in fp32 in rank order and rounded once, as the one-GPU product is) and ``h_new = h + f32(upd)`` for the fp32
  residual stream ``h`` (T, B, D): a new (T, B, D) fp32 tensor, the same on every GPU.  ``parts[rank]`` is this GPU's
  buffer, the others are read through their mappings; the programs wait on ``flag`` (this GPU's row of the site).
* ``tp_prefill_gather(x, parts, flag, rank, epoch, batch) -> h``: the end of a prefill call whose batch is split between
  the GPUs: ``x`` (T, n, D) fp32 holds this GPU's rows (:func:`row_bounds`) of the ``batch`` rows, GPU j's rows (T, n_j,
  D) are the first T n_j rows of its buffer ``parts[j]`` in (t, row) order; ``h`` a new (T, batch, D) fp32 tensor of
  every row, the same on every GPU.  The programs wait on ``flag``.
* ``tp_state_pull(dst, src_peer, flag, epoch)``: ``dst`` (L, R, W) set to another GPU's block ``src_peer`` (L, R, W) of
  the same dtype (any strides but a unit last one, no element in two rows of ``dst``; ``src_peer`` read through that
  GPU's mapping): the recurrent states of a prefill whose batch is split between the GPUs moved between the head split
  and the row split.  The programs wait until the one word ``flag`` (int64 (1,): this GPU's word of the source GPU)
  holds at least the epoch.

Launch constants: ``JOIN_ROWS`` x ``JOIN_BLOCK`` values per program, ``JOIN_WARPS`` warps; ``PULL_BLOCK`` values per
program of a state pull, ``PULL_WARPS`` warps.

References (the kernels' arguments; a flag word a kernel would wait for must hold the epoch already):
:func:`tp_fence_ref`, :func:`tp_release_ref`, :func:`tp_prefill_join_ref`, :func:`tp_prefill_gather_ref`,
:func:`tp_state_pull_ref`.  A one-GPU call stands for every GPU: it sets the other GPUs' words of its own rows before
the call, or passes the same buffer as every GPU's.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ....neuronspark.ops.triton import require_cuda, require_no_grad
from ....neuronspark.ops.triton.tp_exchange import MAX_GPUS, _box

__all__ = ["JOIN_ROWS", "JOIN_BLOCK", "JOIN_WARPS", "PULL_BLOCK", "PULL_WARPS", "row_bounds", "tp_fence",
           "tp_fence_ref", "tp_release", "tp_release_ref", "tp_prefill_join", "tp_prefill_join_ref",
           "tp_prefill_gather", "tp_prefill_gather_ref", "tp_state_pull", "tp_state_pull_ref"]

JOIN_ROWS = 8
JOIN_BLOCK = 512
JOIN_WARPS = 4
PULL_BLOCK = 8192
PULL_WARPS = 4


def row_bounds(batch: int, n: int) -> List[int]:
    """[lo_0, ..., lo_N]: GPU r's rows of a batch split between ``n`` GPUs are [lo_r, lo_(r+1)),
    lo_r = ceil(r B / N)."""
    B, n = int(batch), int(n)
    return [-(-r * B // n) for r in range(n + 1)]


# ================================================================================================ flag words
@triton.jit
def _store_release(ptr, val, m):
    """lanes where ``m`` != 0: fence.proxy.alias, then st.release.sys of the 64-bit ``val`` at ``ptr``"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; @p fence.proxy.alias; @p st.release.sys.global.b64 [$1], $2; "
        "mov.u32 $0, 0; }",
        "=r,l,l,r", [ptr, val, m], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _load_acquire(ptr, m):
    """lanes where ``m`` != 0: ld.acquire.sys of the 64-bit word at ``ptr``; 2^63 - 1 (the minimum's neutral element)
    elsewhere"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b64 $0, 9223372036854775807; "
        "@p ld.acquire.sys.global.b64 $0, [$1]; }",
        "=l,l,r", [ptr, m], dtype=tl.int64, is_pure=False, pack=1)


@triton.jit
def _wait_epoch(FLAG, e, LANES: tl.constexpr):
    """The program waits until the word at FLAG holds at least ``e``: lane 0 loads, the block's minimum (a reduction
    across the program, so a barrier) lets every thread go on together.  LANES: the program's threads."""
    lane = tl.arange(0, LANES)
    first = (lane == 0).to(tl.int32)
    w = _load_acquire(FLAG + lane * 0, first)
    while tl.min(w, axis=0) < e:
        w = _load_acquire(FLAG + lane * 0, first)


@triton.jit
def _wait_all(FLAG, e, RANK: tl.constexpr, NR: tl.constexpr, LANES: tl.constexpr):
    """The program waits until every word of the row FLAG (NR words) but word RANK holds at least ``e``: lane s loads
    word s, the block's minimum lets every thread go on together.  LANES: the program's threads (at least NR)."""
    lane = tl.arange(0, LANES)
    mine = ((lane < NR) & (lane != RANK)).to(tl.int32)
    w = _load_acquire(FLAG + lane, mine)
    while tl.min(w, axis=0) < e:
        w = _load_acquire(FLAG + lane, mine)


@triton.jit
def _release_all(e, LANES: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr, F0, F1, F2, F3, F4, F5, F6, F7):
    """``e`` stored at word RANK of every other GPU's row (F_j), by lane 0"""
    lane = tl.arange(0, LANES)
    first = (lane == 0).to(tl.int32)
    for j in tl.static_range(NR):
        if j != RANK:
            _store_release(_box(j, F0, F1, F2, F3, F4, F5, F6, F7) + RANK + lane * 0, e + lane.to(tl.int64) * 0, first)


# ================================================================================================ kernels
@triton.jit
def _release_kernel(F0, F1, F2, F3, F4, F5, F6, F7, EP, LANES: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr):
    e = tl.load(EP)
    _release_all(e, LANES, RANK, NR, F0, F1, F2, F3, F4, F5, F6, F7)


@triton.jit
def _fence_kernel(C0, C1, C2, C3, C4, C5, C6, C7, EP, LANES: tl.constexpr, RANK: tl.constexpr, NR: tl.constexpr):
    e = tl.load(EP)
    _release_all(e, LANES, RANK, NR, C0, C1, C2, C3, C4, C5, C6, C7)
    _wait_all(_box(RANK, C0, C1, C2, C3, C4, C5, C6, C7), e, RANK, NR, LANES)


@triton.jit
def _prefill_join_kernel(P0, P1, P2, P3, P4, P5, P6, P7, H, OUT, FLAG, EP, T, B, D, s_ht, s_hb, n_rows,
                         ROWS: tl.constexpr, BLOCK: tl.constexpr, LANES: tl.constexpr, RANK: tl.constexpr,
                         NR: tl.constexpr):
    e = tl.load(EP)
    _wait_all(FLAG, e, RANK, NR, LANES)
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)               # output rows t * B + b
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = (r < n_rows)[:, None] & (c < D)[None, :]
    t = r // B
    b = r % B
    po = (b * T + t).to(tl.int64)[:, None] * D + c[None, :]         # partial rows b * T + t
    if RANK == 0:                                                   # the partials summed in rank order
        acc = tl.load(P0 + po, mask=m, other=0.0)
    else:
        acc = tl.load(P0 + po, mask=m, other=0.0, volatile=True)    # another GPU's, written there
    for j in tl.static_range(1, NR):
        if j == RANK:
            acc = acc + tl.load(_box(j, P0, P1, P2, P3, P4, P5, P6, P7) + po, mask=m, other=0.0)
        else:
            acc = acc + tl.load(_box(j, P0, P1, P2, P3, P4, P5, P6, P7) + po, mask=m, other=0.0, volatile=True)
    upd = acc.to(tl.bfloat16).to(tl.float32)
    h = tl.load(H + t.to(tl.int64)[:, None] * s_ht + b.to(tl.int64)[:, None] * s_hb + c[None, :], mask=m, other=0.0)
    tl.store(OUT + r.to(tl.int64)[:, None] * D + c[None, :], h + upd, mask=m)


@triton.jit
def _prefill_gather_kernel(X, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, OUT, FLAG, EP, B, D, s_xt, s_xb, n_rows,
                           ROWS: tl.constexpr, BLOCK: tl.constexpr, LANES: tl.constexpr, RANK: tl.constexpr,
                           NR: tl.constexpr):
    e = tl.load(EP)
    _wait_all(FLAG, e, RANK, NR, LANES)
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)               # output rows t * B + b
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = r // B
    b = r % B
    m = (r < n_rows)[:, None] & (c < D)[None, :]
    v = tl.zeros([ROWS, BLOCK], dtype=tl.float32)
    for j in tl.static_range(NR):                                   # row b's GPU j: lo_j <= b < lo_(j+1)
        lo = (j * B + NR - 1) // NR
        hi = ((j + 1) * B + NR - 1) // NR
        mj = m & ((b >= lo) & (b < hi))[:, None]
        if j == RANK:
            xo = t.to(tl.int64)[:, None] * s_xt + (b - lo).to(tl.int64)[:, None] * s_xb + c[None, :]
            v = tl.where(mj, tl.load(X + xo, mask=mj, other=0.0), v)
        else:
            qo = (t * (hi - lo) + b - lo).to(tl.int64)[:, None] * D + c[None, :]
            v = tl.where(mj, tl.load(_box(j, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7) + qo, mask=mj, other=0.0, volatile=True),
                         v)                                          # another GPU's, written there
    tl.store(OUT + r.to(tl.int64)[:, None] * D + c[None, :], v, mask=m)


@triton.jit
def _state_pull_kernel(DST, SRC, FLAG, EP, R, W, s_dl, s_dr, s_sl, s_sr,
                       BLOCK: tl.constexpr, LANES: tl.constexpr):
    e = tl.load(EP)
    _wait_epoch(FLAG, e, LANES)
    row = tl.program_id(0)                                          # l * R + r
    l = (row // R).to(tl.int64)
    r = (row % R).to(tl.int64)
    c = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = c < W
    v = tl.load(SRC + l * s_sl + r * s_sr + c, mask=m, volatile=True)   # another GPU's, written there
    tl.store(DST + l * s_dl + r * s_dr + c, v, mask=m)


# ================================================================================================ launchers
def _check_word(t: torch.Tensor, name: str, device) -> None:
    if t.dtype != torch.int64 or tuple(t.shape) != (1,) or t.device != device:
        raise RuntimeError(f"neuronspark.tp_bulk: {name} must be an int64 word (1,) on {device}, got {t.dtype} "
                           f"{tuple(t.shape)} on {t.device}")


def _rows_of(ts: Sequence[torch.Tensor], rank: int, name: str, device, what: str, n: int = None) -> Tuple[list, int]:
    """(the eight per-GPU arguments: GPU i's at i, this GPU's in the unused places; N) after checking that ``ts``
    holds 2 to MAX_GPUS tensors on ``device`` and ``rank`` is among them (a flag row: contiguous int64 (N,))."""
    nr = len(ts)
    if not 2 <= nr <= MAX_GPUS or not 0 <= int(rank) < nr or (n is not None and nr != n):
        raise RuntimeError(f"neuronspark.{what}: {name} must hold one tensor per GPU (2 to {MAX_GPUS}) and rank must "
                           f"be among them, got {nr} and rank {rank}")
    for k, t in enumerate(ts):
        if t.device != device:
            raise RuntimeError(f"neuronspark.{what}: {name}[{k}] must be on {device}, got {t.device}")
    return list(ts) + [ts[int(rank)]] * (MAX_GPUS - nr), nr


def _check_row(t: torch.Tensor, nr: int, name: str, device, what: str) -> None:
    if t.dtype != torch.int64 or tuple(t.shape) != (nr,) or not t.is_contiguous() or t.device != device:
        raise RuntimeError(f"neuronspark.{what}: {name} must be a contiguous int64 row ({nr},) on {device}, got "
                           f"{t.dtype} {tuple(t.shape)} on {t.device}")


def tp_fence(calls: Sequence[torch.Tensor], rank: int, epoch: torch.Tensor) -> None:
    """Store ``epoch`` into word ``rank`` of every other GPU's call words, then wait until every other word of this
    GPU's holds at least it (see the module docstring)."""
    require_cuda(epoch, "tp_fence")
    _check_word(epoch, "epoch", epoch.device)
    cs, nr = _rows_of(calls, rank, "calls", epoch.device, "tp_fence")
    for k, t in enumerate(calls):
        _check_row(t, nr, f"calls[{k}]", epoch.device, "tp_fence")
    with torch.cuda.device(epoch.device.index):
        _fence_kernel[(1,)](*cs, epoch, LANES=32, RANK=int(rank), NR=nr, num_warps=1)


def tp_release(flags: Sequence[torch.Tensor], rank: int, epoch: torch.Tensor) -> None:
    """Store ``epoch`` into word ``rank`` of every other GPU's flag row of the site (see the module docstring)."""
    require_cuda(epoch, "tp_release")
    _check_word(epoch, "epoch", epoch.device)
    fs, nr = _rows_of(flags, rank, "flags", epoch.device, "tp_release")
    for k, t in enumerate(flags):
        _check_row(t, nr, f"flags[{k}]", epoch.device, "tp_release")
    with torch.cuda.device(epoch.device.index):
        _release_kernel[(1,)](*fs, epoch, LANES=32, RANK=int(rank), NR=nr, num_warps=1)


def tp_prefill_join(parts: Sequence[torch.Tensor], h: torch.Tensor, flag: torch.Tensor, rank: int,
                    epoch: torch.Tensor) -> torch.Tensor:
    """``h + f32(bf16(parts[0] + ... + parts[N - 1]))`` as a new (T, B, D) fp32 tensor (see the module docstring)."""
    require_cuda(h, "tp_prefill_join")
    if h.dim() != 3 or h.dtype != torch.float32 or h.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_prefill_join: h must be fp32 (T, B, D) with unit last stride, got "
                           f"{h.dtype} {tuple(h.shape)} stride {tuple(h.stride())}")
    T, B, D = (int(s) for s in h.shape)
    ps, nr = _rows_of(parts, rank, "parts", h.device, "tp_prefill_join")
    for k, t in enumerate(parts):
        if tuple(t.shape) != (B * T, D) or t.dtype != torch.float32 or not t.is_contiguous():
            raise RuntimeError(f"neuronspark.tp_prefill_join: parts[{k}] must be contiguous fp32 ({B * T}, {D}) on "
                               f"{h.device}, got {t.dtype} {tuple(t.shape)}")
    _check_row(flag, nr, "flag", h.device, "tp_prefill_join")
    _check_word(epoch, "epoch", h.device)
    require_no_grad("tp_prefill_join", h, *parts)
    out = torch.empty((T, B, D), dtype=torch.float32, device=h.device)
    if T * B and D:
        with torch.cuda.device(h.device.index):
            _prefill_join_kernel[(triton.cdiv(T * B, JOIN_ROWS), triton.cdiv(D, JOIN_BLOCK))](
                *ps, h, out, flag, epoch, T, B, D, h.stride(0), h.stride(1), T * B,
                ROWS=JOIN_ROWS, BLOCK=JOIN_BLOCK, LANES=32 * JOIN_WARPS, RANK=int(rank), NR=nr, num_warps=JOIN_WARPS)
    return out


def tp_prefill_gather(x: torch.Tensor, parts: Sequence[torch.Tensor], flag: torch.Tensor, rank: int,
                      epoch: torch.Tensor, batch: int) -> torch.Tensor:
    """Every row (T, batch, D) fp32 from this GPU's rows ``x`` and the other GPUs' in ``parts`` (see the module
    docstring)."""
    require_cuda(x, "tp_prefill_gather")
    if x.dim() != 3 or x.dtype != torch.float32 or x.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_prefill_gather: x must be fp32 (T, n, D) with unit last stride, got "
                           f"{x.dtype} {tuple(x.shape)} stride {tuple(x.stride())}")
    T, n, D = (int(s) for s in x.shape)
    qs, nr = _rows_of(parts, rank, "parts", x.device, "tp_prefill_gather")
    B = int(batch)
    lo = row_bounds(B, nr)
    if B < nr or n != lo[int(rank) + 1] - lo[int(rank)]:
        raise RuntimeError(f"neuronspark.tp_prefill_gather: a batch of {B} rows split between {nr} GPUs gives GPU "
                           f"{rank} the rows [{lo[int(rank)]}, {lo[int(rank) + 1]}) (row_bounds), got {n} rows")
    for j, t in enumerate(parts):
        nj = lo[j + 1] - lo[j]
        if j != int(rank) and (t.dim() != 2 or int(t.shape[0]) < T * nj or int(t.shape[1]) != D
                               or t.dtype != torch.float32 or not t.is_contiguous()):
            raise RuntimeError(f"neuronspark.tp_prefill_gather: parts[{j}] must be contiguous fp32 (>= {T * nj}, {D}) "
                               f"on {x.device}, got {t.dtype} {tuple(t.shape)}")
    _check_row(flag, nr, "flag", x.device, "tp_prefill_gather")
    _check_word(epoch, "epoch", x.device)
    require_no_grad("tp_prefill_gather", x, *parts)
    out = torch.empty((T, B, D), dtype=torch.float32, device=x.device)
    if T and D:
        with torch.cuda.device(x.device.index):
            _prefill_gather_kernel[(triton.cdiv(T * B, JOIN_ROWS), triton.cdiv(D, JOIN_BLOCK))](
                x, *qs, out, flag, epoch, B, D, x.stride(0), x.stride(1), T * B, ROWS=JOIN_ROWS, BLOCK=JOIN_BLOCK,
                LANES=32 * JOIN_WARPS, RANK=int(rank), NR=nr, num_warps=JOIN_WARPS)
    return out


def _check_block(dst: torch.Tensor, src_peer: torch.Tensor) -> None:
    for name, t in (("dst", dst), ("src_peer", src_peer)):
        if t.dim() != 3 or t.stride(-1) != 1:
            raise RuntimeError(f"neuronspark.tp_state_pull: {name} must be (L, R, W) with unit last stride, got "
                               f"{tuple(t.shape)} stride {tuple(t.stride())}")
    if tuple(src_peer.shape) != tuple(dst.shape) or src_peer.dtype != dst.dtype or src_peer.device != dst.device:
        raise RuntimeError(f"neuronspark.tp_state_pull: src_peer must match dst {dst.dtype} {tuple(dst.shape)} on "
                           f"{dst.device}, got {src_peer.dtype} {tuple(src_peer.shape)} on {src_peer.device}")
    if dst.numel():                                                 # the outer strides nest: no element in two rows
        extent = int(dst.shape[2])
        for stride, size in sorted((int(dst.stride(i)), int(dst.shape[i])) for i in (0, 1) if int(dst.shape[i]) > 1):
            if stride < extent:
                raise RuntimeError(f"neuronspark.tp_state_pull: the rows of dst must not overlap, got "
                                   f"{tuple(dst.shape)} stride {tuple(dst.stride())}")
            extent += stride * (size - 1)


def tp_state_pull(dst: torch.Tensor, src_peer: torch.Tensor, flag: torch.Tensor, epoch: torch.Tensor) -> None:
    """``dst`` set to another GPU's block ``src_peer`` once ``flag`` holds the epoch (see the module docstring)."""
    require_cuda(dst, "tp_state_pull")
    _check_block(dst, src_peer)
    _check_word(flag, "flag", dst.device)
    _check_word(epoch, "epoch", dst.device)
    require_no_grad("tp_state_pull", dst, src_peer)
    L, R, W = (int(s) for s in dst.shape)
    if L * R and W:
        with torch.cuda.device(dst.device.index):
            _state_pull_kernel[(L * R, triton.cdiv(W, PULL_BLOCK))](
                dst, src_peer, flag, epoch, R, W, dst.stride(0), dst.stride(1), src_peer.stride(0),
                src_peer.stride(1), BLOCK=PULL_BLOCK, LANES=32 * PULL_WARPS, num_warps=PULL_WARPS)


# ================================================================================================ reference
def _passed_ref(flag: torch.Tensor, epoch: torch.Tensor, what: str, rank: int = None) -> None:
    """The word ``flag`` (every word of the row ``flag`` but word ``rank``) must hold at least the epoch: the kernel
    waits until it does, a reference cannot."""
    w = flag if rank is None else torch.cat([flag[:int(rank)], flag[int(rank) + 1:]])
    if bool((w < epoch).any()):
        raise RuntimeError(f"neuronspark.{what}: a flag word is below the epoch (the kernel would wait for another "
                           f"GPU)")


def tp_fence_ref(calls: Sequence[torch.Tensor], rank: int, epoch: torch.Tensor) -> None:
    """Pure torch: the epoch stored into word ``rank`` of every other GPU's call words; then every other word of this
    GPU's must hold at least it."""
    for j, t in enumerate(calls):
        if j != int(rank):
            t[int(rank):int(rank) + 1].copy_(epoch)
    _passed_ref(calls[int(rank)], epoch, "tp_fence_ref", rank)


def tp_release_ref(flags: Sequence[torch.Tensor], rank: int, epoch: torch.Tensor) -> None:
    """Pure torch: the epoch stored into word ``rank`` of every other GPU's flag row."""
    for j, t in enumerate(flags):
        if j != int(rank):
            t[int(rank):int(rank) + 1].copy_(epoch)


def tp_prefill_join_ref(parts: Sequence[torch.Tensor], h: torch.Tensor, flag: torch.Tensor, rank: int,
                        epoch: torch.Tensor) -> torch.Tensor:
    """Pure torch: ``h + f32(bf16(parts[0] + ... + parts[N - 1]))`` (summed in rank order) with the partials' (b, t)
    rows viewed as (T, B, D); every other word of ``flag`` must hold at least the epoch."""
    _passed_ref(flag, epoch, "tp_prefill_join_ref", rank)
    T, B, D = (int(s) for s in h.shape)
    acc = parts[0].clone()
    for p in parts[1:]:
        acc = acc + p
    upd = acc.to(torch.bfloat16).float().view(B, T, D).transpose(0, 1)
    return (h + upd).contiguous()


def tp_prefill_gather_ref(x: torch.Tensor, parts: Sequence[torch.Tensor], flag: torch.Tensor, rank: int,
                          epoch: torch.Tensor, batch: int) -> torch.Tensor:
    """Pure torch: GPU j's rows (this GPU's ``x``, another's the first T n_j rows of ``parts[j]`` viewed (T, n_j, D))
    side by side along the batch in rank order (:func:`row_bounds`); every other word of ``flag`` must hold at least
    the epoch."""
    _passed_ref(flag, epoch, "tp_prefill_gather_ref", rank)
    T, n, D = (int(s) for s in x.shape)
    lo = row_bounds(batch, len(parts))
    blocks = [x if j == int(rank) else parts[j][:T * (lo[j + 1] - lo[j])].view(T, lo[j + 1] - lo[j], D)
              for j in range(len(parts))]
    return torch.cat(blocks, dim=1).contiguous()


def tp_state_pull_ref(dst: torch.Tensor, src_peer: torch.Tensor, flag: torch.Tensor, epoch: torch.Tensor) -> None:
    """Pure torch: ``dst`` set to ``src_peer``; ``flag`` must hold at least the epoch."""
    _check_block(dst, src_peer)
    _passed_ref(flag, epoch, "tp_state_pull_ref")
    dst.copy_(src_peer)
