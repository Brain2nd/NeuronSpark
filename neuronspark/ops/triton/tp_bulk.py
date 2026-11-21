"""Bulk exchanges between the two GPUs of a tensor-parallel prefill: a whole buffer in symmetric memory, read by the
other GPU once a flag word says that it is complete.

A GPU writes its buffer (the out-projection's fp32 partial product, or its rows of a stream whose batch is split
between the GPUs) into its symmetric memory; then a kernel of one thread stores the call's epoch into the peer's flag
word for the site (:func:`tp_release`: ``fence.proxy.alias``, then ``st.release.sys``, cumulative over the writes of the
kernels before it in the stream).  The reading kernel's programs each wait until their own flag word holds at least the
epoch (one thread polls with ``ld.acquire.sys``, the program's barrier follows) and then read the peer's buffer through
its mapping (volatile loads).  Both GPUs take a site's buffer of the same size (they grow together).

Aliases.  A buffer is written through its owner's mapping and read through the other GPU's: two virtual addresses of
one allocation, which PTX orders only through an alias proxy fence on the path from the write to the read (and from a
read to the next write of the same buffer).  Every such path passes a release (the writer's for the site, or the
reader's for a later site or the next call's fence), and every release executes one.  A flag word is likewise stored
through one mapping and polled through the other; that a system-scope release and acquire synchronize across the two
mappings is assumed, as PyTorch's symmetric-memory signal pads assume it.

Progress.  Every word a program waits for is stored by the other GPU from a kernel that stores before it waits for
anything (the release waits for nothing; the fence stores, then waits).  The other GPU's release of site s follows its
own wait of site s - 1, which this GPU's release of site s - 1 ends, and that release precedes this GPU's wait of site
s; the fences of a call follow the waits of the call before, which end the same way.  So every wait ends, whatever
order the programs of either GPU are dispatched in, and a waiting grid may be larger than the GPU.

Reuse.  The epoch is a device counter advanced once at the start of every prefill call (inside a captured call as
well), equal on both GPUs, which run the same calls, each GPU's one after another (on one stream, or on streams
ordered by events: the epoch, the flags and the buffers are shared by all of them); a site's flag word is written
once per call and compared with >=, so it is never cleared.  The buffers alternate by site parity (a join's and a
gather's sites alike): within a call a GPU writes a buffer again two sites later, after it has passed the wait of the
site between, which the peer releases only after its read of the earlier site has finished.  Across calls the fence
that starts every prefill call orders them (:func:`tp_fence`): a GPU stores the new epoch into the peer's call word
and waits until its own holds it, which the peer stores only after its previous call has finished, its reads of this
GPU's buffers among them; so a call may have any number of sites, and an eager call may follow a captured one.

Ops (inference, no autograd):

* ``tp_fence(flag_peer, flag, epoch)``: the start of a prefill call: the epoch (int64 (1,)) stored into the peer's call
  word (int64 (1,)), then a wait until this GPU's call word ``flag`` holds at least the epoch; one program.
* ``tp_release(flag_peer, epoch)``: the peer's flag word for the site (int64 (1,)) set to the epoch (int64 (1,)).
* ``tp_prefill_join(part, part_peer, h, flag, epoch) -> h_new``: the end of a split dendritic neuron layer's prefill:
  ``upd = bf16(part + part_peer)`` (the out-projection's two fp32 partial products, (B T, D) in (b, t) row order, summed
  in fp32 and rounded once, as the one-GPU product is) and ``h_new = h + f32(upd)`` for the fp32 residual stream ``h``
  (T, B, D): a new (T, B, D) fp32 tensor, the same on both GPUs (``a + b`` is one fp32 number either way).  The
  programs wait on ``flag`` (this GPU's word for the site) for ``epoch``.
* ``tp_prefill_gather(x, part_peer, flag, epoch, lo, batch) -> h``: the end of a prefill call's layer whose batch is
  split between the GPUs (a hippocampal layer): ``x`` (T, n, D) fp32 holds this GPU's rows [lo, lo + n) of the
  ``batch`` rows, the peer's rows are the rest (those before this GPU's, or those after them), (T, batch - n, D) in
  the first rows of the peer's buffer ``part_peer`` (T (batch - n) rows in (t, row) order); ``h`` a new (T, batch, D)
  fp32 tensor of every row, the same on both GPUs.  The programs wait on ``flag`` for ``epoch``.

Launch constants: ``JOIN_ROWS`` x ``JOIN_BLOCK`` values per program, ``JOIN_WARPS`` warps.

A one-GPU call passes its own buffer as ``part_peer`` and its own word as the flag.  References (the kernels'
arguments; a flag word a kernel would wait for must hold the epoch already): :func:`tp_fence_ref`,
:func:`tp_release_ref`, :func:`tp_prefill_join_ref`, :func:`tp_prefill_gather_ref`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda, require_no_grad

__all__ = ["JOIN_ROWS", "JOIN_BLOCK", "JOIN_WARPS", "tp_fence", "tp_fence_ref", "tp_release", "tp_release_ref",
           "tp_prefill_join", "tp_prefill_join_ref", "tp_prefill_gather", "tp_prefill_gather_ref"]

JOIN_ROWS = 8
JOIN_BLOCK = 512
JOIN_WARPS = 4


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


# ================================================================================================ kernels
@triton.jit
def _release_kernel(FLAG_PEER, EP, LANES: tl.constexpr):
    lane = tl.arange(0, LANES)
    e = tl.load(EP + lane * 0)
    _store_release(FLAG_PEER + lane * 0, e, (lane == 0).to(tl.int32))


@triton.jit
def _fence_kernel(FLAG_PEER, FLAG, EP, LANES: tl.constexpr):
    lane = tl.arange(0, LANES)
    e = tl.load(EP)
    _store_release(FLAG_PEER + lane * 0, e + lane.to(tl.int64) * 0, (lane == 0).to(tl.int32))
    _wait_epoch(FLAG, e, LANES)


@triton.jit
def _prefill_join_kernel(P, Q, H, OUT, FLAG, EP, T, B, D, s_ht, s_hb, n_rows,
                         ROWS: tl.constexpr, BLOCK: tl.constexpr, LANES: tl.constexpr):
    e = tl.load(EP)
    _wait_epoch(FLAG, e, LANES)
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)               # output rows t * B + b
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = (r < n_rows)[:, None] & (c < D)[None, :]
    t = r // B
    b = r % B
    po = (b * T + t).to(tl.int64)[:, None] * D + c[None, :]         # partial rows b * T + t
    p = tl.load(P + po, mask=m, other=0.0)
    q = tl.load(Q + po, mask=m, other=0.0, volatile=True)            # the peer's, written by another GPU
    upd = (p + q).to(tl.bfloat16).to(tl.float32)
    h = tl.load(H + t.to(tl.int64)[:, None] * s_ht + b.to(tl.int64)[:, None] * s_hb + c[None, :], mask=m, other=0.0)
    tl.store(OUT + r.to(tl.int64)[:, None] * D + c[None, :], h + upd, mask=m)


@triton.jit
def _prefill_gather_kernel(X, Q, OUT, FLAG, EP, B, D, lo, hi, peer_lo, n_peer, s_xt, s_xb, n_rows,
                           ROWS: tl.constexpr, BLOCK: tl.constexpr, LANES: tl.constexpr):
    e = tl.load(EP)
    _wait_epoch(FLAG, e, LANES)
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)               # output rows t * B + b
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = r // B
    b = r % B
    m = (r < n_rows)[:, None] & (c < D)[None, :]
    mo = m & ((b >= lo) & (b < hi))[:, None]                        # this GPU's rows
    mp = m & ((b < lo) | (b >= hi))[:, None]                        # the peer's
    xo = t.to(tl.int64)[:, None] * s_xt + (b - lo).to(tl.int64)[:, None] * s_xb + c[None, :]
    qo = (t * n_peer + b - peer_lo).to(tl.int64)[:, None] * D + c[None, :]
    x = tl.load(X + xo, mask=mo, other=0.0)
    q = tl.load(Q + qo, mask=mp, other=0.0, volatile=True)          # the peer's, written by another GPU
    tl.store(OUT + r.to(tl.int64)[:, None] * D + c[None, :], tl.where(mo, x, q), mask=m)


# ================================================================================================ launchers
def _check_word(t: torch.Tensor, name: str, device) -> None:
    if t.dtype != torch.int64 or tuple(t.shape) != (1,) or t.device != device:
        raise RuntimeError(f"neuronspark.tp_bulk: {name} must be an int64 word (1,) on {device}, got {t.dtype} "
                           f"{tuple(t.shape)} on {t.device}")


def tp_fence(flag_peer: torch.Tensor, flag: torch.Tensor, epoch: torch.Tensor) -> None:
    """Store ``epoch`` into the peer's call word ``flag_peer``, then wait until ``flag`` holds at least it (see the
    module docstring)."""
    require_cuda(epoch, "tp_fence")
    for name, t in (("flag_peer", flag_peer), ("flag", flag), ("epoch", epoch)):
        _check_word(t, name, epoch.device)
    with torch.cuda.device(epoch.device.index):
        _fence_kernel[(1,)](flag_peer, flag, epoch, LANES=32, num_warps=1)


def tp_release(flag_peer: torch.Tensor, epoch: torch.Tensor) -> None:
    """Store ``epoch`` into the peer's flag word ``flag_peer`` (see the module docstring)."""
    require_cuda(epoch, "tp_release")
    _check_word(flag_peer, "flag_peer", epoch.device)
    _check_word(epoch, "epoch", epoch.device)
    with torch.cuda.device(epoch.device.index):
        _release_kernel[(1,)](flag_peer, epoch, LANES=32, num_warps=1)


def tp_prefill_join(part: torch.Tensor, part_peer: torch.Tensor, h: torch.Tensor, flag: torch.Tensor,
                    epoch: torch.Tensor) -> torch.Tensor:
    """``h + f32(bf16(part + part_peer))`` as a new (T, B, D) fp32 tensor (see the module docstring)."""
    require_cuda(h, "tp_prefill_join")
    if h.dim() != 3 or h.dtype != torch.float32 or h.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_prefill_join: h must be fp32 (T, B, D) with unit last stride, got "
                           f"{h.dtype} {tuple(h.shape)} stride {tuple(h.stride())}")
    T, B, D = (int(s) for s in h.shape)
    for name, t in (("part", part), ("part_peer", part_peer)):
        if tuple(t.shape) != (B * T, D) or t.dtype != torch.float32 or not t.is_contiguous() or t.device != h.device:
            raise RuntimeError(f"neuronspark.tp_prefill_join: {name} must be contiguous fp32 ({B * T}, {D}) on "
                               f"{h.device}, got {t.dtype} {tuple(t.shape)} on {t.device}")
    _check_word(flag, "flag", h.device)
    _check_word(epoch, "epoch", h.device)
    require_no_grad("tp_prefill_join", part, part_peer, h)
    out = torch.empty((T, B, D), dtype=torch.float32, device=h.device)
    if T * B and D:
        with torch.cuda.device(h.device.index):
            _prefill_join_kernel[(triton.cdiv(T * B, JOIN_ROWS), triton.cdiv(D, JOIN_BLOCK))](
                part, part_peer, h, out, flag, epoch, T, B, D, h.stride(0), h.stride(1), T * B,
                ROWS=JOIN_ROWS, BLOCK=JOIN_BLOCK, LANES=32 * JOIN_WARPS, num_warps=JOIN_WARPS)
    return out


def tp_prefill_gather(x: torch.Tensor, part_peer: torch.Tensor, flag: torch.Tensor, epoch: torch.Tensor, lo: int,
                      batch: int) -> torch.Tensor:
    """Every row (T, batch, D) fp32 from this GPU's rows ``x`` = rows [lo, lo + n) and the peer's in ``part_peer``
    (see the module docstring)."""
    require_cuda(x, "tp_prefill_gather")
    if x.dim() != 3 or x.dtype != torch.float32 or x.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.tp_prefill_gather: x must be fp32 (T, n, D) with unit last stride, got "
                           f"{x.dtype} {tuple(x.shape)} stride {tuple(x.stride())}")
    T, n, D = (int(s) for s in x.shape)
    B, lo = int(batch), int(lo)
    n_peer = B - n
    if lo < 0 or n < 1 or n_peer < 1 or not (lo == 0 or lo + n == B):
        raise RuntimeError(f"neuronspark.tp_prefill_gather: this GPU's rows [{lo}, {lo + n}) must be the first or the "
                           f"last rows of the batch of {B}, at least one row on each GPU")
    if (part_peer.dim() != 2 or int(part_peer.shape[0]) < T * n_peer or int(part_peer.shape[1]) != D
            or part_peer.dtype != torch.float32 or not part_peer.is_contiguous() or part_peer.device != x.device):
        raise RuntimeError(f"neuronspark.tp_prefill_gather: part_peer must be contiguous fp32 (>= {T * n_peer}, {D}) "
                           f"on {x.device}, got {part_peer.dtype} {tuple(part_peer.shape)} on {part_peer.device}")
    _check_word(flag, "flag", x.device)
    _check_word(epoch, "epoch", x.device)
    require_no_grad("tp_prefill_gather", x, part_peer)
    out = torch.empty((T, B, D), dtype=torch.float32, device=x.device)
    if T and D:
        with torch.cuda.device(x.device.index):
            _prefill_gather_kernel[(triton.cdiv(T * B, JOIN_ROWS), triton.cdiv(D, JOIN_BLOCK))](
                x, part_peer, out, flag, epoch, B, D, lo, lo + n, n if lo == 0 else 0, n_peer, x.stride(0),
                x.stride(1), T * B, ROWS=JOIN_ROWS, BLOCK=JOIN_BLOCK, LANES=32 * JOIN_WARPS, num_warps=JOIN_WARPS)
    return out


# ================================================================================================ reference
def _passed_ref(flag: torch.Tensor, epoch: torch.Tensor, what: str) -> None:
    """The word ``flag`` must hold at least the epoch: the kernel waits until it does, a reference cannot."""
    if bool((flag < epoch).any()):
        raise RuntimeError(f"neuronspark.{what}: the flag word is below the epoch (the kernel would wait for the peer)")


def tp_fence_ref(flag_peer: torch.Tensor, flag: torch.Tensor, epoch: torch.Tensor) -> None:
    """Pure torch: the epoch stored into the peer's call word ``flag_peer``; then ``flag`` must hold at least it."""
    flag_peer.copy_(epoch)
    _passed_ref(flag, epoch, "tp_fence_ref")


def tp_release_ref(flag_peer: torch.Tensor, epoch: torch.Tensor) -> None:
    """Pure torch: the epoch stored into the peer's flag word ``flag_peer``."""
    flag_peer.copy_(epoch)


def tp_prefill_join_ref(part: torch.Tensor, part_peer: torch.Tensor, h: torch.Tensor, flag: torch.Tensor,
                        epoch: torch.Tensor) -> torch.Tensor:
    """Pure torch: ``h + f32(bf16(part + part_peer))`` with the partials' (b, t) rows viewed as (T, B, D); ``flag``
    must hold at least the epoch."""
    _passed_ref(flag, epoch, "tp_prefill_join_ref")
    T, B, D = (int(s) for s in h.shape)
    upd = (part + part_peer).to(torch.bfloat16).float().view(B, T, D).transpose(0, 1)
    return (h + upd).contiguous()


def tp_prefill_gather_ref(x: torch.Tensor, part_peer: torch.Tensor, flag: torch.Tensor, epoch: torch.Tensor, lo: int,
                          batch: int) -> torch.Tensor:
    """Pure torch: this GPU's rows (T, n, D) and the peer's (the first T (batch - n) rows of ``part_peer`` viewed (T,
    batch - n, D)) side by side along the batch, this GPU's first when ``lo`` is 0; ``flag`` must hold at least the
    epoch."""
    _passed_ref(flag, epoch, "tp_prefill_gather_ref")
    T, n, D = (int(s) for s in x.shape)
    peer = part_peer[:T * (int(batch) - n)].view(T, int(batch) - n, D)
    return torch.cat([x, peer] if int(lo) == 0 else [peer, x], dim=1).contiguous()
