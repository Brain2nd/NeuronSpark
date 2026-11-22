"""Whole-row read of the model-level lexical n-gram memory (training): one forward kernel; a backward of one sort and
one kernel.

``lexical_train`` is the whole-row form of :meth:`neuronspark.modules.lexical_memory.LexicalMemory.forward`: per row b
and position t (positions before the row start read the pad id V)::

    key_n  = base-(V + 1) number of (x_{t-n+1}, ..., x_t)                 int64 (order n >= 2)
    h      = key_n * mult + add  (mod 2^64);   h ^= h >> 29              the table's hash
    bucket = h mod rows                                                    unsigned residue (the id itself for order 1)
    m[t, b, off_k : off_k + d_k] = table_k[bucket] * scale                 fp32, (T, B, M) contiguous

* forward: one kernel, ``FWD_POSITIONS`` positions per program and every table; the integer arithmetic of
  :meth:`LexicalMemory.buckets` on int64 (two's-complement wrap of the product, a masked arithmetic shift for the
  logical one, the unsigned residue fixed with ``2^64 mod rows`` for negative values), so the buckets are the same
  integers, and the same gather and product (the same bits).  It also writes each (table, position)'s row in the
  concatenation of the tables (int32 key = the table's first row there + bucket), kept for the backward.
* backward from ``g = dL/dm``: the keys sorted (``torch.sort``, stable).  Table k's keys lie in its own range of the
  concatenation, so table k holds sorted slots [k R, (k + 1) R) (R = B T), and equal keys keep the order of their
  positions.  One kernel, ``BWD_SLOTS`` sorted slots of one table per program: each slot's g row times ``scale`` (the
  product of the reference's backward), summed over each run of equal keys by a segmented scan over the block; the
  program holding a run's first slot writes the row, reading the next blocks while the run goes on.  Every row is
  written by one program and nothing is atomic: the same bits at every call.  The rows go into one zeroed fp32 buffer
  (a dense view per table) returned to autograd as the tables' gradients; a table whose gradient buffer is bound
  (:func:`neuronspark.ops.linear.bind_grad_`: an optimizer that owns the table's ``.grad``) has the rows added into that
  buffer instead (``grad[row] = grad[row] + sum``: what adding the dense gradient adds, the other rows' zeros adding
  nothing), and autograd receives no gradient for it.  A bound table armed to overwrite
  (:func:`neuronspark.ops.linear.arm_grad_overwrite_`) is refused: the backward writes only the rows read.

Numerics: the forward is exact (the reference's integers, gathers and products); the backward's sums are fp32 in
another order than the reference's (torch's embedding backward per table).  Tests: the forward against
:func:`lexical_train_ref` (the torch chain of the module) exactly, the gradients with the per-operator tolerance rule of
:mod:`neuronspark.utils.testing`, and two backwards bit for bit.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from neuronspark.ops.linear import armed_grad_overwrites, bound_grad
from neuronspark.ops.triton import require_cuda

__all__ = ["LexicalTrainFn", "lexical_train", "lexical_train_ref"]

# ================================================================================================ fixed constants
#: forward: positions per program, warps
FWD_POSITIONS = 32
FWD_WARPS = 4
#: backward: sorted slots of one table per program (a run longer than this is continued by the program holding its
#: first slot, a block at a time), warps
BWD_SLOTS = 64
BWD_WARPS = 4
_MASK35 = (1 << 35) - 1                # a logical right shift by 29 of a 64-bit value keeps 35 bits


# ================================================================================================ kernels
@triton.jit
def _lex_train_table_fwd(IDS, TAB, MULT, ADD, WRAP, OUT, KEYS, p, pm, t, row0, x0, kR, ids_s1, V, scale,
                         ORDER: tl.constexpr, NR: tl.constexpr, D: tl.constexpr, DB: tl.constexpr, C0: tl.constexpr,
                         R0: tl.constexpr, HI: tl.constexpr, M: tl.constexpr):
    """one table's rows of the positions p (its channels [C0, C0 + D) of m) and its keys KEYS[kR + p]"""
    if ORDER == 1:
        bucket = x0
    else:
        key = tl.load(IDS + row0 + (t - (ORDER - 1)) * ids_s1, mask=pm & (t >= ORDER - 1), other=V)
        for i in tl.static_range(1, ORDER):                             # x_{t-j}, j = ORDER - 1 - i, oldest first
            if ORDER - 1 - i == 0:
                xj = x0
            else:
                xj = tl.load(IDS + row0 + (t - (ORDER - 1 - i)) * ids_s1, mask=pm & (t >= ORDER - 1 - i), other=V)
            key = key * (V + 1) + xj
        h = key * tl.load(MULT + HI) + tl.load(ADD + HI)               # wraps mod 2^64
        h = h ^ ((h >> 29) & 34359738367)                              # 2^35 - 1: the logical shift's 35 bits
        r = h % NR
        r = tl.where(r < 0, r + NR, r)                                 # the non-negative remainder
        bucket = tl.where(h < 0, (r + tl.load(WRAP + HI)) % NR, r)
    dd = tl.arange(0, DB)
    m2 = pm[:, None] & (dd < D)[None, :]
    rows = tl.load(TAB + bucket[:, None] * D + dd[None, :], mask=m2, other=0.0)
    tl.store(OUT + p[:, None] * M + (C0 + dd)[None, :], rows * scale, mask=m2)
    tl.store(KEYS + kR + p, (bucket + R0).to(tl.int32), mask=pm)


@triton.jit
def _lex_train_fwd_kernel(IDS, TABS, MULT, ADD, WRAP, OUT, KEYS, R, B, ids_s0, ids_s1, V, scale,
                          ORDERS: tl.constexpr, ROWS: tl.constexpr, DIMS: tl.constexpr, DBS: tl.constexpr,
                          NT: tl.constexpr, M: tl.constexpr, BP: tl.constexpr):
    """m rows of BP positions p = t B + b for every table; KEYS[k R + p] = the table's first row + bucket."""
    p = tl.program_id(0).to(tl.int64) * BP + tl.arange(0, BP)
    pm = p < R
    t = p // B
    b = p - t * B
    row0 = b * ids_s0
    x0 = tl.load(IDS + row0 + t * ids_s1, mask=pm, other=0)
    c0 = 0                                                             # the table's first channel of m
    r0 = 0                                                             # the table's first row of the concatenation
    hi = 0                                                             # the table's hash constants
    for k in tl.static_range(NT):
        _lex_train_table_fwd(IDS, TABS[k], MULT, ADD, WRAP, OUT, KEYS, p, pm, t, row0, x0, k * R, ids_s1, V, scale,
                             ORDERS[k], ROWS[k], DIMS[k], DBS[k], c0, r0, hi, M)
        c0 += DIMS[k]
        r0 += ROWS[k]
        hi += ORDERS[k] > 1


@triton.jit
def _segment_add(fa, va, fb, vb):
    """the segmented sum: a run start (flag) begins a new sum"""
    return fa | fb, tl.where(fb != 0, vb, va + vb)


@triton.jit
def _lex_train_table_bwd(G, SK, PERM, DST, j, s0, R, B, g_s0, g_s1, g_s2, scale,
                         C0: tl.constexpr, R0: tl.constexpr, D: tl.constexpr, DB: tl.constexpr, ACC: tl.constexpr,
                         BS: tl.constexpr):
    """Sorted slots [j BS, (j + 1) BS) of one table (its slots start at s0 = k R): the rows of the runs this block
    holds the first slot of (DST + (key - R0) D: written, or added into for ACC)."""
    i = j * BS + tl.arange(0, BS)
    im = i < R
    key = tl.load(SK + s0 + i, mask=im, other=-1)
    prev = tl.load(SK + s0 + i - 1, mask=im & (i > 0), other=-1)
    nxt = tl.load(SK + s0 + i + 1, mask=(i + 1) < R, other=-1)
    key0 = tl.load(SK + s0 + j * BS)                                   # the block's first slot
    first_start = key0 != tl.load(SK + s0 + j * BS - 1, mask=j > 0, other=-1)
    own = im & ((key != key0) | first_start)                           # the run's first slot is in this block
    pos = tl.load(PERM + s0 + i, mask=im, other=s0) - s0                # the slot's position t B + b
    t = pos // B
    b = pos - t * B
    dd = tl.arange(0, DB)
    dm = dd < D
    m2 = im[:, None] & dm[None, :]
    ch = (C0 + dd).to(tl.int64) * g_s2
    g = tl.load(G + (t * g_s0 + b * g_s1)[:, None] + ch[None, :], mask=m2, other=0.0) * scale
    flag = tl.broadcast_to(((key != prev) & im).to(tl.int32)[:, None], (BS, DB))
    _, s = tl.associative_scan((flag, g), 0, _segment_add)            # the run's sum up to each slot
    done = own & (nxt != key)                                          # the run ends in this block
    dst = DST + (key - R0).to(tl.int64)[:, None] * D + dd[None, :]
    w2 = done[:, None] & dm[None, :]
    if ACC:
        s += tl.load(dst, mask=w2, other=0.0)
    tl.store(dst, s, mask=w2)
    # the block's last run, if it began here and goes on: continued over the next blocks
    il = tl.minimum(j * BS + BS, R) - 1
    kl = tl.load(SK + s0 + il)
    if ((kl != key0) | first_start) & (tl.load(SK + s0 + il + 1, mask=il + 1 < R, other=-1) == kl):
        acc = tl.sum(tl.where((i == il)[:, None], s, 0.0), axis=0)
        q = j * BS + BS
        go = q < R
        while go:
            ii = q + tl.arange(0, BS)
            inr = (ii < R) & (tl.load(SK + s0 + ii, mask=ii < R, other=-1) == kl)
            pos2 = tl.load(PERM + s0 + ii, mask=inr, other=s0) - s0
            t2 = pos2 // B
            b2 = pos2 - t2 * B
            g2 = tl.load(G + (t2 * g_s0 + b2 * g_s1)[:, None] + ch[None, :], mask=inr[:, None] & dm[None, :],
                         other=0.0) * scale
            acc += tl.sum(g2, axis=0)
            q += BS
            go = (tl.sum(inr.to(tl.int32), axis=0) == BS) & (q < R)
        dl = DST + (kl - R0).to(tl.int64) * D + dd
        if ACC:
            acc += tl.load(dl, mask=dm, other=0.0)
        tl.store(dl, acc, mask=dm)


@triton.jit
def _lex_train_bwd_kernel(G, SK, PERM, DSTS, R, B, g_s0, g_s1, g_s2, scale,
                          ROWS: tl.constexpr, DIMS: tl.constexpr, DBS: tl.constexpr, ACCS: tl.constexpr,
                          NT: tl.constexpr, BS: tl.constexpr):
    """program (j, k): block j of table k's sorted slots"""
    j = tl.program_id(0)
    kp = tl.program_id(1)
    c0 = 0
    r0 = 0
    for k in tl.static_range(NT):
        if kp == k:
            _lex_train_table_bwd(G, SK, PERM, DSTS[k], j, (k * R).to(tl.int64), R, B, g_s0, g_s1, g_s2, scale,
                                 c0, r0, DIMS[k], DBS[k], ACCS[k], BS)
        c0 += DIMS[k]
        r0 += ROWS[k]


# ================================================================================================ launchers
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.lexical_train: " + msg)


def _check(ids, tables, orders, hash_mult, hash_add, hash_wrap) -> None:
    require_cuda(ids, "lexical_train")
    _need(ids.dim() == 2 and ids.dtype == torch.int64, f"ids must be (B, T) int64, got {ids.dtype} {tuple(ids.shape)}")
    _need(len(tables) == len(orders) >= 1, "one order per table")
    _need(all(int(n) >= 1 for n in orders), f"orders {list(orders)} must be positive")
    n_hashed = sum(1 for n in orders if int(n) > 1)
    for name, t in (("hash_mult", hash_mult), ("hash_add", hash_add), ("hash_wrap", hash_wrap)):
        _need(t.dim() == 1 and t.dtype == torch.int64 and t.is_contiguous() and int(t.shape[0]) >= max(n_hashed, 1)
              and t.device == ids.device, f"{name} must be a contiguous int64 vector with one entry per hashed table")
    for t in tables:
        _need(t.dim() == 2 and t.dtype == torch.float32 and t.is_contiguous() and t.device == ids.device,
              f"a table must be a contiguous fp32 (rows, d) tensor on the ids' device, got {t.dtype} {tuple(t.shape)}")
    _need(sum(int(t.shape[0]) for t in tables) < 2 ** 31, "the tables' rows must sum to less than 2^31")
    _need(len(tables) * ids.numel() < 2 ** 31, "tables x positions must be less than 2^31")


def _layout(tables, orders):
    """(orders, rows, dims, dims rounded up to powers of two) as constexpr tuples"""
    return (tuple(int(n) for n in orders), tuple(int(t.shape[0]) for t in tables),
            tuple(int(t.shape[1]) for t in tables), tuple(int(triton.next_power_of_2(int(t.shape[1]))) for t in tables))


@torch.library.custom_op("neuronspark::lexical_train_fwd", mutates_args=())
def _lexical_train_fwd(ids: torch.Tensor, tables: List[torch.Tensor], orders: List[int], hash_mult: torch.Tensor,
                       hash_add: torch.Tensor, hash_wrap: torch.Tensor, vocab_size: int,
                       scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(m (T, B, M) fp32 contiguous, keys (len(tables) B T,) int32)."""
    B, T = (int(v) for v in ids.shape)
    R = B * T
    ords, rows, dims, dbs = _layout(tables, orders)
    M = sum(dims)
    out = torch.empty((T, B, M), dtype=torch.float32, device=ids.device)
    keys = torch.empty((len(tables) * R,), dtype=torch.int32, device=ids.device)
    if R:
        _lex_train_fwd_kernel[(triton.cdiv(R, FWD_POSITIONS),)](
            ids, tuple(tables), hash_mult, hash_add, hash_wrap, out, keys, R, B, int(ids.stride(0)),
            int(ids.stride(1)), int(vocab_size), float(scale), ORDERS=ords, ROWS=rows, DIMS=dims, DBS=dbs,
            NT=len(tables), M=M, BP=FWD_POSITIONS, num_warps=FWD_WARPS)
    return out, keys


@_lexical_train_fwd.register_fake
def _lexical_train_fwd_fake(ids, tables, orders, hash_mult, hash_add, hash_wrap, vocab_size, scale):
    B, T = ids.shape
    M = sum(int(t.shape[1]) for t in tables)
    return (tables[0].new_empty((T, B, M), dtype=torch.float32),
            ids.new_empty((len(tables) * B * T,), dtype=torch.int32))


def _lexical_train_bwd(g: torch.Tensor, keys: torch.Tensor, tables: Sequence[torch.Tensor], orders: Sequence[int],
                       B: int, T: int, scale: float,
                       bound: Sequence[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
    """The tables' gradients from g (T, B, M) fp32: a dense fp32 tensor per table, None for a table with a bound
    buffer (the rows added into it)."""
    R = B * T
    ords, rows, dims, dbs = _layout(tables, orders)
    dense = [k for k in range(len(tables)) if bound[k] is None]
    buf = torch.zeros(sum(rows[k] * dims[k] for k in dense), dtype=torch.float32, device=g.device)
    grads: List[Optional[torch.Tensor]] = []
    off = 0
    for k in range(len(tables)):
        if bound[k] is None:
            grads.append(buf[off:off + rows[k] * dims[k]].view(rows[k], dims[k]))
            off += rows[k] * dims[k]
        else:
            grads.append(None)
    if R:
        sk, perm = torch.sort(keys, stable=True)
        dsts = tuple(bound[k] if bound[k] is not None else grads[k] for k in range(len(tables)))
        _lex_train_bwd_kernel[(triton.cdiv(R, BWD_SLOTS), len(tables))](
            g, sk, perm, dsts, R, B, int(g.stride(0)), int(g.stride(1)), int(g.stride(2)), float(scale),
            ROWS=rows, DIMS=dims, DBS=dbs, ACCS=tuple(int(x is not None) for x in bound), NT=len(tables),
            BS=BWD_SLOTS, num_warps=BWD_WARPS)
    return grads


class LexicalTrainFn(torch.autograd.Function):
    """m (T, B, M) fp32 from the token ids and the tables; gradients for the tables (see the module docstring)."""

    @staticmethod
    def forward(ctx, ids, hash_mult, hash_add, hash_wrap, orders, vocab_size, scale, *tables):
        m, keys = _lexical_train_fwd(ids, list(tables), list(orders), hash_mult, hash_add, hash_wrap, int(vocab_size),
                                     float(scale))
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(keys)
        ctx.tables = tables
        ctx.orders = tuple(orders)
        ctx.shape = (int(ids.shape[0]), int(ids.shape[1]))
        ctx.scale = float(scale)
        return m

    @staticmethod
    def backward(ctx, g):
        n = 7 + len(ctx.tables)
        if g is None:
            return (None,) * n
        (keys,) = ctx.saved_tensors
        B, T = ctx.shape
        M = sum(int(t.shape[1]) for t in ctx.tables)
        if g.dtype != torch.float32 or tuple(g.shape) != (T, B, M):
            raise RuntimeError(f"neuronspark.lexical_train backward: gradient {g.dtype} {tuple(g.shape)} for "
                               f"({T}, {B}, {M})")
        need = ctx.needs_input_grad[7:]
        bound = [bound_grad(t) if need[k] else None for k, t in enumerate(ctx.tables)]
        armed = armed_grad_overwrites([t for t, buf in zip(ctx.tables, bound) if buf is not None])
        if armed:
            raise RuntimeError("neuronspark.lexical_train backward: a lexical table armed to overwrite its bound "
                               "gradient buffer (arm_grad_overwrite_); the backward adds the rows it reads, so its "
                               "owner zeroes the buffer instead")
        grads = _lexical_train_bwd(g, keys, ctx.tables, ctx.orders, B, T, ctx.scale, bound)
        return (None,) * 7 + tuple(gr if need[k] else None for k, gr in enumerate(grads))


# ================================================================================================ public function
def lexical_train(ids: torch.Tensor, tables: List[torch.Tensor], orders: List[int], hash_mult: torch.Tensor,
                  hash_add: torch.Tensor, hash_wrap: torch.Tensor, vocab_size: int, scale: float) -> torch.Tensor:
    """(T, B, M) fp32 lexical vectors of whole rows, differentiable in the tables: ``ids`` (B, T) int64 token ids
    (0 <= id < V: the kernel gathers on the device; any strides), ``tables`` the fp32 (rows_k, d_k) tables in m order
    (contiguous), ``orders`` the n-gram order of each table (1 = the exact-id table, first), ``hash_mult`` /
    ``hash_add`` / ``hash_wrap`` the int64 hash constants (one entry per hashed table, in table order), ``vocab_size``
    V, ``scale`` the output scale.  Training regime."""
    _check(ids, tables, orders, hash_mult, hash_add, hash_wrap)
    return LexicalTrainFn.apply(ids, hash_mult, hash_add, hash_wrap, tuple(int(n) for n in orders), int(vocab_size),
                                float(scale), *tables)


# ================================================================================================ reference
def lexical_train_ref(ids: torch.Tensor, tables: List[torch.Tensor], orders: List[int], hash_mult: torch.Tensor,
                      hash_add: torch.Tensor, hash_wrap: torch.Tensor, vocab_size: int, scale: float) -> torch.Tensor:
    """The torch chain of the whole-row read (same arguments and output as :func:`lexical_train`): the int64 keys over
    the (T, B) ids with pad ids before the row start, the hashes, ``F.embedding`` per table, the concatenation times
    ``scale`` (differentiable in the tables through the gathers)."""
    x = ids.transpose(0, 1).to(torch.int64)                               # (T, B)
    T = int(x.shape[0])
    n_max = max(int(n) for n in orders)
    V1 = int(vocab_size) + 1
    pad = torch.full((n_max - 1, int(x.shape[1])), int(vocab_size), dtype=torch.int64, device=x.device)
    ext = torch.cat([pad, x], dim=0)                                       # ext[n_max - 1 + t] = x_t
    parts, hi = [], 0
    for t, n in zip(tables, orders):
        n = int(n)
        if n == 1:
            bucket = x
        else:
            key = torch.zeros_like(x)
            for j in range(n - 1, -1, -1):                                 # x_{t-j}, oldest first
                key = key * V1 + ext[n_max - 1 - j:n_max - 1 - j + T]
            h = key * hash_mult[hi] + hash_add[hi]                         # wraps mod 2^64 (two's complement)
            h = torch.bitwise_xor(h, torch.bitwise_and(torch.bitwise_right_shift(h, 29), _MASK35))
            r = int(t.shape[0])
            bucket = torch.remainder(h, r)
            bucket = torch.where(h < 0, torch.remainder(bucket + hash_wrap[hi], r), bucket)
            hi += 1
        parts.append(F.embedding(bucket, t).float())
    return torch.cat(parts, dim=-1) * scale
