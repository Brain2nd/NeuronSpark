"""Decode-step read of the model-level lexical n-gram memory (T = 1): one kernel launch per table.

``lexical_step`` forms, per row, the lexical vector of one token from the carried window of the row's last ids::

    key_n  = base-(V + 1) number of (x_{t-n+1}, ..., x_t)                 int64, the older ids read from the window
    h      = key_n * mult + add  (mod 2^64);   h ^= h >> 29              the table's hash (order n >= 2)
    bucket = h mod rows                                                    unsigned residue (the id itself for order 1)
    m[off : off + d] = table[bucket] * scale                               fp32

and rolls the window to the last ids (the launch of the last table).  The integer arithmetic is that of
:meth:`neuronspark.modules.lexical_memory.LexicalMemory.buckets` on int64 (two's-complement wrap of the product, a
masked arithmetic shift for the logical one, the unsigned residue fixed with ``2^64 mod rows`` for negative values), so
the buckets are the same integers; the vector is the same gather and the same product.  Reference
``lexical_step_ref``: the torch chain of the session read.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

from typing import List

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ....neuronspark.ops.triton import require_cuda

__all__ = ["LEX_STEP_WARPS", "lexical_step", "lexical_step_ref"]

# ================================================================================================ fixed constants
#: one program per row and table, warps (one: its threads read the window together before they roll it)
LEX_STEP_WARPS = 1
_MASK35 = (1 << 35) - 1                # a logical right shift by 29 of a 64-bit value keeps 35 bits


# ================================================================================================ kernel
@triton.jit
def _lex_table_step_kernel(IDS, WIN, TAB, OUT, MULT, ADD, WRAP, hi, off, M, rows, V1, scale, D,
                           KW: tl.constexpr, ORDER: tl.constexpr, DB: tl.constexpr, ROLL: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    x = tl.load(IDS + b)
    if ORDER > 1:
        key = tl.load(WIN + b * KW + (KW - (ORDER - 1)))                   # x_{t-n+1}, the oldest id of the key
        for i in tl.static_range(ORDER - 2):
            key = key * V1 + tl.load(WIN + b * KW + (KW - (ORDER - 2 - i)))
        key = key * V1 + x
        h = key * tl.load(MULT + hi) + tl.load(ADD + hi)                   # wraps mod 2^64
        h = h ^ ((h >> 29) & 34359738367)                                  # 2^35 - 1: the logical shift's 35 bits
        r = h % rows
        r = tl.where(r < 0, r + rows, r)                                   # the non-negative remainder
        bucket = tl.where(h < 0, (r + tl.load(WRAP + hi)) % rows, r)
    else:
        bucket = x
    dd = tl.arange(0, DB)
    dm = dd < D
    tl.store(OUT + b * M + off + dd, tl.load(TAB + bucket * D + dd, mask=dm, other=0.0) * scale, mask=dm)
    if ROLL:
        for j in tl.static_range(KW - 1):
            tl.store(WIN + b * KW + j, tl.load(WIN + b * KW + j + 1))
        tl.store(WIN + b * KW + (KW - 1), x)


# ================================================================================================ launcher
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.lexical_step: " + msg)


def _lexical_step_fwd(ids: torch.Tensor, win: torch.Tensor, tables: List[torch.Tensor], orders: List[int],
                      hash_mult: torch.Tensor, hash_add: torch.Tensor, hash_wrap: torch.Tensor, vocab_size: int,
                      scale: float) -> torch.Tensor:
    require_cuda(ids, "lexical_step")
    _need(ids.dim() == 1 and ids.dtype == torch.int64 and ids.is_contiguous(),
          f"ids must be a contiguous (B,) int64 tensor, got {ids.dtype} {tuple(ids.shape)}")
    B = int(ids.shape[0])
    _need(win.dim() == 2 and int(win.shape[0]) == B and win.dtype == torch.int64 and win.is_contiguous()
          and int(win.shape[1]) >= 1 and win.device == ids.device,
          f"the window must be a contiguous ({B}, n_max - 1) int64 tensor, got {win.dtype} {tuple(win.shape)}")
    kw = int(win.shape[1])
    _need(len(tables) == len(orders) >= 1, "one order per table")
    _need(all(1 <= int(n) <= kw + 1 for n in orders), f"orders {list(orders)} for a window of {kw} ids")
    n_hashed = sum(1 for n in orders if int(n) > 1)
    for name, t in (("hash_mult", hash_mult), ("hash_add", hash_add), ("hash_wrap", hash_wrap)):
        _need(t.dim() == 1 and t.dtype == torch.int64 and t.is_contiguous() and int(t.shape[0]) >= max(n_hashed, 1)
              and t.device == ids.device,
              f"{name} must be a contiguous int64 vector with one entry per hashed table")
    for t in tables:
        _need(t.dim() == 2 and t.dtype == torch.float32 and t.is_contiguous() and t.device == ids.device,
              f"a table must be a contiguous fp32 (rows, d) tensor, got {t.dtype} {tuple(t.shape)}")
    M = sum(int(t.shape[1]) for t in tables)
    out = torch.empty((B, M), dtype=torch.float32, device=ids.device)
    off, hi = 0, 0
    for k, (t, n) in enumerate(zip(tables, orders)):
        d = int(t.shape[1])
        _lex_table_step_kernel[(B,)](ids, win, t, out, hash_mult, hash_add, hash_wrap, hi, off, M, int(t.shape[0]),
                                     int(vocab_size) + 1, float(scale), d, KW=kw, ORDER=int(n),
                                     DB=triton.next_power_of_2(d), ROLL=int(k == len(tables) - 1),
                                     num_warps=LEX_STEP_WARPS, enable_fp_fusion=False)
        off += d
        hi += int(int(n) > 1)
    return out


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::lexical_step", mutates_args=("win",))
def _lexical_step_op(ids: torch.Tensor, win: torch.Tensor, tables: List[torch.Tensor], orders: List[int],
                     hash_mult: torch.Tensor, hash_add: torch.Tensor, hash_wrap: torch.Tensor, vocab_size: int,
                     scale: float) -> torch.Tensor:
    return _lexical_step_fwd(ids, win, list(tables), list(orders), hash_mult, hash_add, hash_wrap, vocab_size, scale)


@_lexical_step_op.register_fake
def _lexical_step_fake(ids, win, tables, orders, hash_mult, hash_add, hash_wrap, vocab_size, scale):
    return tables[0].new_empty((ids.shape[0], sum(int(t.shape[1]) for t in tables)), dtype=torch.float32)


# ================================================================================================ public function
def lexical_step(ids: torch.Tensor, win: torch.Tensor, tables: List[torch.Tensor], orders: List[int],
                 hash_mult: torch.Tensor, hash_add: torch.Tensor, hash_wrap: torch.Tensor, vocab_size: int,
                 scale: float) -> torch.Tensor:
    """(B, M) fp32 lexical vector of one token per row: ``ids`` (B,) int64, ``win`` (B, n_max - 1) int64 the rows' last
    ids, oldest first (rolled in place to end with ``ids``), ``tables`` the fp32 (rows_k, d_k) tables in m order,
    ``orders`` the n-gram order of each table (1 = the exact-id table), ``hash_mult`` / ``hash_add`` / ``hash_wrap``
    the int64 hash constants (one entry per hashed table, in table order), ``vocab_size`` V, ``scale`` the output
    scale.  ``ids`` are token ids (0 <= id < V: the kernel gathers on the device) and share no memory with ``win``."""
    return _KERNEL_OPS.lexical_step(ids, win, list(tables), [int(n) for n in orders], hash_mult, hash_add,
                                              hash_wrap, int(vocab_size), float(scale))


# ================================================================================================ pure-torch reference
@torch.no_grad()
def lexical_step_ref(ids: torch.Tensor, win: torch.Tensor, tables: List[torch.Tensor], orders: List[int],
                     hash_mult: torch.Tensor, hash_add: torch.Tensor, hash_wrap: torch.Tensor, vocab_size: int,
                     scale: float) -> torch.Tensor:
    """The torch chain of the session read for one token: the int64 keys and hashes, ``F.embedding`` per table, the
    concatenation times ``scale``; ``win`` rolled in place."""
    kw = int(win.shape[1])
    V1 = int(vocab_size) + 1
    ext = torch.cat([win, ids.view(-1, 1)], dim=1)                        # (B, kw + 1): ext[:, kw] = x_t
    parts, hi = [], 0
    for t, n in zip(tables, orders):
        n = int(n)
        if n == 1:
            bucket = ids
        else:
            key = torch.zeros_like(ids)
            for j in range(n - 1, -1, -1):                                 # x_{t-j}, oldest first
                key = key * V1 + ext[:, kw - j]
            h = key * hash_mult[hi] + hash_add[hi]                         # wraps mod 2^64 (two's complement)
            h = torch.bitwise_xor(h, torch.bitwise_and(torch.bitwise_right_shift(h, 29), _MASK35))
            r = int(t.shape[0])
            bucket = torch.remainder(h, r)
            bucket = torch.where(h < 0, torch.remainder(bucket + hash_wrap[hi], r), bucket)
            hi += 1
        parts.append(F.embedding(bucket, t).float())
    win.copy_(ext[:, 1:])
    return torch.cat(parts, dim=-1) * scale
