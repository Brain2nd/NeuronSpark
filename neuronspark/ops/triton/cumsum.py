"""Cumulative sums over the time axis and the carried running means of the memory layer.

Operators (registry names):

* ``cumsum0(x)`` (training): the inclusive ``x.cumsum(dim=0)`` of an fp32 CUDA tensor (T, ...), with autograd's
  backward (the reverse scan of the incoming gradient).  Used for the pool clocks (narrow (T, B) scans) and the
  value-residual gate's running sums (wide (T, B, C) scans).
* ``exact_cumsum0(x)`` (prefill, no autograd): ``torch.cumsum(x, dim=0)`` of an fp32 (T, ...) tensor with more than one
  column, as ATen's outer-dim scan computes it.
* ``centred_running(x, run_sum, count, out_dtype)`` (prefill, no autograd, T >= 2): the strictly-past running-mean
  centring ``c_t = (x_t - S_t / max(n + t, 1)) * [n + t > 0]`` with ``S`` the carried fp32 (B, C) sum (``run_sum``,
  advanced in place to the sum through the call's last token) and ``n = count`` (B,) int64 the row position of the
  call's first token; returns (T, B, C) in ``out_dtype``.

Numerics contract
-----------------
* ``cumsum0``, narrow scans (at most 32 columns): one lane per column runs ATen's outer-dim recurrence ``acc = +0;
  acc = add.rn(acc, x[t]); y[t] = acc`` strictly left to right (right to left for the backward), 64 loads in flight
  per tile -- ATen's order of additions.
* ``cumsum0``, wide scans (more than 32 columns): the T steps in 32 chunks; lane (chunk, column) runs the sequential
  fp32 chain over its chunk (pass 1 the chunk totals, an exclusive prefix over the totals, pass 2 the chain again from
  that offset).  Within a chunk the order is ATen's; the chunk offsets make the result the per-operator rule class.
* A single column (``numel == size(0)``) is ``x.cumsum(dim=0)`` itself (ATen scans it with cub's inclusive scan).
* ``exact_cumsum0``: the same IEEE fp32 operations as ``torch.cumsum`` in the same order (``add.rn``; the accumulator
  from +0).
* ``centred_running``: the tokens in tiles; a tile's prefix sums by a parallel scan plus the carried sum of the earlier
  tiles (the order of the fp32 additions differs from the sequential chain: the per-operator rule class), then the
  chain's per-element ``div.rn`` by max(n + t, 1), the difference and the mask, the output cast fused (round to nearest
  even).

Launchers are ``torch.library`` custom ops (``neuronspark::cumsum0``, ``::exact_cumsum0``, ``::centred_running``,
the last declaring its in-place update of ``run_sum``) with fake implementations.  References: ``torch.cumsum``
and the torch chain of the running-mean centring (:func:`centred_running_ref`).
"""
from typing import Optional

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (jit helpers resolved through this module's globals)
    _rn_add_f32,
    _rn_add_f32 as _add_rn_f32,
    _rn_div_f32,
    _rn_fma_f32,
    _rn_mul_f32,
    _rn_sub_f32,
)

__all__ = ["Cumsum0Fn", "cumsum0", "cumsum0_ref", "exact_cumsum0", "exact_cumsum0_ref", "centred_running",
           "centred_running_ref"]

# ------------------------------------------------------------------------------------------------ fixed constants
# Training: narrow widths take the sequential kernel, wide widths the chunked kernel.
NARROW_N = 32                          # widths up to 32 columns are "narrow"
SEQ_BLOCK_N, SEQ_WARPS = 32, 1         # sequential kernel for the narrow scans (one warp, 32 columns per program)
CHUNK_CH, CHUNK_BLOCK_N, CHUNK_WARPS, CHUNK_STAGES = 32, 32, 8, 4   # chunked kernel for the wide scans
# Prefill exact cumsum: one warp per program, one column per lane.
PREFILL_BLOCK = 32
# centred_running (TT tokens, BC columns, warps) by the call's width N = B * C: narrow (one prefill row) long tiles over few
# columns, wide (eight rows) short tiles over more
CENTRED_NARROW_N = 4096
CENTRED_NARROW = (512, 8, 4)
CENTRED_WIDE = (128, 32, 4)
_OUT_TL = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def _two_level(x: torch.Tensor):
    """(C, SXB) such that element (t, n) of ``x`` (inner axes flattened row-major into n) sits at
    t * stride(0) + (n // C) * SXB + (n % C), or None when the inner axes do not collapse to that form."""
    dims = [(s, st) for s, st in zip(x.shape[1:], x.stride()[1:]) if s != 1]
    C = 1
    i = len(dims) - 1
    while i >= 0 and dims[i][1] == C:
        C *= dims[i][0]
        i -= 1
    if i < 0:
        return C, 0
    if C == 1:
        return None
    sxb = dims[i][1]
    expected = sxb * dims[i][0]
    for j in range(i - 1, -1, -1):
        if dims[j][1] != expected:
            return None
        expected *= dims[j][0]
    return C, sxb


# ================================================================================================ training kernels
# The sequential column scan (narrow widths): per column acc = +0, then one add.rn per time step, the loads of 64
# consecutive steps issued before the 64 dependent additions; STEP = -1 scans backward (the gradient).
@triton.jit
def _cumsum0_u64_kernel(X, Y, T, N, C, SXB, SXT: tl.constexpr, SYT: tl.constexpr, REVERSE: tl.constexpr,
                        STEP: tl.constexpr, BLOCK_N: tl.constexpr):
    """64-step unrolled column scan (see the module docstring)."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cm = cols < N
    xp = X + ((cols // C).to(tl.int64) * SXB + (cols % C))
    yp = Y + cols
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    n_full = (T // 64) * 64
    for tile in range(0, n_full, 64):
        if REVERSE:
            t0 = T - 1 - tile
        else:
            t0 = tile
        xb = xp + t0.to(tl.int64) * SXT
        yb = yp + t0.to(tl.int64) * SYT
        x0 = tl.load(xb + 0 * STEP * SXT, mask=cm, other=0.0)
        x1 = tl.load(xb + 1 * STEP * SXT, mask=cm, other=0.0)
        x2 = tl.load(xb + 2 * STEP * SXT, mask=cm, other=0.0)
        x3 = tl.load(xb + 3 * STEP * SXT, mask=cm, other=0.0)
        x4 = tl.load(xb + 4 * STEP * SXT, mask=cm, other=0.0)
        x5 = tl.load(xb + 5 * STEP * SXT, mask=cm, other=0.0)
        x6 = tl.load(xb + 6 * STEP * SXT, mask=cm, other=0.0)
        x7 = tl.load(xb + 7 * STEP * SXT, mask=cm, other=0.0)
        x8 = tl.load(xb + 8 * STEP * SXT, mask=cm, other=0.0)
        x9 = tl.load(xb + 9 * STEP * SXT, mask=cm, other=0.0)
        x10 = tl.load(xb + 10 * STEP * SXT, mask=cm, other=0.0)
        x11 = tl.load(xb + 11 * STEP * SXT, mask=cm, other=0.0)
        x12 = tl.load(xb + 12 * STEP * SXT, mask=cm, other=0.0)
        x13 = tl.load(xb + 13 * STEP * SXT, mask=cm, other=0.0)
        x14 = tl.load(xb + 14 * STEP * SXT, mask=cm, other=0.0)
        x15 = tl.load(xb + 15 * STEP * SXT, mask=cm, other=0.0)
        x16 = tl.load(xb + 16 * STEP * SXT, mask=cm, other=0.0)
        x17 = tl.load(xb + 17 * STEP * SXT, mask=cm, other=0.0)
        x18 = tl.load(xb + 18 * STEP * SXT, mask=cm, other=0.0)
        x19 = tl.load(xb + 19 * STEP * SXT, mask=cm, other=0.0)
        x20 = tl.load(xb + 20 * STEP * SXT, mask=cm, other=0.0)
        x21 = tl.load(xb + 21 * STEP * SXT, mask=cm, other=0.0)
        x22 = tl.load(xb + 22 * STEP * SXT, mask=cm, other=0.0)
        x23 = tl.load(xb + 23 * STEP * SXT, mask=cm, other=0.0)
        x24 = tl.load(xb + 24 * STEP * SXT, mask=cm, other=0.0)
        x25 = tl.load(xb + 25 * STEP * SXT, mask=cm, other=0.0)
        x26 = tl.load(xb + 26 * STEP * SXT, mask=cm, other=0.0)
        x27 = tl.load(xb + 27 * STEP * SXT, mask=cm, other=0.0)
        x28 = tl.load(xb + 28 * STEP * SXT, mask=cm, other=0.0)
        x29 = tl.load(xb + 29 * STEP * SXT, mask=cm, other=0.0)
        x30 = tl.load(xb + 30 * STEP * SXT, mask=cm, other=0.0)
        x31 = tl.load(xb + 31 * STEP * SXT, mask=cm, other=0.0)
        x32 = tl.load(xb + 32 * STEP * SXT, mask=cm, other=0.0)
        x33 = tl.load(xb + 33 * STEP * SXT, mask=cm, other=0.0)
        x34 = tl.load(xb + 34 * STEP * SXT, mask=cm, other=0.0)
        x35 = tl.load(xb + 35 * STEP * SXT, mask=cm, other=0.0)
        x36 = tl.load(xb + 36 * STEP * SXT, mask=cm, other=0.0)
        x37 = tl.load(xb + 37 * STEP * SXT, mask=cm, other=0.0)
        x38 = tl.load(xb + 38 * STEP * SXT, mask=cm, other=0.0)
        x39 = tl.load(xb + 39 * STEP * SXT, mask=cm, other=0.0)
        x40 = tl.load(xb + 40 * STEP * SXT, mask=cm, other=0.0)
        x41 = tl.load(xb + 41 * STEP * SXT, mask=cm, other=0.0)
        x42 = tl.load(xb + 42 * STEP * SXT, mask=cm, other=0.0)
        x43 = tl.load(xb + 43 * STEP * SXT, mask=cm, other=0.0)
        x44 = tl.load(xb + 44 * STEP * SXT, mask=cm, other=0.0)
        x45 = tl.load(xb + 45 * STEP * SXT, mask=cm, other=0.0)
        x46 = tl.load(xb + 46 * STEP * SXT, mask=cm, other=0.0)
        x47 = tl.load(xb + 47 * STEP * SXT, mask=cm, other=0.0)
        x48 = tl.load(xb + 48 * STEP * SXT, mask=cm, other=0.0)
        x49 = tl.load(xb + 49 * STEP * SXT, mask=cm, other=0.0)
        x50 = tl.load(xb + 50 * STEP * SXT, mask=cm, other=0.0)
        x51 = tl.load(xb + 51 * STEP * SXT, mask=cm, other=0.0)
        x52 = tl.load(xb + 52 * STEP * SXT, mask=cm, other=0.0)
        x53 = tl.load(xb + 53 * STEP * SXT, mask=cm, other=0.0)
        x54 = tl.load(xb + 54 * STEP * SXT, mask=cm, other=0.0)
        x55 = tl.load(xb + 55 * STEP * SXT, mask=cm, other=0.0)
        x56 = tl.load(xb + 56 * STEP * SXT, mask=cm, other=0.0)
        x57 = tl.load(xb + 57 * STEP * SXT, mask=cm, other=0.0)
        x58 = tl.load(xb + 58 * STEP * SXT, mask=cm, other=0.0)
        x59 = tl.load(xb + 59 * STEP * SXT, mask=cm, other=0.0)
        x60 = tl.load(xb + 60 * STEP * SXT, mask=cm, other=0.0)
        x61 = tl.load(xb + 61 * STEP * SXT, mask=cm, other=0.0)
        x62 = tl.load(xb + 62 * STEP * SXT, mask=cm, other=0.0)
        x63 = tl.load(xb + 63 * STEP * SXT, mask=cm, other=0.0)
        acc = _add_rn_f32(acc, x0)
        tl.store(yb + 0 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x1)
        tl.store(yb + 1 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x2)
        tl.store(yb + 2 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x3)
        tl.store(yb + 3 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x4)
        tl.store(yb + 4 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x5)
        tl.store(yb + 5 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x6)
        tl.store(yb + 6 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x7)
        tl.store(yb + 7 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x8)
        tl.store(yb + 8 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x9)
        tl.store(yb + 9 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x10)
        tl.store(yb + 10 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x11)
        tl.store(yb + 11 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x12)
        tl.store(yb + 12 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x13)
        tl.store(yb + 13 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x14)
        tl.store(yb + 14 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x15)
        tl.store(yb + 15 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x16)
        tl.store(yb + 16 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x17)
        tl.store(yb + 17 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x18)
        tl.store(yb + 18 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x19)
        tl.store(yb + 19 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x20)
        tl.store(yb + 20 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x21)
        tl.store(yb + 21 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x22)
        tl.store(yb + 22 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x23)
        tl.store(yb + 23 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x24)
        tl.store(yb + 24 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x25)
        tl.store(yb + 25 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x26)
        tl.store(yb + 26 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x27)
        tl.store(yb + 27 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x28)
        tl.store(yb + 28 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x29)
        tl.store(yb + 29 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x30)
        tl.store(yb + 30 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x31)
        tl.store(yb + 31 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x32)
        tl.store(yb + 32 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x33)
        tl.store(yb + 33 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x34)
        tl.store(yb + 34 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x35)
        tl.store(yb + 35 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x36)
        tl.store(yb + 36 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x37)
        tl.store(yb + 37 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x38)
        tl.store(yb + 38 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x39)
        tl.store(yb + 39 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x40)
        tl.store(yb + 40 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x41)
        tl.store(yb + 41 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x42)
        tl.store(yb + 42 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x43)
        tl.store(yb + 43 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x44)
        tl.store(yb + 44 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x45)
        tl.store(yb + 45 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x46)
        tl.store(yb + 46 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x47)
        tl.store(yb + 47 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x48)
        tl.store(yb + 48 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x49)
        tl.store(yb + 49 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x50)
        tl.store(yb + 50 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x51)
        tl.store(yb + 51 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x52)
        tl.store(yb + 52 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x53)
        tl.store(yb + 53 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x54)
        tl.store(yb + 54 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x55)
        tl.store(yb + 55 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x56)
        tl.store(yb + 56 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x57)
        tl.store(yb + 57 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x58)
        tl.store(yb + 58 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x59)
        tl.store(yb + 59 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x60)
        tl.store(yb + 60 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x61)
        tl.store(yb + 61 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x62)
        tl.store(yb + 62 * STEP * SYT, acc, mask=cm)
        acc = _add_rn_f32(acc, x63)
        tl.store(yb + 63 * STEP * SYT, acc, mask=cm)
    if n_full < T:
        if REVERSE:
            t0 = T - 1 - n_full
        else:
            t0 = n_full
        xb = xp + t0.to(tl.int64) * SXT
        yb = yp + t0.to(tl.int64) * SYT
        m0 = cm & (t0 + 0 * STEP >= 0) & (t0 + 0 * STEP < T)
        m1 = cm & (t0 + 1 * STEP >= 0) & (t0 + 1 * STEP < T)
        m2 = cm & (t0 + 2 * STEP >= 0) & (t0 + 2 * STEP < T)
        m3 = cm & (t0 + 3 * STEP >= 0) & (t0 + 3 * STEP < T)
        m4 = cm & (t0 + 4 * STEP >= 0) & (t0 + 4 * STEP < T)
        m5 = cm & (t0 + 5 * STEP >= 0) & (t0 + 5 * STEP < T)
        m6 = cm & (t0 + 6 * STEP >= 0) & (t0 + 6 * STEP < T)
        m7 = cm & (t0 + 7 * STEP >= 0) & (t0 + 7 * STEP < T)
        m8 = cm & (t0 + 8 * STEP >= 0) & (t0 + 8 * STEP < T)
        m9 = cm & (t0 + 9 * STEP >= 0) & (t0 + 9 * STEP < T)
        m10 = cm & (t0 + 10 * STEP >= 0) & (t0 + 10 * STEP < T)
        m11 = cm & (t0 + 11 * STEP >= 0) & (t0 + 11 * STEP < T)
        m12 = cm & (t0 + 12 * STEP >= 0) & (t0 + 12 * STEP < T)
        m13 = cm & (t0 + 13 * STEP >= 0) & (t0 + 13 * STEP < T)
        m14 = cm & (t0 + 14 * STEP >= 0) & (t0 + 14 * STEP < T)
        m15 = cm & (t0 + 15 * STEP >= 0) & (t0 + 15 * STEP < T)
        m16 = cm & (t0 + 16 * STEP >= 0) & (t0 + 16 * STEP < T)
        m17 = cm & (t0 + 17 * STEP >= 0) & (t0 + 17 * STEP < T)
        m18 = cm & (t0 + 18 * STEP >= 0) & (t0 + 18 * STEP < T)
        m19 = cm & (t0 + 19 * STEP >= 0) & (t0 + 19 * STEP < T)
        m20 = cm & (t0 + 20 * STEP >= 0) & (t0 + 20 * STEP < T)
        m21 = cm & (t0 + 21 * STEP >= 0) & (t0 + 21 * STEP < T)
        m22 = cm & (t0 + 22 * STEP >= 0) & (t0 + 22 * STEP < T)
        m23 = cm & (t0 + 23 * STEP >= 0) & (t0 + 23 * STEP < T)
        m24 = cm & (t0 + 24 * STEP >= 0) & (t0 + 24 * STEP < T)
        m25 = cm & (t0 + 25 * STEP >= 0) & (t0 + 25 * STEP < T)
        m26 = cm & (t0 + 26 * STEP >= 0) & (t0 + 26 * STEP < T)
        m27 = cm & (t0 + 27 * STEP >= 0) & (t0 + 27 * STEP < T)
        m28 = cm & (t0 + 28 * STEP >= 0) & (t0 + 28 * STEP < T)
        m29 = cm & (t0 + 29 * STEP >= 0) & (t0 + 29 * STEP < T)
        m30 = cm & (t0 + 30 * STEP >= 0) & (t0 + 30 * STEP < T)
        m31 = cm & (t0 + 31 * STEP >= 0) & (t0 + 31 * STEP < T)
        m32 = cm & (t0 + 32 * STEP >= 0) & (t0 + 32 * STEP < T)
        m33 = cm & (t0 + 33 * STEP >= 0) & (t0 + 33 * STEP < T)
        m34 = cm & (t0 + 34 * STEP >= 0) & (t0 + 34 * STEP < T)
        m35 = cm & (t0 + 35 * STEP >= 0) & (t0 + 35 * STEP < T)
        m36 = cm & (t0 + 36 * STEP >= 0) & (t0 + 36 * STEP < T)
        m37 = cm & (t0 + 37 * STEP >= 0) & (t0 + 37 * STEP < T)
        m38 = cm & (t0 + 38 * STEP >= 0) & (t0 + 38 * STEP < T)
        m39 = cm & (t0 + 39 * STEP >= 0) & (t0 + 39 * STEP < T)
        m40 = cm & (t0 + 40 * STEP >= 0) & (t0 + 40 * STEP < T)
        m41 = cm & (t0 + 41 * STEP >= 0) & (t0 + 41 * STEP < T)
        m42 = cm & (t0 + 42 * STEP >= 0) & (t0 + 42 * STEP < T)
        m43 = cm & (t0 + 43 * STEP >= 0) & (t0 + 43 * STEP < T)
        m44 = cm & (t0 + 44 * STEP >= 0) & (t0 + 44 * STEP < T)
        m45 = cm & (t0 + 45 * STEP >= 0) & (t0 + 45 * STEP < T)
        m46 = cm & (t0 + 46 * STEP >= 0) & (t0 + 46 * STEP < T)
        m47 = cm & (t0 + 47 * STEP >= 0) & (t0 + 47 * STEP < T)
        m48 = cm & (t0 + 48 * STEP >= 0) & (t0 + 48 * STEP < T)
        m49 = cm & (t0 + 49 * STEP >= 0) & (t0 + 49 * STEP < T)
        m50 = cm & (t0 + 50 * STEP >= 0) & (t0 + 50 * STEP < T)
        m51 = cm & (t0 + 51 * STEP >= 0) & (t0 + 51 * STEP < T)
        m52 = cm & (t0 + 52 * STEP >= 0) & (t0 + 52 * STEP < T)
        m53 = cm & (t0 + 53 * STEP >= 0) & (t0 + 53 * STEP < T)
        m54 = cm & (t0 + 54 * STEP >= 0) & (t0 + 54 * STEP < T)
        m55 = cm & (t0 + 55 * STEP >= 0) & (t0 + 55 * STEP < T)
        m56 = cm & (t0 + 56 * STEP >= 0) & (t0 + 56 * STEP < T)
        m57 = cm & (t0 + 57 * STEP >= 0) & (t0 + 57 * STEP < T)
        m58 = cm & (t0 + 58 * STEP >= 0) & (t0 + 58 * STEP < T)
        m59 = cm & (t0 + 59 * STEP >= 0) & (t0 + 59 * STEP < T)
        m60 = cm & (t0 + 60 * STEP >= 0) & (t0 + 60 * STEP < T)
        m61 = cm & (t0 + 61 * STEP >= 0) & (t0 + 61 * STEP < T)
        m62 = cm & (t0 + 62 * STEP >= 0) & (t0 + 62 * STEP < T)
        m63 = cm & (t0 + 63 * STEP >= 0) & (t0 + 63 * STEP < T)
        x0 = tl.load(xb + 0 * STEP * SXT, mask=m0, other=0.0)
        x1 = tl.load(xb + 1 * STEP * SXT, mask=m1, other=0.0)
        x2 = tl.load(xb + 2 * STEP * SXT, mask=m2, other=0.0)
        x3 = tl.load(xb + 3 * STEP * SXT, mask=m3, other=0.0)
        x4 = tl.load(xb + 4 * STEP * SXT, mask=m4, other=0.0)
        x5 = tl.load(xb + 5 * STEP * SXT, mask=m5, other=0.0)
        x6 = tl.load(xb + 6 * STEP * SXT, mask=m6, other=0.0)
        x7 = tl.load(xb + 7 * STEP * SXT, mask=m7, other=0.0)
        x8 = tl.load(xb + 8 * STEP * SXT, mask=m8, other=0.0)
        x9 = tl.load(xb + 9 * STEP * SXT, mask=m9, other=0.0)
        x10 = tl.load(xb + 10 * STEP * SXT, mask=m10, other=0.0)
        x11 = tl.load(xb + 11 * STEP * SXT, mask=m11, other=0.0)
        x12 = tl.load(xb + 12 * STEP * SXT, mask=m12, other=0.0)
        x13 = tl.load(xb + 13 * STEP * SXT, mask=m13, other=0.0)
        x14 = tl.load(xb + 14 * STEP * SXT, mask=m14, other=0.0)
        x15 = tl.load(xb + 15 * STEP * SXT, mask=m15, other=0.0)
        x16 = tl.load(xb + 16 * STEP * SXT, mask=m16, other=0.0)
        x17 = tl.load(xb + 17 * STEP * SXT, mask=m17, other=0.0)
        x18 = tl.load(xb + 18 * STEP * SXT, mask=m18, other=0.0)
        x19 = tl.load(xb + 19 * STEP * SXT, mask=m19, other=0.0)
        x20 = tl.load(xb + 20 * STEP * SXT, mask=m20, other=0.0)
        x21 = tl.load(xb + 21 * STEP * SXT, mask=m21, other=0.0)
        x22 = tl.load(xb + 22 * STEP * SXT, mask=m22, other=0.0)
        x23 = tl.load(xb + 23 * STEP * SXT, mask=m23, other=0.0)
        x24 = tl.load(xb + 24 * STEP * SXT, mask=m24, other=0.0)
        x25 = tl.load(xb + 25 * STEP * SXT, mask=m25, other=0.0)
        x26 = tl.load(xb + 26 * STEP * SXT, mask=m26, other=0.0)
        x27 = tl.load(xb + 27 * STEP * SXT, mask=m27, other=0.0)
        x28 = tl.load(xb + 28 * STEP * SXT, mask=m28, other=0.0)
        x29 = tl.load(xb + 29 * STEP * SXT, mask=m29, other=0.0)
        x30 = tl.load(xb + 30 * STEP * SXT, mask=m30, other=0.0)
        x31 = tl.load(xb + 31 * STEP * SXT, mask=m31, other=0.0)
        x32 = tl.load(xb + 32 * STEP * SXT, mask=m32, other=0.0)
        x33 = tl.load(xb + 33 * STEP * SXT, mask=m33, other=0.0)
        x34 = tl.load(xb + 34 * STEP * SXT, mask=m34, other=0.0)
        x35 = tl.load(xb + 35 * STEP * SXT, mask=m35, other=0.0)
        x36 = tl.load(xb + 36 * STEP * SXT, mask=m36, other=0.0)
        x37 = tl.load(xb + 37 * STEP * SXT, mask=m37, other=0.0)
        x38 = tl.load(xb + 38 * STEP * SXT, mask=m38, other=0.0)
        x39 = tl.load(xb + 39 * STEP * SXT, mask=m39, other=0.0)
        x40 = tl.load(xb + 40 * STEP * SXT, mask=m40, other=0.0)
        x41 = tl.load(xb + 41 * STEP * SXT, mask=m41, other=0.0)
        x42 = tl.load(xb + 42 * STEP * SXT, mask=m42, other=0.0)
        x43 = tl.load(xb + 43 * STEP * SXT, mask=m43, other=0.0)
        x44 = tl.load(xb + 44 * STEP * SXT, mask=m44, other=0.0)
        x45 = tl.load(xb + 45 * STEP * SXT, mask=m45, other=0.0)
        x46 = tl.load(xb + 46 * STEP * SXT, mask=m46, other=0.0)
        x47 = tl.load(xb + 47 * STEP * SXT, mask=m47, other=0.0)
        x48 = tl.load(xb + 48 * STEP * SXT, mask=m48, other=0.0)
        x49 = tl.load(xb + 49 * STEP * SXT, mask=m49, other=0.0)
        x50 = tl.load(xb + 50 * STEP * SXT, mask=m50, other=0.0)
        x51 = tl.load(xb + 51 * STEP * SXT, mask=m51, other=0.0)
        x52 = tl.load(xb + 52 * STEP * SXT, mask=m52, other=0.0)
        x53 = tl.load(xb + 53 * STEP * SXT, mask=m53, other=0.0)
        x54 = tl.load(xb + 54 * STEP * SXT, mask=m54, other=0.0)
        x55 = tl.load(xb + 55 * STEP * SXT, mask=m55, other=0.0)
        x56 = tl.load(xb + 56 * STEP * SXT, mask=m56, other=0.0)
        x57 = tl.load(xb + 57 * STEP * SXT, mask=m57, other=0.0)
        x58 = tl.load(xb + 58 * STEP * SXT, mask=m58, other=0.0)
        x59 = tl.load(xb + 59 * STEP * SXT, mask=m59, other=0.0)
        x60 = tl.load(xb + 60 * STEP * SXT, mask=m60, other=0.0)
        x61 = tl.load(xb + 61 * STEP * SXT, mask=m61, other=0.0)
        x62 = tl.load(xb + 62 * STEP * SXT, mask=m62, other=0.0)
        x63 = tl.load(xb + 63 * STEP * SXT, mask=m63, other=0.0)
        acc = _add_rn_f32(acc, x0)
        tl.store(yb + 0 * STEP * SYT, acc, mask=m0)
        acc = _add_rn_f32(acc, x1)
        tl.store(yb + 1 * STEP * SYT, acc, mask=m1)
        acc = _add_rn_f32(acc, x2)
        tl.store(yb + 2 * STEP * SYT, acc, mask=m2)
        acc = _add_rn_f32(acc, x3)
        tl.store(yb + 3 * STEP * SYT, acc, mask=m3)
        acc = _add_rn_f32(acc, x4)
        tl.store(yb + 4 * STEP * SYT, acc, mask=m4)
        acc = _add_rn_f32(acc, x5)
        tl.store(yb + 5 * STEP * SYT, acc, mask=m5)
        acc = _add_rn_f32(acc, x6)
        tl.store(yb + 6 * STEP * SYT, acc, mask=m6)
        acc = _add_rn_f32(acc, x7)
        tl.store(yb + 7 * STEP * SYT, acc, mask=m7)
        acc = _add_rn_f32(acc, x8)
        tl.store(yb + 8 * STEP * SYT, acc, mask=m8)
        acc = _add_rn_f32(acc, x9)
        tl.store(yb + 9 * STEP * SYT, acc, mask=m9)
        acc = _add_rn_f32(acc, x10)
        tl.store(yb + 10 * STEP * SYT, acc, mask=m10)
        acc = _add_rn_f32(acc, x11)
        tl.store(yb + 11 * STEP * SYT, acc, mask=m11)
        acc = _add_rn_f32(acc, x12)
        tl.store(yb + 12 * STEP * SYT, acc, mask=m12)
        acc = _add_rn_f32(acc, x13)
        tl.store(yb + 13 * STEP * SYT, acc, mask=m13)
        acc = _add_rn_f32(acc, x14)
        tl.store(yb + 14 * STEP * SYT, acc, mask=m14)
        acc = _add_rn_f32(acc, x15)
        tl.store(yb + 15 * STEP * SYT, acc, mask=m15)
        acc = _add_rn_f32(acc, x16)
        tl.store(yb + 16 * STEP * SYT, acc, mask=m16)
        acc = _add_rn_f32(acc, x17)
        tl.store(yb + 17 * STEP * SYT, acc, mask=m17)
        acc = _add_rn_f32(acc, x18)
        tl.store(yb + 18 * STEP * SYT, acc, mask=m18)
        acc = _add_rn_f32(acc, x19)
        tl.store(yb + 19 * STEP * SYT, acc, mask=m19)
        acc = _add_rn_f32(acc, x20)
        tl.store(yb + 20 * STEP * SYT, acc, mask=m20)
        acc = _add_rn_f32(acc, x21)
        tl.store(yb + 21 * STEP * SYT, acc, mask=m21)
        acc = _add_rn_f32(acc, x22)
        tl.store(yb + 22 * STEP * SYT, acc, mask=m22)
        acc = _add_rn_f32(acc, x23)
        tl.store(yb + 23 * STEP * SYT, acc, mask=m23)
        acc = _add_rn_f32(acc, x24)
        tl.store(yb + 24 * STEP * SYT, acc, mask=m24)
        acc = _add_rn_f32(acc, x25)
        tl.store(yb + 25 * STEP * SYT, acc, mask=m25)
        acc = _add_rn_f32(acc, x26)
        tl.store(yb + 26 * STEP * SYT, acc, mask=m26)
        acc = _add_rn_f32(acc, x27)
        tl.store(yb + 27 * STEP * SYT, acc, mask=m27)
        acc = _add_rn_f32(acc, x28)
        tl.store(yb + 28 * STEP * SYT, acc, mask=m28)
        acc = _add_rn_f32(acc, x29)
        tl.store(yb + 29 * STEP * SYT, acc, mask=m29)
        acc = _add_rn_f32(acc, x30)
        tl.store(yb + 30 * STEP * SYT, acc, mask=m30)
        acc = _add_rn_f32(acc, x31)
        tl.store(yb + 31 * STEP * SYT, acc, mask=m31)
        acc = _add_rn_f32(acc, x32)
        tl.store(yb + 32 * STEP * SYT, acc, mask=m32)
        acc = _add_rn_f32(acc, x33)
        tl.store(yb + 33 * STEP * SYT, acc, mask=m33)
        acc = _add_rn_f32(acc, x34)
        tl.store(yb + 34 * STEP * SYT, acc, mask=m34)
        acc = _add_rn_f32(acc, x35)
        tl.store(yb + 35 * STEP * SYT, acc, mask=m35)
        acc = _add_rn_f32(acc, x36)
        tl.store(yb + 36 * STEP * SYT, acc, mask=m36)
        acc = _add_rn_f32(acc, x37)
        tl.store(yb + 37 * STEP * SYT, acc, mask=m37)
        acc = _add_rn_f32(acc, x38)
        tl.store(yb + 38 * STEP * SYT, acc, mask=m38)
        acc = _add_rn_f32(acc, x39)
        tl.store(yb + 39 * STEP * SYT, acc, mask=m39)
        acc = _add_rn_f32(acc, x40)
        tl.store(yb + 40 * STEP * SYT, acc, mask=m40)
        acc = _add_rn_f32(acc, x41)
        tl.store(yb + 41 * STEP * SYT, acc, mask=m41)
        acc = _add_rn_f32(acc, x42)
        tl.store(yb + 42 * STEP * SYT, acc, mask=m42)
        acc = _add_rn_f32(acc, x43)
        tl.store(yb + 43 * STEP * SYT, acc, mask=m43)
        acc = _add_rn_f32(acc, x44)
        tl.store(yb + 44 * STEP * SYT, acc, mask=m44)
        acc = _add_rn_f32(acc, x45)
        tl.store(yb + 45 * STEP * SYT, acc, mask=m45)
        acc = _add_rn_f32(acc, x46)
        tl.store(yb + 46 * STEP * SYT, acc, mask=m46)
        acc = _add_rn_f32(acc, x47)
        tl.store(yb + 47 * STEP * SYT, acc, mask=m47)
        acc = _add_rn_f32(acc, x48)
        tl.store(yb + 48 * STEP * SYT, acc, mask=m48)
        acc = _add_rn_f32(acc, x49)
        tl.store(yb + 49 * STEP * SYT, acc, mask=m49)
        acc = _add_rn_f32(acc, x50)
        tl.store(yb + 50 * STEP * SYT, acc, mask=m50)
        acc = _add_rn_f32(acc, x51)
        tl.store(yb + 51 * STEP * SYT, acc, mask=m51)
        acc = _add_rn_f32(acc, x52)
        tl.store(yb + 52 * STEP * SYT, acc, mask=m52)
        acc = _add_rn_f32(acc, x53)
        tl.store(yb + 53 * STEP * SYT, acc, mask=m53)
        acc = _add_rn_f32(acc, x54)
        tl.store(yb + 54 * STEP * SYT, acc, mask=m54)
        acc = _add_rn_f32(acc, x55)
        tl.store(yb + 55 * STEP * SYT, acc, mask=m55)
        acc = _add_rn_f32(acc, x56)
        tl.store(yb + 56 * STEP * SYT, acc, mask=m56)
        acc = _add_rn_f32(acc, x57)
        tl.store(yb + 57 * STEP * SYT, acc, mask=m57)
        acc = _add_rn_f32(acc, x58)
        tl.store(yb + 58 * STEP * SYT, acc, mask=m58)
        acc = _add_rn_f32(acc, x59)
        tl.store(yb + 59 * STEP * SYT, acc, mask=m59)
        acc = _add_rn_f32(acc, x60)
        tl.store(yb + 60 * STEP * SYT, acc, mask=m60)
        acc = _add_rn_f32(acc, x61)
        tl.store(yb + 61 * STEP * SYT, acc, mask=m61)
        acc = _add_rn_f32(acc, x62)
        tl.store(yb + 62 * STEP * SYT, acc, mask=m62)
        acc = _add_rn_f32(acc, x63)
        tl.store(yb + 63 * STEP * SYT, acc, mask=m63)


# The chunked column scan (wide widths).
@triton.jit
def _cumsum0_chunk_kernel(X, Y, T, N, C, SXB, SXT, SYT, TC, REVERSE: tl.constexpr, CH: tl.constexpr,
                          BLOCK_N: tl.constexpr, STAGES: tl.constexpr):
    """y[t, n] = sum_{s <= t} x[s, n] (REVERSE: sum_{s >= t}) for BLOCK_N columns per program.  The T steps are CH
    chunks of TC consecutive steps; lane (c, n) runs the sequential fp32 chain over its chunk (pass 1: the chunk
    totals; the exclusive prefix over the CH totals is a tl.cumsum; pass 2: the chain again from that offset, storing).
    Within a chunk the order is ATen's; the chunk offsets make the result the per-operator rule class."""
    pid = tl.program_id(0)
    cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cm = cols < N
    xcol = (cols // C).to(tl.int64) * SXB + (cols % C)
    base = tl.arange(0, CH) * TC                                  # (CH,) first step (in scan order) of each chunk
    acc = tl.zeros([CH, BLOCK_N], dtype=tl.float32)
    for s in tl.range(0, TC, num_stages=STAGES):
        tl_ = base + s
        tm = tl_ < T
        if REVERSE:
            t = T - 1 - tl_
        else:
            t = tl_
        m = tm[:, None] & cm[None, :]
        x = tl.load(X + t.to(tl.int64)[:, None] * SXT + xcol[None, :], mask=m, other=0.0)
        acc = acc + x
    acc = tl.cumsum(acc, axis=0) - acc                            # exclusive prefix of the chunk totals: the offsets
    for s in tl.range(0, TC, num_stages=STAGES):
        tl_ = base + s
        tm = tl_ < T
        if REVERSE:
            t = T - 1 - tl_
        else:
            t = tl_
        m = tm[:, None] & cm[None, :]
        x = tl.load(X + t.to(tl.int64)[:, None] * SXT + xcol[None, :], mask=m, other=0.0)
        acc = acc + x
        tl.store(Y + t.to(tl.int64)[:, None] * SYT + cols[None, :], acc, mask=m)


# ================================================================================================ prefill kernels
# (_rn_* are the IEEE fp32 round-to-nearest helpers of neuronspark.ops.triton._ptx)


@triton.jit
def _cum0_step(acc, x, OUT, off, m):
    """one row of torch's outer-dim cumsum: acc = acc + x (fp32 RN), out = acc"""
    nxt = _rn_add_f32(acc, x)
    tl.store(OUT + off, nxt, mask=m)
    return tl.where(m, nxt, acc)


@triton.jit
def _cumsum0_kernel(X, OUT, L, N, BLOCK: tl.constexpr):
    """OUT[t, j] = ((0 + X[0, j]) + X[1, j]) + ... + X[t, j] in fp32 RN, sequentially over t, X / OUT (L, N) fp32
    contiguous; 16 rows per tile, the next tile's loads issued before the current tile is summed"""
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    cm = cols < N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    x0 = tl.load(X + (0 + 0) * N + cols, mask=cm & (0 + 0 < L), other=0.0).to(tl.float32)
    x1 = tl.load(X + (0 + 1) * N + cols, mask=cm & (0 + 1 < L), other=0.0).to(tl.float32)
    x2 = tl.load(X + (0 + 2) * N + cols, mask=cm & (0 + 2 < L), other=0.0).to(tl.float32)
    x3 = tl.load(X + (0 + 3) * N + cols, mask=cm & (0 + 3 < L), other=0.0).to(tl.float32)
    x4 = tl.load(X + (0 + 4) * N + cols, mask=cm & (0 + 4 < L), other=0.0).to(tl.float32)
    x5 = tl.load(X + (0 + 5) * N + cols, mask=cm & (0 + 5 < L), other=0.0).to(tl.float32)
    x6 = tl.load(X + (0 + 6) * N + cols, mask=cm & (0 + 6 < L), other=0.0).to(tl.float32)
    x7 = tl.load(X + (0 + 7) * N + cols, mask=cm & (0 + 7 < L), other=0.0).to(tl.float32)
    x8 = tl.load(X + (0 + 8) * N + cols, mask=cm & (0 + 8 < L), other=0.0).to(tl.float32)
    x9 = tl.load(X + (0 + 9) * N + cols, mask=cm & (0 + 9 < L), other=0.0).to(tl.float32)
    x10 = tl.load(X + (0 + 10) * N + cols, mask=cm & (0 + 10 < L), other=0.0).to(tl.float32)
    x11 = tl.load(X + (0 + 11) * N + cols, mask=cm & (0 + 11 < L), other=0.0).to(tl.float32)
    x12 = tl.load(X + (0 + 12) * N + cols, mask=cm & (0 + 12 < L), other=0.0).to(tl.float32)
    x13 = tl.load(X + (0 + 13) * N + cols, mask=cm & (0 + 13 < L), other=0.0).to(tl.float32)
    x14 = tl.load(X + (0 + 14) * N + cols, mask=cm & (0 + 14 < L), other=0.0).to(tl.float32)
    x15 = tl.load(X + (0 + 15) * N + cols, mask=cm & (0 + 15 < L), other=0.0).to(tl.float32)
    for tile in range(0, L, 16):
        n0 = tl.load(X + (tile + 16 + 0) * N + cols, mask=cm & (tile + 16 + 0 < L), other=0.0).to(tl.float32)
        n1 = tl.load(X + (tile + 16 + 1) * N + cols, mask=cm & (tile + 16 + 1 < L), other=0.0).to(tl.float32)
        n2 = tl.load(X + (tile + 16 + 2) * N + cols, mask=cm & (tile + 16 + 2 < L), other=0.0).to(tl.float32)
        n3 = tl.load(X + (tile + 16 + 3) * N + cols, mask=cm & (tile + 16 + 3 < L), other=0.0).to(tl.float32)
        n4 = tl.load(X + (tile + 16 + 4) * N + cols, mask=cm & (tile + 16 + 4 < L), other=0.0).to(tl.float32)
        n5 = tl.load(X + (tile + 16 + 5) * N + cols, mask=cm & (tile + 16 + 5 < L), other=0.0).to(tl.float32)
        n6 = tl.load(X + (tile + 16 + 6) * N + cols, mask=cm & (tile + 16 + 6 < L), other=0.0).to(tl.float32)
        n7 = tl.load(X + (tile + 16 + 7) * N + cols, mask=cm & (tile + 16 + 7 < L), other=0.0).to(tl.float32)
        n8 = tl.load(X + (tile + 16 + 8) * N + cols, mask=cm & (tile + 16 + 8 < L), other=0.0).to(tl.float32)
        n9 = tl.load(X + (tile + 16 + 9) * N + cols, mask=cm & (tile + 16 + 9 < L), other=0.0).to(tl.float32)
        n10 = tl.load(X + (tile + 16 + 10) * N + cols, mask=cm & (tile + 16 + 10 < L), other=0.0).to(tl.float32)
        n11 = tl.load(X + (tile + 16 + 11) * N + cols, mask=cm & (tile + 16 + 11 < L), other=0.0).to(tl.float32)
        n12 = tl.load(X + (tile + 16 + 12) * N + cols, mask=cm & (tile + 16 + 12 < L), other=0.0).to(tl.float32)
        n13 = tl.load(X + (tile + 16 + 13) * N + cols, mask=cm & (tile + 16 + 13 < L), other=0.0).to(tl.float32)
        n14 = tl.load(X + (tile + 16 + 14) * N + cols, mask=cm & (tile + 16 + 14 < L), other=0.0).to(tl.float32)
        n15 = tl.load(X + (tile + 16 + 15) * N + cols, mask=cm & (tile + 16 + 15 < L), other=0.0).to(tl.float32)
        acc = _cum0_step(acc, x0, OUT, (tile + 0) * N + cols, cm & (tile + 0 < L))
        acc = _cum0_step(acc, x1, OUT, (tile + 1) * N + cols, cm & (tile + 1 < L))
        acc = _cum0_step(acc, x2, OUT, (tile + 2) * N + cols, cm & (tile + 2 < L))
        acc = _cum0_step(acc, x3, OUT, (tile + 3) * N + cols, cm & (tile + 3 < L))
        acc = _cum0_step(acc, x4, OUT, (tile + 4) * N + cols, cm & (tile + 4 < L))
        acc = _cum0_step(acc, x5, OUT, (tile + 5) * N + cols, cm & (tile + 5 < L))
        acc = _cum0_step(acc, x6, OUT, (tile + 6) * N + cols, cm & (tile + 6 < L))
        acc = _cum0_step(acc, x7, OUT, (tile + 7) * N + cols, cm & (tile + 7 < L))
        acc = _cum0_step(acc, x8, OUT, (tile + 8) * N + cols, cm & (tile + 8 < L))
        acc = _cum0_step(acc, x9, OUT, (tile + 9) * N + cols, cm & (tile + 9 < L))
        acc = _cum0_step(acc, x10, OUT, (tile + 10) * N + cols, cm & (tile + 10 < L))
        acc = _cum0_step(acc, x11, OUT, (tile + 11) * N + cols, cm & (tile + 11 < L))
        acc = _cum0_step(acc, x12, OUT, (tile + 12) * N + cols, cm & (tile + 12 < L))
        acc = _cum0_step(acc, x13, OUT, (tile + 13) * N + cols, cm & (tile + 13 < L))
        acc = _cum0_step(acc, x14, OUT, (tile + 14) * N + cols, cm & (tile + 14 < L))
        acc = _cum0_step(acc, x15, OUT, (tile + 15) * N + cols, cm & (tile + 15 < L))
        x0 = n0
        x1 = n1
        x2 = n2
        x3 = n3
        x4 = n4
        x5 = n5
        x6 = n6
        x7 = n7
        x8 = n8
        x9 = n9
        x10 = n10
        x11 = n11
        x12 = n12
        x13 = n13
        x14 = n14
        x15 = n15


@triton.jit
def _centred_tiled_kernel(X, S, CNT, OUT, L, N, C, sxt, sxb, sxc, sot, sob, soc, OUT_DTYPE: tl.constexpr,
                          TT: tl.constexpr, BC: tl.constexpr):
    """centred_running: BC columns (j = b * C + c of N = B * C) per program, the T tokens in tiles of TT; per tile the prefix
    sums by a parallel scan plus the carried sum of the earlier tiles, then the per-element division, difference and mask;
    X (L, B, C) of strides (sxt, sxb, sxc) read as fp32, S (B, C) fp32 contiguous (read, then written with the sum through
    the last token), CNT (B,) int64, OUT (L, B, C) OUT_DTYPE of strides (sot, sob, soc)"""
    pid = tl.program_id(0)
    cols = pid * BC + tl.arange(0, BC)
    cm = cols < N
    bb = cols // C
    cc = cols - bb * C
    xcol = bb * sxb + cc * sxc
    ocol = bb * sob + cc * soc
    cnt = tl.load(CNT + bb, mask=cm, other=0)
    carry = tl.load(S + cols, mask=cm, other=0.0)
    rows = tl.arange(0, TT)
    for t0 in range(0, L, TT):
        t = t0 + rows
        m = (t < L)[:, None] & cm[None, :]
        x = tl.load(X + t[:, None] * sxt + xcol[None, :], mask=m, other=0.0).to(tl.float32)
        inc = tl.cumsum(x, axis=0)
        run = carry[None, :] + (inc - x)                                 # the sum before token t
        pos = cnt[None, :] + t[:, None]
        q = _rn_div_f32(run, tl.maximum(pos.to(tl.float32), 1.0))
        o = (x - q) * tl.where(pos > 0, 1.0, 0.0)
        tl.store(OUT + t[:, None] * sot + ocol[None, :], o.to(OUT_DTYPE), mask=m)
        carry = carry + tl.sum(x, axis=0)
    tl.store(S + cols, carry, mask=cm)


# ================================================================================================ training launchers
def _launch_seq(x: torch.Tensor, y: torch.Tensor, reverse: bool) -> None:
    """Sequential scan of ``x`` along axis 0 into the contiguous ``y`` (same shape); ``x`` has the two-level layout."""
    T = int(x.shape[0])
    n_cols = x.numel() // T
    c, sxb = _two_level(x)
    grid = (triton.cdiv(n_cols, SEQ_BLOCK_N),)
    _cumsum0_u64_kernel[grid](x, y, T, n_cols, c, sxb, SXT=int(x.stride(0)), SYT=n_cols, REVERSE=int(bool(reverse)),
                              STEP=-1 if reverse else 1, BLOCK_N=SEQ_BLOCK_N, num_warps=SEQ_WARPS)


def _launch_chunk(x: torch.Tensor, y: torch.Tensor, reverse: bool) -> None:
    """Chunked scan of ``x`` along axis 0 into the contiguous ``y`` (same shape); ``x`` has the two-level layout."""
    T = int(x.shape[0])
    n_cols = x.numel() // T
    c, sxb = _two_level(x)
    tc = (T + CHUNK_CH - 1) // CHUNK_CH
    grid = (triton.cdiv(n_cols, CHUNK_BLOCK_N),)
    _cumsum0_chunk_kernel[grid](x, y, T, n_cols, c, sxb, int(x.stride(0)), n_cols, tc, REVERSE=int(bool(reverse)),
                                CH=CHUNK_CH, BLOCK_N=CHUNK_BLOCK_N, STAGES=CHUNK_STAGES, num_warps=CHUNK_WARPS)


@torch.library.custom_op("neuronspark::cumsum0", mutates_args=(), schema="(Tensor x, bool reverse) -> Tensor")
def _cumsum0_op(x, reverse):
    """Contiguous fp32 scan of ``x`` along axis 0 (ascending t, or descending t when ``reverse``): the sequential
    kernel for at most 32 columns, the chunked kernel otherwise.  ``x``: CUDA fp32 with more than one column."""
    if _two_level(x) is None:
        x = x.contiguous()        # ATen's expect_contiguous: a copy, the per-column order is unchanged
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n_cols = x.numel() // int(x.shape[0])
    if n_cols <= NARROW_N:
        _launch_seq(x, y, reverse)
    else:
        _launch_chunk(x, y, reverse)
    return y


@_cumsum0_op.register_fake
def _cumsum0_fake(x, reverse):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


class Cumsum0Fn(torch.autograd.Function):
    """cumsum(dim=0); backward = the reverse scan of the incoming gradient (autograd's flip-cumsum-flip structure; the
    gradient itself when it has <= 1 element or one time step, as cumsum_backward returns it)."""

    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return _cumsum0_op(x, False)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None
        if g.numel() <= 1 or g.shape[0] == 1:
            return g
        return _cumsum0_op(g, True)


def cumsum0(x: torch.Tensor) -> torch.Tensor:
    """``x.cumsum(dim=0)`` for a CUDA fp32 tensor with at least one axis and one element (training regime; see the
    module docstring for the narrow / wide / single-column forms)."""
    require_cuda(x, "cumsum0")
    if x.dtype != torch.float32:
        raise RuntimeError(f"neuronspark.cumsum0: a CUDA float32 tensor is required, got {x.dtype}")
    if x.dim() < 1 or x.numel() == 0:
        raise RuntimeError(f"neuronspark.cumsum0: at least one axis and one element are required, got {tuple(x.shape)}")
    if x.numel() == x.shape[0]:
        return x.cumsum(dim=0)
    return Cumsum0Fn.apply(x)


def cumsum0_ref(x: torch.Tensor) -> torch.Tensor:
    return x.cumsum(dim=0)


# ================================================================================================ prefill launchers
def _prefill_grid(n: int):
    return (triton.cdiv(int(n), PREFILL_BLOCK),)


def _dense(t: torch.Tensor) -> bool:
    """non-overlapping and dense: its strides are a permutation of a contiguous layout"""
    if t.is_contiguous():
        return True
    dims = sorted(range(t.dim()), key=lambda d: (int(t.stride(d)), -int(t.shape[d])))
    expect = 1
    for d in dims:
        if int(t.shape[d]) == 1:
            continue
        if int(t.stride(d)) != expect:
            return False
        expect *= int(t.shape[d])
    return True


def _cumsum0_takes(x: torch.Tensor) -> bool:
    """the inputs whose torch kernel is tensor_kernel_scan_outer_dim over dim 0 with fp32 accumulation: fp32 CUDA,
    >= 2 dimensions, int32 offsets, and not a tensor whose elements all lie along dim 0 (torch sends numel == size(0)
    to cub's parallel inclusive scan)"""
    return (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32 and x.dim() >= 2
            and x.numel() > 0 and x.numel() != int(x.shape[0])
            and x.numel() + 32 * (x.numel() // max(int(x.shape[0]), 1)) < 2 ** 31)


@torch.library.custom_op("neuronspark::exact_cumsum0", mutates_args=(), schema="(Tensor x) -> Tensor")
def _exact_cumsum0_op(x):
    xc = x if x.is_contiguous() else x.contiguous()       # torch's scan_dim scans a contiguous copy
    L = int(x.shape[0])
    N = x.numel() // L
    out = torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
    _cumsum0_kernel[_prefill_grid(N)](xc, out, L, N, BLOCK=PREFILL_BLOCK, num_warps=1, num_stages=1,
                                      enable_fp_fusion=False)
    return out


@_exact_cumsum0_op.register_fake
def _exact_cumsum0_fake(x):
    return torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)


def exact_cumsum0(x: torch.Tensor) -> torch.Tensor:
    """``torch.cumsum(x, dim=0)`` for an fp32 CUDA (T, N...) tensor with at least two dimensions and more than one
    column, one kernel (prefill regime, no autograd); returns a contiguous fp32 tensor."""
    if not _cumsum0_takes(x):
        raise RuntimeError(f"neuronspark.exact_cumsum0: an fp32 CUDA tensor of >= 2 dimensions with more than one "
                           f"column is required, got {getattr(x, 'dtype', None)} {tuple(getattr(x, 'shape', ()))}")
    return _exact_cumsum0_op(x)


def exact_cumsum0_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(x, dim=0)


def _centred_out(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    T, B, C = (int(s) for s in x.shape)
    # the torch chain's result has x's layout (x.float() keeps x's strides, the products take the dimension order of
    # their first operand, the casts preserve strides); for a dense x so does this one
    if _dense(x):
        return torch.empty_strided(tuple(x.shape), tuple(x.stride()), dtype=out_dtype, device=x.device)
    return torch.empty((T, B, C), dtype=out_dtype, device=x.device)


@torch.library.custom_op("neuronspark::centred_running", mutates_args=("run_sum",),
                         schema="(Tensor x, Tensor(a!) run_sum, Tensor count, ScalarType out_dtype) -> Tensor")
def _centred_running_op(x, run_sum, count, out_dtype):
    T, B, C = (int(s) for s in x.shape)
    out = _centred_out(x, out_dtype)
    sx, so = [int(v) for v in x.stride()], [int(v) for v in out.stride()]
    if (T + 512) * max(sx[0], so[0]) + B * max(sx[1], so[1]) + C * max(sx[2], so[2]) >= 2 ** 31 or min(sx + so) < 0:
        raise RuntimeError("neuronspark.centred_running: offsets beyond the int32 range")   # (a tile reaches T + 511)
    tt, bc, nw = CENTRED_NARROW if B * C < CENTRED_NARROW_N else CENTRED_WIDE
    _centred_tiled_kernel[(triton.cdiv(B * C, bc),)](x, run_sum, count, out, T, B * C, C, sx[0], sx[1], sx[2], so[0], so[1],
                                                     so[2], OUT_DTYPE=_OUT_TL[out_dtype], TT=tt, BC=bc, num_warps=nw)
    return out


@_centred_running_op.register_fake
def _centred_running_fake(x, run_sum, count, out_dtype):
    return _centred_out(x, out_dtype)


def _check_centred(x, run_sum, count, out_dtype, what):
    if not isinstance(x, torch.Tensor) or x.dim() != 3:
        raise RuntimeError(f"neuronspark.{what}: x must be a (T, B, C) tensor")
    if (not isinstance(run_sum, torch.Tensor) or tuple(run_sum.shape) != tuple(x.shape[1:])
            or run_sum.dtype != torch.float32):
        raise RuntimeError(f"neuronspark.{what}: input {tuple(x.shape)} for a carried sum "
                           f"{tuple(getattr(run_sum, 'shape', ()))} {getattr(run_sum, 'dtype', None)} "
                           "(expected fp32 (B, C))")
    B = int(x.shape[1])
    if not isinstance(count, torch.Tensor) or tuple(count.shape) != (B,) or count.dtype != torch.int64:
        raise RuntimeError(f"neuronspark.{what}: the count must be a ({B},) int64 tensor")
    if out_dtype not in _OUT_TL or x.dtype not in _OUT_TL:
        raise RuntimeError(f"neuronspark.{what}: dtypes {x.dtype} -> {out_dtype}")


def centred_running(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor,
                    out_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """The strictly-past running-mean centring of a prefill call (see the module docstring): ``x`` (T, B, C) CUDA fp32
    / bf16 / fp16 with T >= 2, ``run_sum`` the carried fp32 (B, C) contiguous sum (advanced in place), ``count`` the
    (B,) int64 contiguous row positions of the call's first token (not advanced), ``out_dtype`` the result dtype
    (default ``x.dtype``).  One pass, no autograd."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    _check_centred(x, run_sum, count, out_dtype, "centred_running")
    T = int(x.shape[0])
    if T < 2 or not (x.is_cuda and run_sum.is_cuda and count.is_cuda and run_sum.is_contiguous()
                     and count.is_contiguous()):
        raise RuntimeError(f"neuronspark.centred_running: T >= 2 CUDA operands with a contiguous carried sum / count "
                           f"only (T {T}, contiguous {run_sum.is_contiguous()} / {count.is_contiguous()})")
    return _centred_running_op(x, run_sum, count, out_dtype)


def centred_running_ref(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor,
                        out_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """The torch chain of the session running mean cast to ``out_dtype``; ``run_sum`` advanced in place."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    _check_centred(x, run_sum, count, out_dtype, "centred_running_ref")
    T, B = int(x.shape[0]), int(x.shape[1])
    xf = x.float()
    if T == 1:
        xcum = run_sum.unsqueeze(0)
        run = None
    else:
        run = torch.cat([run_sum.unsqueeze(0), xf], dim=0).cumsum(dim=0)      # (T + 1, B, C): S_t .. S_{t+T}
        xcum = run[:-1]
    pos = count.view(1, B, 1) + torch.arange(T, device=x.device, dtype=torch.int64).view(T, 1, 1)
    xn = pos.to(torch.float32).clamp_min(1.0)
    has_past = (pos > 0).to(torch.float32)
    out = (xf - xcum / xn) * has_past
    if run is None:
        run_sum.add_(xf[0])
    else:
        run_sum.copy_(run[-1])
    return out.to(out_dtype)
