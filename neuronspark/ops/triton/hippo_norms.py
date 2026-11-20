"""Row RMS norms of the hippocampal layer.

Operators (registry names):

* ``row_rmsnorm(x, weight, eps)`` (training): ``y = x * rsqrt(mean(x^2, -1) + eps)``, times ``weight`` when given, on a
  dense fp32 ``(T, B, C)`` x with a unit inner stride -> fp32, row-major (contiguous) whatever x's layout.
* ``row_rmsnorm_tee(x, weight, eps, fp32, bf16)`` (training): the same norm with the forms its consumers take:
  ``(x, y, ys16)`` = x itself (the residual stream handed on), the fp32 norm (when ``fp32``) and ``bf16`` (0, 1, 2)
  bf16 roundings of it (the operands of projections: no cast in the projection).  The norm's outputs are row-major
  (contiguous) whatever x's layout: the consumers that take (T * B, C) rows view them without a copy.  One forward
  kernel; one backward kernel sums the gradients of every output, the residual stream's included.  Used for
  the lexical-read norm (bf16 only), the frozen attention norm (fp32) and the FFN norm (``weight`` = the fp32 gain;
  fp32 for the neuron, two bf16 operands for the PKM projections).
* ``out_norm_gate(attn, w, gate_logits, eps)`` (training): ``sigmoid(gate_logits) * ((1 + w) * rmsnorm(attn))`` with a
  dense fp32 ``attn``, the zero-centred weight ``w`` (fp32 ``(C,)``) and bf16 gate logits -> fp32.
* ``rmsnorm_rows(x, eps, weight=None, zero_centered=False, out_dtype=None)`` (prefill / decode, no autograd):
  ``(float(x) * rsqrt(mean(float(x)^2, -1) + eps) [* w1]).to(out_dtype)`` over the last dim with ``w1 = weight`` or
  ``1 + weight`` (``zero_centered``); rows of width ``N % 128 == 0``, ``N <= 4096``; other widths raise (the 64-wide
  lexical q / k rows are formed by the module's torch lines).

Numerics contract
-----------------
Elementwise arithmetic is inline PTX with explicit IEEE rounding: ``add / sub / mul / div .rn.f32`` (no FMA contraction,
no flush-to-zero), exact conversions ``cvt.f32.bf16`` / ``cvt.rn.bf16.f32``, ``rsqrt.approx.f32`` (non-FTZ: the
instruction nvcc emits for ATen's ``::rsqrt(float)``), the sigmoid ``1 / (1 + expf(-x))`` with libdevice ``expf``; every
training kernel is compiled with ``enable_fp_fusion=False``.

* ``row_rmsnorm`` / ``row_rmsnorm_tee``: plain fp32 register sums and products in both directions (Triton's
  arithmetic; another summation order than the torch chain's reductions).
* ``out_norm_gate`` forward: the mean of squares is formed in the order of ATen's ``reduce_kernel`` for a reduction
  over the contiguous last dim with one warp per row (VEC-wide loads, VEC running sums per lane combined in slot order,
  then the ``shfl_down`` tree), times the host's fp32 ``1 / C``, i.e. in the order of operations of torch's ``pow +
  mean`` chain.  The layout conditions of that reduction path are checked; other layouts raise.
* ``out_norm_gate`` backward: one kernel forms the row sums of the ``r`` gradient's summands in registers (a register
  tree, not ATen's order), both contributions to ``x`` (the multiply's and the square's edges of torch's graph), and per
  program the column partial sums of the ``(C,)`` weight gradient, which one torch ``sum`` reduces: fp32
  reduction-order class.  The gate gradient follows ATen's bf16 ``SigmoidBackward0`` (three bf16 roundings).
* ``rmsnorm_rows``: one kernel per call: ``x * x`` (``mul.rn``), the sum of squares in ATen's vectorised 4-wide reduction
  order for a contiguous row, the host fp32 factor ``1 / N``, the fp32 ``eps`` add, libdevice ``rsqrt``, the products,
  the final rounding: the eager torch chain's order of operations.

Each launcher is registered as a ``torch.library`` custom op (``neuronspark::hippo_*``) with a fake implementation that
allocates the same layouts.  Pure-torch references ``*_ref`` (same parameters) are the eager torch chains: the
RMSNorm eager path, the frozen norm, the zero-centred out norm with its output gate.
"""
import struct
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (the glue arithmetic under short local names)
    _add_rn,
    _mul_rn,
    _rn_add_f32 as _g_add,
    _rn_div_f32 as _g_div,
    _rn_fma_f32 as _g_fma,
    _rn_mul_f32 as _g_mul,
    _rn_sqrt_f32 as _g_sqrt,
    _rn_sub_f32 as _g_sub,
    _sr_bf16_to_f32 as _g_w,
    _sr_f32_to_bf16 as _g_n,
    _sr_max as _g_max,
    _sr_neg as _g_neg,
    _sr_sigmoid as _g_sigmoid,
)

__all__ = [
    "RowNormFn", "row_rmsnorm", "row_rmsnorm_ref", "row_rmsnorm_tee", "row_rmsnorm_tee_ref",
    "OutNormGateFn", "out_norm_gate", "out_norm_gate_ref",
    "rmsnorm_rows", "rmsnorm_rows_ref", "rmsnorm_gate_rows", "rmsnorm_gate_rows_ref",
]

# ------------------------------------------------------------------------------------------------ fixed constants
# Launch configuration of the training regime (fixed):
FWDX_ROWS, FWDX_WARPS = 1, 4          # forwards of row_rmsnorm / out_norm_gate: rows per program, warps
ROW_BR = 4                            # rows per program of the backward kernels (fixes the weight-gradient partials)
ROW_WARPS = 8                         # warps of the backward kernels (row-loop form for row_rmsnorm / out_norm_gate)
XVEC = 4                              # reduce_kernel's input vector width for the fp32 mean (exact-order row sums)
XUP = True                            # shfl_down offsets 1, 2, .., 16 (adjacent pairs)
# Inference regime (rmsnorm_rows): ATen's vectorised reduction order (ORDER 0), 4 rows per program, 4 warps.
RMS_ROWS_PER_PROGRAM = 4
RMS_WARPS = 4
RMS_MAX_N = 4096


# ================================================================================================ glue arithmetic
@triton.jit
def _g_rsqrt(a):
    """rsqrt.approx.f32 (no FTZ): nvcc's code for ATen's rsqrt_kernel_cuda (::rsqrt(float))."""
    return tl.inline_asm_elementwise(asm="rsqrt.approx.f32 $0, $1;", constraints="=f,f", args=[a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _g_r16(x):
    """fp32 value rounded to bf16 and widened back (a bf16 op's result as the next op sees it)."""
    return _g_w(_g_n(x))


@triton.jit
def _g_bits(b):
    """the fp32 value of an int32 bit pattern passed from the host (exact constants: eps, 1 / C)."""
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _g_clamp_min(v, lim):
    """ATen launch_clamp_scalar (Min): isnan(v) ? v : ::max(v, lim)."""
    return tl.where(v != v, v, _g_max(v, lim))


@triton.jit
def _g_silu(x):
    """ATen silu functor (opmath float): x / (1 + exp(-x))."""
    return _g_div(x, _g_add(1.0, libdevice.exp(_g_neg(x))))


@triton.jit
def _g_silu_bwd(dy, x):
    """ATen silu_backward functor: s = 1 / (1 + exp(-x)); dy * s * (1 + x * (1 - s)), the 1 + x * (.) one fma."""
    sg = _g_div(1.0, _g_add(1.0, libdevice.exp(_g_neg(x))))
    return _g_mul(_g_mul(dy, sg), _g_fma(x, _g_sub(1.0, sg), 1.0))


# ================================================================================================ exact-order row sum
@triton.jit
def _xpair(v, N: tl.constexpr, UP: tl.constexpr):
    """one shfl_down level on (BR, N) lane values -> (BR, N / 2): out[m] = v[2m] + v[2m + 1] (UP: offset 1 of this
    level's lane spacing) or v[m] + v[m + N / 2] (halves), each output a tl.sum with exactly two non-zero terms
    (= one IEEE add; the other terms are +0)."""
    i = tl.arange(0, N)
    m = tl.arange(0, N // 2)
    if UP:
        sel = (i[None, :] // 2) == m[:, None]
    else:
        sel = (i[None, :] % (N // 2)) == m[:, None]
    return tl.sum(tl.where(sel[None, :, :], v[:, None, :], 0.0), axis=2)


@triton.jit
def _xrow_sum2(P, roff, rm, RHO, MODE: tl.constexpr, IS16: tl.constexpr, NJ: tl.constexpr, VEC: tl.constexpr,
               UP: tl.constexpr, BR: tl.constexpr):
    """The fp32 row sums of BR channel-contiguous rows of NJ * 32 * VEC elements (row r at P + roff[r]) in the order of
    at::native::reduce_kernel for a reduction over the contiguous last dim with one warp per output row: lane l reads
    the VEC-vectors l, l + 32, ... and keeps VEC running sums (one per vector slot, from 0), combines them in slot order
    (((s0 + s1) + s2) + ...), then the warp_shfl_down tree (_xpair per level).  The (BR, 32, VEC) running sums are a 3-D
    tile; slot k = a masked sum with one non-zero term (exact).  Summands: MODE 0 = the element (bf16 widened when IS16),
    1 = its square (x * x rounded, ATen's pow(x, 2)), 2 = bf16(float(y) * rho[r])."""
    lane = tl.arange(0, 32)
    sl = tl.arange(0, VEC)
    off = (lane[:, None] * VEC + sl[None, :]).to(tl.int64)
    acc = tl.zeros([BR, 32, VEC], dtype=tl.float32)
    if MODE == 2:
        rho = tl.broadcast_to(RHO[:, None, None], (BR, 32, VEC))
    for j in tl.static_range(NJ):
        v = tl.load(P + (roff[:, None, None] + (j * 32 * VEC + off)[None, :, :]), mask=rm[:, None, None], other=0.0)
        if IS16:
            v = _g_w(v)
        if MODE == 1:
            v = _g_mul(v, v)
        if MODE == 2:
            v = _g_r16(_g_mul(v, rho))
        acc = _g_add(acc, v)
    s3 = sl[None, None, :]
    v = tl.sum(tl.where(s3 == 0, acc, 0.0), axis=2)
    for k in tl.static_range(1, VEC):
        v = _g_add(v, tl.sum(tl.where(s3 == k, acc, 0.0), axis=2))
    v = _xpair(v, 32, UP)
    v = _xpair(v, 16, UP)
    v = _xpair(v, 8, UP)
    v = _xpair(v, 4, UP)
    v = _xpair(v, 2, UP)
    return tl.sum(v, axis=1)


# ================================================================================================ training kernels
@triton.jit
def _rn_fwd_kernel(X, W, Y32, Y16, RR, B, D, eps, xs0, xs1, ys0, ys1, hs0, hs1,
                   HAS_W: tl.constexpr, OUT32: tl.constexpr, OUT16: tl.constexpr, BD: tl.constexpr):
    """One (t, b) row: y = x * rsqrt(mean(x^2) + eps) [* w], written as fp32 (Y32) and / or rounded to bf16 (Y16); the
    row factor stored at RR[t * B + b]."""
    r = tl.program_id(0).to(tl.int64)
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD)
    cm = c < D
    x = tl.load(X + (t * xs0 + b * xs1) + c, mask=cm, other=0.0)
    rr = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    tl.store(RR + r, rr)
    y = x * rr
    if HAS_W:
        y = tl.load(W + c, mask=cm, other=0.0) * y
    if OUT32:
        tl.store(Y32 + (t * ys0 + b * ys1) + c, y, mask=cm)
    if OUT16:
        tl.store(Y16 + (t * hs0 + b * hs1) + c, y.to(tl.bfloat16), mask=cm)


@triton.jit
def _rn_bwd_kernel(G32, G16A, G16B, GRES, X, RR, W, GX, PW, R, B, D,
                   as0, as1, as2, bs0, bs1, bs2, cs0, cs1, cs2, ds0, ds1, ds2, xs0, xs1, os0, os1,
                   HAS_W: tl.constexpr, HAS32: tl.constexpr, N16: tl.constexpr, HAS_RES: tl.constexpr,
                   BR: tl.constexpr, BD: tl.constexpr):
    """BR rows per program.  g = the sum of the gradients of y (fp32 G32, bf16 G16A / G16B widened); with a weight
    PW[program, c] = sum_rows g (x rr) and g <- g w; gx = g rr - (sum_c g x) rr^3 x / D [+ GRES, the gradient that
    reaches x beside the norm]."""
    pid = tl.program_id(0).to(tl.int64)
    r = pid * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD)
    cm = c < D
    m = rm[:, None] & cm[None, :]
    c64 = c.to(tl.int64)
    g = tl.zeros([BR, BD], dtype=tl.float32)
    if HAS32:
        g += tl.load(G32 + (t * as0 + b * as1)[:, None] + c64[None, :] * as2, mask=m, other=0.0)
    if N16 > 0:
        g += tl.load(G16A + (t * bs0 + b * bs1)[:, None] + c64[None, :] * bs2, mask=m, other=0.0).to(tl.float32)
    if N16 > 1:
        g += tl.load(G16B + (t * cs0 + b * cs1)[:, None] + c64[None, :] * cs2, mask=m, other=0.0).to(tl.float32)
    x = tl.load(X + (t * xs0 + b * xs1)[:, None] + c[None, :], mask=m, other=0.0)
    rr = tl.load(RR + r, mask=rm, other=0.0)
    if HAS_W:
        tl.store(PW + pid * D + c, tl.sum(g * (x * rr[:, None]), axis=0), mask=cm)
        g = g * tl.load(W + c, mask=cm, other=0.0)[None, :]
    sr = tl.sum(g * x, axis=1) * (rr * rr * rr) / D
    gx = g * rr[:, None] - sr[:, None] * x
    if HAS_RES:
        gx += tl.load(GRES + (t * ds0 + b * ds1)[:, None] + c64[None, :] * ds2, mask=m, other=0.0)
    tl.store(GX + (t * os0 + b * os1)[:, None] + c[None, :], gx, mask=m)


@triton.jit
def _og_fwd_kernel(X, W1, SG, O, VAR, R, B, D, EPSB, MINV,
                   xs0, xs1, xs2, ss0, ss1, ss2, os0, os1, os2, vs0, vs1,
                   BR: tl.constexpr, BD: tl.constexpr,
                   NJ: tl.constexpr, XVEC: tl.constexpr, XUP: tl.constexpr):
    """o = bf16(sigmoid(s)) * (w1 * (x * rsqrt(mean(x^2) + eps))); var written."""
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < D
    m = rm[:, None] & cm[None, :]
    var = _g_mul(_xrow_sum2(X, t * xs0 + b * xs1, rm, MINV, 1, False, NJ, XVEC, XUP, BR), _g_bits(MINV))
    x = tl.load(X + (t[:, None] * xs0 + b[:, None] * xs1 + c[None, :] * xs2), mask=m, other=0.0)
    tl.store(VAR + (t * vs0 + b * vs1), var, mask=rm)
    rr = tl.broadcast_to(_g_rsqrt(_g_add(var, _g_bits(EPSB)))[:, None], (BR, BD))
    w1 = tl.broadcast_to(tl.load(W1 + c, mask=cm, other=0.0)[None, :], (BR, BD))
    y = _g_mul(w1, _g_mul(x, rr))
    sg = _g_r16(_g_sigmoid(_g_w(tl.load(SG + (t[:, None] * ss0 + b[:, None] * ss1 + c[None, :] * ss2), mask=m,
                                        other=0.0))))
    tl.store(O + (t[:, None] * os0 + b[:, None] * os1 + c[None, :] * os2), _g_mul(sg, y), mask=m)


@triton.jit
def _og_bwd_loop_kernel(G, X, VAR, W1, SG, DS, GX, PW, R, B, D, EPSB, INVB,
                        gs0, gs1, gs2, xs0, xs1, xs2, vs0, vs1, ss0, ss1, ss2, ds0, ds1, ds2, os0, os1, os2,
                        BR: tl.constexpr, BD: tl.constexpr):
    """out_norm_gate backward, one [1, BD] row per iteration: the gate-logit gradient (bf16 SigmoidBackward0), the
    weight's column partial (row by row from row 0), x's gradient (both edges)."""
    pid = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < D
    w1 = tl.broadcast_to(tl.load(W1 + c, mask=cm, other=0.0)[None, :], (1, BD))
    for i in tl.static_range(BR):
        r = pid * BR + i
        rv = r < R
        t = r // B
        b = r - t * B
        m = cm[None, :] & rv
        g = tl.load(G + (t * gs0 + b * gs1 + c[None, :] * gs2), mask=m, other=0.0)
        x = tl.load(X + (t * xs0 + b * xs1 + c[None, :] * xs2), mask=m, other=0.0)
        var = tl.load(VAR + (t * vs0 + b * vs1) + tl.zeros([1], dtype=tl.int64),
                      mask=rv & (tl.zeros([1], dtype=tl.int32) == 0), other=1.0)
        r1 = _g_rsqrt(_g_add(var, _g_bits(EPSB)))
        rr = tl.broadcast_to(r1[:, None], (1, BD))
        xn = _g_mul(x, rr)
        y = _g_mul(w1, xn)
        sg = _g_r16(_g_sigmoid(_g_w(tl.load(SG + (t * ss0 + b * ss1 + c[None, :] * ss2), mask=m, other=0.0))))
        gsg = _g_r16(_g_mul(g, y))
        # SigmoidBackward0 on bf16: ATen's functor a * (1 - b) * b in c10::BFloat16 arithmetic (three roundings)
        ds = _g_r16(_g_mul(_g_r16(_g_mul(gsg, _g_r16(_g_sub(1.0, sg)))), sg))
        tl.store(DS + (t * ds0 + b * ds1 + c[None, :] * ds2), _g_n(ds), mask=m)
        gy = _g_mul(g, sg)
        pr = _g_mul(gy, xn)
        if i == 0:
            pacc = pr
        else:
            pacc = _g_add(pacc, pr)
        g2 = _g_mul(gy, w1)
        sr = tl.sum(_g_mul(g2, x), axis=1)
        gt = _g_mul(_g_mul(sr, -0.5), _g_mul(_g_mul(r1, r1), r1))
        gm = tl.broadcast_to(_g_mul(gt, _g_bits(INVB))[:, None], (1, BD))
        gx = _g_add(_g_mul(g2, rr), _g_mul(gm, _g_mul(x, 2.0)))
        tl.store(GX + (t * os0 + b * os1 + c[None, :] * os2), gx, mask=m)
    tl.store(PW + (pid * D + c[None, :]), pacc, mask=cm[None, :])


# ================================================================================================ inference kernel
@triton.jit
def _lane_tree(v, R: tl.constexpr):
    a, b = tl.split(tl.reshape(v, [R, 16, 2]))
    v = _add_rn(a, b)
    a, b = tl.split(tl.reshape(v, [R, 8, 2]))
    v = _add_rn(a, b)
    a, b = tl.split(tl.reshape(v, [R, 4, 2]))
    v = _add_rn(a, b)
    a, b = tl.split(tl.reshape(v, [R, 2, 2]))
    v = _add_rn(a, b)
    a, b = tl.split(tl.reshape(v, [R, 1, 2]))
    v = _add_rn(a, b)
    return tl.reshape(v, [R])


@triton.jit
def _four_slots(acc, R: tl.constexpr):
    # acc: (R, 32, 4) [lane, slot] -> (R, 32): ((s0 + s1) + s2) + s3
    e, o = tl.split(tl.reshape(acc, [R, 32, 2, 2]))
    s0, s2 = tl.split(e)
    s1, s3 = tl.split(o)
    return _add_rn(_add_rn(_add_rn(s0, s1), s2), s3)


@triton.jit
def _rmsnorm_kernel(X, W, Y, sx, sy, N, factor, eps, R: tl.constexpr, C: tl.constexpr, HAS_W: tl.constexpr,
                    OUT_BF16: tl.constexpr, G=None, sg=0, HAS_G: tl.constexpr = False):
    """R rows of C chunks of 128: the sum of squares with lane l, slot i, chunk c -> element 128 c + 4 l + i (ATen's
    vectorised order), the slots combined in order, the lane tree; then the scaled row [* w] [* bf16(sigmoid(g))],
    cast."""
    pid = tl.program_id(axis=0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < N
    offs128 = tl.arange(0, 128)
    base = X + rows.to(tl.int64)[:, None] * sx
    acc = tl.zeros([R, 128], dtype=tl.float32)
    for c in tl.static_range(C):
        xc = tl.load(base + c * 128 + offs128[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        acc = _add_rn(acc, _mul_rn(xc, xc))
    s = _lane_tree(_four_slots(tl.reshape(acc, [R, 32, 4]), R), R)                 # (R,)
    ms = _mul_rn(s, factor)                                                        # MeanOps::project
    r = libdevice.rsqrt(_add_rn(ms, eps))                                          # torch.rsqrt(ms + eps)
    for c in tl.static_range(C):
        xc = tl.load(base + c * 128 + offs128[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        y = _mul_rn(xc, r[:, None])
        if HAS_W:
            w = tl.load(W + c * 128 + offs128)
            y = _mul_rn(y, w[None, :])
        if HAS_G:
            # the output gate: torch.sigmoid of the bf16 gate logits (one / (one + expf(-x)) in fp32, rounded to bf16)
            # times the fp32 norm
            gx = tl.load(G + rows.to(tl.int64)[:, None] * sg + c * 128 + offs128[None, :], mask=rmask[:, None],
                         other=0.0).to(tl.float32)
            y = _mul_rn(_g_sigmoid(gx).to(tl.bfloat16).to(tl.float32), y)
        yo = Y + rows.to(tl.int64)[:, None] * sy + c * 128 + offs128[None, :]
        if OUT_BF16:
            tl.store(yo, y.to(tl.bfloat16), mask=rmask[:, None])
        else:
            tl.store(yo, y, mask=rmask[:, None])


# ================================================================================================ host helpers
def _f32_bits(v) -> int:
    """int32 bit pattern of float32(v) (the double -> float rounding of C)."""
    return struct.unpack("<i", struct.pack("<f", float(np.float32(v))))[0]


def _invb(D: int) -> int:
    """bit pattern of the host's fp32 factor 1 / D (ATen's MeanOps factor)."""
    return _f32_bits(np.float32(1.0) / np.float32(D))


def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


def _meta(t: torch.Tensor, shape=None, stride=None, dtype=None) -> torch.Tensor:
    """a meta tensor of t's (or the given) shape / strides / dtype: torch's ops on it give torch's output layouts."""
    return torch.empty_strided(tuple(t.shape) if shape is None else shape,
                               tuple(t.stride()) if stride is None else stride,
                               dtype=t.dtype if dtype is None else dtype, device="meta")


def _rows(shape) -> Tuple[int, int, int, int]:
    T, B, D = (int(v) for v in shape)
    return T, B, D, T * B


def _s3(t: torch.Tensor):
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2))


def _s2(t: torch.Tensor):
    """strides of a (T, B, 1) tensor on (t, b)."""
    return int(t.stride(0)), int(t.stride(1))


def _var_like(xm: torch.Tensor, device) -> torch.Tensor:
    """the (T, B, 1) fp32 var tensor with the layout x.pow(2).mean(-1, keepdim=True) gives (xm: x on meta)."""
    vm = xm.pow(2).mean(-1, keepdim=True)
    return torch.empty_strided(tuple(vm.shape), tuple(vm.stride()), dtype=torch.float32, device=device)


def _xcfg(red_m: torch.Tensor, src: torch.Tensor, D: int, R: int, vec: int) -> Tuple[int, int]:
    """(NJ, VEC) of the exact-order row sum for a row sum that ATen forms with reduce_kernel on red_m (meta tensor of the
    reduced operand) while the kernel reads src; raises when reduce_kernel would not take the one-warp-per-row
    vectorized path the kernel reproduces."""
    es = red_m.element_size()
    st = tuple(int(v) for v in red_m.stride())
    lp2 = 1 << (max(int(R), 1).bit_length() - 1)
    if not (st[2] == 1 and int(src.stride(2)) == 1 and D > 128 and D % (32 * vec) == 0
            and (st[0] * es) % (vec * es) == 0 and (st[1] * es) % (vec * es) == 0
            and D // 32 < min(256, 16 * min(16, lp2)) and R >= 16):
        raise RuntimeError(f"neuronspark.hippo_norms: reduce layout {tuple(red_m.shape)}/{st} (source stride "
                           f"{tuple(src.stride())}, D {D}, vec {vec}) is not the one-warp-per-row vectorized case "
                           "the exact-order row sum reproduces")
    return D // (32 * vec), vec


def _check_dense3(x: torch.Tensor, dtype: torch.dtype, what: str) -> None:
    require_cuda(x, what)
    if not (x.dtype == dtype and x.dim() == 3 and x.numel() > 0 and x.numel() < 2 ** 31):
        raise RuntimeError(f"neuronspark.{what}: a non-empty 3-D {dtype} CUDA tensor below 2^31 elements is required, "
                           f"got {x.dtype} {tuple(x.shape)}")
    if torch.empty_like(x).stride() != x.stride():
        raise RuntimeError(f"neuronspark.{what}: strides {tuple(x.stride())} are not dense (empty_like does not keep "
                           "them)")


# ================================================================================================ row_rmsnorm
def _zero(device, dtype) -> torch.Tensor:
    return torch.empty((0,), dtype=dtype, device=device)


@torch.library.custom_op("neuronspark::hippo_rownorm_fwd", mutates_args=(),
                         schema="(Tensor x, Tensor? weight, float eps, bool out32, bool out16) -> (Tensor, Tensor, Tensor)")
def _rownorm_fwd(x, weight, eps, out32, out16):
    """(y fp32, y bf16, rr): the norm, row-major (T, B, C), as fp32 and / or bf16 (an output that is not asked for is
    empty) and the (T, B) fp32 row factors."""
    T, B, D, R = _rows(x.shape)
    y32 = torch.empty((T, B, D) if out32 else (0,), dtype=torch.float32, device=x.device)
    y16 = torch.empty((T, B, D) if out16 else (0,), dtype=torch.bfloat16, device=x.device)
    rr = torch.empty((T, B), dtype=torch.float32, device=x.device)
    _rn_fwd_kernel[(R,)](x, weight if weight is not None else x, y32, y16, rr, B, D, float(eps), *_s2(x),
                         *(_s2(y32) if out32 else (0, 0)), *(_s2(y16) if out16 else (0, 0)),
                         HAS_W=weight is not None, OUT32=bool(out32), OUT16=bool(out16), BD=_pow2(D),
                         num_warps=FWDX_WARPS)
    return y32, y16, rr


@_rownorm_fwd.register_fake
def _rownorm_fwd_fake(x, weight, eps, out32, out16):
    shape = tuple(x.shape)
    return (x.new_empty(shape if out32 else (0,), dtype=torch.float32),
            x.new_empty(shape if out16 else (0,), dtype=torch.bfloat16),
            x.new_empty(shape[:2], dtype=torch.float32))


@torch.library.custom_op("neuronspark::hippo_rownorm_bwd", mutates_args=(),
                         schema="(Tensor? g32, Tensor? g16a, Tensor? g16b, Tensor? gres, Tensor x, Tensor rr, "
                                "Tensor? weight) -> (Tensor, Tensor)")
def _rownorm_bwd(g32, g16a, g16b, gres, x, rr, weight):
    """(gx, pw): x's gradient in x's layout (the norm's edges and gres summed in the kernel) and the (D,) weight
    gradient ((0,) without weight).  g16a is given whenever g16b is."""
    T, B, D, R = _rows(x.shape)
    has_w = weight is not None
    gx = torch.empty_like(x)
    np_ = triton.cdiv(R, ROW_BR)
    pwp = torch.empty((np_, D), dtype=torch.float32, device=x.device) if has_w else None
    n16 = (g16a is not None) + (g16b is not None)
    _rn_bwd_kernel[(np_,)](
        g32 if g32 is not None else x, g16a if g16a is not None else x, g16b if g16b is not None else x,
        gres if gres is not None else x, x, rr, weight if has_w else x, gx, pwp if has_w else gx, R, B, D,
        *(_s3(g32) if g32 is not None else (0, 0, 0)), *(_s3(g16a) if g16a is not None else (0, 0, 0)),
        *(_s3(g16b) if g16b is not None else (0, 0, 0)), *(_s3(gres) if gres is not None else (0, 0, 0)),
        *_s2(x), *_s2(gx), HAS_W=has_w, HAS32=g32 is not None, N16=n16, HAS_RES=gres is not None,
        BR=ROW_BR, BD=_pow2(D), num_warps=ROW_WARPS)
    return gx, (pwp.sum(0) if has_w else _zero(x.device, torch.float32))


@_rownorm_bwd.register_fake
def _rownorm_bwd_fake(g32, g16a, g16b, gres, x, rr, weight):
    return torch.empty_like(x), x.new_empty((int(x.shape[-1]) if weight is not None else 0,), dtype=torch.float32)


class RowNormFn(torch.autograd.Function):
    """The row norm y = x * rsqrt(mean(x^2, -1) + eps) [* weight] of a dense fp32 (T, B, C) x with the outputs
    ``(x, [y fp32], [y bf16] * n16)``: x itself (the residual stream, handed on so that its gradient returns here), the
    norm as fp32 when ``fp32``, and ``n16`` (0, 1 or 2) bf16 roundings of the norm that share one buffer (one per bf16
    consumer: each consumer's gradient arrives on its own).  Backward: one kernel sums the gradients of the norm's
    outputs in fp32, forms the norm's two edges to x, adds the residual stream's gradient and writes x's gradient; the
    weight's column partials are reduced by one torch sum."""

    @staticmethod
    def forward(ctx, x, weight, eps, fp32, n16):
        y32, y16, rr = _rownorm_fwd(x, weight, float(eps), bool(fp32), n16 > 0)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, rr, weight)
        ctx.outs = (bool(fp32), int(n16))
        outs = [x]
        if fp32:
            outs.append(y32)
        if n16 > 0:
            outs.append(y16)
        if n16 > 1:
            outs.append(y16.detach())                 # the same buffer as a second output
        return tuple(outs)

    @staticmethod
    def backward(ctx, gres, *gs):
        fp32, n16 = ctx.outs
        x, rr, weight = ctx.saved_tensors
        g32 = gs[0] if fp32 else None
        g16 = [g for g in gs[(1 if fp32 else 0):] if g is not None]
        if g32 is None and not g16:
            return gres, None, None, None, None
        for g, dt in ((g32, torch.float32), *((g, torch.bfloat16) for g in g16), (gres, torch.float32)):
            if g is not None and (g.dtype != dt or g.shape != x.shape):
                raise RuntimeError(f"neuronspark.row_rmsnorm backward: gradient {g.dtype} {tuple(g.shape)} for "
                                   f"{dt} {tuple(x.shape)}")
        gx, pw = _rownorm_bwd(g32, g16[0] if g16 else None, g16[1] if len(g16) > 1 else None, gres, x, rr, weight)
        return gx, (pw if weight is not None else None), None, None, None


def _check_rownorm(x: torch.Tensor, weight: Optional[torch.Tensor], what: str) -> None:
    _check_dense3(x, torch.float32, what)
    if x.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.{what}: x needs a unit inner stride, got {tuple(x.stride())}")
    if weight is not None and (weight.dtype != torch.float32 or weight.dim() != 1 or not weight.is_contiguous()
                               or int(weight.shape[0]) != int(x.shape[-1]) or weight.device != x.device):
        raise RuntimeError(f"neuronspark.{what}: weight must be a contiguous fp32 ({x.shape[-1]},) tensor on x's "
                           f"device, got {weight.dtype} {tuple(weight.shape)}")


def row_rmsnorm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """``x * rsqrt(mean(x^2, -1) + eps)`` [``* weight``] for a dense fp32 (T, B, C) CUDA ``x`` (unit inner stride, any
    (t, b) strides: the trunk's (T, B, C) view of a (B, T, C) tensor) -> fp32, row-major.  ``weight``: None (gain 1)
    or an fp32 (C,) tensor.  Training regime."""
    _check_rownorm(x, weight, "row_rmsnorm")
    return RowNormFn.apply(x, weight, float(eps), True, 0)[1]


def row_rmsnorm_tee(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float, fp32: bool,
                    bf16: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]:
    """The row norm of the residual stream with its consumers' forms: ``(x, y, ys16)``.

    ``x`` as for :func:`row_rmsnorm`.  Returns ``x`` (the same values; use it as the residual stream after this call,
    so that its gradient and the norm's are summed in one backward kernel), ``y`` the fp32 norm, row-major (None
    unless ``fp32``) and ``ys16``, ``bf16`` (0, 1 or 2) bf16 roundings of the norm, row-major (one buffer; one
    tensor per bf16 consumer, e.g. the operand of a projection).  Training regime."""
    _check_rownorm(x, weight, "row_rmsnorm_tee")
    bf16 = int(bf16)
    if bf16 not in (0, 1, 2) or not (fp32 or bf16):
        raise RuntimeError(f"neuronspark.row_rmsnorm_tee: fp32 = {fp32}, bf16 = {bf16}: at least one output, at most two "
                           "bf16 outputs")
    outs = RowNormFn.apply(x, weight, float(eps), bool(fp32), bf16)
    return outs[0], (outs[1] if fp32 else None), tuple(outs[(2 if fp32 else 1):])


def row_rmsnorm_ref(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """The eager torch chain (frozen norm / RMSNorm eager path): fp32 result."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = weight * y
    return y


def row_rmsnorm_tee_ref(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float, fp32: bool,
                        bf16: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]]:
    """The torch form of :func:`row_rmsnorm_tee`: x itself, the eager chain's norm, and one bf16 cast of it per bf16
    output."""
    y = row_rmsnorm_ref(x, weight, eps)
    return x, (y if fp32 else None), tuple(y.to(torch.bfloat16) for _ in range(int(bf16)))


# ================================================================================================ out_norm_gate
def _ogate_fwd_alloc(x, w1, sgl):
    var = _var_like(_meta(x), x.device)
    om = torch.sigmoid(_meta(sgl)) * (_meta(w1) * (_meta(x) * _meta(var)))
    out = torch.empty_strided(tuple(x.shape), om.stride(), dtype=torch.float32, device=x.device)
    return out, var


@torch.library.custom_op("neuronspark::hippo_outnorm_gate_fwd", mutates_args=(),
                         schema="(Tensor x, Tensor w1, Tensor sgl, float eps) -> (Tensor, Tensor)")
def _ogate_fwd(x, w1, sgl, eps):
    T, B, D, R = _rows(x.shape)
    nj, vec = _xcfg(_meta(x).pow(2), x, D, R, XVEC)
    out, var = _ogate_fwd_alloc(x, w1, sgl)
    _og_fwd_kernel[(triton.cdiv(R, FWDX_ROWS),)](
        x, w1, sgl, out, var, R, B, D, _f32_bits(eps), _invb(D), *_s3(x), *_s3(sgl), *_s3(out), *_s2(var),
        BR=FWDX_ROWS, BD=_pow2(D), NJ=nj, XVEC=vec, XUP=XUP, num_warps=FWDX_WARPS, enable_fp_fusion=False)
    return out, var


@_ogate_fwd.register_fake
def _ogate_fwd_fake(x, w1, sgl, eps):
    return _ogate_fwd_alloc(x, w1, sgl)


def _ogate_bwd_alloc(g, x, var, w1, sgl):
    gm_, xm, vm, wm, sm = _meta(g), _meta(x), _meta(var), _meta(w1), _meta(sgl)
    sigm = torch.sigmoid(sm)
    gsg = (gm_ * (wm * (xm * vm))).to(torch.bfloat16)
    ds_st = torch.ops.aten.sigmoid_backward(gsg, sigm).stride()
    gx_st = ((gm_ * sigm) * wm * vm).stride()
    ds = torch.empty_strided(tuple(x.shape), ds_st, dtype=torch.bfloat16, device=x.device)
    gx = torch.empty_strided(tuple(x.shape), gx_st, dtype=torch.float32, device=x.device)
    pw = torch.empty((int(x.shape[-1]),), dtype=torch.float32, device=x.device)
    return gx, pw, ds


@torch.library.custom_op("neuronspark::hippo_outnorm_gate_bwd", mutates_args=(),
                         schema="(Tensor g, Tensor x, Tensor var, Tensor w1, Tensor sgl, float eps) "
                                "-> (Tensor, Tensor, Tensor)")
def _ogate_bwd(g, x, var, w1, sgl, eps):
    """(gx, pw, ds): x's gradient, the (D,) gradient of (1 + w), the bf16 gate-logit gradient."""
    T, B, D, R = _rows(x.shape)
    gx, _, ds = _ogate_bwd_alloc(g, x, var, w1, sgl)
    np_ = triton.cdiv(R, ROW_BR)
    pwp = torch.empty((np_, D), dtype=torch.float32, device=x.device)
    _og_bwd_loop_kernel[(np_,)](g, x, var, w1, sgl, ds, gx, pwp, R, B, D, _f32_bits(eps), _invb(D), *_s3(g), *_s3(x),
                                *_s2(var), *_s3(sgl), *_s3(ds), *_s3(gx), BR=ROW_BR, BD=_pow2(D),
                                num_warps=ROW_WARPS, enable_fp_fusion=False)
    return gx, pwp.sum(0), ds


@_ogate_bwd.register_fake
def _ogate_bwd_fake(g, x, var, w1, sgl, eps):
    return _ogate_bwd_alloc(g, x, var, w1, sgl)


class OutNormGateFn(torch.autograd.Function):
    """sigmoid(s) * (w1 * (x * rsqrt(mean(x^2, -1) + eps))): x fp32 dense 3-D, w1 = (1 + w) fp32 (D,), s the bf16 gate
    logits (the sigmoid evaluated in bf16, as ATen's bf16 functor)."""

    @staticmethod
    def forward(ctx, x, w1, sgl, eps):
        out, var = _ogate_fwd(x, w1, sgl, float(eps))
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, var, w1, sgl)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None, None
        x, var, w1, sgl = ctx.saved_tensors
        if g.dtype != torch.float32 or g.shape != x.shape:
            raise RuntimeError(f"neuronspark.out_norm_gate backward: gradient {g.dtype} {tuple(g.shape)} for "
                               f"{tuple(x.shape)}")
        gx, pw, ds = _ogate_bwd(g, x, var, w1, sgl, ctx.eps)
        return gx, pw, ds, None


def out_norm_gate(attn: torch.Tensor, w: torch.Tensor, gate_logits: torch.Tensor, eps: float) -> torch.Tensor:
    """``sigmoid(gate_logits) * ((1 + w) * (attn * rsqrt(mean(attn^2, -1) + eps)))`` -> fp32.

    ``attn`` dense fp32 (T, B, C) (the concatenated head reads), ``w`` the zero-centred norm weight fp32 (C,),
    ``gate_logits`` bf16 (T, B, C) (the output-gate projection of the layer input ``h``).  ``1 + w`` is formed here by
    torch (its gradient passes to ``w`` unchanged).  Training regime."""
    _check_dense3(attn, torch.float32, "out_norm_gate")
    D = int(attn.shape[-1])
    if not (isinstance(w, torch.Tensor) and w.dtype == torch.float32 and w.dim() == 1 and int(w.shape[0]) == D
            and w.device == attn.device):
        raise RuntimeError(f"neuronspark.out_norm_gate: w must be an fp32 ({D},) tensor on attn's device")
    if not (isinstance(gate_logits, torch.Tensor) and gate_logits.dtype == torch.bfloat16
            and gate_logits.shape == attn.shape and gate_logits.device == attn.device):
        raise RuntimeError(f"neuronspark.out_norm_gate: the gate logits must be bf16 {tuple(attn.shape)}, got "
                           f"{getattr(gate_logits, 'dtype', None)} {tuple(getattr(gate_logits, 'shape', ()))}")
    w1 = 1.0 + w
    return OutNormGateFn.apply(attn, w1, gate_logits, float(eps))


def out_norm_gate_ref(attn: torch.Tensor, w: torch.Tensor, gate_logits: torch.Tensor, eps: float) -> torch.Tensor:
    """The eager torch lines of the zero-centred out norm and the output gate (fp32 result)."""
    xf = attn.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = (1.0 + w) * xf
    return torch.sigmoid(gate_logits) * y


# ================================================================================================ rmsnorm_rows
def _w1_of(weight: Optional[torch.Tensor], zero_centered: bool) -> Optional[torch.Tensor]:
    """the (N,) fp32 factor of the inference chains: the norm weight, or 1 + weight for a zero-centred norm."""
    if weight is None:
        if zero_centered:
            raise RuntimeError("neuronspark.rmsnorm_rows: zero_centered needs the weight")
        return None
    return ((1.0 + weight) if zero_centered else weight).float().contiguous()


@torch.library.custom_op("neuronspark::hippo_rmsnorm_rows", mutates_args=(),
                         schema="(Tensor x, float eps, Tensor? w1, ScalarType out_dtype) -> Tensor")
def _rmsnorm_rows_op(x, eps, w1, out_dtype):
    N = int(x.shape[-1])
    x2 = x.reshape(-1, N)
    if x2.stride(1) != 1 or (x2.shape[0] > 1 and x2.stride(0) != N):
        x2 = x2.contiguous()                   # a reshape of a non-row-major view copies; the kernel wants row-major rows
    rows = int(x2.shape[0])
    y = torch.empty(tuple(x.shape), dtype=out_dtype, device=x.device)    # contiguous: row stride N
    factor = float(np.float32(1.0) / np.float32(N))                       # the host's float division in MeanOps' factor
    grid = (triton.cdiv(rows, RMS_ROWS_PER_PROGRAM),)
    _rmsnorm_kernel[grid](x2, w1 if w1 is not None else y, y, x2.stride(0), N, rows, factor,
                          float(np.float32(eps)), R=RMS_ROWS_PER_PROGRAM, C=N // 128, HAS_W=w1 is not None,
                          OUT_BF16=(out_dtype == torch.bfloat16), num_warps=RMS_WARPS)
    return y


@_rmsnorm_rows_op.register_fake
def _rmsnorm_rows_fake(x, eps, w1, out_dtype):
    return torch.empty(tuple(x.shape), dtype=out_dtype, device=x.device)


def rmsnorm_rows(x: torch.Tensor, eps: float, weight: Optional[torch.Tensor] = None, zero_centered: bool = False,
                 out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``(x.float() * rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)) [* w1]).to(out_dtype)`` over the last dim,
    one kernel per call (inference: prefill and decode, no autograd).

    ``x``: CUDA fp32 / bf16 (..., N) with unit last stride, N a multiple of 128 and at most 4096 (other widths raise);
    rows that are not row-major are copied first.  ``weight``: None, the fp32 (N,) norm weight (``w1 = weight``) or,
    with ``zero_centered``, the zero-centred weight (``w1 = 1 + weight``).  ``out_dtype``: fp32 or bf16 (default
    ``x.dtype``).  Returns a contiguous tensor of x's shape."""
    require_cuda(x, "rmsnorm_rows")
    if x.dim() < 1 or x.stride(-1) != 1 or x.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"neuronspark.rmsnorm_rows: CUDA fp32 / bf16 with unit last stride, got {x.dtype} "
                           f"{tuple(x.shape)} {tuple(x.stride())}")
    N = int(x.shape[-1])
    if N % 128 or N > RMS_MAX_N or N == 0:
        raise RuntimeError(f"neuronspark.rmsnorm_rows: N must be a positive multiple of 128 <= {RMS_MAX_N}, got {N}")
    out_dtype = x.dtype if out_dtype is None else out_dtype
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"neuronspark.rmsnorm_rows: out dtype {out_dtype}")
    w1 = _w1_of(weight, zero_centered)
    if w1 is not None and (w1.numel() != N or w1.device != x.device):
        raise RuntimeError(f"neuronspark.rmsnorm_rows: the weight must be an (N,) = ({N},) tensor on x's device")
    return _rmsnorm_rows_op(x, float(eps), w1, out_dtype)


@torch.library.custom_op("neuronspark::hippo_rmsnorm_gate_rows", mutates_args=(),
                         schema="(Tensor x, float eps, Tensor w1, Tensor gate) -> Tensor")
def _rmsnorm_gate_rows_op(x, eps, w1, gate):
    N = int(x.shape[-1])
    x2 = x.reshape(-1, N)
    if x2.stride(1) != 1 or (x2.shape[0] > 1 and x2.stride(0) != N):
        x2 = x2.contiguous()
    g2 = gate.reshape(-1, N)
    if g2.stride(1) != 1 or (g2.shape[0] > 1 and g2.stride(0) != N):
        g2 = g2.contiguous()
    rows = int(x2.shape[0])
    y = torch.empty(tuple(x.shape), dtype=torch.bfloat16, device=x.device)
    factor = float(np.float32(1.0) / np.float32(N))
    _rmsnorm_kernel[(triton.cdiv(rows, RMS_ROWS_PER_PROGRAM),)](
        x2, w1, y, x2.stride(0), N, rows, factor, float(np.float32(eps)), R=RMS_ROWS_PER_PROGRAM, C=N // 128,
        HAS_W=True, OUT_BF16=True, G=g2, sg=g2.stride(0), HAS_G=True, num_warps=RMS_WARPS)
    return y


@_rmsnorm_gate_rows_op.register_fake
def _rmsnorm_gate_rows_fake(x, eps, w1, gate):
    return torch.empty(tuple(x.shape), dtype=torch.bfloat16, device=x.device)


def rmsnorm_gate_rows(x: torch.Tensor, eps: float, weight: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The hippocampal layer's inference output norm and gate in one kernel: ``bf16(sigmoid(gate) * rmsnorm(x) * (1 +
    weight))`` (the zero-centred norm in fp32, torch's bf16 sigmoid of the gate logits, the product rounded to bf16 --
    the output projection's input).  ``x`` CUDA fp32 / bf16 (..., N) with unit last stride, N a multiple of 128 and at
    most 4096; ``weight`` the zero-centred fp32 (N,) norm weight; ``gate`` bf16 of x's shape.  Inference, no autograd."""
    require_cuda(x, "rmsnorm_gate_rows")
    if x.dim() < 1 or x.stride(-1) != 1 or x.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"neuronspark.rmsnorm_gate_rows: CUDA fp32 / bf16 with unit last stride, got {x.dtype} "
                           f"{tuple(x.shape)} {tuple(x.stride())}")
    N = int(x.shape[-1])
    if N % 128 or N > RMS_MAX_N or N == 0:
        raise RuntimeError(f"neuronspark.rmsnorm_gate_rows: N must be a positive multiple of 128 <= {RMS_MAX_N}, got {N}")
    if (not isinstance(gate, torch.Tensor) or gate.dtype != torch.bfloat16 or tuple(gate.shape) != tuple(x.shape)
            or gate.stride(-1) != 1 or gate.device != x.device):
        raise RuntimeError(f"neuronspark.rmsnorm_gate_rows: the gate must be bf16 {tuple(x.shape)} with unit last stride")
    w1 = _w1_of(weight, True)
    if w1.numel() != N or w1.device != x.device:
        raise RuntimeError(f"neuronspark.rmsnorm_gate_rows: the weight must be an (N,) = ({N},) tensor on x's device")
    return _rmsnorm_gate_rows_op(x, float(eps), w1, gate)


def rmsnorm_gate_rows_ref(x: torch.Tensor, eps: float, weight: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """The torch lines of the inference output norm and gate: the zero-centred norm in fp32, sigmoid(gate) * it, bf16."""
    return (torch.sigmoid(gate) * rmsnorm_rows_ref(x, eps, weight, True, torch.float32)).to(torch.bfloat16)


def rmsnorm_rows_ref(x: torch.Tensor, eps: float, weight: Optional[torch.Tensor] = None, zero_centered: bool = False,
                     out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """The torch chain of the inference norm sites."""
    w1 = _w1_of(weight, zero_centered)
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w1 is not None:
        y = y * w1
    return y.to(x.dtype if out_dtype is None else out_dtype)
