"""dendritic-layer causal depthwise conv (width 4) + SiLU on the x|B|C channels of the in-projection.  Training (forward
and backward) and inference prefill.

Ops:

* training (whole row): ``dendritic_conv_silu(xBC, conv_w, conv_b) -> out`` with ``DendriticConvSiluFn``. ``xBC`` (B, L,
  C) bf16 with unit channel stride (the strided x|B|C slice of the projection, row stride 18832 in the default
  configuration), ``conv_w`` (C, 4) fp32 contiguous (column 3 = lag 0), ``conv_b`` (C,) fp32; ``out`` (B, L, C) bf16
  contiguous. Backward: ``dxBC`` (B, L, C) bf16, ``dconv_w`` (C, 4) fp32, ``dconv_b`` (C,) fp32. The combined
  dendritic-layer Function (:mod:`neuronspark.ops.dendritic_combined`) calls the launchers ``_conv_fwd`` / ``_conv_bwd``
  directly (its backward writes dx into the projection gradient).
* prefill: ``dendritic_conv_silu_prefill(xBC, conv_w, conv_b, tail) -> (xbc, tail_new)`` with ``tail`` (B, C, 3) fp32 =
  the three previous conv inputs (lags t-3, t-2, t-1; bf16-exact values): ``x_ext = cat([bf16(tail), xBC], time)``, the
  training forward kernel on x_ext, the first three output rows dropped; ``xbc`` (B, T, C) bf16 contiguous, ``tail_new``
  (B, C, 3) fp32 contiguous = the last three rows of x_ext.

Numerics.  Forward, per element: ``pre = bias + w0 x[t-3]; pre += w1 x[t-2]; pre += w2 x[t-1]; pre += w3 x[t]`` in fp32
(x[t < 0] = 0; Triton's FMA contraction), ``out = bf16(pre / (1 + exp(-pre)))``: the causal_conv1d 1.7.0 forward's
chain.  Backward: ``dact = dout sig(pre) (1 + pre (1 - sig(pre)))`` with pre recomputed from x,
``dx[l] = sum_w w[w] dact[l + 3 - w]``, and the weight / bias gradients as per-program fp32 partial sums in a
(programs, 5 C) workspace reduced by one ``torch.sum(0)`` (fixed order, deterministic; the reduction order differs
from causal_conv1d's atomics: fp32 reduction-order class).  The prefill form computes each kept output element from
the same four inputs as the whole-row form at that position.

Launch constants: forward (segments of 16 rows, 2 segments per program, 128 channels, 2 warps); backward (8-row
segments with carried halos and prefetch, 16 segments per program, 128 channels, 2 warps).

Contract (anything else raises): CUDA bf16 activations whose channel axis has unit stride, a contiguous fp32 (C, 4)
weight and a contiguous fp32 (C,) bias; ``tail`` fp32 (B, C, 3). The combined dendritic-layer Function passes its
operands through ``_ensure_stride`` (row strides a multiple of 8 elements).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = [
    "FWD_CFG", "BWD_CFG", "DendriticConvSiluFn", "dendritic_conv_silu", "dendritic_conv_silu_ref",
    "dendritic_conv_silu_prefill",
    "dendritic_conv_silu_prefill_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
FWD_CFG = (16, 2, 128, 2)        # forward kernel _conv_silu_fwd_ls16: LS rows / segment, SPP segments / program, BC, warps
BWD_CFG = (8, 16, 128, 2)        # backward kernel _conv_silu_bwd_ls8: LS, SPP, BC, warps
CONV_WIDTH = 4


# ================================================================================================================
# Kernels: forward _conv_silu_fwd_ls16 (LS 16 rows per segment), backward _conv_silu_bwd_ls8 (LS 8 rows per segment, carried halos);
# the per-row taps are written out.
# ================================================================================================================
@triton.jit
def _conv_silu_fwd_ls16(X, W, Bias, OUT, sxb, sxl, sob, sol, L, nseg, nprog_l, C, SPP, BC: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_bp = tl.program_id(1)
    b = pid_bp // nprog_l
    q = pid_bp - b * nprog_l
    c = pid_c * BC + tl.arange(0, BC)
    cm = c < C
    w0 = tl.load(W + c * 4 + 0, mask=cm, other=0.)
    w1 = tl.load(W + c * 4 + 1, mask=cm, other=0.)
    w2 = tl.load(W + c * 4 + 2, mask=cm, other=0.)
    w3 = tl.load(W + c * 4 + 3, mask=cm, other=0.)
    bias = tl.load(Bias + c, mask=cm, other=0.)
    xb = X + b * sxb + c
    ob = OUT + b * sob + c
    seg0 = q * SPP
    seg1 = tl.minimum(seg0 + SPP, nseg)
    for seg in range(seg0, seg1):
        l0 = seg * 16
        lx0 = l0 + (-3)
        xr0 = tl.load(xb + lx0 * sxl, mask=cm & (lx0 >= 0), other=0.).to(tl.float32)
        lx1 = l0 + (-2)
        xr1 = tl.load(xb + lx1 * sxl, mask=cm & (lx1 >= 0), other=0.).to(tl.float32)
        lx2 = l0 + (-1)
        xr2 = tl.load(xb + lx2 * sxl, mask=cm & (lx2 >= 0), other=0.).to(tl.float32)
        lx3 = l0 + (0)
        xr3 = tl.load(xb + lx3 * sxl, mask=cm & (lx3 < L), other=0.).to(tl.float32)
        lx4 = l0 + (1)
        xr4 = tl.load(xb + lx4 * sxl, mask=cm & (lx4 < L), other=0.).to(tl.float32)
        lx5 = l0 + (2)
        xr5 = tl.load(xb + lx5 * sxl, mask=cm & (lx5 < L), other=0.).to(tl.float32)
        lx6 = l0 + (3)
        xr6 = tl.load(xb + lx6 * sxl, mask=cm & (lx6 < L), other=0.).to(tl.float32)
        lx7 = l0 + (4)
        xr7 = tl.load(xb + lx7 * sxl, mask=cm & (lx7 < L), other=0.).to(tl.float32)
        lx8 = l0 + (5)
        xr8 = tl.load(xb + lx8 * sxl, mask=cm & (lx8 < L), other=0.).to(tl.float32)
        lx9 = l0 + (6)
        xr9 = tl.load(xb + lx9 * sxl, mask=cm & (lx9 < L), other=0.).to(tl.float32)
        lx10 = l0 + (7)
        xr10 = tl.load(xb + lx10 * sxl, mask=cm & (lx10 < L), other=0.).to(tl.float32)
        lx11 = l0 + (8)
        xr11 = tl.load(xb + lx11 * sxl, mask=cm & (lx11 < L), other=0.).to(tl.float32)
        lx12 = l0 + (9)
        xr12 = tl.load(xb + lx12 * sxl, mask=cm & (lx12 < L), other=0.).to(tl.float32)
        lx13 = l0 + (10)
        xr13 = tl.load(xb + lx13 * sxl, mask=cm & (lx13 < L), other=0.).to(tl.float32)
        lx14 = l0 + (11)
        xr14 = tl.load(xb + lx14 * sxl, mask=cm & (lx14 < L), other=0.).to(tl.float32)
        lx15 = l0 + (12)
        xr15 = tl.load(xb + lx15 * sxl, mask=cm & (lx15 < L), other=0.).to(tl.float32)
        lx16 = l0 + (13)
        xr16 = tl.load(xb + lx16 * sxl, mask=cm & (lx16 < L), other=0.).to(tl.float32)
        lx17 = l0 + (14)
        xr17 = tl.load(xb + lx17 * sxl, mask=cm & (lx17 < L), other=0.).to(tl.float32)
        lx18 = l0 + (15)
        xr18 = tl.load(xb + lx18 * sxl, mask=cm & (lx18 < L), other=0.).to(tl.float32)
        p0 = bias + w0 * xr0
        p0 += w1 * xr1
        p0 += w2 * xr2
        p0 += w3 * xr3
        tl.store(ob + lx3 * sol, p0 / (1.0 + tl.exp(-p0)), mask=cm & (lx3 < L))
        p1 = bias + w0 * xr1
        p1 += w1 * xr2
        p1 += w2 * xr3
        p1 += w3 * xr4
        tl.store(ob + lx4 * sol, p1 / (1.0 + tl.exp(-p1)), mask=cm & (lx4 < L))
        p2 = bias + w0 * xr2
        p2 += w1 * xr3
        p2 += w2 * xr4
        p2 += w3 * xr5
        tl.store(ob + lx5 * sol, p2 / (1.0 + tl.exp(-p2)), mask=cm & (lx5 < L))
        p3 = bias + w0 * xr3
        p3 += w1 * xr4
        p3 += w2 * xr5
        p3 += w3 * xr6
        tl.store(ob + lx6 * sol, p3 / (1.0 + tl.exp(-p3)), mask=cm & (lx6 < L))
        p4 = bias + w0 * xr4
        p4 += w1 * xr5
        p4 += w2 * xr6
        p4 += w3 * xr7
        tl.store(ob + lx7 * sol, p4 / (1.0 + tl.exp(-p4)), mask=cm & (lx7 < L))
        p5 = bias + w0 * xr5
        p5 += w1 * xr6
        p5 += w2 * xr7
        p5 += w3 * xr8
        tl.store(ob + lx8 * sol, p5 / (1.0 + tl.exp(-p5)), mask=cm & (lx8 < L))
        p6 = bias + w0 * xr6
        p6 += w1 * xr7
        p6 += w2 * xr8
        p6 += w3 * xr9
        tl.store(ob + lx9 * sol, p6 / (1.0 + tl.exp(-p6)), mask=cm & (lx9 < L))
        p7 = bias + w0 * xr7
        p7 += w1 * xr8
        p7 += w2 * xr9
        p7 += w3 * xr10
        tl.store(ob + lx10 * sol, p7 / (1.0 + tl.exp(-p7)), mask=cm & (lx10 < L))
        p8 = bias + w0 * xr8
        p8 += w1 * xr9
        p8 += w2 * xr10
        p8 += w3 * xr11
        tl.store(ob + lx11 * sol, p8 / (1.0 + tl.exp(-p8)), mask=cm & (lx11 < L))
        p9 = bias + w0 * xr9
        p9 += w1 * xr10
        p9 += w2 * xr11
        p9 += w3 * xr12
        tl.store(ob + lx12 * sol, p9 / (1.0 + tl.exp(-p9)), mask=cm & (lx12 < L))
        p10 = bias + w0 * xr10
        p10 += w1 * xr11
        p10 += w2 * xr12
        p10 += w3 * xr13
        tl.store(ob + lx13 * sol, p10 / (1.0 + tl.exp(-p10)), mask=cm & (lx13 < L))
        p11 = bias + w0 * xr11
        p11 += w1 * xr12
        p11 += w2 * xr13
        p11 += w3 * xr14
        tl.store(ob + lx14 * sol, p11 / (1.0 + tl.exp(-p11)), mask=cm & (lx14 < L))
        p12 = bias + w0 * xr12
        p12 += w1 * xr13
        p12 += w2 * xr14
        p12 += w3 * xr15
        tl.store(ob + lx15 * sol, p12 / (1.0 + tl.exp(-p12)), mask=cm & (lx15 < L))
        p13 = bias + w0 * xr13
        p13 += w1 * xr14
        p13 += w2 * xr15
        p13 += w3 * xr16
        tl.store(ob + lx16 * sol, p13 / (1.0 + tl.exp(-p13)), mask=cm & (lx16 < L))
        p14 = bias + w0 * xr14
        p14 += w1 * xr15
        p14 += w2 * xr16
        p14 += w3 * xr17
        tl.store(ob + lx17 * sol, p14 / (1.0 + tl.exp(-p14)), mask=cm & (lx17 < L))
        p15 = bias + w0 * xr15
        p15 += w1 * xr16
        p15 += w2 * xr17
        p15 += w3 * xr18
        tl.store(ob + lx18 * sol, p15 / (1.0 + tl.exp(-p15)), mask=cm & (lx18 < L))



@triton.jit
def _conv_silu_bwd_ls8(X, W, Bias, DOUT, DX, PART, sxb, sxl, sob, sol, sdb, sdl, L, nseg, nprog_l, C, SPP, BC: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_bp = tl.program_id(1)
    b = pid_bp // nprog_l
    q = pid_bp - b * nprog_l
    c = pid_c * BC + tl.arange(0, BC)
    cm = c < C
    w0 = tl.load(W + c * 4 + 0, mask=cm, other=0.)
    w1 = tl.load(W + c * 4 + 1, mask=cm, other=0.)
    w2 = tl.load(W + c * 4 + 2, mask=cm, other=0.)
    w3 = tl.load(W + c * 4 + 3, mask=cm, other=0.)
    bias = tl.load(Bias + c, mask=cm, other=0.)
    xb = X + b * sxb + c
    ob = DOUT + b * sob + c
    db_ = DX + b * sdb + c
    aw0 = tl.zeros((BC,), dtype=tl.float32)
    aw1 = tl.zeros((BC,), dtype=tl.float32)
    aw2 = tl.zeros((BC,), dtype=tl.float32)
    aw3 = tl.zeros((BC,), dtype=tl.float32)
    ab = tl.zeros((BC,), dtype=tl.float32)
    seg0 = q * SPP
    seg1 = tl.minimum(seg0 + SPP, nseg)
    lp = seg0 * 8
    xc0 = tl.load(xb + (lp + (-3)) * sxl, mask=cm & ((lp + (-3)) >= 0) & ((lp + (-3)) < L), other=0.).to(tl.float32)
    xc1 = tl.load(xb + (lp + (-2)) * sxl, mask=cm & ((lp + (-2)) >= 0) & ((lp + (-2)) < L), other=0.).to(tl.float32)
    xc2 = tl.load(xb + (lp + (-1)) * sxl, mask=cm & ((lp + (-1)) >= 0) & ((lp + (-1)) < L), other=0.).to(tl.float32)
    xc3 = tl.load(xb + (lp + (0)) * sxl, mask=cm & ((lp + (0)) >= 0) & ((lp + (0)) < L), other=0.).to(tl.float32)
    xc4 = tl.load(xb + (lp + (1)) * sxl, mask=cm & ((lp + (1)) >= 0) & ((lp + (1)) < L), other=0.).to(tl.float32)
    xc5 = tl.load(xb + (lp + (2)) * sxl, mask=cm & ((lp + (2)) >= 0) & ((lp + (2)) < L), other=0.).to(tl.float32)
    dp0 = tl.load(ob + (lp + 0) * sol, mask=cm & ((lp + 0) >= 0) & ((lp + 0) < L), other=0.).to(tl.float32)
    dp1 = tl.load(ob + (lp + 1) * sol, mask=cm & ((lp + 1) >= 0) & ((lp + 1) < L), other=0.).to(tl.float32)
    dp2 = tl.load(ob + (lp + 2) * sol, mask=cm & ((lp + 2) >= 0) & ((lp + 2) < L), other=0.).to(tl.float32)
    pp0 = bias + w0 * xc0
    pp0 += w1 * xc1
    pp0 += w2 * xc2
    pp0 += w3 * xc3
    sp0 = 1.0 / (1.0 + tl.exp(-pp0))
    ac0 = dp0 * (sp0 * (1 + pp0 * (1 - sp0)))
    pp1 = bias + w0 * xc1
    pp1 += w1 * xc2
    pp1 += w2 * xc3
    pp1 += w3 * xc4
    sp1 = 1.0 / (1.0 + tl.exp(-pp1))
    ac1 = dp1 * (sp1 * (1 + pp1 * (1 - sp1)))
    pp2 = bias + w0 * xc2
    pp2 += w1 * xc3
    pp2 += w2 * xc4
    pp2 += w3 * xc5
    sp2 = 1.0 / (1.0 + tl.exp(-pp2))
    ac2 = dp2 * (sp2 * (1 + pp2 * (1 - sp2)))
    xn0 = tl.load(xb + (lp + 3) * sxl, mask=cm & ((lp + 3) >= 0) & ((lp + 3) < L), other=0.).to(tl.float32)
    dn0 = tl.load(ob + (lp + 3) * sol, mask=cm & ((lp + 3) >= 0) & ((lp + 3) < L), other=0.).to(tl.float32)
    xn1 = tl.load(xb + (lp + 4) * sxl, mask=cm & ((lp + 4) >= 0) & ((lp + 4) < L), other=0.).to(tl.float32)
    dn1 = tl.load(ob + (lp + 4) * sol, mask=cm & ((lp + 4) >= 0) & ((lp + 4) < L), other=0.).to(tl.float32)
    xn2 = tl.load(xb + (lp + 5) * sxl, mask=cm & ((lp + 5) >= 0) & ((lp + 5) < L), other=0.).to(tl.float32)
    dn2 = tl.load(ob + (lp + 5) * sol, mask=cm & ((lp + 5) >= 0) & ((lp + 5) < L), other=0.).to(tl.float32)
    xn3 = tl.load(xb + (lp + 6) * sxl, mask=cm & ((lp + 6) >= 0) & ((lp + 6) < L), other=0.).to(tl.float32)
    dn3 = tl.load(ob + (lp + 6) * sol, mask=cm & ((lp + 6) >= 0) & ((lp + 6) < L), other=0.).to(tl.float32)
    xn4 = tl.load(xb + (lp + 7) * sxl, mask=cm & ((lp + 7) >= 0) & ((lp + 7) < L), other=0.).to(tl.float32)
    dn4 = tl.load(ob + (lp + 7) * sol, mask=cm & ((lp + 7) >= 0) & ((lp + 7) < L), other=0.).to(tl.float32)
    xn5 = tl.load(xb + (lp + 8) * sxl, mask=cm & ((lp + 8) >= 0) & ((lp + 8) < L), other=0.).to(tl.float32)
    dn5 = tl.load(ob + (lp + 8) * sol, mask=cm & ((lp + 8) >= 0) & ((lp + 8) < L), other=0.).to(tl.float32)
    xn6 = tl.load(xb + (lp + 9) * sxl, mask=cm & ((lp + 9) >= 0) & ((lp + 9) < L), other=0.).to(tl.float32)
    dn6 = tl.load(ob + (lp + 9) * sol, mask=cm & ((lp + 9) >= 0) & ((lp + 9) < L), other=0.).to(tl.float32)
    xn7 = tl.load(xb + (lp + 10) * sxl, mask=cm & ((lp + 10) >= 0) & ((lp + 10) < L), other=0.).to(tl.float32)
    dn7 = tl.load(ob + (lp + 10) * sol, mask=cm & ((lp + 10) >= 0) & ((lp + 10) < L), other=0.).to(tl.float32)
    for seg in range(seg0, seg1):
        l0 = seg * 8
        hn = seg + 1 < seg1
        xq0 = tl.load(xb + (l0 + 11) * sxl, mask=cm & hn & ((l0 + 11) >= 0) & ((l0 + 11) < L), other=0.).to(tl.float32)
        dq0 = tl.load(ob + (l0 + 11) * sol, mask=cm & hn & ((l0 + 11) >= 0) & ((l0 + 11) < L), other=0.).to(tl.float32)
        xq1 = tl.load(xb + (l0 + 12) * sxl, mask=cm & hn & ((l0 + 12) >= 0) & ((l0 + 12) < L), other=0.).to(tl.float32)
        dq1 = tl.load(ob + (l0 + 12) * sol, mask=cm & hn & ((l0 + 12) >= 0) & ((l0 + 12) < L), other=0.).to(tl.float32)
        xq2 = tl.load(xb + (l0 + 13) * sxl, mask=cm & hn & ((l0 + 13) >= 0) & ((l0 + 13) < L), other=0.).to(tl.float32)
        dq2 = tl.load(ob + (l0 + 13) * sol, mask=cm & hn & ((l0 + 13) >= 0) & ((l0 + 13) < L), other=0.).to(tl.float32)
        xq3 = tl.load(xb + (l0 + 14) * sxl, mask=cm & hn & ((l0 + 14) >= 0) & ((l0 + 14) < L), other=0.).to(tl.float32)
        dq3 = tl.load(ob + (l0 + 14) * sol, mask=cm & hn & ((l0 + 14) >= 0) & ((l0 + 14) < L), other=0.).to(tl.float32)
        xq4 = tl.load(xb + (l0 + 15) * sxl, mask=cm & hn & ((l0 + 15) >= 0) & ((l0 + 15) < L), other=0.).to(tl.float32)
        dq4 = tl.load(ob + (l0 + 15) * sol, mask=cm & hn & ((l0 + 15) >= 0) & ((l0 + 15) < L), other=0.).to(tl.float32)
        xq5 = tl.load(xb + (l0 + 16) * sxl, mask=cm & hn & ((l0 + 16) >= 0) & ((l0 + 16) < L), other=0.).to(tl.float32)
        dq5 = tl.load(ob + (l0 + 16) * sol, mask=cm & hn & ((l0 + 16) >= 0) & ((l0 + 16) < L), other=0.).to(tl.float32)
        xq6 = tl.load(xb + (l0 + 17) * sxl, mask=cm & hn & ((l0 + 17) >= 0) & ((l0 + 17) < L), other=0.).to(tl.float32)
        dq6 = tl.load(ob + (l0 + 17) * sol, mask=cm & hn & ((l0 + 17) >= 0) & ((l0 + 17) < L), other=0.).to(tl.float32)
        xq7 = tl.load(xb + (l0 + 18) * sxl, mask=cm & hn & ((l0 + 18) >= 0) & ((l0 + 18) < L), other=0.).to(tl.float32)
        dq7 = tl.load(ob + (l0 + 18) * sol, mask=cm & hn & ((l0 + 18) >= 0) & ((l0 + 18) < L), other=0.).to(tl.float32)
        p3 = bias + w0 * xc3
        p3 += w1 * xc4
        p3 += w2 * xc5
        p3 += w3 * xn0
        s3 = 1.0 / (1.0 + tl.exp(-p3))
        a3 = dn0 * (s3 * (1 + p3 * (1 - s3)))
        p4 = bias + w0 * xc4
        p4 += w1 * xc5
        p4 += w2 * xn0
        p4 += w3 * xn1
        s4 = 1.0 / (1.0 + tl.exp(-p4))
        a4 = dn1 * (s4 * (1 + p4 * (1 - s4)))
        p5 = bias + w0 * xc5
        p5 += w1 * xn0
        p5 += w2 * xn1
        p5 += w3 * xn2
        s5 = 1.0 / (1.0 + tl.exp(-p5))
        a5 = dn2 * (s5 * (1 + p5 * (1 - s5)))
        p6 = bias + w0 * xn0
        p6 += w1 * xn1
        p6 += w2 * xn2
        p6 += w3 * xn3
        s6 = 1.0 / (1.0 + tl.exp(-p6))
        a6 = dn3 * (s6 * (1 + p6 * (1 - s6)))
        p7 = bias + w0 * xn1
        p7 += w1 * xn2
        p7 += w2 * xn3
        p7 += w3 * xn4
        s7 = 1.0 / (1.0 + tl.exp(-p7))
        a7 = dn4 * (s7 * (1 + p7 * (1 - s7)))
        p8 = bias + w0 * xn2
        p8 += w1 * xn3
        p8 += w2 * xn4
        p8 += w3 * xn5
        s8 = 1.0 / (1.0 + tl.exp(-p8))
        a8 = dn5 * (s8 * (1 + p8 * (1 - s8)))
        p9 = bias + w0 * xn3
        p9 += w1 * xn4
        p9 += w2 * xn5
        p9 += w3 * xn6
        s9 = 1.0 / (1.0 + tl.exp(-p9))
        a9 = dn6 * (s9 * (1 + p9 * (1 - s9)))
        p10 = bias + w0 * xn4
        p10 += w1 * xn5
        p10 += w2 * xn6
        p10 += w3 * xn7
        s10 = 1.0 / (1.0 + tl.exp(-p10))
        a10 = dn7 * (s10 * (1 + p10 * (1 - s10)))
        aw0 += ac0 * xc0
        aw1 += ac0 * xc1
        aw2 += ac0 * xc2
        aw3 += ac0 * xc3
        ab += ac0
        aw0 += ac1 * xc1
        aw1 += ac1 * xc2
        aw2 += ac1 * xc3
        aw3 += ac1 * xc4
        ab += ac1
        aw0 += ac2 * xc2
        aw1 += ac2 * xc3
        aw2 += ac2 * xc4
        aw3 += ac2 * xc5
        ab += ac2
        aw0 += a3 * xc3
        aw1 += a3 * xc4
        aw2 += a3 * xc5
        aw3 += a3 * xn0
        ab += a3
        aw0 += a4 * xc4
        aw1 += a4 * xc5
        aw2 += a4 * xn0
        aw3 += a4 * xn1
        ab += a4
        aw0 += a5 * xc5
        aw1 += a5 * xn0
        aw2 += a5 * xn1
        aw3 += a5 * xn2
        ab += a5
        aw0 += a6 * xn0
        aw1 += a6 * xn1
        aw2 += a6 * xn2
        aw3 += a6 * xn3
        ab += a6
        aw0 += a7 * xn1
        aw1 += a7 * xn2
        aw2 += a7 * xn3
        aw3 += a7 * xn4
        ab += a7
        dv0 = w0 * a3
        dv0 += w1 * ac2
        dv0 += w2 * ac1
        dv0 += w3 * ac0
        tl.store(db_ + (l0 + 0) * sdl, dv0, mask=cm & (l0 + 0 < L))
        dv1 = w0 * a4
        dv1 += w1 * a3
        dv1 += w2 * ac2
        dv1 += w3 * ac1
        tl.store(db_ + (l0 + 1) * sdl, dv1, mask=cm & (l0 + 1 < L))
        dv2 = w0 * a5
        dv2 += w1 * a4
        dv2 += w2 * a3
        dv2 += w3 * ac2
        tl.store(db_ + (l0 + 2) * sdl, dv2, mask=cm & (l0 + 2 < L))
        dv3 = w0 * a6
        dv3 += w1 * a5
        dv3 += w2 * a4
        dv3 += w3 * a3
        tl.store(db_ + (l0 + 3) * sdl, dv3, mask=cm & (l0 + 3 < L))
        dv4 = w0 * a7
        dv4 += w1 * a6
        dv4 += w2 * a5
        dv4 += w3 * a4
        tl.store(db_ + (l0 + 4) * sdl, dv4, mask=cm & (l0 + 4 < L))
        dv5 = w0 * a8
        dv5 += w1 * a7
        dv5 += w2 * a6
        dv5 += w3 * a5
        tl.store(db_ + (l0 + 5) * sdl, dv5, mask=cm & (l0 + 5 < L))
        dv6 = w0 * a9
        dv6 += w1 * a8
        dv6 += w2 * a7
        dv6 += w3 * a6
        tl.store(db_ + (l0 + 6) * sdl, dv6, mask=cm & (l0 + 6 < L))
        dv7 = w0 * a10
        dv7 += w1 * a9
        dv7 += w2 * a8
        dv7 += w3 * a7
        tl.store(db_ + (l0 + 7) * sdl, dv7, mask=cm & (l0 + 7 < L))
        xc0 = xn2
        xc1 = xn3
        xc2 = xn4
        xc3 = xn5
        xc4 = xn6
        xc5 = xn7
        ac0 = a8
        ac1 = a9
        ac2 = a10
        xn0 = xq0
        dn0 = dq0
        xn1 = xq1
        dn1 = dq1
        xn2 = xq2
        dn2 = dq2
        xn3 = xq3
        dn3 = dq3
        xn4 = xq4
        dn4 = dq4
        xn5 = xq5
        dn5 = dq5
        xn6 = xq6
        dn6 = dq6
        xn7 = xq7
        dn7 = dq7
    pb = PART + pid_bp * (5 * C)
    tl.store(pb + c * 4 + 0, aw0, mask=cm)
    tl.store(pb + c * 4 + 1, aw1, mask=cm)
    tl.store(pb + c * 4 + 2, aw2, mask=cm)
    tl.store(pb + c * 4 + 3, aw3, mask=cm)
    tl.store(pb + 4 * C + c, ab, mask=cm)


# ================================================================================================================
# end of the kernels
# ================================================================================================================


# ------------------------------------------------------------------------------------------------ launchers
def _conv_check(x, weight, bias, *others):
    """x (B, C, L) channel-last view (stride(1) == 1); weight contiguous fp32 (C, 4); bias contiguous fp32 (C,)."""
    if x.dim() != 3 or x.stride(1) != 1:
        raise RuntimeError("neuronspark.dendritic_conv: x must be a (B, C, L) channel-last view (stride(1) == 1)")
    B, C, L = x.shape
    if tuple(weight.shape) != (C, 4) or not weight.is_contiguous() or weight.dtype != torch.float32:
        raise RuntimeError("neuronspark.dendritic_conv: weight must be a contiguous fp32 (C, 4) tensor")
    if bias is None or tuple(bias.shape) != (C,) or bias.stride(0) != 1 or bias.dtype != torch.float32:
        raise RuntimeError("neuronspark.dendritic_conv: bias must be a contiguous fp32 (C,) tensor")
    for t in others:
        if tuple(t.shape) != (B, C, L) or t.stride(1) != 1 or t.dtype != x.dtype:
            raise RuntimeError("neuronspark.dendritic_conv: every (B, C, L) operand must be channel-last with x's "
                               "dtype")
    return B, C, L


def _conv_fwd(x, weight, bias, out=None):
    """silu(causal depthwise conv1d) of a channel-last (B, C, L) view; ``out`` (new, contiguous (B, L, C) seen as
    (B, C, L)) unless given."""
    B, C, L = _conv_check(x, weight, bias)
    LS, SPP, BC, nw = FWD_CFG
    if out is None:
        out = torch.empty((B, L, C), dtype=x.dtype, device=x.device).transpose(1, 2)
    _conv_check(x, weight, bias, out)
    nseg = math.ceil(L / LS)
    nprog_l = math.ceil(nseg / SPP)
    grid = (math.ceil(C / BC), B * nprog_l)
    with torch.cuda.device(x.device.index):
        _conv_silu_fwd_ls16[grid](x, weight, bias, out, x.stride(0), x.stride(2), out.stride(0), out.stride(2), L, nseg,
                            nprog_l, C, SPP, BC=BC, num_warps=nw)
    return out


def _conv_bwd(x, weight, bias, dout, dx):
    """(dx, dwb): dx written into the given channel-last buffer; dwb (5 C,) fp32 = the deterministic partial sums of
    one (programs, 5 C) workspace reduced by one ``sum(0)``: ``dwb[:4 C].view(C, 4)`` is the weight gradient and
    ``dwb[4 C:]`` the bias gradient (``split_dwb``)."""
    B, C, L = _conv_check(x, weight, bias, dout, dx)
    LS, SPP, BC, nw = BWD_CFG
    nseg = math.ceil(L / LS)
    nprog_l = math.ceil(nseg / SPP)
    P = B * nprog_l
    part = torch.empty((P, 5 * C), dtype=torch.float32, device=x.device)
    grid = (math.ceil(C / BC), P)
    with torch.cuda.device(x.device.index):
        _conv_silu_bwd_ls8[grid](x, weight, bias, dout, dx, part, x.stride(0), x.stride(2), dout.stride(0), dout.stride(2),
                           dx.stride(0), dx.stride(2), L, nseg, nprog_l, C, SPP, BC=BC, num_warps=nw)
    return dx, part.sum(0)


def split_dwb(dwb: torch.Tensor, C: int):
    """(dweight (C, 4), dbias (C,)) views of the (5 C,) conv parameter gradient."""
    return dwb[:4 * C].view(C, 4), dwb[4 * C:]


def _ensure_stride(inp: torch.Tensor) -> torch.Tensor:
    """(B, L, C) with C % 8 == 0: the tensor itself when its row stride is a multiple of 8 elements, else a contiguous
    copy (never for the projection slice: row stride 18832)."""
    if inp.shape[2] % 8 != 0:
        raise RuntimeError("neuronspark.dendritic_conv: the number of convolution channels must be a multiple of 8")
    return inp if inp.stride(1) % 8 == 0 else inp.contiguous()


def _check_operands(xBC, conv_w, conv_b, what):
    require_cuda(xBC, what)
    if xBC.dim() != 3 or xBC.dtype != torch.bfloat16 or xBC.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.{what}: xBC must be a CUDA bf16 (B, L, C) tensor with unit channel stride, got "
                           f"{xBC.dtype} {tuple(xBC.shape)} strides {tuple(xBC.stride())}")
    C = int(xBC.shape[-1])
    if conv_w.device != xBC.device or conv_b.device != xBC.device:
        raise RuntimeError(f"neuronspark.{what}: the conv weight and bias must be on {xBC.device}")
    if tuple(conv_w.shape) != (C, CONV_WIDTH) or conv_w.dtype != torch.float32 or not conv_w.is_contiguous():
        raise RuntimeError(f"neuronspark.{what}: conv_w must be a contiguous fp32 ({C}, 4) tensor, got {conv_w.dtype} "
                           f"{tuple(conv_w.shape)}")
    if tuple(conv_b.shape) != (C,) or conv_b.dtype != torch.float32 or not conv_b.is_contiguous():
        raise RuntimeError(f"neuronspark.{what}: conv_b must be a contiguous fp32 ({C},) tensor")


# ------------------------------------------------------------------------------------------------ custom ops
@torch.library.custom_op("neuronspark::dendritic_conv_silu_fwd", mutates_args=(),
                         schema="(Tensor xBC, Tensor conv_w, Tensor conv_b) -> Tensor")
def _dendritic_conv_silu_fwd_op(xBC, conv_w, conv_b):
    """out (B, L, C) bf16 contiguous"""
    return _conv_fwd(_ensure_stride(xBC).transpose(1, 2), conv_w, conv_b).transpose(1, 2)


@_dendritic_conv_silu_fwd_op.register_fake
def _dendritic_conv_silu_fwd_fake(xBC, conv_w, conv_b):
    B, L, C = (int(s) for s in xBC.shape)
    return torch.empty((B, L, C), dtype=xBC.dtype, device=xBC.device)


@torch.library.custom_op("neuronspark::dendritic_conv_silu_bwd", mutates_args=(),
                         schema="(Tensor xBC, Tensor conv_w, Tensor conv_b, Tensor dout) -> (Tensor, Tensor)")
def _dendritic_conv_silu_bwd_op(xBC, conv_w, conv_b, dout):
    """(dxBC (B, L, C) bf16 contiguous, dwb (5 C,) fp32: the weight gradient (C, 4) flattened, then the bias
    gradient (C,))"""
    x = _ensure_stride(xBC)
    dx = torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)
    _, dwb = _conv_bwd(x.transpose(1, 2), conv_w, conv_b, _ensure_stride(dout).transpose(1, 2), dx.transpose(1, 2))
    return dx, dwb


@_dendritic_conv_silu_bwd_op.register_fake
def _dendritic_conv_silu_bwd_fake(xBC, conv_w, conv_b, dout):
    B, L, C = (int(s) for s in xBC.shape)
    return (torch.empty((B, L, C), dtype=xBC.dtype, device=xBC.device),
            torch.empty((5 * C,), dtype=torch.float32, device=xBC.device))


def _prefill_alloc(xBC):
    B, T, C = (int(s) for s in xBC.shape)
    return (torch.empty((B, T, C), dtype=torch.bfloat16, device=xBC.device),
            torch.empty((B, C, CONV_WIDTH - 1), dtype=torch.float32, device=xBC.device))


@torch.library.custom_op("neuronspark::dendritic_conv_silu_prefill", mutates_args=(),
                         schema="(Tensor xBC, Tensor conv_w, Tensor conv_b, Tensor tail) -> (Tensor, Tensor)")
def _dendritic_conv_silu_prefill_op(xBC, conv_w, conv_b, tail):
    """(xbc (B, T, C) bf16 contiguous, tail_new (B, C, 3) fp32 contiguous)"""
    k = CONV_WIDTH - 1
    x_ext = torch.cat([tail.to(torch.bfloat16).transpose(1, 2), xBC], dim=1)          # (B, k + T, C) bf16
    out = _conv_fwd(x_ext.transpose(1, 2), conv_w, conv_b)                             # (B, C, k + T) view
    xbc = out.transpose(1, 2)[:, k:, :].contiguous()
    tail_new = x_ext[:, -k:, :].transpose(1, 2).to(torch.float32).contiguous()
    return xbc, tail_new


@_dendritic_conv_silu_prefill_op.register_fake
def _dendritic_conv_silu_prefill_fake(xBC, conv_w, conv_b, tail):
    return _prefill_alloc(xBC)


# ------------------------------------------------------------------------------------------------ autograd Function
class DendriticConvSiluFn(torch.autograd.Function):
    """out = bf16(silu(causal depthwise conv1d(xBC))) (B, L, C); saves (xBC, conv_w, conv_b).  Backward: dxBC
    (B, L, C) bf16 contiguous, dconv_w (C, 4) fp32, dconv_b (C,) fp32."""

    @staticmethod
    def forward(ctx, xBC, conv_w, conv_b):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(xBC, conv_w, conv_b)
        return torch.ops.neuronspark.dendritic_conv_silu_fwd(xBC, conv_w, conv_b)

    @staticmethod
    def backward(ctx, dout):
        if dout is None:
            return None, None, None
        xBC, conv_w, conv_b = ctx.saved_tensors
        if dout.dtype != xBC.dtype or tuple(dout.shape) != tuple(xBC.shape):
            raise RuntimeError(f"neuronspark.dendritic_conv_silu: the incoming gradient must be {xBC.dtype} of shape "
                               f"{tuple(xBC.shape)}, got {dout.dtype} {tuple(dout.shape)}")
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        dx, dwb = torch.ops.neuronspark.dendritic_conv_silu_bwd(xBC, conv_w, conv_b, dout)
        dw, db = split_dwb(dwb, int(xBC.shape[-1]))
        need = ctx.needs_input_grad
        return (dx if need[0] else None, dw if need[1] else None, db if need[2] else None)


# ------------------------------------------------------------------------------------------------ public
def dendritic_conv_silu(xBC: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor) -> torch.Tensor:
    """Training form: (B, L, C) bf16 contiguous conv + SiLU output (see the module docstring)."""
    _check_operands(xBC, conv_w, conv_b, "dendritic_conv_silu")
    return DendriticConvSiluFn.apply(xBC, conv_w, conv_b)


def dendritic_conv_silu_prefill(xBC: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor, tail: torch.Tensor):
    """Prefill form: ``(xbc (B, T, C) bf16 contiguous, tail_new (B, C, 3) fp32)`` from the carried ``tail`` (see the
    module docstring).  No autograd."""
    _check_operands(xBC, conv_w, conv_b, "dendritic_conv_silu_prefill")
    B, T, C = (int(s) for s in xBC.shape)
    if (tail.device != xBC.device or tail.dtype != torch.float32
            or tuple(tail.shape) != (B, C, CONV_WIDTH - 1)):
        raise RuntimeError(f"neuronspark.dendritic_conv_silu_prefill: tail must be fp32 ({B}, {C}, {CONV_WIDTH - 1}) "
                           f"on {xBC.device}, got {tail.dtype} {tuple(tail.shape)} {tail.device}")
    if T < 1:
        raise RuntimeError("neuronspark.dendritic_conv_silu_prefill: at least one time step is required")
    return torch.ops.neuronspark.dendritic_conv_silu_prefill(xBC, conv_w, conv_b, tail)


# ------------------------------------------------------------------------------------------------ references
def _conv_silu_chain(x_bcl: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor) -> torch.Tensor:
    """fp32 causal conv tap chain + SiLU on a (B, C, L) fp32 tensor (zero history): pre = b + w0 x[t-3] + w1 x[t-2]
    + w2 x[t-1] + w3 x[t], in this order."""
    L = int(x_bcl.shape[-1])
    k = CONV_WIDTH - 1
    xp = F.pad(x_bcl, (k, 0))
    w = conv_w.float()
    pre = conv_b.float()[None, :, None].expand_as(x_bcl)
    for j in range(CONV_WIDTH):
        pre = pre + w[None, :, j, None] * xp[:, :, j:j + L]
    return F.silu(pre)


def dendritic_conv_silu_ref(xBC: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor) -> torch.Tensor:
    """Pure torch: the fp32 tap chain on the bf16 inputs, SiLU, rounded to bf16; (B, L, C) contiguous.
    Differentiable."""
    out = _conv_silu_chain(xBC.float().transpose(1, 2), conv_w, conv_b)
    return out.to(torch.bfloat16).transpose(1, 2).contiguous()


def dendritic_conv_silu_prefill_ref(xBC: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor, tail: torch.Tensor):
    """Pure torch: the fp32 tap chain over ``cat([tail, xBC])`` with the three tail columns as the history
    (the chunked inference form); returns (xbc (B, T, C) bf16 contiguous, tail_new (B, C, 3) fp32 = the last three
    inputs)."""
    k = CONV_WIDTH - 1
    x_ext = torch.cat([tail.to(torch.bfloat16).float(), xBC.float().transpose(1, 2)], dim=-1)    # (B, C, k + T)
    out = _conv_silu_chain(x_ext, conv_w, conv_b)[..., k:]
    return out.to(torch.bfloat16).transpose(1, 2).contiguous(), x_ext[..., -k:].contiguous()
