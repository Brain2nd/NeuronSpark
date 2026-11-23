"""Triton kernels around the pool read of a hippocampal layer (training): the operands of the read formed in the
read's own layouts, and the backward of that formation.

Launchers (used by :mod:`neuronspark.ops.pool_attend` and :mod:`neuronspark.ops.pool_read`):

* :func:`pack_heads_fwd`: for every head ``h < H`` (channels ``[h dk, (h + 1) dk)``) the unit keys
  ``k / max(|k|, eps)``, the gained unit queries ``(q / max(|q|, eps)) gain_h`` and the tagged values
  ``bf16(sigmoid(gate + wbias_h - sparsity)) v``, each rounded once to bf16 and written into slot ``h`` of the
  contiguous ``(T, B, HQ, dk)`` tensors the read's kernels take, HQ = H + M; the last head's keys and tagged values
  are also written into slots ``H .. HQ - 1``, one copy per extra query (see "Key slots" below); the fp32 norms are
  kept for the backward.  One launch.
* :func:`pack_xq_fwd`: the M extra queries of the last head, ``normalize(rope(silu(x_m))) exp(clamp(psi_m, -3, 3))
  gain_{H-1}``, written into slots ``H .. H + M - 1`` of the query tensor.  One launch.
* :func:`pack_heads_bwd`: from the read's fp32 query / key / value gradients and clock gradient per slot, the gradients of q, k, v (bf16), of the gate (fp32), of the H clocks (fp32) and the per-program partial sums
  of the gain and write-bias gradients; the last head's key, value and clock gradients are the sums over its slots.
  One launch.
* :func:`pack_xq_bwd`: the gradients of the M projections (bf16), of psi (bf16) and the partial sums of the last
  head's gain gradient.  One launch.
* :func:`fold_gain_bias`: the gain gradient (bf16) and the write-bias gradient (fp32) from those partial sums, and
  the beta-slot sums of the H heads from the sums per slot.  One launch.
* :func:`read_delta`: ``D[b, i, t] = sum_c dO[t, b, i, c] O[t, b, i, c]`` (the softmax backward's row term).  One
  launch.
* :func:`read_unpack`: the bf16 reads (T, B, H + M, dv) as the H heads' fp32 (T, B, H dv) and the extra queries'
  bf16 (T, B, M dv) (what their consumers take).  One launch.
* :func:`read_delta_split`: the bf16 gradient of the reads from the gradients of those two tensors, with the row term.
  One launch.

Key slots: the read's key-major backward kernels run one CTA per (key tile, row, key head) and step through the query
heads that read that key head.  With the last head's keys in one slot its CTAs would serve 1 + M query heads and take
1 + M times the steps of the others, and the kernel would last as long as its longest CTA.  With one key slot per
query head every CTA serves one query head: the same work in CTAs of equal length (640 instead of 384 at T 1024, B 8,
H 6, M 4), spread over the SMs.  The repeated keys and values are the same bf16 values, and the gradients of the
slots are summed in fp32.

Numerics: fp32 arithmetic on the widened bf16 operands; a value is rounded to bf16 where the torch chain holds a bf16
tensor (the SiLU output, each product and the sum of the rotation, the sigmoid tag, the stored operands, the two terms
and the sum of a normalisation's input gradient, the rotation's backward, each contribution to the gain's gradient
and their accumulation).  The norms are in-kernel fp32 sums.  The
products and sums are separate fp32 operations (no FMA contraction).  The rule of
:mod:`neuronspark.utils.testing` against :func:`neuronspark.ops.pool_attend.pool_attend_ref`.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ....neuronspark.ops.triton._ptx import _sr_bf16_to_f32 as _w, _sr_f32_to_bf16 as _n  # noqa: F401  (jit helpers)

__all__ = ["pack_heads_fwd", "pack_xq_fwd", "pack_heads_bwd", "pack_xq_bwd", "fold_gain_bias", "read_delta",
           "read_unpack", "read_delta_split", "MAX_M", "NORM_EPS", "PSI_CLAMP"]

# ------------------------------------------------------------------------------------------------ fixed constants
PACK_BT, PACK_WARPS = 4, 4            # positions per program, warps (heads and extra queries, forward and backward)
DELTA_BT, DELTA_WARPS = 8, 4          # positions per program, warps of the row term
FOLD_BLOCK, FOLD_WARPS = 1024, 4      # partial sums per pass, warps of the gain / write-bias fold
MAX_M = 8                             # the extra-query kernels take up to eight projection pointers
NORM_EPS = 1e-12                      # F.normalize's eps
PSI_CLAMP = 3.0                       # clamp of the extra queries' log gain


# ================================================================================================ kernels
@triton.jit
def _sel8(h, P0, P1, P2, P3, P4, P5, P6, P7):
    """the address of the h-th of eight pointers (int64)."""
    a = tl.where(h == 7, P7.to(tl.int64), P0.to(tl.int64))
    a = tl.where(h == 6, P6.to(tl.int64), a)
    a = tl.where(h == 5, P5.to(tl.int64), a)
    a = tl.where(h == 4, P4.to(tl.int64), a)
    a = tl.where(h == 3, P3.to(tl.int64), a)
    a = tl.where(h == 2, P2.to(tl.int64), a)
    a = tl.where(h == 1, P1.to(tl.int64), a)
    return a


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _r16(x):
    """an fp32 value rounded to bf16 and widened back (a bf16 tensor's value as the next operation reads it)."""
    return _w(_n(x))


@triton.jit
def _heads_fwd_kernel(Q, K, G, V, GAIN, BIAS, QA, KA, GV, QN, KN, T, B, H, HQ, DK, SP, EPS,
                      qs0, qs1, qs2, ks0, ks1, ks2, gs0, gs1, gs2, vs0, vs1, vs2,
                      BT: tl.constexpr, BD: tl.constexpr, MAXM: tl.constexpr):
    """head h, BT positions of one b: unit keys, gained unit queries, tagged values (bf16) into slot h of the
    (T, B, HQ, dk) tensors (the last head's keys and values also into slots H .. HQ - 1) and the two fp32 norms."""
    b = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2).to(tl.int64)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DK)[None, :]
    ch = h * DK + c
    q = _w(tl.load(Q + (t[:, None] * qs0 + b * qs1 + ch[None, :] * qs2), mask=m, other=0.0))
    k = _w(tl.load(K + (t[:, None] * ks0 + b * ks1 + ch[None, :] * ks2), mask=m, other=0.0))
    qn = tl.sqrt(tl.sum(q * q, axis=1))
    kn = tl.sqrt(tl.sum(k * k, axis=1))
    no = (t * B + b) * H + h
    tl.store(QN + no, qn, mask=tm)
    tl.store(KN + no, kn, mask=tm)
    gain = _w(tl.load(GAIN + h))
    qo = (q / tl.maximum(qn, EPS)[:, None]) * gain
    ko = k / tl.maximum(kn, EPS)[:, None]
    row = (t[:, None] * B + b) * HQ
    ho = (row + h) * DK + c[None, :]
    tl.store(QA + ho, _n(qo), mask=m)
    k16 = _n(ko)
    tl.store(KA + ho, k16, mask=m)
    g = tl.load(G + (t[:, None] * gs0 + b * gs1 + ch[None, :] * gs2), mask=m, other=0.0)
    s = _w(_n(_sigmoid((g + tl.load(BIAS + h)) - SP)))
    v = _w(tl.load(V + (t[:, None] * vs0 + b * vs1 + ch[None, :] * vs2), mask=m, other=0.0))
    gv16 = _n(s * v)
    tl.store(GV + ho, gv16, mask=m)
    for j in tl.static_range(MAXM):                      # the last head's keys and values, one slot per extra query
        mx = m & ((h == H - 1) & (H + j < HQ))
        hx = (row + (H + j)) * DK + c[None, :]
        tl.store(KA + hx, k16, mask=mx)
        tl.store(GV + hx, gv16, mask=mx)


@triton.jit
def _heads_bwd_kernel(DQ, DKK, DGV, DCUM, Q, K, G, V, GAIN, BIAS, QN, KN, OQ, OK, OV, OG, OCUM, PG, PB, T, B, H, HQ,
                      DK, NP, SP, EPS, qs0, qs1, qs2, ks0, ks1, ks2, gs0, gs1, gs2, vs0, vs1, vs2,
                      BT: tl.constexpr, BD: tl.constexpr, MAXM: tl.constexpr):
    """head h, BT positions of one b: the gradients of q, k, v (bf16) and of the gate (fp32) into contiguous
    (T, B, H dk) tensors and of the clock into (T, B, H), from the read's gradients per slot (T, B, HQ, .) (the last
    head: the fp32 sum over slots H - 1 .. HQ - 1); PG[h, prog] = sum dq (q / d), PB[h, prog] = sum dgate."""
    b = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2).to(tl.int64)
    prog = tl.program_id(0) * B + tl.program_id(1)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DK)[None, :]
    ch = h * DK + c
    ho = ((t[:, None] * B + b) * H + h) * DK + c[None, :]
    no = (t * B + b) * H + h
    row = (t * B + b) * HQ
    so = (row[:, None] + h) * DK + c[None, :]
    gk = tl.load(DKK + so, mask=m, other=0.0)
    dgv = tl.load(DGV + so, mask=m, other=0.0)
    dc = tl.load(DCUM + (row + h), mask=tm, other=0.0)
    for j in tl.static_range(MAXM):                      # the last head: the other slots of its keys
        ex = (h == H - 1) & (H + j < HQ)
        sx = (row[:, None] + (H + j)) * DK + c[None, :]
        gk += tl.load(DKK + sx, mask=m & ex, other=0.0)
        dgv += tl.load(DGV + sx, mask=m & ex, other=0.0)
        dc += tl.load(DCUM + (row + (H + j)), mask=tm & ex, other=0.0)
    tl.store(OCUM + no, dc, mask=tm)
    # keys: y = k / d, d = max(n, eps)
    k = _w(tl.load(K + (t[:, None] * ks0 + b * ks1 + ch[None, :] * ks2), mask=m, other=0.0))
    kn = tl.load(KN + no, mask=tm, other=1.0)
    kd = tl.maximum(kn, EPS)
    kh = k / kd[:, None]
    sk = tl.where(kn >= EPS, tl.sum(gk * kh, axis=1), 0.0) / kd
    tl.store(OK + ho, _n(_r16(gk / kd[:, None]) - _r16(sk[:, None] * kh)), mask=m)
    # queries: y = (q / d) gain
    gq = tl.load(DQ + ((t[:, None] * B + b) * HQ + h) * DK + c[None, :], mask=m, other=0.0)
    q = _w(tl.load(Q + (t[:, None] * qs0 + b * qs1 + ch[None, :] * qs2), mask=m, other=0.0))
    qn = tl.load(QN + no, mask=tm, other=1.0)
    qd = tl.maximum(qn, EPS)
    qh = q / qd[:, None]
    gain = _w(tl.load(GAIN + h))
    dots = tl.sum(gq * qh, axis=1)
    sq = tl.where(qn >= EPS, dots * gain, 0.0) / qd
    tl.store(OQ + ho, _n(_r16((gq * gain) / qd[:, None]) - _r16(sq[:, None] * qh)), mask=m)
    tl.store(PG + (h * NP + prog), tl.sum(dots, axis=0))
    # tagged values: gv = bf16(s16 v), s16 = bf16(s), s = sigmoid(gate + bias - sparsity); the read's gradient of gv
    # is rounded to bf16 once, after the sum over the slots
    dgv = _r16(dgv)
    g = tl.load(G + (t[:, None] * gs0 + b * gs1 + ch[None, :] * gs2), mask=m, other=0.0)
    s = _sigmoid((g + tl.load(BIAS + h)) - SP)
    v = _w(tl.load(V + (t[:, None] * vs0 + b * vs1 + ch[None, :] * vs2), mask=m, other=0.0))
    tl.store(OV + ho, _n(dgv * _w(_n(s))), mask=m)
    dg = _w(_n(dgv * v)) * ((1.0 - s) * s)
    dg = tl.where(m, dg, 0.0)
    tl.store(OG + ho, dg, mask=m)
    tl.store(PB + (h * NP + prog), tl.sum(tl.sum(dg, axis=1), axis=0))


@triton.jit
def _xq_formed(X, PSI, GAIN, C, S, t, tm, b, mq, H, DK, EPS, CLAMP, xs0, xs1, ps0, ps1, ps2, cs0, cs1, ss0, ss1,
               BT: tl.constexpr, BD: tl.constexpr):
    """the extra query mq at BT positions of one b: (x even / odd, sigmoid(x) even / odd, cos, sin, the rotated SiLU
    even / odd (bf16 values), its norm, psi, exp(clamp(psi)), the last head's gain)."""
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DK)[None, :]
    p = tl.arange(0, BD // 2).to(tl.int64)
    mp = tm[:, None] & (p < DK // 2)[None, :]
    x = _w(tl.load(X + (t[:, None] * xs0 + b * xs1 + c[None, :]), mask=m, other=0.0))
    sg = _sigmoid(x)
    a = _r16(x / (1.0 + tl.exp(-x)))
    xe, xo = tl.split(tl.reshape(x, [BT, BD // 2, 2]))
    sge, sgo = tl.split(tl.reshape(sg, [BT, BD // 2, 2]))
    ae, ao = tl.split(tl.reshape(a, [BT, BD // 2, 2]))
    cs = _w(tl.load(C + (t[:, None] * cs0 + p[None, :] * cs1), mask=mp, other=0.0))
    sn = _w(tl.load(S + (t[:, None] * ss0 + p[None, :] * ss1), mask=mp, other=0.0))
    re = _r16(_r16(ae * cs) - _r16(ao * sn))
    ro = _r16(_r16(ao * cs) + _r16(ae * sn))
    rn = tl.sqrt(tl.sum(re * re + ro * ro, axis=1))
    psi = _w(tl.load(PSI + (t * ps0 + b * ps1 + mq * ps2), mask=tm, other=0.0))
    e = tl.exp(tl.minimum(tl.maximum(psi, -CLAMP), CLAMP))
    gain = _w(tl.load(GAIN + (H - 1)))
    return xe, xo, sge, sgo, cs, sn, re, ro, rn, psi, e, gain


@triton.jit
def _xq_fwd_kernel(X0, X1, X2, X3, X4, X5, X6, X7, PSI, GAIN, C, S, QA, T, B, H, HQ, DK, EPS, CLAMP,
                   xs0, xs1, ps0, ps1, ps2, cs0, cs1, ss0, ss1, BT: tl.constexpr, BD: tl.constexpr):
    """extra query mq, BT positions of one b: normalize(rope(silu(x))) exp(clamp(psi)) gain into slot H + mq."""
    b = tl.program_id(1).to(tl.int64)
    mq = tl.program_id(2).to(tl.int64)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    X = _sel8(mq, X0, X1, X2, X3, X4, X5, X6, X7).to(tl.pointer_type(tl.bfloat16))
    xe, xo, sge, sgo, cs, sn, re, ro, rn, psi, e, gain = _xq_formed(
        X, PSI, GAIN, C, S, t, tm, b, mq, H, DK, EPS, CLAMP, xs0, xs1, ps0, ps1, ps2, cs0, cs1, ss0, ss1, BT, BD)
    f = (e * gain) / tl.maximum(rn, EPS)
    y = tl.reshape(tl.join(_n(re * f[:, None]), _n(ro * f[:, None])), [BT, BD])
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DK)[None, :]
    tl.store(QA + ((t[:, None] * B + b) * HQ + (H + mq)) * DK + c[None, :], y, mask=m)


@triton.jit
def _xq_bwd_kernel(DQ, X0, X1, X2, X3, X4, X5, X6, X7, PSI, GAIN, C, S, OX, OPSI, PX, T, B, H, HQ, M, DK, NP,
                   EPS, CLAMP, xs0, xs1, ps0, ps1, ps2, cs0, cs1, ss0, ss1, BT: tl.constexpr, BD: tl.constexpr):
    """extra query mq, BT positions of one b: the projection's gradient (bf16) into OX (M, T, B, dk), psi's gradient
    (bf16) into OPSI (T, B, M), PX[mq, prog] = sum (dq . unit query) exp(clamp(psi))."""
    b = tl.program_id(1).to(tl.int64)
    mq = tl.program_id(2).to(tl.int64)
    prog = tl.program_id(0) * B + tl.program_id(1)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    X = _sel8(mq, X0, X1, X2, X3, X4, X5, X6, X7).to(tl.pointer_type(tl.bfloat16))
    xe, xo, sge, sgo, cs, sn, re, ro, rn, psi, e, gain = _xq_formed(
        X, PSI, GAIN, C, S, t, tm, b, mq, H, DK, EPS, CLAMP, xs0, xs1, ps0, ps1, ps2, cs0, cs1, ss0, ss1, BT, BD)
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DK)[None, :]
    gq = tl.load(DQ + ((t[:, None] * B + b) * HQ + (H + mq)) * DK + c[None, :], mask=m, other=0.0)
    ge, go = tl.split(tl.reshape(gq, [BT, BD // 2, 2]))
    rd = tl.maximum(rn, EPS)
    he = re / rd[:, None]
    ho = ro / rd[:, None]
    dots = tl.sum(ge * he + go * ho, axis=1)                    # dq . unit query = the gradient of the gain product
    f = e * gain
    coef = tl.where(rn >= EPS, dots * f, 0.0) / rd
    dre = _r16(_r16((ge * f[:, None]) / rd[:, None]) - _r16(coef[:, None] * he))
    dro = _r16(_r16((go * f[:, None]) / rd[:, None]) - _r16(coef[:, None] * ho))
    dae = _r16(_r16(dre * cs) + _r16(dro * sn))                 # the rotation's transpose
    dao = _r16(_r16(dro * cs) - _r16(dre * sn))
    dxe = dae * sge * (1.0 + xe * (1.0 - sge))                  # SiLU backward
    dxo = dao * sgo * (1.0 + xo * (1.0 - sgo))
    tl.store(OX + ((mq * T + t[:, None]) * B + b) * DK + c[None, :],
             tl.reshape(tl.join(_n(dxe), _n(dxo)), [BT, BD]), mask=m)
    inside = (psi >= -CLAMP) & (psi <= CLAMP)
    tl.store(OPSI + ((t * B + b) * M + mq), _n(tl.where(inside, dots * f, 0.0)), mask=tm)
    tl.store(PX + (mq * NP + prog), tl.sum(tl.where(tm, dots * e, 0.0), axis=0))


@triton.jit
def _fold_kernel(PG, PB, PX, DSL, DGAIN, DBIAS, OSL, H, M, NP, NB, BLOCK: tl.constexpr, MAXM: tl.constexpr,
                 NBP: tl.constexpr):
    """head h: DBIAS[h] = sum PB[h] (fp32) and DGAIN[h] = bf16(sum PG[h]); for the last head the extra queries'
    contributions come first, each sum PX[m] rounded to bf16 and added in bf16 from the last query to the first (the
    order in which the gain's gradient is accumulated), then the head's own.  OSL[h] = the beta-slot sums DSL of
    slot h (the last head: summed over slots H - 1 .. H + M - 1)."""
    h = tl.program_id(0)
    n = tl.arange(0, NBP)
    nm = n < NB
    sl = tl.load(DSL + (h * NB + n), mask=nm, other=0.0)
    for j in tl.static_range(MAXM):
        sl += tl.load(DSL + ((H + j) * NB + n), mask=nm & ((h == H - 1) & (j < M)), other=0.0)
    tl.store(OSL + (h * NB + n), sl, mask=nm)
    i = tl.arange(0, BLOCK)
    ag = tl.zeros([BLOCK], dtype=tl.float32)
    ab = tl.zeros([BLOCK], dtype=tl.float32)
    for o in range(0, NP, BLOCK):
        m = (o + i) < NP
        ag += tl.load(PG + (h * NP + o + i), mask=m, other=0.0)
        ab += tl.load(PB + (h * NP + o + i), mask=m, other=0.0)
    tl.store(DBIAS + h, tl.sum(ab, axis=0))
    last = h == H - 1
    acc = tl.sum(tl.zeros([BLOCK], dtype=tl.float32), axis=0)
    for j in tl.static_range(MAXM):
        mq = MAXM - 1 - j
        use = last & (mq < M)
        ax = tl.zeros([BLOCK], dtype=tl.float32)
        for o in range(0, NP, BLOCK):
            ax += tl.load(PX + (mq * NP + o + i), mask=((o + i) < NP) & use, other=0.0)
        acc = tl.where(use, _r16(acc + _r16(tl.sum(ax, axis=0))), acc)
    head = _r16(tl.sum(ag, axis=0))
    tl.store(DGAIN + h, _n(tl.where(last & (M > 0), acc + head, head)))


@triton.jit
def _delta_kernel(DO, O32, DELTA, T, NR, DV, BT: tl.constexpr, BD: tl.constexpr):
    """row r = b HQ + i, BT positions: DELTA[r, t] = sum_c dO[t, r, c] O[t, r, c]."""
    r = tl.program_id(1).to(tl.int64)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DV)[None, :]
    off = (t[:, None] * NR + r) * DV + c[None, :]
    do = _w(tl.load(DO + off, mask=m, other=0.0))
    o = tl.load(O32 + off, mask=m, other=0.0)
    tl.store(DELTA + (r * T + t), tl.sum(do * o, axis=1), mask=tm)


@triton.jit
def _unpack_kernel(O16, HD, XR, T, B, H, M, DV, BT: tl.constexpr, BD: tl.constexpr):
    """read i of (b, BT positions): heads i < H widened into HD (T, B, H dv) fp32, the extra queries' reads copied
    into XR (T, B, M dv) bf16."""
    b = tl.program_id(1).to(tl.int64)
    i = tl.program_id(2).to(tl.int64)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DV)[None, :]
    o = tl.load(O16 + ((t[:, None] * B + b) * (H + M) + i) * DV + c[None, :], mask=m, other=0.0)
    tl.store(HD + ((t[:, None] * B + b) * H + i) * DV + c[None, :], _w(o), mask=m & (i < H))
    tl.store(XR + ((t[:, None] * B + b) * M + (i - H)) * DV + c[None, :], o, mask=m & (i >= H))


@triton.jit
def _delta_split_kernel(DH, DX, O32, DO, DELTA, T, B, H, M, DV, BT: tl.constexpr, BD: tl.constexpr):
    """read i of (b, BT positions): dO (T, B, H + M, dv) bf16 from the heads' fp32 gradient DH (T, B, H dv) (rounded
    to bf16) or the extra queries' bf16 gradient DX (T, B, M dv), and DELTA[b (H + M) + i, t] = sum_c dO O."""
    b = tl.program_id(1).to(tl.int64)
    i = tl.program_id(2).to(tl.int64)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD).to(tl.int64)
    m = tm[:, None] & (c < DV)[None, :]
    gh = tl.load(DH + ((t[:, None] * B + b) * H + i) * DV + c[None, :], mask=m & (i < H), other=0.0)
    gx = tl.load(DX + ((t[:, None] * B + b) * M + (i - H)) * DV + c[None, :], mask=m & (i >= H), other=0.0)
    g16 = tl.where(i < H, _n(gh), gx)
    off = ((t[:, None] * B + b) * (H + M) + i) * DV + c[None, :]
    tl.store(DO + off, g16, mask=m)
    o = tl.load(O32 + off, mask=m, other=0.0)
    tl.store(DELTA + ((b * (H + M) + i) * T + t), tl.sum(_w(g16) * o, axis=1), mask=tm)


# ================================================================================================ launchers
def _s3(t: torch.Tensor) -> Tuple[int, int, int]:
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2))


def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


def _dims(q: torch.Tensor, H: int) -> Tuple[int, int, int]:
    T, B, D = (int(s) for s in q.shape)
    if D % H or (D // H) % 2 or q.numel() >= 2 ** 31:
        raise RuntimeError(f"neuronspark.pool_pack: {H} heads of an even width below 2^31 elements, got "
                           f"{tuple(q.shape)}")
    return T, B, D // H


def pack_heads_fwd(q, k, gate, v, gain, wbias, sparsity: float, q16, k16, gv) -> Tuple[torch.Tensor, torch.Tensor]:
    """Writes slots 0 .. H - 1 of q16 and every slot of k16 and gv (contiguous bf16 (T, B, HQ, dk); slots H .. HQ - 1
    of k16 and gv repeat the last head) from q, k, v (T, B, H dk) bf16, gate (T, B, H dk) fp32, gain (H,) bf16, wbias
    (H,) fp32 (any strides for the 3-D operands); returns the fp32 norms (qn, kn), (T, B, H) each."""
    H = int(gain.shape[0])
    T, B, dk = _dims(q, H)
    HQ = int(q16.shape[2])
    if not 0 <= HQ - H <= MAX_M or k16.shape != q16.shape or gv.shape != q16.shape:
        raise RuntimeError(f"neuronspark.pool_pack: {H} heads and up to {MAX_M} extra queries in (T, B, HQ, dk) "
                           f"tensors, got {tuple(q16.shape)}, {tuple(k16.shape)}, {tuple(gv.shape)}")
    qn = torch.empty((T, B, H), dtype=torch.float32, device=q.device)
    kn = torch.empty((T, B, H), dtype=torch.float32, device=q.device)
    _heads_fwd_kernel[(triton.cdiv(T, PACK_BT), B, H)](
        q, k, gate, v, gain, wbias, q16, k16, gv, qn, kn, T, B, H, HQ, dk, float(sparsity), NORM_EPS,
        *_s3(q), *_s3(k), *_s3(gate), *_s3(v), BT=PACK_BT, BD=_pow2(dk), MAXM=MAX_M, num_warps=PACK_WARPS, enable_fp_fusion=False)
    return qn, kn


def pack_heads_bwd(dq, dkk, dgv, dcum, q, k, gate, v, gain, wbias, sparsity: float, qn, kn):
    """(dq_in, dk_in, dv_in bf16 (T, B, H dk), dgate fp32 (T, B, H dk), dcum fp32 (T, B, H), PG, PB (H, programs)
    fp32 partial sums of the gain and write-bias gradients) from the read's gradients per slot: dq, dk, dgv
    (T, B, HQ, dk) fp32 and dcum (T, B, HQ) fp32."""
    H = int(gain.shape[0])
    T, B, dk = _dims(q, H)
    HQ = int(dq.shape[2])
    if (tuple(dkk.shape) != (T, B, HQ, dk) or tuple(dgv.shape) != (T, B, HQ, dk) or tuple(dcum.shape) != (T, B, HQ)
            or dgv.dtype != torch.float32 or not 0 <= HQ - H <= MAX_M):
        raise RuntimeError(f"neuronspark.pool_pack: the read's gradients per slot ({T}, {B}, {HQ}, {dk}) required, "
                           f"got {tuple(dkk.shape)}, {tuple(dgv.shape)}, {tuple(dcum.shape)}")
    dev = q.device
    oq = torch.empty((T, B, H * dk), dtype=torch.bfloat16, device=dev)
    ok = torch.empty((T, B, H * dk), dtype=torch.bfloat16, device=dev)
    ov = torch.empty((T, B, H * dk), dtype=torch.bfloat16, device=dev)
    og = torch.empty((T, B, H * dk), dtype=torch.float32, device=dev)
    ocum = torch.empty((T, B, H), dtype=torch.float32, device=dev)
    npt = triton.cdiv(T, PACK_BT)
    NP = npt * B
    pg = torch.empty((H, NP), dtype=torch.float32, device=dev)
    pb = torch.empty((H, NP), dtype=torch.float32, device=dev)
    _heads_bwd_kernel[(npt, B, H)](
        dq, dkk, dgv, dcum, q, k, gate, v, gain, wbias, qn, kn, oq, ok, ov, og, ocum, pg, pb, T, B, H, HQ, dk, NP,
        float(sparsity), NORM_EPS, *_s3(q), *_s3(k), *_s3(gate), *_s3(v),
        BT=PACK_BT, BD=_pow2(dk), MAXM=MAX_M, num_warps=PACK_WARPS, enable_fp_fusion=False)
    return oq, ok, ov, og, ocum, pg, pb


def _xq_operands(xq: Sequence[torch.Tensor], psi, cos, sin, T: int, B: int, dk: int) -> List[torch.Tensor]:
    M = len(xq)
    if not 1 <= M <= MAX_M:
        raise RuntimeError(f"neuronspark.pool_pack: 1 .. {MAX_M} extra queries supported, got {M}")
    xs = []
    for x in xq:
        if tuple(x.shape) != (T, B, dk) or x.dtype != torch.bfloat16:
            raise RuntimeError(f"neuronspark.pool_pack: an extra-query projection must be bf16 ({T}, {B}, {dk}), got "
                               f"{x.dtype} {tuple(x.shape)}")
        xs.append(x.contiguous())
    if tuple(psi.shape) != (T, B, M) or psi.dtype != torch.bfloat16:
        raise RuntimeError(f"neuronspark.pool_pack: psi must be bf16 ({T}, {B}, {M}), got {psi.dtype} "
                           f"{tuple(psi.shape)}")
    for name, tab in (("cos", cos), ("sin", sin)):
        if tuple(tab.shape) != (T, dk // 2) or tab.dtype != torch.bfloat16:
            raise RuntimeError(f"neuronspark.pool_pack: {name} must be the bf16 ({T}, {dk // 2}) band of the last "
                               f"head, got {tab.dtype} {tuple(tab.shape)}")
    return xs


def pack_xq_fwd(xq: Sequence[torch.Tensor], psi, gain, cos, sin, q16) -> None:
    """Writes slots H .. H + M - 1 of q16 (T, B, H + M, dk) from the M projections (T, B, dk) bf16, psi (T, B, M)
    bf16, the bf16 (H,) gain and the last head's bf16 (T, dk / 2) RoPE band."""
    T, B, HQ, dk = (int(s) for s in q16.shape)
    xs = _xq_operands(xq, psi, cos, sin, T, B, dk)
    M = len(xs)
    H = HQ - M
    pad = xs + [xs[0]] * (MAX_M - M)
    _xq_fwd_kernel[(triton.cdiv(T, PACK_BT), B, M)](
        *pad, psi, gain, cos, sin, q16, T, B, H, HQ, dk, NORM_EPS, PSI_CLAMP,
        int(xs[0].stride(0)), int(xs[0].stride(1)), *_s3(psi), int(cos.stride(0)), int(cos.stride(1)),
        int(sin.stride(0)), int(sin.stride(1)), BT=PACK_BT, BD=_pow2(dk), num_warps=PACK_WARPS, enable_fp_fusion=False)


def pack_xq_bwd(dq, xq: Sequence[torch.Tensor], psi, gain, cos, sin):
    """(dx (M, T, B, dk) bf16, dpsi (T, B, M) bf16, PX (M, programs) fp32 partial sums of the last head's gain
    gradient) from the read's dq (T, B, H + M, dk) fp32."""
    T, B, HQ, dk = (int(s) for s in dq.shape)
    xs = _xq_operands(xq, psi, cos, sin, T, B, dk)
    M = len(xs)
    H = HQ - M
    dev = dq.device
    ox = torch.empty((M, T, B, dk), dtype=torch.bfloat16, device=dev)
    opsi = torch.empty((T, B, M), dtype=torch.bfloat16, device=dev)
    npt = triton.cdiv(T, PACK_BT)
    NP = npt * B
    px = torch.empty((M, NP), dtype=torch.float32, device=dev)
    pad = xs + [xs[0]] * (MAX_M - M)
    _xq_bwd_kernel[(npt, B, M)](
        dq, *pad, psi, gain, cos, sin, ox, opsi, px, T, B, H, HQ, M, dk, NP, NORM_EPS, PSI_CLAMP,
        int(xs[0].stride(0)), int(xs[0].stride(1)), *_s3(psi), int(cos.stride(0)), int(cos.stride(1)),
        int(sin.stride(0)), int(sin.stride(1)), BT=PACK_BT, BD=_pow2(dk), num_warps=PACK_WARPS, enable_fp_fusion=False)
    return ox, opsi, px


def fold_gain_bias(pg: torch.Tensor, pb: torch.Tensor, px, dslots: torch.Tensor):
    """(the gain gradient (H,) bf16, the write-bias gradient (H,) fp32, the beta-slot sums (H, NB) fp32) from the
    partial sums PG, PB (H, programs) of :func:`pack_heads_bwd`, PX (M, programs) of :func:`pack_xq_bwd` (``None``
    without extra queries) and the read's beta-slot sums per slot (H + M, NB)."""
    H, NP = int(pg.shape[0]), int(pg.shape[1])
    M = 0 if px is None else int(px.shape[0])
    NB = int(dslots.shape[1])
    if tuple(dslots.shape) != (H + M, NB) or dslots.dtype != torch.float32 or not dslots.is_contiguous():
        raise RuntimeError(f"neuronspark.pool_pack: the beta-slot sums must be contiguous fp32 ({H + M}, NB), got "
                           f"{dslots.dtype} {tuple(dslots.shape)}")
    d_gain = torch.empty((H,), dtype=torch.bfloat16, device=pg.device)
    d_bias = torch.empty((H,), dtype=torch.float32, device=pg.device)
    oslots = torch.empty((H, NB), dtype=torch.float32, device=pg.device)
    _fold_kernel[(H,)](pg, pb, pg if px is None else px, dslots, d_gain, d_bias, oslots, H, M, NP, NB,
                       BLOCK=FOLD_BLOCK, MAXM=MAX_M, NBP=_pow2(NB), num_warps=FOLD_WARPS)
    return d_gain, d_bias, oslots


def read_delta(do16: torch.Tensor, out32: torch.Tensor) -> torch.Tensor:
    """(B, HQ, T) fp32 ``sum_c dO O`` from the contiguous bf16 dO and fp32 O, (T, B, HQ, dv) each."""
    T, B, HQ, dv = (int(s) for s in do16.shape)
    if (do16.dtype != torch.bfloat16 or out32.dtype != torch.float32 or out32.shape != do16.shape
            or not do16.is_contiguous() or not out32.is_contiguous()):
        raise RuntimeError("neuronspark.pool_pack: read_delta takes the contiguous bf16 gradient and fp32 output")
    delta = torch.empty((B, HQ, T), dtype=torch.float32, device=do16.device)
    _delta_kernel[(triton.cdiv(T, DELTA_BT), B * HQ)](do16, out32, delta, T, B * HQ, dv, BT=DELTA_BT, BD=_pow2(dv),
                                                      num_warps=DELTA_WARPS)
    return delta


def read_unpack(out: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(the H heads' reads widened to fp32 (T, B, H dv), the other M reads bf16 (T, B, M dv)) of the contiguous bf16
    reads (T, B, H + M, dv)."""
    T, B, HQ, dv = (int(s) for s in out.shape)
    M = HQ - int(H)
    heads = torch.empty((T, B, H * dv), dtype=torch.float32, device=out.device)
    xr = torch.empty((T, B, M * dv), dtype=torch.bfloat16, device=out.device)
    _unpack_kernel[(triton.cdiv(T, DELTA_BT), B, HQ)](out, heads, xr, T, B, H, M, dv, BT=DELTA_BT, BD=_pow2(dv),
                                                      num_warps=DELTA_WARPS)
    return heads, xr


def read_delta_split(dh: torch.Tensor, dx: torch.Tensor, out32: torch.Tensor, H: int):
    """(dO (T, B, H + M, dv) bf16, D (B, H + M, T) fp32) from the heads' fp32 gradient (T, B, H dv), the extra
    queries' bf16 gradient (T, B, M dv) and the fp32 output."""
    T, B, HQ, dv = (int(s) for s in out32.shape)
    M = HQ - int(H)
    dh, dx = dh.contiguous(), dx.contiguous()
    if (dh.dtype != torch.float32 or tuple(dh.shape) != (T, B, H * dv) or dx.dtype != torch.bfloat16
            or tuple(dx.shape) != (T, B, M * dv)):
        raise RuntimeError(f"neuronspark.pool_pack: read_delta_split takes the fp32 ({T}, {B}, {H * dv}) and the bf16 "
                           f"({T}, {B}, {M * dv}) gradients, got {dh.dtype} {tuple(dh.shape)}, {dx.dtype} "
                           f"{tuple(dx.shape)}")
    do16 = torch.empty((T, B, HQ, dv), dtype=torch.bfloat16, device=out32.device)
    delta = torch.empty((B, HQ, T), dtype=torch.float32, device=out32.device)
    _delta_split_kernel[(triton.cdiv(T, DELTA_BT), B, HQ)](dh, dx, out32, do16, delta, T, B, H, M, dv, BT=DELTA_BT,
                                                           BD=_pow2(dv), num_warps=DELTA_WARPS)
    return do16, delta
