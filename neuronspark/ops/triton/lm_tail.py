"""Fused readout balance + cross entropy of the training loss (forward and backward) on the dense (R, V) logits.

The readout of the training regime is ``logits = bf16(rows @ E^T)`` (R, V) followed by the readout balance: per width
``w`` in (16, 32), in order, ``logits[r, v] = bf16(fp32(logits[r, v]) + fp32(c_w[r, v % w]))`` with the bf16
contributions ``c_w = bf16(rows @ W_w^T)`` (R, w) -- two separate roundings -- and the per-token loss
``F.cross_entropy(balanced.float(), labels, ignore_index=-100, reduction="none")``.  The two kernels here compute that
loss and its backward without materialising the balanced logits:

* forward (``_lf_fwd_kernel``, one program per row): every balanced value is formed on the fly (the fp32 add and the
  RNE to bf16 per width, in order), then a per-lane online softmax over lane groups of KV consecutive columns
  (``s <- s 2^((m - m') log2 e) + sum_k 2^((x_k - m') log2 e)``, ``ex2.approx.ftz``).  Outputs: the row max (a max is
  order free), ``logsum = logf(sum)`` with libdevice ``__nv_logf``'s instruction sequence (the sum in this kernel's
  order: about 1 ulp from ATen's ``cunn_SoftMaxForward`` order), and the per-token loss ``-1 * ((x_t - max) -
  logsum)`` in ATen's nll instruction sequence (+0 for ignored rows; an out-of-range target traps, as ATen's device
  assert).
* backward (``_lf_bwd_kernel``, one program per row): each balanced value again, then ATen's ``cunn_SoftMaxBackward``
  chain on the saved max / logsum: ``logp = (x - max) - logsum``, ``e = expf(logp)`` (libdevice's sequence),
  ``gI = fma(e, -sum, gO)`` with ``gO`` the nll gradient (``-1 * g`` at the target), rounded to bf16 into a new dense
  (R, V) gradient; and the gradient of each contribution, the width sum
  ``grad[:, :head].view(R, head // w, w).sum(1)`` plus the tail add ``bf16(out + grad[:, head:])``, accumulated in
  registers in the order of ATen's ``reduce_kernel<128, 4>`` on the contiguous head copy (the reduce configuration and
  the 32-bit row split are recomputed on the host exactly as ATen's ``setReduceConfig`` / ``SplitUntil32Bit`` choose
  them; a configuration outside the kernel's envelope raises).

Numerics: balanced values and the row max follow the reference chain's rounding points; logsum and the loss differ
from the ATen chain by the summation order of the softmax denominator (about 1e-7 relative); the logits gradient is
the ATen chain on the fused max / logsum; the width sums are ATen's reduction order on that gradient.  The reference
``balance_ce_per_token_ref`` is the torch chain itself.  Tests: the per-operator tolerance rule of
:mod:`neuronspark.utils.testing` against the reference.

Layout: the logits are the dense (R, V) bf16 tensor of the readout (unit column stride, any row stride >= V); the
gradient is a new dense (R, V) bf16 tensor, and the loop bounds run over V columns.  The launch configurations are the
fixed constants below.
"""
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from neuronspark.ops.triton.balance_add import balance_add_ref
from neuronspark.ops.triton._ptx import (  # noqa: F401  (Triton resolves the helpers through this module's globals)
    _ex2ftz as _lf_ex2,
    _rn_add_f32 as _lt_add_rn,
    _rn_fma_f32 as _lt_fma_rn,
    _rn_mul_f32 as _lt_mul_rn,
    _rn_sub_f32 as _lt_sub_rn,
    _sr_bf16_to_f32 as _lt_bf16_to_f32,
    _sr_f32_to_bf16 as _lt_f32_to_bf16,
    _sr_neg as _lt_neg,
)

__all__ = [
    "IGNORE_INDEX", "FWD_TILE_GROUPS", "FWD_TILE_COLS", "FWD_WARPS", "BWD_TILE", "BWD_WARPS", "BWD_MAXNREG",
    "BWD_UNROLL", "aten_mid_sum_config", "aten_mid_sum_chunks",
    "BalanceCEFn", "balance_ce_per_token", "balance_ce_per_token_ref",
]

IGNORE_INDEX = -100
LOG2E = tl.constexpr(1.4426950408889634)

# ================================================================================================ fixed launch constants
# forward: tiles of FWD_TILE_GROUPS lane groups x FWD_TILE_COLS consecutive columns, FWD_WARPS warps per row
FWD_TILE_GROUPS = 128
FWD_TILE_COLS = 8
FWD_WARPS = 4
# backward: columns per register tile, warps per row, register cap per thread, accumulator periods per loop iteration
BWD_TILE = 512
BWD_WARPS = 8
BWD_MAXNREG = 40
BWD_UNROLL = 2

# documented first-call cache: the backward launch plan per (rows, V, widths, tile, device)
_PLAN_CACHE: dict = {}


# ================================================================================================ PTX helpers

@triton.jit
def _lt_nll_neg(x):
    """nll_loss (weight-less) ``-cur_weight * x`` as the sm_100 SASS computes it: FADD -x, -RZ."""
    return tl.inline_asm_elementwise(asm="{ .reg .f32 t; neg.f32 t, $1; add.rn.f32 $0, t, 0f80000000; }",
                                     constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _lt_exp_parts(d):
    """libdevice __nv_expf(d) (no fast-math) up to its last multiply: returns (ex2, 2^j) with expf(d) = RN(ex2 * 2^j).
    PTX: fma.rn + cvt.sat (FFMA.SAT), fma.rm, add, neg, fma.rn x2, shl 23, ex2.approx.ftz."""
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .f32 t1, t2, t3, t4, t5;
        .reg .b32 sb;
        fma.rn.f32 t1, $2, 0f3BBB989D, 0f3F000000;
        cvt.sat.f32.f32 t1, t1;
        fma.rm.f32 t2, t1, 0f437C0000, 0f4B400001;
        add.rn.f32 t3, t2, 0fCB40007F;
        neg.f32 t3, t3;
        fma.rn.f32 t4, $2, 0f3FB8AA3B, t3;
        fma.rn.f32 t5, $2, 0f32A57060, t4;
        mov.b32 sb, t2;
        shl.b32 sb, sb, 23;
        mov.b32 $1, sb;
        ex2.approx.ftz.f32 $0, t5;
        }""",
        constraints="=f,=f,f", args=[d], dtype=(tl.float32, tl.float32), is_pure=True, pack=1)


@triton.jit
def _lt_logf(x):
    """libdevice __nv_logf (no fast-math, non-FTZ), instruction for instruction."""
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .pred p1, p2, p3;
        .reg .f32 xs, e0, ef, ex, f, p, q, r, t;
        .reg .b32 xb, i, eb, mb;
        setp.lt.f32 p1, $1, 0f00800000;
        mul.rn.f32 t, $1, 0f4B000000;
        selp.f32 xs, t, $1, p1;
        selp.f32 e0, 0fC1B80000, 0f00000000, p1;
        mov.b32 xb, xs;
        add.s32 i, xb, -1059760811;
        and.b32 eb, i, -8388608;
        sub.s32 mb, xb, eb;
        cvt.rn.f32.s32 ef, eb;
        fma.rn.f32 ex, ef, 0f34000000, e0;
        mov.b32 f, mb;
        add.rn.f32 f, f, 0fBF800000;
        fma.rn.f32 p, f, 0fBE055027, 0f3E1039F6;
        fma.rn.f32 p, p, f, 0fBDF8CDCC;
        fma.rn.f32 p, p, f, 0f3E0F2955;
        fma.rn.f32 p, p, f, 0fBE2AD8B9;
        fma.rn.f32 p, p, f, 0f3E4CED0B;
        fma.rn.f32 p, p, f, 0fBE7FFF22;
        fma.rn.f32 p, p, f, 0f3EAAAA78;
        fma.rn.f32 p, p, f, 0fBF000000;
        mul.rn.f32 q, f, p;
        fma.rn.f32 q, q, f, f;
        fma.rn.f32 r, ex, 0f3F317218, q;
        setp.gt.u32 p2, xb, 2139095039;
        fma.rn.f32 t, xs, 0f7F800000, 0f7F800000;
        selp.f32 r, t, r, p2;
        setp.eq.f32 p3, xs, 0f00000000;
        selp.f32 $0, 0fFF800000, r, p3;
        }""",
        constraints="=f,f", args=[x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _lt_trap_if(flag):
    """Device-side abort (the reference's device assert on an out-of-range target) when flag != 0."""
    return tl.inline_asm_elementwise(asm="{ .reg .pred p; setp.ne.s32 p, $1, 0; @p trap; mov.b32 $0, 0; }",
                                     constraints="=r,r", args=[flag], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _lt_row_of(v, i, W: tl.constexpr):
    """v[i, :] as a (1, W) tensor (a gather of the stored values)."""
    return tl.gather(v, tl.full([1, W], i, tl.int32), 0)


@triton.jit
def _lt_tree_y(v, base, H: tl.constexpr, W: tl.constexpr):
    """Reduce.cuh block_y_reduce over blockDim.y = H of the rows v[base + y]: for offset = H/2 .. 1,
    value[y] = combine(value[y], value[y + offset]) for y < offset; thread y = 0's value."""
    if H == 1:
        res = _lt_row_of(v, base, W)
    elif H == 2:
        res = _lt_add_rn(_lt_row_of(v, base, W), _lt_row_of(v, base + 1, W))
    else:
        tl.static_assert(H == 4)
        res = _lt_add_rn(_lt_add_rn(_lt_row_of(v, base, W), _lt_row_of(v, base + 2, W)),
                         _lt_add_rn(_lt_row_of(v, base + 1, W), _lt_row_of(v, base + 3, W)))
    return res


@triton.jit
def _lt_tree_y4(g0, g1, g2, g3, H: tl.constexpr):
    """block_y_reduce over H thread values given as separate tensors."""
    if H == 1:
        res = g0
    elif H == 2:
        res = _lt_add_rn(g0, g1)
    else:
        res = _lt_add_rn(_lt_add_rn(g0, g2), _lt_add_rn(g1, g3))
    return res


# ================================================================================================ forward

@triton.jit
def _lf_bal(x, c0, c1, NW: tl.constexpr):
    """The balance chain per width: fp32 add, RNE to bf16, back to fp32."""
    x = (x + c0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    if NW > 1:
        x = (x + c1).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    return x


@triton.jit
def _lf_fwd_kernel(X, C0, C1, T, MX, LS, LOSS, SX, IGNORE,
                   V: tl.constexpr, W0: tl.constexpr, W1: tl.constexpr, NW: tl.constexpr,
                   G: tl.constexpr, KV: tl.constexpr):
    """One row per program: balanced values on the fly, per-lane-group online softmax, row max / logsum / loss."""
    r = tl.program_id(0).to(tl.int64)
    xr = X + r * SX
    BLOCK: tl.constexpr = G * KV
    offs = tl.arange(0, G)[:, None] * KV + tl.arange(0, KV)[None, :]          # (G, KV) column offsets in a tile
    c0 = tl.load(C0 + r * W0 + (offs % W0)).to(tl.float32)                    # tile starts are multiples of BLOCK
    if NW > 1:
        c1 = tl.load(C1 + r * W1 + (offs % W1)).to(tl.float32)
    else:
        c1 = c0
    NEG_MAX: tl.constexpr = -3.4028234663852886e38
    m = tl.full([G], NEG_MAX, tl.float32)
    s = tl.zeros([G], tl.float32)
    NFULL: tl.constexpr = V // BLOCK
    for i in range(0, NFULL):
        x = _lf_bal(tl.load(xr + i * BLOCK + offs).to(tl.float32), c0, c1, NW)
        mn = tl.maximum(m, tl.max(x, axis=1))
        nb = mn * (-LOG2E)
        s = s * _lf_ex2((m - mn) * LOG2E) + tl.sum(_lf_ex2(x * LOG2E + nb[:, None]), axis=1)
        m = mn
    if V % BLOCK != 0:
        cols = NFULL * BLOCK + offs
        ok = cols < V
        x = _lf_bal(tl.load(xr + cols, mask=ok, other=0.0).to(tl.float32), c0, c1, NW)
        x = tl.where(ok, x, float("-inf"))
        mn = tl.maximum(m, tl.max(x, axis=1))
        nb = mn * (-LOG2E)
        s = s * _lf_ex2((m - mn) * LOG2E) + tl.sum(_lf_ex2(x * LOG2E + nb[:, None]), axis=1)
        m = mn
    mx = tl.max(m, axis=0, keep_dims=True)                                    # (1,): the row max
    ssum = tl.sum(s * _lf_ex2((m - mx) * LOG2E), axis=0, keep_dims=True)
    ls = _lt_logf(ssum)
    # the row's nll term on the balanced value of the target column
    one = tl.arange(0, 1)
    tg = tl.load(T + r + one)                                                 # (1,) int64
    valid = tg != IGNORE
    inb = (tg >= 0) & (tg < V)
    _lt_trap_if((valid & (inb == 0)).to(tl.int32))
    tgc = tl.where(valid & inb, tg, 0)
    xt = tl.load(xr + tgc).to(tl.float32)
    ct0 = tl.load(C0 + r * W0 + (tgc % W0)).to(tl.float32)
    if NW > 1:
        ct1 = tl.load(C1 + r * W1 + (tgc % W1)).to(tl.float32)
    else:
        ct1 = ct0
    xt = _lf_bal(xt, ct0, ct1, NW)
    lp = _lt_sub_rn(_lt_sub_rn(xt, mx), ls)
    loss = tl.where(valid, _lt_nll_neg(lp), 0.0)
    tl.store(MX + r + one, mx)
    tl.store(LS + r + one, ls)
    tl.store(LOSS + r + one, loss)


# ================================================================================================ backward

@triton.jit
def _lf_grad(x, c0, c1, cols, tg32, mx, ls, nsum, v, NW: tl.constexpr):
    """raw bf16 logits x -> fp32 gradient gI before the bf16 rounding (the cunn_SoftMaxBackward chain)."""
    xf = _lf_bal(x.to(tl.float32), c0, c1, NW)
    lp = _lt_sub_rn(_lt_sub_rn(xf, mx), ls)
    e2, sc = _lt_exp_parts(lp)
    e = _lt_mul_rn(e2, sc)                                                    # expf(logp)
    go = tl.where(cols == tg32, v, 0.0)
    return _lt_fma_rn(e, nsum, go)                                            # gradOutput - exp(output) * sum


@triton.jit
def _lf_bwd_tile(X, DX, rx, rd, cols, c0, c1, tg32, mx, ls, nsum, v,
                 V: tl.constexpr, P: tl.constexpr, MASKED: tl.constexpr, NW: tl.constexpr):
    """one tile: load, gradient, bf16 store; returns the stored values as fp32."""
    if MASKED:
        x = tl.load(X + rx + cols, mask=cols < V, other=0.0)
    else:
        x = tl.load(X + rx + cols)
    gi = _lf_grad(x, c0, c1, cols, tg32, mx, ls, nsum, v, NW)
    if MASKED:
        gi = tl.where(cols < V, gi, 0.0)
    gb = gi.to(tl.bfloat16, fp_downcast_rounding="rtne")
    if MASKED:
        tl.store(DX + rd + cols, gb, mask=cols < P)
    else:
        tl.store(DX + rd + cols, gb)
    return gb.to(tl.float32)


@triton.jit
def _lf_bwd_x(DX, rd, cols, x, c0, c1, tg32, mx, ls, nsum, v, NW: tl.constexpr):
    """one full tile from its loaded logits: gradient, bf16 store; returns the stored values as fp32."""
    gb = _lf_grad(x, c0, c1, cols, tg32, mx, ls, nsum, v, NW).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(DX + rd + cols, gb)
    return gb.to(tl.float32)


@triton.jit
def _lf_load8(xb, TILE: tl.constexpr, NSUB: tl.constexpr):
    """the NSUB tiles of one period (unused slots alias tile 0; never read)."""
    x0 = tl.load(xb, eviction_policy="evict_first")
    x1 = x0
    x2 = x0
    x3 = x0
    x4 = x0
    x5 = x0
    x6 = x0
    x7 = x0
    if NSUB > 1:
        x1 = tl.load(xb + TILE, eviction_policy="evict_first")
    if NSUB > 2:
        x2 = tl.load(xb + 2 * TILE, eviction_policy="evict_first")
    if NSUB > 3:
        x3 = tl.load(xb + 3 * TILE, eviction_policy="evict_first")
    if NSUB > 4:
        x4 = tl.load(xb + 4 * TILE, eviction_policy="evict_first")
    if NSUB > 5:
        x5 = tl.load(xb + 5 * TILE, eviction_policy="evict_first")
    if NSUB > 6:
        x6 = tl.load(xb + 6 * TILE, eviction_policy="evict_first")
    if NSUB > 7:
        x7 = tl.load(xb + 7 * TILE, eviction_policy="evict_first")
    return x0, x1, x2, x3, x4, x5, x6, x7


@triton.jit
def _lf_period8(DX, rd, cb, x0, x1, x2, x3, x4, x5, x6, x7, a0, a1, a2, a3, a4, a5, a6, a7,
                c0, c1, tg32, mx, ls, nsum, v,
                TILE: tl.constexpr, NSUB: tl.constexpr, NW: tl.constexpr):
    """one full period from its loaded tiles: gradients stored, tile s's values added to accumulator s."""
    a0 = _lt_add_rn(a0, _lf_bwd_x(DX, rd, cb, x0, c0, c1, tg32, mx, ls, nsum, v, NW))
    if NSUB > 1:
        a1 = _lt_add_rn(a1, _lf_bwd_x(DX, rd, cb + TILE, x1, c0, c1, tg32, mx, ls, nsum, v, NW))
    if NSUB > 2:
        a2 = _lt_add_rn(a2, _lf_bwd_x(DX, rd, cb + 2 * TILE, x2, c0, c1, tg32, mx, ls, nsum, v, NW))
    if NSUB > 3:
        a3 = _lt_add_rn(a3, _lf_bwd_x(DX, rd, cb + 3 * TILE, x3, c0, c1, tg32, mx, ls, nsum, v, NW))
    if NSUB > 4:
        a4 = _lt_add_rn(a4, _lf_bwd_x(DX, rd, cb + 4 * TILE, x4, c0, c1, tg32, mx, ls, nsum, v, NW))
    if NSUB > 5:
        a5 = _lt_add_rn(a5, _lf_bwd_x(DX, rd, cb + 5 * TILE, x5, c0, c1, tg32, mx, ls, nsum, v, NW))
    if NSUB > 6:
        a6 = _lt_add_rn(a6, _lf_bwd_x(DX, rd, cb + 6 * TILE, x6, c0, c1, tg32, mx, ls, nsum, v, NW))
    if NSUB > 7:
        a7 = _lt_add_rn(a7, _lf_bwd_x(DX, rd, cb + 7 * TILE, x7, c0, c1, tg32, mx, ls, nsum, v, NW))
    return a0, a1, a2, a3, a4, a5, a6, a7


@triton.jit
def _lf_last_add(acc, gl, cl, HEAD: tl.constexpr):
    """the last period's term of each accumulator (columns >= HEAD are not part of the width's head)."""
    return _lt_add_rn(acc, tl.where(cl < HEAD, gl, 0.0))


@triton.jit
def _lf_width_out(X, C0, C1, D, rx, r, tg32, mx, ls, nsum, v,
                  a0, a1, a2, a3, a4, a5, a6, a7,
                  V: tl.constexpr, W: tl.constexpr, W0: tl.constexpr, W1: tl.constexpr, NW: tl.constexpr,
                  H: tl.constexpr, CC: tl.constexpr, GP: tl.constexpr, TILE: tl.constexpr):
    """One width's value: the 4 S W accumulators (slot = position in the period = (j S + m) W + c, held as tile
    s = slot // TILE, lane slot % TILE) combined in ATen's reduce order (four interleaved accumulators, block_y_reduce,
    the CTA combine), then the tail add and the store."""
    S: tl.constexpr = H * CC
    j0 = tl.join(a0, a1)
    j1 = tl.join(a2, a3)
    j2 = tl.join(a4, a5)
    j3 = tl.join(a6, a7)
    flat = tl.reshape(tl.join(tl.join(j0, j1), tl.join(j2, j3)), (TILE * 8,))   # [t*8 + b*4 + a*2 + c] = tile 4c+2a+b
    e = tl.arange(0, GP * W)
    m = e // W
    cw = e % W
    ok = m < S
    slot = m * W + cw
    tv = tl.zeros([GP * W], tl.float32)
    for j in tl.static_range(4):
        sl = slot + j * S * W
        sub = sl // TILE
        t = sl % TILE
        idx = tl.where(ok, t * 8 + (sub % 2) * 4 + ((sub // 2) % 2) * 2 + sub // 4, 0)
        aj = tl.where(ok, tl.gather(flat, idx, 0), 0.0)
        if j == 0:
            tv = aj
        else:
            tv = _lt_add_rn(tv, aj)                                           # ((a0 + a1) + a2) + a3
    tv = tl.reshape(tv, (GP, W))
    if CC == 1:
        res = _lt_tree_y(tv, 0, H, W)
    else:
        g0 = tl.zeros([1, W], tl.float32)
        g1 = tl.zeros([1, W], tl.float32)
        g2 = tl.zeros([1, W], tl.float32)
        g3 = tl.zeros([1, W], tl.float32)
        for cta in tl.static_range(CC):
            u = _lt_tree_y(tv, H * cta, H, W)
            if cta % H == 0:
                g0 = _lt_add_rn(g0, u)
            elif cta % H == 1:
                g1 = _lt_add_rn(g1, u)
            elif cta % H == 2:
                g2 = _lt_add_rn(g2, u)
            else:
                g3 = _lt_add_rn(g3, u)
        res = _lt_tree_y4(g0, g1, g2, g3, H)
    ob = _lt_f32_to_bf16(res)                                                 # (1, W)
    HEAD: tl.constexpr = V // W * W
    c = tl.arange(0, W)[None, :]
    if HEAD < V:
        # the tail add: out = bf16(out + grad[r, HEAD + c]) (grad +0 for c >= tail); the tail's gradient values are
        # formed again from the logits (the same chain as the stored values)
        cols = HEAD + c
        okt = cols < V
        x = tl.load(X + rx + cols, mask=okt, other=0.0)
        ct0 = tl.load(C0 + r * W0 + (cols % W0)).to(tl.float32)
        if NW > 1:
            ct1 = tl.load(C1 + r * W1 + (cols % W1)).to(tl.float32)
        else:
            ct1 = ct0
        gi = _lf_grad(x, ct0, ct1, cols, tg32, mx, ls, nsum, v, NW)
        pad = tl.where(okt, gi.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32), 0.0)
        ob = _lt_f32_to_bf16(_lt_add_rn(_lt_bf16_to_f32(ob), pad))
    tl.store(D + r * W + c, ob)


@triton.jit
def _lf_bwd_kernel(X, C0, C1, T, MX, LS, G, DX, D0, D1, R0, SX, SD, SG, IGNORE,
                   V: tl.constexpr, P: tl.constexpr, W0: tl.constexpr, W1: tl.constexpr, NW: tl.constexpr,
                   H0: tl.constexpr, CC0: tl.constexpr, GP0: tl.constexpr,
                   H1: tl.constexpr, CC1: tl.constexpr, GP1: tl.constexpr,
                   TILE: tl.constexpr, NSUB: tl.constexpr, UNR: tl.constexpr):
    """One row per program (rows R0 + program id): the bf16 logits gradient and both width sums.  P is the row length
    of the gradient (= V for the dense layout)."""
    r = (tl.program_id(0) + R0).to(tl.int64)
    rx = r * SX
    rd = r * SD
    PERIOD: tl.constexpr = TILE * NSUB
    NPER: tl.constexpr = (P + PERIOD - 1) // PERIOD
    LAST: tl.constexpr = (NPER - 1) * PERIOD
    HEAD0: tl.constexpr = V // W0 * W0
    HEAD1: tl.constexpr = V // W1 * W1
    ar = tl.arange(0, TILE)
    c0 = tl.load(C0 + r * W0 + (ar % W0)).to(tl.float32)                      # tile starts are multiples of TILE
    if NW > 1:
        c1 = tl.load(C1 + r * W1 + (ar % W1)).to(tl.float32)
    else:
        c1 = c0
    z = tl.zeros([TILE], tl.int32)
    mx = tl.load(MX + r + z)
    ls = tl.load(LS + r + z)
    tg = tl.load(T + r + z)
    g = tl.load(G + r * SG + z)
    valid = tg != IGNORE
    v = _lt_nll_neg(g)                                                        # NllLossBackward0: -1 * g
    sumk = tl.where(valid, _lt_add_rn(tl.zeros_like(v), v), 0.0)              # the row sum of gO, exactly
    nsum = _lt_neg(sumk)
    tg32 = tl.where(valid, tg, -1).to(tl.int32)
    a0 = tl.zeros([TILE], tl.float32)
    a1 = tl.zeros([TILE], tl.float32)
    a2 = tl.zeros([TILE], tl.float32)
    a3 = tl.zeros([TILE], tl.float32)
    a4 = tl.zeros([TILE], tl.float32)
    a5 = tl.zeros([TILE], tl.float32)
    a6 = tl.zeros([TILE], tl.float32)
    a7 = tl.zeros([TILE], tl.float32)
    NFULL: tl.constexpr = NPER - 1                                            # full periods: every column < every head
    for q in range(0, NFULL // UNR):
        # the tile loads of UNR periods are issued first (the stores in between may alias X for the compiler)
        pb = q * (UNR * PERIOD)
        x0, x1, x2, x3, x4, x5, x6, x7 = _lf_load8(X + rx + pb + ar, TILE, NSUB)
        if UNR > 1:
            y0, y1, y2, y3, y4, y5, y6, y7 = _lf_load8(X + rx + pb + PERIOD + ar, TILE, NSUB)
        a0, a1, a2, a3, a4, a5, a6, a7 = _lf_period8(DX, rd, pb + ar, x0, x1, x2, x3, x4, x5, x6, x7,
                                                     a0, a1, a2, a3, a4, a5, a6, a7, c0, c1, tg32, mx, ls, nsum,
                                                     v, TILE, NSUB, NW)
        if UNR > 1:
            a0, a1, a2, a3, a4, a5, a6, a7 = _lf_period8(DX, rd, pb + PERIOD + ar, y0, y1, y2, y3, y4, y5, y6, y7,
                                                         a0, a1, a2, a3, a4, a5, a6, a7, c0, c1, tg32, mx, ls,
                                                         nsum, v, TILE, NSUB, NW)
    if NFULL % UNR != 0:
        pb = (NFULL // UNR) * (UNR * PERIOD)
        x0, x1, x2, x3, x4, x5, x6, x7 = _lf_load8(X + rx + pb + ar, TILE, NSUB)
        a0, a1, a2, a3, a4, a5, a6, a7 = _lf_period8(DX, rd, pb + ar, x0, x1, x2, x3, x4, x5, x6, x7,
                                                     a0, a1, a2, a3, a4, a5, a6, a7, c0, c1, tg32, mx, ls, nsum,
                                                     v, TILE, NSUB, NW)
    # last period (masked): its term goes to width w's accumulator only below that width's head
    b0 = a0
    b1 = a1
    b2 = a2
    b3 = a3
    b4 = a4
    b5 = a5
    b6 = a6
    b7 = a7
    for s in tl.static_range(NSUB):
        if LAST + s * TILE < P:
            cl = LAST + s * TILE + ar
            gl = _lf_bwd_tile(X, DX, rx, rd, cl, c0, c1, tg32, mx, ls, nsum, v, V, P, True, NW)
            if s == 0:
                a0 = _lf_last_add(a0, gl, cl, HEAD0)
                b0 = _lf_last_add(b0, gl, cl, HEAD1)
            elif s == 1:
                a1 = _lf_last_add(a1, gl, cl, HEAD0)
                b1 = _lf_last_add(b1, gl, cl, HEAD1)
            elif s == 2:
                a2 = _lf_last_add(a2, gl, cl, HEAD0)
                b2 = _lf_last_add(b2, gl, cl, HEAD1)
            elif s == 3:
                a3 = _lf_last_add(a3, gl, cl, HEAD0)
                b3 = _lf_last_add(b3, gl, cl, HEAD1)
            elif s == 4:
                a4 = _lf_last_add(a4, gl, cl, HEAD0)
                b4 = _lf_last_add(b4, gl, cl, HEAD1)
            elif s == 5:
                a5 = _lf_last_add(a5, gl, cl, HEAD0)
                b5 = _lf_last_add(b5, gl, cl, HEAD1)
            elif s == 6:
                a6 = _lf_last_add(a6, gl, cl, HEAD0)
                b6 = _lf_last_add(b6, gl, cl, HEAD1)
            else:
                a7 = _lf_last_add(a7, gl, cl, HEAD0)
                b7 = _lf_last_add(b7, gl, cl, HEAD1)
    # the row's values again as (1, 1) tensors for the tail recompute (broadcast against (1, W))
    z1 = tl.zeros([1, 1], tl.int32)
    mx1 = tl.load(MX + r + z1)
    ls1 = tl.load(LS + r + z1)
    g1 = tl.load(G + r * SG + z1)
    tgs = tl.load(T + r + z1)
    valid1 = tgs != IGNORE
    v1 = _lt_nll_neg(g1)
    sumk1 = tl.where(valid1, _lt_add_rn(tl.zeros_like(v1), v1), 0.0)
    nsum1 = _lt_neg(sumk1)
    tg1 = tl.where(valid1, tgs, -1).to(tl.int32)
    _lf_width_out(X, C0, C1, D0, rx, r, tg1, mx1, ls1, nsum1, v1, a0, a1, a2, a3, a4, a5, a6, a7,
                  V, W0, W0, W1, NW, H0, CC0, GP0, TILE)
    if NW > 1:
        _lf_width_out(X, C0, C1, D1, rx, r, tg1, mx1, ls1, nsum1, v1, b0, b1, b2, b3, b4, b5, b6, b7,
                      V, W1, W0, W1, NW, H1, CC1, GP1, TILE)


# ================================================================================================ ATen reduce emulation

def _last_pow2(n: int) -> int:
    for sh in (1, 2, 4, 8, 16):
        n |= n >> sh
    return max(1, n - (n >> 1))


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def aten_mid_sum_config(rows: int, reps: int, w: int, elem_offset: int, num_mp: int, max_threads_mp: int) -> dict:
    """setReduceConfig<float, BFloat16, vt0 4, input_vec_size 4> (Reduce.cuh of torch 2.9.1) for
    x.view(rows, reps, w).sum(1) of a contiguous bf16 (rows, reps, w) tensor whose data pointer is elem_offset elements
    past a 16-byte aligned address: TensorIterator dims [reps (reduced, stride w), w (stride 1), rows (stride reps w)],
    no coalescing; the reduction is not on the fastest dimension, so "vectorize along output", set_block_dimension,
    split_output(block_width), split across warps or not, then the global-reduction CTA count.  Returns the per-output
    order parameters H (warps splitting the input), C (CTAs per output) and the launch shape."""
    num_outputs, ipo = rows * w, reps
    vec = 4
    for n in (elem_offset, w, w, reps * w):          # base address, shape[output dim], input strides of dims 0 and 2
        while n % vec:
            vec //= 2
    dim0, dim1 = num_outputs // vec, ipo
    mnt = 512 // vec
    d0 = _last_pow2(dim0) if dim0 < mnt else mnt
    d1 = _last_pow2(dim1) if dim1 < mnt else mnt
    bw = min(d0, 32)
    bh = min(d1, mnt // bw)
    bw = min(d0, mnt // bh)
    step_input, step_output = 1, bw
    split = _div_up(ipo, step_input) >= min(bh * 16, 256)
    if split:
        H = bh
        step_input *= bh
    else:
        H = 1
        step_output *= bh
    target = num_mp * (max_threads_mp // (bw * bh))
    grid_x = _div_up(num_outputs // vec, step_output)
    C = 1
    vpt = _div_up(ipo, step_input)
    if split and vpt >= 256 and grid_x <= target:
        C = max(min(_div_up(target, grid_x), _div_up(vpt, 16)), _div_up(vpt, 256))
    return {"vec": vec, "block": (bw, bh), "H": H, "C": C, "S": H * C, "grid": (grid_x, C), "split": split}


def aten_mid_sum_chunks(rows: int, reps: int, w: int) -> list:
    """The row ranges of the 32-bit sub-iterators ATen reduces separately (SplitUntil32Bit: split the dimension of
    largest extent -- rows -- in halves, first half first, until can_use_32bit_indexing holds)."""
    imax = 2**31 - 1

    def ok(nr):
        return (nr * reps * w <= imax and 1 + (reps - 1) * 2 * w + (w - 1) * 2 + (nr - 1) * 2 * reps * w <= imax
                and 1 + (w - 1) * 2 + (nr - 1) * 2 * w <= imax)
    out = []

    def rec(start, nr):
        if ok(nr):
            out.append((start, nr))
            return
        if not (nr - 1) * reps * w >= max(reps - 1, w - 1):
            raise RuntimeError("balance_ce_per_token: the 32-bit split would not split the rows dimension")
        cs = nr // 2
        rec(start, cs)
        rec(start + cs, nr - cs)
    rec(0, rows)
    return out


# ================================================================================================ host side

def _check(logits, labels, contribs, widths) -> Tuple[int, int, Tuple[int, ...]]:
    if not isinstance(logits, torch.Tensor) or not logits.is_cuda or logits.dtype != torch.bfloat16:
        raise RuntimeError("balance_ce_per_token: logits must be a CUDA bfloat16 tensor")
    if logits.dim() != 2 or logits.stride(1) != 1 or (logits.shape[0] > 1 and logits.stride(0) < logits.shape[1]):
        raise RuntimeError("balance_ce_per_token: logits must be a 2-D (R, V) tensor with unit column stride and a "
                           f"row stride >= V, got shape {tuple(logits.shape)} strides {tuple(logits.stride())}")
    R, V = int(logits.shape[0]), int(logits.shape[1])
    if R < 2 or R >= 2**31 - 1 or R * max(V, int(logits.stride(0))) >= 2**62:
        raise RuntimeError(f"balance_ce_per_token: unsupported size R={R} V={V}")
    widths = tuple(int(w) for w in widths)
    if not 1 <= len(widths) <= 2 or len(contribs) != len(widths):
        raise RuntimeError(f"balance_ce_per_token: 1 or 2 width contributions are supported, got widths {widths}, "
                           f"{len(contribs)} contributions")
    for w, t in zip(widths, contribs):
        if w < 4 or w & (w - 1) or V // w < 2:
            raise RuntimeError(f"balance_ce_per_token: width {w} must be a power of two >= 4 with V // w >= 2")
        if (not isinstance(t, torch.Tensor) or t.dtype != torch.bfloat16 or t.device != logits.device
                or tuple(t.shape) != (R, w) or not t.is_contiguous()):
            raise RuntimeError(f"balance_ce_per_token: the contribution of width {w} must be a contiguous CUDA bf16 "
                               f"({R}, {w}) tensor, got {getattr(t, 'dtype', None)} {tuple(getattr(t, 'shape', ()))}")
    if (not isinstance(labels, torch.Tensor) or labels.device != logits.device or labels.dtype != torch.int64
            or labels.dim() != 1 or int(labels.shape[0]) != R or not labels.is_contiguous()):
        raise RuntimeError("balance_ce_per_token: labels must be a contiguous int64 (rows,) tensor on the logits' "
                           "device")
    return R, V, widths


def _bwd_plan(R: int, V: int, widths: Tuple[int, ...], tile: int, device) -> list:
    """Per launch: (row start, rows, ((H, C, GP) per width), NSUB): ATen's reduce configuration of every 32-bit row
    chunk, consecutive chunks with equal configurations merged into one launch."""
    props = torch.cuda.get_device_properties(device)
    per_chunk = []
    for w in widths:
        reps = V // w
        rows = []
        for start, nr in aten_mid_sum_chunks(R, reps, w):
            cfg = aten_mid_sum_config(nr, reps, w, start * reps * w, props.multi_processor_count,
                                      props.max_threads_per_multi_processor)
            if cfg["vec"] != 4 or cfg["H"] not in (1, 2, 4) or cfg["C"] > 64 or reps < cfg["S"]:
                raise RuntimeError(f"balance_ce_per_token: rows {start}..{start + nr} reps {reps} width {w}: ATen's "
                                   f"reduce config {cfg} is outside the kernel's envelope")
            rows.append((start, nr, cfg))
        per_chunk.append(rows)
    # split points = union of every width's chunk starts
    cuts = sorted({s for rows in per_chunk for (s, _, _) in rows} | {R})
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        cfgs = []
        for w, rows in zip(widths, per_chunk):
            cfg = next(c for (s, n, c) in rows if s <= a < s + n)
            gp = 1
            while gp < cfg["S"]:
                gp *= 2
            cfgs.append((cfg["H"], cfg["C"], gp, 4 * cfg["S"] * w))
        periods = {c[3] for c in cfgs}
        if len(periods) != 1:
            raise RuntimeError(f"balance_ce_per_token: rows {a}..{b}: the widths' accumulator periods differ {cfgs}")
        period = periods.pop()
        if period % tile or period // tile > 8:
            raise RuntimeError(f"balance_ce_per_token: period {period} is not 1..8 tiles of {tile}")
        P = V                                                                  # the dense gradient's row length
        nper = (P + period - 1) // period
        hmin = min(V // w * w for w in widths)
        if (nper - 1) * period > hmin:
            raise RuntimeError(f"balance_ce_per_token: the last period ({(nper - 1) * period}) starts past a width's "
                               f"head ({hmin})")
        key = tuple(c[:3] for c in cfgs)
        if segs and segs[-1][2] == key and segs[-1][0] + segs[-1][1] == a:
            segs[-1] = (segs[-1][0], segs[-1][1] + (b - a), key, period // tile)
        else:
            segs.append((a, b - a, key, period // tile))
    return segs


def _balance_ce_fwd(logits: torch.Tensor, labels: torch.Tensor, contribs: Sequence[torch.Tensor],
                    widths: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (loss (R,) fp32, rowmax (R,) fp32, logsum (R,) fp32)."""
    R, V, widths = _check(logits, labels, contribs, widths)
    G, KV, WARPS = FWD_TILE_GROUPS, FWD_TILE_COLS, FWD_WARPS
    if (G * KV) % 32 or (G * KV) > V:
        raise RuntimeError(f"balance_ce_per_token: forward tile {G} x {KV} must be a multiple of 32 and <= V")
    dev = logits.device
    loss = torch.empty(R, dtype=torch.float32, device=dev)
    mx = torch.empty(R, dtype=torch.float32, device=dev)
    ls = torch.empty(R, dtype=torch.float32, device=dev)
    t1 = contribs[1] if len(contribs) > 1 else contribs[0]
    _lf_fwd_kernel[(R,)](logits, contribs[0], t1, labels, mx, ls, loss, int(logits.stride(0)), IGNORE_INDEX,
                         V=V, W0=widths[0], W1=widths[-1], NW=len(widths), G=G, KV=KV, num_warps=WARPS)
    return loss, mx, ls


def _balance_ce_bwd(logits: torch.Tensor, labels: torch.Tensor, mx: torch.Tensor, ls: torch.Tensor, g: torch.Tensor,
                    contribs: Sequence[torch.Tensor], widths: Sequence[int]) -> List[torch.Tensor]:
    """-> [the bf16 logits gradient (R, V) dense, the width sums (R, w) bf16 per width]."""
    R, V, widths = _check(logits, labels, contribs, widths)
    if g.dim() != 1 or int(g.shape[0]) != R or g.dtype != torch.float32 or g.device != logits.device:
        raise RuntimeError(f"balance_ce_per_token: the upstream gradient must be (rows,) float32, got {g.dtype} "
                           f"{tuple(g.shape)} {g.device}")
    for name, t in (("rowmax", mx), ("logsum", ls)):
        if t.dtype != torch.float32 or tuple(t.shape) != (R,) or not t.is_contiguous():
            raise RuntimeError(f"balance_ce_per_token: the saved {name} must be a contiguous ({R},) fp32 tensor")
    TILE, UNR = BWD_TILE, BWD_UNROLL
    key = (R, V, widths, TILE, logits.device.index)
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _bwd_plan(R, V, widths, TILE, logits.device)
        _PLAN_CACHE[key] = plan
    dev = logits.device
    P = V
    grad = torch.empty((R, P), dtype=torch.bfloat16, device=dev)
    outs = [torch.empty((R, w), dtype=torch.bfloat16, device=dev) for w in widths]
    t1 = contribs[1] if len(contribs) > 1 else contribs[0]
    o1 = outs[1] if len(outs) > 1 else outs[0]
    for start, nr, ccfg, nsub in plan:
        (h0, c0, gp0) = ccfg[0]
        (h1, c1, gp1) = ccfg[-1]
        _lf_bwd_kernel[(nr,)](logits, contribs[0], t1, labels, mx, ls, g, grad, outs[0], o1, start,
                              int(logits.stride(0)), P, int(g.stride(0)), IGNORE_INDEX,
                              V=V, P=P, W0=widths[0], W1=widths[-1], NW=len(widths),
                              H0=h0, CC0=c0, GP0=gp0, H1=h1, CC1=c1, GP1=gp1,
                              TILE=TILE, NSUB=nsub, UNR=UNR, num_warps=BWD_WARPS, enable_fp_fusion=False,
                              maxnreg=BWD_MAXNREG)
    return [grad] + outs


@torch.library.custom_op("neuronspark::balance_ce_fwd", mutates_args=())
def _balance_ce_fwd_op(logits: torch.Tensor, labels: torch.Tensor, contribs: List[torch.Tensor],
                       widths: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _balance_ce_fwd(logits, labels, list(contribs), list(widths))


@_balance_ce_fwd_op.register_fake
def _balance_ce_fwd_fake(logits, labels, contribs, widths):
    R = logits.shape[0]
    return (logits.new_empty((R,), dtype=torch.float32), logits.new_empty((R,), dtype=torch.float32),
            logits.new_empty((R,), dtype=torch.float32))


@torch.library.custom_op("neuronspark::balance_ce_bwd", mutates_args=())
def _balance_ce_bwd_op(logits: torch.Tensor, labels: torch.Tensor, mx: torch.Tensor, ls: torch.Tensor,
                       g: torch.Tensor, contribs: List[torch.Tensor], widths: List[int]) -> List[torch.Tensor]:
    return _balance_ce_bwd(logits, labels, mx, ls, g, list(contribs), list(widths))


@_balance_ce_bwd_op.register_fake
def _balance_ce_bwd_fake(logits, labels, mx, ls, g, contribs, widths):
    R, V = logits.shape[0], logits.shape[1]
    return ([logits.new_empty((R, V), dtype=torch.bfloat16)]
            + [logits.new_empty((R, int(w)), dtype=torch.bfloat16) for w in widths])


class BalanceCEFn(torch.autograd.Function):
    """Per-token cross entropy of the balanced logits; backward: the logits gradient (dense (R, V) bf16) and every
    contribution's gradient (the width sums, (R, w) bf16)."""

    @staticmethod
    def forward(ctx, logits, labels, widths, *contribs):
        ctx.set_materialize_grads(False)
        loss, mx, ls = torch.ops.neuronspark.balance_ce_fwd(logits, labels, list(contribs), list(widths))
        ctx.widths = tuple(widths)
        ctx.save_for_backward(logits, labels, mx, ls, *contribs)
        return loss

    @staticmethod
    def backward(ctx, g):
        n = len(ctx.widths)
        if g is None:
            return (None, None, None) + (None,) * n
        logits, labels, mx, ls, *contribs = ctx.saved_tensors
        grad, *outs = torch.ops.neuronspark.balance_ce_bwd(logits, labels, mx, ls, g, list(contribs),
                                                            list(ctx.widths))
        return (grad if ctx.needs_input_grad[0] else None, None, None,
                *(o if ctx.needs_input_grad[3 + i] else None for i, o in enumerate(outs)))


def balance_ce_per_token(logits: torch.Tensor, labels: torch.Tensor, contribs: Sequence[torch.Tensor],
                         widths: Sequence[int]) -> torch.Tensor:
    """Per-token training loss (R,) fp32 of the balanced logits.

    logits (R, V) bf16 (unit column stride, row stride >= V; R >= 2): the raw readout; labels (R,) int64 contiguous,
    -100 ignored (+0 loss), every other value in [0, V); contribs ((R, 16), (R, 32)) bf16 contiguous: the balance
    contributions; widths (16, 32), in application order.  Differentiable in the logits and the contributions."""
    widths = tuple(int(w) for w in widths)
    contribs = tuple(contribs)
    _check(logits, labels, contribs, widths)
    return BalanceCEFn.apply(logits, labels, widths, *contribs)


def balance_ce_per_token_ref(logits: torch.Tensor, labels: torch.Tensor, contribs: Sequence[torch.Tensor],
                             widths: Sequence[int]) -> torch.Tensor:
    """``F.cross_entropy(balance_add_ref(logits).float(), labels, ignore_index=-100, reduction="none")``: the torch
    chain the fused kernels replace (differentiable through autograd).  Same arguments and output as
    :func:`balance_ce_per_token`."""
    with torch.autocast("cuda", enabled=False):
        x = balance_add_ref(logits, contribs, widths)
        return F.cross_entropy(x.float(), labels, ignore_index=IGNORE_INDEX, reduction="none")
