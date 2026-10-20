"""Residual and value glue of the hippocampal layer (training).

Operators (registry names):

* ``value_residual(v, lam, carry, glu_add)``: the value residual and the GLU tail of the last head::

      v' = bf16(float(v) * (1 + lam))           (hippocampal layer 0: carry is None)
      v' = bf16(float(v) + lam * float(carry))  (later layers: carry = layer 0's value stream)
      out = cat([v'[..., :C - d], bf16(float(v'[..., C - d:]) + glu_add)], -1)    (d = glu_add.shape[-1])

  ``v`` bf16 (T, B, C), ``lam`` fp32 broadcastable to (T, B, C), ``carry`` bf16 (T, B, C) or None, ``glu_add`` fp32
  (T, B, d).
* ``zc_center(z, zcum)``: ``((z - zcum / max(t, 1)) * (t > 0)).contiguous()`` for fp32 (T, B, C) ``z`` and its
  strictly-past running sum ``zcum`` (the value-residual gate's centred input).
* ``read_gain_residual(h, r, f)``: ``h + bf16(float(r) * (1 + f))`` for fp32 ``h`` (T, B, C), the centred bf16
  attention residual ``r`` and the fp32 (T, B) read-gain factor ``f``.  The gained residual is one kernel; the residual
  add is the torch line ``h + rg``.
* ``pkm_tail(h, ffn, y, rho)``: ``h + (ffn + (y_g - mean_c(y_g)))`` with ``y_g = bf16(float(y) * rho)`` (bf16 sub / add,
  fp32 residual); ``rho`` fp32 (T, B) without gradient (the fixed copy-gate strength).

Numerics contract
-----------------
Forwards: the same IEEE operations at the same rounding points as the torch lines (inline PTX ``add / sub / mul / div
.rn.f32``, exact conversions, no FMA contraction, ``enable_fp_fusion=False``); ``pkm_tail``'s bf16 mean is formed in
ATen's ``reduce_kernel`` order (one warp per row), so the forwards follow the eager chain's order of operations.
Backwards: the autograd chain's roundings (the bf16 casts, the "+ 0" of zero-filled slice gradients), with the row
sums of the (T, B, 1) factors formed in registers (fp32 reduction-order class); ``value_residual`` / ``zc_center``
backwards are elementwise.  ``read_gain_residual``'s factor gradient takes the bf16 gradient of ``rg`` that autograd
delivers for the fp32 add.

Launchers are ``torch.library`` custom ops (``neuronspark::hippo_*``) with fake implementations of the same layouts.
Pure-torch references ``*_ref`` (same parameters): the torch lines of the centring, the value residual and GLU tail,
the read gain and the PKM tail.
"""
from typing import Optional

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton.hippo_norms import (  # noqa: F401  (jit helpers resolved through this module's globals)
    XUP,
    XVEC,
    _g_add,
    _g_bits,
    _g_div,
    _g_max,
    _g_mul,
    _g_n,
    _g_neg,
    _g_r16,
    _g_sub,
    _g_w,
    _invb,
    _meta,
    _pow2,
    _rows,
    _s2,
    _s3,
    _xcfg,
    _xrow_sum2,
)

__all__ = [
    "ValueResidualFn", "value_residual", "value_residual_ref",
    "ZcCenterFn", "zc_center", "zc_center_ref",
    "ReadGainFn", "read_gain_residual", "read_gain_residual_ref",
    "PkmTailFn", "pkm_tail", "pkm_tail_ref", "PKM_TAIL_MIN_ROWS",
]

# ------------------------------------------------------------------------------------------------ fixed constants
VRES_BT, VRES_BC, VRES_WARPS = 64, 64, 4   # value_residual (t tile, channel tile, warps)
FLAT_BLOCK, FLAT_WARPS = 1024, 4           # flat elementwise kernels (zc_center, the read-gain forward)
ROW_BR, ROW_WARPS = 4, 8                   # row kernels of the backwards (read gain, PKM tail)
PKT_FWDX_ROWS, PKT_FWDX_WARPS = 1, 4       # PKM tail fused forward (exact-order bf16 mean)


# ================================================================================================ kernels
@triton.jit
def _vres_fwd_kernel(V, XC, LAM, GV, O, T, B, D, DT,
                     vs0, vs1, vs2, xs0, xs1, xs2, ls0, ls1, ls2, gs0, gs1, gs2, os0, os1, os2,
                     FIRST: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """out = v' (channels < D - DT) | bf16(float(v') + gv) (the last DT channels), v' = bf16(v + lam xc) or
    bf16(v (1 + lam)); program (t tile, b, c tile)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    b = tl.program_id(1).to(tl.int64)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    t2 = t.to(tl.int64)[:, None]
    c2 = c.to(tl.int64)[None, :]
    m = (t < T)[:, None] & (c < D)[None, :]
    vf = _g_w(tl.load(V + (t2 * vs0 + b * vs1 + c2 * vs2), mask=m, other=0.0))
    lam = tl.load(LAM + (t2 * ls0 + b * ls1 + c2 * ls2), mask=m, other=0.0)
    if FIRST:
        val = _g_r16(_g_mul(vf, _g_add(1.0, lam)))
    else:
        xf = _g_w(tl.load(XC + (t2 * xs0 + b * xs1 + c2 * xs2), mask=m, other=0.0))
        val = _g_r16(_g_add(vf, _g_mul(lam, xf)))
    tail = (c2 >= D - DT) & m
    gv = tl.load(GV + (t2 * gs0 + b * gs1 + (c2 - (D - DT)) * gs2), mask=tail, other=0.0)
    val = tl.where(tail, _g_add(val, gv), val)
    tl.store(O + (t2 * os0 + b * os1 + c2 * os2), _g_n(val), mask=m)


@triton.jit
def _vres_bwd_kernel(G, V, XC, LAM, GVO, GXC, GL, GGV, T, B, D, DT,
                     gs0, gs1, gs2, vs0, vs1, vs2, xs0, xs1, xs2, ls0, ls1, ls2,
                     as0, as1, as2, cs0, cs1, cs2, ps0, ps1, ps2, qs0, qs1, qs2,
                     FIRST: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """gv's gradient float(g) of the tail; v''s gradient canon(g) (the two slice gradients summed, -0 -> +0); v's
    float(canon(g)) cast back (later layers) or bf16(gf (1 + lam)) (layer 0); the carry's bf16(gf lam); lam's gf xc
    (layer 0: gf v)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    b = tl.program_id(1).to(tl.int64)
    c = tl.program_id(2) * BC + tl.arange(0, BC)
    t2 = t.to(tl.int64)[:, None]
    c2 = c.to(tl.int64)[None, :]
    m = (t < T)[:, None] & (c < D)[None, :]
    gf = _g_w(tl.load(G + (t2 * gs0 + b * gs1 + c2 * gs2), mask=m, other=0.0))
    tail = (c2 >= D - DT) & m
    tl.store(GGV + (t2 * qs0 + b * qs1 + (c2 - (D - DT)) * qs2), gf, mask=tail)    # gv: float(g) of the tail
    g2 = _g_r16(_g_add(gf, 0.0))                       # v''s gradient: the two slice gradients summed (canon)
    lam = tl.load(LAM + (t2 * ls0 + b * ls1 + c2 * ls2), mask=m, other=0.0)
    vf = _g_w(tl.load(V + (t2 * vs0 + b * vs1 + c2 * vs2), mask=m, other=0.0))
    if FIRST:
        tl.store(GVO + (t2 * as0 + b * as1 + c2 * as2), _g_n(_g_mul(g2, _g_add(1.0, lam))), mask=m)
        tl.store(GL + (t2 * ps0 + b * ps1 + c2 * ps2), _g_mul(g2, vf), mask=m)
    else:
        xf = _g_w(tl.load(XC + (t2 * xs0 + b * xs1 + c2 * xs2), mask=m, other=0.0))
        tl.store(GVO + (t2 * as0 + b * as1 + c2 * as2), _g_n(g2), mask=m)
        tl.store(GXC + (t2 * cs0 + b * cs1 + c2 * cs2), _g_n(_g_mul(g2, lam)), mask=m)
        tl.store(GL + (t2 * ps0 + b * ps1 + c2 * ps2), _g_mul(g2, xf), mask=m)


@triton.jit
def _zc_fwd_kernel(Z, ZC, O, N, S0, S1, T0, zs0, zs1, zs2, cs0, cs1, cs2, os0, os1, os2, TAX: tl.constexpr,
                   BLOCK: tl.constexpr):
    """o = (z - zcum / max(t, 1)) * (t > 0) over flat elements (TAX = the position of the time axis in the order)."""
    lin = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = lin < N
    i0 = lin % S0
    rr = lin // S0
    i1 = rr % S1
    i2 = rr // S1
    if TAX == 0:
        t = i0
    elif TAX == 1:
        t = i1
    else:
        t = i2
    tf = (t + T0).to(tl.float32)
    zn = _g_max(tf, 1.0)                                  # arange(T).clamp_min(1.0)
    mk = tl.where(t + T0 > 0, 1.0, 0.0)                   # (arange(T) > 0).to(fp32)
    zv = tl.load(Z + (i0 * zs0 + i1 * zs1 + i2 * zs2), mask=m, other=0.0)
    cv = tl.load(ZC + (i0 * cs0 + i1 * cs1 + i2 * cs2), mask=m, other=0.0)
    tl.store(O + (i0 * os0 + i1 * os1 + i2 * os2), _g_mul(_g_sub(zv, _g_div(cv, zn)), mk), mask=m)


@triton.jit
def _zc_bwd_kernel(G, GZ, GC, N, S0, S1, T0, gs0, gs1, gs2, as0, as1, as2, cs0, cs1, cs2, TAX: tl.constexpr,
                   BLOCK: tl.constexpr):
    """gz = g * mask (MulBackward0, passed on by SubBackward0); gc = (-(g * mask)) / zn (DivBackward0)."""
    lin = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = lin < N
    i0 = lin % S0
    rr = lin // S0
    i1 = rr % S1
    i2 = rr // S1
    if TAX == 0:
        t = i0
    elif TAX == 1:
        t = i1
    else:
        t = i2
    zn = _g_max((t + T0).to(tl.float32), 1.0)
    mk = tl.where(t + T0 > 0, 1.0, 0.0)
    gm = _g_mul(tl.load(G + (i0 * gs0 + i1 * gs1 + i2 * gs2), mask=m, other=0.0), mk)   # MulBackward0: grad * mask
    tl.store(GZ + (i0 * as0 + i1 * as1 + i2 * as2), gm, mask=m)                          # SubBackward0: self
    tl.store(GC + (i0 * cs0 + i1 * cs1 + i2 * cs2), _g_div(_g_neg(gm), zn), mask=m)      # other: (-g) / zn


@triton.jit
def _rg_only_fwd_kernel(R, F, RO, N, S0, S1, rs0, rs1, rs2, fs0, fs1, fs2, qs0, qs1, qs2, BLOCK: tl.constexpr):
    """rg = bf16(float(r) (1 + f)): add.rn(1, f), mul.rn, cvt.rn.bf16."""
    lin = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = lin < N
    i0 = lin % S0
    rr = lin // S0
    i1 = rr % S1
    i2 = rr // S1
    rv = _g_w(tl.load(R + (i0 * rs0 + i1 * rs1 + i2 * rs2), mask=m, other=0.0))
    f1 = _g_add(1.0, tl.load(F + (i0 * fs0 + i1 * fs1 + i2 * fs2), mask=m, other=0.0))   # 1.0 + f
    tl.store(RO + (i0 * qs0 + i1 * qs1 + i2 * qs2), _g_n(_g_mul(rv, f1)), mask=m)        # (r.float() * (1 + f)).to(bf16)


@triton.jit
def _rg_bwd16_kernel(G, RR, F, GR, PF, R, B, D, gs0, gs1, gs2, rs0, rs1, rs2, fs0, fs1, as0, as1, as2, ps0, ps1,
                     BR: tl.constexpr, BD: tl.constexpr):
    """G = the bf16 gradient of rg: gf = float(G); gr = bf16(gf (1 + f)); PF[t, b] = sum_c gf r."""
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < D
    m = rm[:, None] & cm[None, :]
    gf = _g_w(tl.load(G + (t[:, None] * gs0 + b[:, None] * gs1 + c[None, :] * gs2), mask=m, other=0.0))
    rv = _g_w(tl.load(RR + (t[:, None] * rs0 + b[:, None] * rs1 + c[None, :] * rs2), mask=m, other=0.0))
    f1 = tl.broadcast_to(_g_add(1.0, tl.load(F + (t * fs0 + b * fs1), mask=rm, other=0.0))[:, None], (BR, BD))
    tl.store(GR + (t[:, None] * as0 + b[:, None] * as1 + c[None, :] * as2), _g_n(_g_mul(gf, f1)), mask=m)
    tl.store(PF + (t * ps0 + b * ps1), tl.sum(_g_mul(gf, rv), axis=1), mask=rm)


@triton.jit
def _pk_fwd_kernel(H, F, Y, RHO, HO, MEAN, R, B, D, MINV,
                   hs0, hs1, hs2, fs0, fs1, fs2, ys0, ys1, ys2, rs0, rs1, os0, os1, os2, ms0, ms1,
                   BR: tl.constexpr, BD: tl.constexpr,
                   NJ: tl.constexpr, XVEC: tl.constexpr, XUP: tl.constexpr):
    """y_g = bf16(y rho); m = bf16(sum(y_g) / D) (the sum in reduce_kernel's order); ho = h + bf16(ffn + bf16(y_g - m));
    m written (bf16, (T, B, 1))."""
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < D
    m = rm[:, None] & cm[None, :]
    yv = _g_w(tl.load(Y + (t[:, None] * ys0 + b[:, None] * ys1 + c[None, :] * ys2), mask=m, other=0.0))
    rho1 = tl.load(RHO + (t * rs0 + b * rs1), mask=rm, other=0.0)
    rho = tl.broadcast_to(rho1[:, None], (BR, BD))
    yg = _g_r16(_g_mul(yv, rho))
    mv = _g_r16(_g_mul(_xrow_sum2(Y, t * ys0 + b * ys1, rm, rho1, 2, True, NJ, XVEC, XUP, BR), _g_bits(MINV)))
    tl.store(MEAN + (t * ms0 + b * ms1), _g_n(mv), mask=rm)
    d = _g_r16(_g_sub(yg, tl.broadcast_to(mv[:, None], (BR, BD))))
    fv = _g_w(tl.load(F + (t[:, None] * fs0 + b[:, None] * fs1 + c[None, :] * fs2), mask=m, other=0.0))
    ff = _g_r16(_g_add(fv, d))
    hv = tl.load(H + (t[:, None] * hs0 + b[:, None] * hs1 + c[None, :] * hs2), mask=m, other=0.0)
    tl.store(HO + (t[:, None] * os0 + b[:, None] * os1 + c[None, :] * os2), _g_add(hv, ff), mask=m)


@triton.jit
def _pk_bwd_kernel(G, RHO, GB, GY, R, B, D, INVB, gs0, gs1, gs2, rs0, rs1, bs0, bs1, bs2, ys0, ys1, ys2,
                   BR: tl.constexpr, BD: tl.constexpr):
    """gb = bf16(g); s = bf16(sum_c -gb); gy = bf16(bf16(gb + bf16(s / D)) rho)."""
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < D
    m = rm[:, None] & cm[None, :]
    gb = _g_r16(tl.load(G + (t[:, None] * gs0 + b[:, None] * gs1 + c[None, :] * gs2), mask=m, other=0.0))
    tl.store(GB + (t[:, None] * bs0 + b[:, None] * bs1 + c[None, :] * bs2), _g_n(gb), mask=m)
    p = _g_r16(_g_neg(gb))                                                          # SubBackward0: -grad (bf16)
    sv = _g_r16(tl.sum(p, axis=1))                                                  # torch's bf16 sum (fp32 acc)
    mg = _g_r16(_g_mul(tl.broadcast_to(sv[:, None], (BR, BD)), _g_bits(INVB)))     # MeanBackward1
    gyg = _g_r16(_g_add(gb, mg))                                                    # y_g's input buffer
    rho = tl.broadcast_to(tl.load(RHO + (t * rs0 + b * rs1), mask=rm, other=0.0)[:, None], (BR, BD))
    tl.store(GY + (t[:, None] * ys0 + b[:, None] * ys1 + c[None, :] * ys2), _g_n(_g_mul(gyg, rho)), mask=m)


# ================================================================================================ host helpers
def _bcast(t: torch.Tensor, shape):
    """t's strides seen over shape (stride 0 on its size-1 / missing leading axes)."""
    st = list(t.stride())
    sh = list(t.shape)
    pad = len(shape) - len(sh)
    st = [0] * pad + st
    sh = [1] * pad + sh
    return [0 if sh[i] == 1 and shape[i] != 1 else st[i] for i in range(len(shape))]


def _axes3(ref_strides, shape, tensors):
    """Iteration axes of a 3-D elementwise launch ordered by ref_strides (innermost = smallest; ties: last axis first);
    returns (S0, S1, N) and, per tensor (a list of 3 strides), its strides in that order."""
    order = sorted(range(3), key=lambda a: (ref_strides[a], -a))
    S = [int(shape[a]) for a in order]
    out = []
    for st in tensors:
        out.extend(int(st[a]) for a in order)
    return S[0], S[1], S[0] * S[1] * S[2], out


def _flat_grid(n: int):
    return (triton.cdiv(int(n), FLAT_BLOCK),)


# ================================================================================================ value_residual
def _vres_fwd_alloc(v, xc, lam, gv, first):
    T, B, D = (int(x) for x in v.shape)
    dt = int(gv.shape[-1])
    vm = _meta(v)
    lm = _meta(lam)
    if first:
        vpm = (vm.float() * (1.0 + lm)).to(torch.bfloat16)
    else:
        vpm = (vm.float() + lm * _meta(xc).float()).to(torch.bfloat16)
    om = torch.cat([vpm[..., :D - dt], (vpm[..., D - dt:].float() + _meta(gv)).to(torch.bfloat16)], dim=-1)
    return torch.empty_strided((T, B, D), om.stride(), dtype=torch.bfloat16, device=v.device)


@torch.library.custom_op("neuronspark::hippo_vres_fwd", mutates_args=(),
                         schema="(Tensor v, Tensor? xc, Tensor lam, Tensor gv, bool first) -> Tensor")
def _vres_fwd(v, xc, lam, gv, first):
    T, B, D = (int(x) for x in v.shape)
    dt = int(gv.shape[-1])
    out = _vres_fwd_alloc(v, xc, lam, gv, first)
    lst = _bcast(lam, (T, B, D))
    xst = list(xc.stride()) if xc is not None else [0, 0, 0]
    grid = (triton.cdiv(T, VRES_BT), B, triton.cdiv(D, VRES_BC))
    _vres_fwd_kernel[grid](v, xc if xc is not None else v, lam, gv, out, T, B, D, dt, *v.stride(), *xst, *lst,
                           *gv.stride(), *out.stride(), FIRST=bool(first), BT=VRES_BT, BC=VRES_BC,
                           num_warps=VRES_WARPS, enable_fp_fusion=False)
    return out


@_vres_fwd.register_fake
def _vres_fwd_fake(v, xc, lam, gv, first):
    return _vres_fwd_alloc(v, xc, lam, gv, first)


def _vres_bwd_alloc(g, v, xc, lam, dt, first):
    T, B, D = (int(x) for x in v.shape)
    dev = v.device
    gm_ = _meta(g)
    # torch's layouts: the cat's narrow views, .to(fp32) of the tail (ToCopyBackward0), the slice gradients' zero
    # fills (contiguous (T, B, D)) and their sum, .to(fp32) of it, the products and their casts
    ggv_st = gm_[..., D - dt:].to(torch.float32).stride()
    g2m = torch.empty((T, B, D), dtype=torch.bfloat16, device="meta").to(torch.float32)
    vm, lm = _meta(v), _meta(lam)
    if first:
        gvo_st = (g2m * (1.0 + lm)).to(torch.bfloat16).stride()
        gl_st = (g2m * vm.float()).stride()
        gxc = torch.empty((0,), dtype=torch.bfloat16, device=dev)
    else:
        xm = _meta(xc)
        gvo_st = g2m.to(torch.bfloat16).stride()
        gl_st = (g2m * xm.float()).stride()
        gxc = torch.empty_strided((T, B, D), (g2m * lm).to(torch.bfloat16).stride(), dtype=torch.bfloat16, device=dev)
    ggv = torch.empty_strided((T, B, dt), ggv_st, dtype=torch.float32, device=dev)
    gvo = torch.empty_strided((T, B, D), gvo_st, dtype=torch.bfloat16, device=dev)
    gl = torch.empty_strided((T, B, D), gl_st, dtype=torch.float32, device=dev)
    return gvo, gxc, gl, ggv


@torch.library.custom_op("neuronspark::hippo_vres_bwd", mutates_args=(),
                         schema="(Tensor g, Tensor v, Tensor? xc, Tensor lam, int dt, bool first) "
                                "-> (Tensor, Tensor, Tensor, Tensor)")
def _vres_bwd(g, v, xc, lam, dt, first):
    """(gv_in, gcarry, glam, gglu): gcarry is empty for layer 0."""
    T, B, D = (int(x) for x in v.shape)
    gvo, gxc, gl, ggv = _vres_bwd_alloc(g, v, xc, lam, dt, first)
    lst = _bcast(lam, (T, B, D))
    xst = list(xc.stride()) if xc is not None else [0, 0, 0]
    grid = (triton.cdiv(T, VRES_BT), B, triton.cdiv(D, VRES_BC))
    _vres_bwd_kernel[grid](g, v, xc if xc is not None else v, lam, gvo, gxc if not first else gvo, gl, ggv,
                           T, B, D, dt, *g.stride(), *v.stride(), *xst, *lst, *gvo.stride(),
                           *(gxc.stride() if not first else [0, 0, 0]), *gl.stride(), *ggv.stride(),
                           FIRST=bool(first), BT=VRES_BT, BC=VRES_BC, num_warps=VRES_WARPS, enable_fp_fusion=False)
    return gvo, gxc, gl, ggv


@_vres_bwd.register_fake
def _vres_bwd_fake(g, v, xc, lam, dt, first):
    return _vres_bwd_alloc(g, v, xc, lam, dt, first)


class ValueResidualFn(torch.autograd.Function):
    """The value residual with the GLU tail (see the module docstring); ``xc`` None = hippocampal layer 0."""

    @staticmethod
    def forward(ctx, v, xc, lam, gv, first, dt):
        out = _vres_fwd(v, xc, lam, gv, bool(first))
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(v, xc, lam)
        ctx.first, ctx.dt = bool(first), int(dt)
        return out

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None, None, None, None
        v, xc, lam = ctx.saved_tensors
        gvo, gxc, gl, ggv = _vres_bwd(g, v, xc, lam, ctx.dt, ctx.first)
        return gvo, (None if ctx.first else gxc), gl, ggv, None, None


def value_residual(v: torch.Tensor, lam: torch.Tensor, carry: Optional[torch.Tensor],
                   glu_add: torch.Tensor) -> torch.Tensor:
    """The value residual of a hippocampal layer with the GLU addend on the last ``glu_add.shape[-1]`` channels (see the
    module docstring); ``carry`` None for hippocampal layer 0.  bf16 (T, B, C) result.  Training regime."""
    require_cuda(v, "value_residual")
    if not (v.dtype == torch.bfloat16 and v.dim() == 3):
        raise RuntimeError("neuronspark.value_residual: v must be a 3-D bf16 CUDA tensor")
    T, B, D = (int(x) for x in v.shape)
    first = carry is None
    if not first and not (isinstance(carry, torch.Tensor) and carry.dtype == torch.bfloat16 and carry.shape == v.shape
                          and carry.device == v.device):
        raise RuntimeError("neuronspark.value_residual: the carry must be bf16 of v's shape")
    if not (isinstance(lam, torch.Tensor) and lam.dtype == torch.float32 and lam.device == v.device and lam.dim() <= 3
            and all(a in (1, b_) for a, b_ in zip((1,) * (3 - lam.dim()) + tuple(lam.shape), (T, B, D)))):
        raise RuntimeError(f"neuronspark.value_residual: lam must be fp32 broadcastable to {(T, B, D)}, got "
                           f"{getattr(lam, 'dtype', None)} {tuple(getattr(lam, 'shape', ()))}")
    dt = int(glu_add.shape[-1]) if isinstance(glu_add, torch.Tensor) and glu_add.dim() == 3 else -1
    if not (isinstance(glu_add, torch.Tensor) and glu_add.dtype == torch.float32 and tuple(glu_add.shape) == (T, B, dt)
            and glu_add.device == v.device and 0 < dt <= D):
        raise RuntimeError("neuronspark.value_residual: glu_add must be fp32 (T, B, d) with 0 < d <= C")
    if v.numel() >= 2 ** 31:
        raise RuntimeError("neuronspark.value_residual: tensor too large")
    return ValueResidualFn.apply(v, None if first else carry, lam, glu_add, first, dt)


def value_residual_ref(v: torch.Tensor, lam: torch.Tensor, carry: Optional[torch.Tensor],
                       glu_add: torch.Tensor) -> torch.Tensor:
    D = int(v.shape[-1])
    dt = int(glu_add.shape[-1])
    if carry is None:
        vp = (v.float() * (1.0 + lam)).to(v.dtype)
    else:
        vp = (v.float() + lam * carry.float()).to(v.dtype)
    return torch.cat([vp[..., :D - dt], (vp[..., D - dt:].float() + glu_add).to(v.dtype)], dim=-1)


# ================================================================================================ zc_center
def _zc_fwd_alloc(z, zcum):
    T, B, D = (int(v) for v in z.shape)
    znm = torch.empty((T, 1, 1), dtype=torch.float32, device="meta")
    o_st = ((_meta(z) - _meta(zcum) / znm) * znm).contiguous().stride()
    return torch.empty_strided((T, B, D), o_st, dtype=torch.float32, device=z.device)


@torch.library.custom_op("neuronspark::hippo_zc_fwd", mutates_args=(), schema="(Tensor z, Tensor zcum) -> Tensor")
def _zc_fwd(z, zcum):
    T, B, D = (int(v) for v in z.shape)
    out = _zc_fwd_alloc(z, zcum)
    S0, S1, N, st = _axes3(out.stride(), (T, B, D), [z.stride(), zcum.stride(), out.stride()])
    order = sorted(range(3), key=lambda a: (out.stride()[a], -a))
    _zc_fwd_kernel[_flat_grid(N)](z, zcum, out, N, S0, S1, 0, *st, TAX=order.index(0), BLOCK=FLAT_BLOCK,
                                  num_warps=FLAT_WARPS, enable_fp_fusion=False)
    return out


@_zc_fwd.register_fake
def _zc_fwd_fake(z, zcum):
    return _zc_fwd_alloc(z, zcum)


def _zc_bwd_alloc(g):
    T, B, D = (int(v) for v in g.shape)
    znm = torch.empty((T, 1, 1), dtype=torch.float32, device="meta")
    gmm = _meta(g) * znm                              # MulBackward0: grad * mask (mask (T, 1, 1))
    gc_st = ((-gmm) / znm).stride()                   # SubBackward0 other: -grad; DivBackward0: grad / zn
    gz = torch.empty_strided((T, B, D), gmm.stride(), dtype=torch.float32, device=g.device)
    gc = torch.empty_strided((T, B, D), gc_st, dtype=torch.float32, device=g.device)
    return gz, gc


@torch.library.custom_op("neuronspark::hippo_zc_bwd", mutates_args=(), schema="(Tensor g) -> (Tensor, Tensor)")
def _zc_bwd(g):
    T, B, D = (int(v) for v in g.shape)
    gz, gc = _zc_bwd_alloc(g)
    S0, S1, N, st = _axes3(g.stride(), (T, B, D), [g.stride(), gz.stride(), gc.stride()])
    order = sorted(range(3), key=lambda a: (g.stride()[a], -a))
    _zc_bwd_kernel[_flat_grid(N)](g, gz, gc, N, S0, S1, 0, *st, TAX=order.index(0), BLOCK=FLAT_BLOCK,
                                  num_warps=FLAT_WARPS, enable_fp_fusion=False)
    return gz, gc


@_zc_bwd.register_fake
def _zc_bwd_fake(g):
    return _zc_bwd_alloc(g)


class ZcCenterFn(torch.autograd.Function):
    """((z - zcum / max(t, 1)) * (t > 0)).contiguous() for fp32 (T, B, C) z / zcum."""

    @staticmethod
    def forward(ctx, z, zcum):
        ctx.set_materialize_grads(False)
        return _zc_fwd(z, zcum)

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None
        return _zc_bwd(g)


def zc_center(z: torch.Tensor, zcum: torch.Tensor) -> torch.Tensor:
    """``((z - zcum / arange(T).clamp_min(1)) * (arange(T) > 0)).contiguous()`` for fp32 (T, B, C) CUDA ``z`` and its
    strictly-past running sum ``zcum`` -> contiguous fp32.  Training regime (grad-enabled steps; no-grad forwards of
    the module use the same arithmetic in torch)."""
    for n, t in (("z", z), ("zcum", zcum)):
        require_cuda(t, "zc_center")
        if not (t.dtype == torch.float32 and t.dim() == 3 and t.numel() < 2 ** 31):
            raise RuntimeError(f"neuronspark.zc_center: {n} must be a 3-D fp32 CUDA tensor")
    if z.shape != zcum.shape:
        raise RuntimeError("neuronspark.zc_center: shapes differ")
    return ZcCenterFn.apply(z, zcum)


def zc_center_ref(z: torch.Tensor, zcum: torch.Tensor) -> torch.Tensor:
    T = int(z.shape[0])
    zn = torch.arange(T, device=z.device, dtype=torch.float32).clamp_min(1.0).view(T, 1, 1)
    mask = (torch.arange(T, device=z.device) > 0).view(T, 1, 1).to(torch.float32)
    return ((z - zcum / zn) * mask).contiguous()


# ================================================================================================ read_gain_residual
def _rg_fwd_alloc(r, f):
    shp = tuple(int(v) for v in r.shape)
    rgm = (_meta(r).float() * (1.0 + _meta(f))).to(torch.bfloat16)
    return torch.empty_strided(shp, rgm.stride(), dtype=torch.bfloat16, device=r.device)


@torch.library.custom_op("neuronspark::hippo_read_gain_fwd", mutates_args=(), schema="(Tensor r, Tensor f) -> Tensor")
def _rg_fwd(r, f):
    """rg = bf16(float(r) (1 + f)) for bf16 r (T, B, D) and fp32 f (T, B, 1)."""
    shp = tuple(int(v) for v in r.shape)
    rg = _rg_fwd_alloc(r, f)
    S0, S1, N, st = _axes3(r.stride(), shp, [r.stride(), _bcast(f, shp), rg.stride()])
    _rg_only_fwd_kernel[_flat_grid(N)](r, f, rg, N, S0, S1, *st, BLOCK=FLAT_BLOCK, num_warps=FLAT_WARPS,
                                       enable_fp_fusion=False)
    return rg


@_rg_fwd.register_fake
def _rg_fwd_fake(r, f):
    return _rg_fwd_alloc(r, f)


def _rg_bwd_alloc(grg, r, f):
    T, B, D = (int(v) for v in r.shape)
    gfm = _meta(grg).float()
    gr = torch.empty_strided(tuple(r.shape), (gfm * (1.0 + _meta(f))).to(torch.bfloat16).stride(),
                             dtype=torch.bfloat16, device=r.device)
    pf = torch.empty_strided((T, B, 1), (B, 1, 1), dtype=torch.float32, device=r.device)
    return gr, pf


@torch.library.custom_op("neuronspark::hippo_read_gain_bwd", mutates_args=(),
                         schema="(Tensor grg, Tensor r, Tensor f) -> (Tensor, Tensor)")
def _rg_bwd(grg, r, f):
    """(gr bf16 (T, B, D), pf fp32 (T, B, 1)) from the bf16 gradient of rg."""
    T, B, D, R = _rows(r.shape)
    gr, pf = _rg_bwd_alloc(grg, r, f)
    _rg_bwd16_kernel[(triton.cdiv(R, ROW_BR),)](grg, r, f, gr, pf, R, B, D, *_s3(grg), *_s3(r), *_s2(f), *_s3(gr),
                                                *_s2(pf), BR=ROW_BR, BD=_pow2(D), num_warps=ROW_WARPS,
                                                enable_fp_fusion=False)
    return gr, pf


@_rg_bwd.register_fake
def _rg_bwd_fake(grg, r, f):
    return _rg_bwd_alloc(grg, r, f)


class ReadGainFn(torch.autograd.Function):
    """rg = bf16(float(r) (1 + f)): r bf16 (T, B, D), f fp32 (T, B, 1).  Backward from rg's bf16 gradient: gr =
    bf16(gf (1 + f)), the factor's gradient the in-kernel row sum of gf r."""

    @staticmethod
    def forward(ctx, r, f):
        rg = _rg_fwd(r, f)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(r, f)
        return rg

    @staticmethod
    def backward(ctx, grg):
        if grg is None:
            return None, None
        r, f = ctx.saved_tensors
        if grg.dtype != torch.bfloat16 or tuple(grg.shape) != tuple(r.shape):
            raise RuntimeError(f"neuronspark.read_gain_residual backward: expected the bf16 gradient of the gained "
                               f"residual, got {grg.dtype} {tuple(grg.shape)}")
        return _rg_bwd(grg, r, f)


def read_gain_residual(h: torch.Tensor, r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """``h + bf16(r.float() * (1 + f))`` for fp32 (T, B, D) ``h``, bf16 ``r`` of h's shape and the fp32 (T, B) read-gain
    factor ``f`` -> fp32.  The gained residual is :class:`ReadGainFn`; the add is torch's (an exact widening of the
    bf16 value plus one fp32 rounding; h's gradient passes through, rg's gradient arrives as bf16).  Every regime (the
    inference calls run without autograd)."""
    require_cuda(h, "read_gain_residual")
    if not (h.dtype == torch.float32 and h.dim() == 3 and h.numel() < 2 ** 31):
        raise RuntimeError("neuronspark.read_gain_residual: h must be a 3-D fp32 CUDA tensor")
    if not (isinstance(r, torch.Tensor) and r.dtype == torch.bfloat16 and r.shape == h.shape and r.device == h.device):
        raise RuntimeError("neuronspark.read_gain_residual: r must be bf16 of h's shape")
    if not (isinstance(f, torch.Tensor) and f.dtype == torch.float32 and tuple(f.shape) == tuple(h.shape[:2])
            and f.device == h.device):
        raise RuntimeError(f"neuronspark.read_gain_residual: f must be fp32 (T, B), got {getattr(f, 'dtype', None)} "
                           f"{tuple(getattr(f, 'shape', ()))}")
    rg = ReadGainFn.apply(r, f.unsqueeze(-1))
    return h + rg


def read_gain_residual_ref(h: torch.Tensor, r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return h + (r.float() * (1.0 + f).unsqueeze(-1)).to(r.dtype)


# ================================================================================================ pkm_tail
def _pkt_fwd_alloc(h, ffn, y, rho):
    ygm = (_meta(y).float() * _meta(rho)).to(torch.bfloat16)
    mean_m = ygm.mean(dim=-1, keepdim=True)
    mean = torch.empty_strided(tuple(mean_m.shape), tuple(mean_m.stride()), dtype=torch.bfloat16, device=h.device)
    hom = _meta(h) + (_meta(ffn) + (ygm - mean_m))
    ho = torch.empty_strided(tuple(h.shape), hom.stride(), dtype=torch.float32, device=h.device)
    return ho, mean, ygm


@torch.library.custom_op("neuronspark::hippo_pkm_tail_fwd", mutates_args=(),
                         schema="(Tensor h, Tensor ffn, Tensor y, Tensor rho) -> (Tensor, Tensor)")
def _pkt_fwd(h, ffn, y, rho):
    """(ho fp32, mean bf16 (T, B, 1))."""
    T, B, D, R = _rows(h.shape)
    ho, mean, ygm = _pkt_fwd_alloc(h, ffn, y, rho)
    nj, vec = _xcfg(ygm, y, D, R, XVEC)
    _pk_fwd_kernel[(triton.cdiv(R, PKT_FWDX_ROWS),)](
        h, ffn, y, rho, ho, mean, R, B, D, _invb(D), *_s3(h), *_s3(ffn), *_s3(y), *_s2(rho), *_s3(ho), *_s2(mean),
        BR=PKT_FWDX_ROWS, BD=_pow2(D), NJ=nj, XVEC=vec, XUP=XUP, num_warps=PKT_FWDX_WARPS, enable_fp_fusion=False)
    return ho, mean


@_pkt_fwd.register_fake
def _pkt_fwd_fake(h, ffn, y, rho):
    ho, mean, _ = _pkt_fwd_alloc(h, ffn, y, rho)
    return ho, mean


def _pkt_bwd_alloc(g, rho):
    gbm = _meta(g).to(torch.bfloat16)
    D = int(g.shape[-1])
    sm = (-gbm).sum([2], keepdim=True)
    accm = gbm + sm.expand(tuple(g.shape)) / D
    gym = (accm.float() * _meta(rho)).to(torch.bfloat16)
    gb = torch.empty_strided(tuple(g.shape), gbm.stride(), dtype=torch.bfloat16, device=g.device)
    gy = torch.empty_strided(tuple(g.shape), gym.stride(), dtype=torch.bfloat16, device=g.device)
    return gb, gy


@torch.library.custom_op("neuronspark::hippo_pkm_tail_bwd", mutates_args=(),
                         schema="(Tensor g, Tensor rho) -> (Tensor, Tensor)")
def _pkt_bwd(g, rho):
    """(gb bf16, gy bf16), both (T, B, D): the gradients of ffn and y."""
    T, B, D, R = _rows(g.shape)
    gb, gy = _pkt_bwd_alloc(g, rho)
    _pk_bwd_kernel[(triton.cdiv(R, ROW_BR),)](g, rho, gb, gy, R, B, D, _invb(D), *_s3(g), *_s2(rho), *_s3(gb),
                                              *_s3(gy), BR=ROW_BR, BD=_pow2(D), num_warps=ROW_WARPS,
                                              enable_fp_fusion=False)
    return gb, gy


@_pkt_bwd.register_fake
def _pkt_bwd_fake(g, rho):
    return _pkt_bwd_alloc(g, rho)


class PkmTailFn(torch.autograd.Function):
    """h + (ffn + (y_g - y_g.mean(-1, keepdim=True))), y_g = bf16(float(y) * rho); h fp32, ffn / y bf16 (T, B, D),
    rho fp32 (T, B, 1) without gradient."""

    @staticmethod
    def forward(ctx, h, ffn, y, rho):
        ho, mean = _pkt_fwd(h, ffn, y, rho)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(rho)
        return ho

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None, None
        (rho,) = ctx.saved_tensors
        gb, gy = _pkt_bwd(g, rho)
        return g, gb, gy, None


#: the fewest (T x B) rows pkm_tail takes: below it ATen's row mean leaves the one-warp-per-row vectorized reduction the
#: kernel reproduces, and the caller keeps the torch lines (the hippocampal layer's decode at B < 16)
PKM_TAIL_MIN_ROWS = 16


def pkm_tail(h: torch.Tensor, ffn: torch.Tensor, y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """``h + (ffn + (y_g - mean_c(y_g)))`` with ``y_g = bf16(y.float() * rho)``: fp32 (T, B, D) ``h``, bf16 ``ffn`` /
    ``y`` of h's shape, fp32 (T, B) ``rho`` without gradient -> fp32.  Every regime (the inference calls run without
    autograd)."""
    require_cuda(h, "pkm_tail")
    if not (h.dtype == torch.float32 and h.dim() == 3 and h.numel() < 2 ** 31):
        raise RuntimeError("neuronspark.pkm_tail: h must be a 3-D fp32 CUDA tensor")
    for n, t in (("ffn", ffn), ("y", y)):
        if not (isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16 and t.shape == h.shape
                and t.device == h.device):
            raise RuntimeError(f"neuronspark.pkm_tail: {n} must be bf16 of h's shape, got {getattr(t, 'dtype', None)} "
                               f"{tuple(getattr(t, 'shape', ()))}")
    if not (isinstance(rho, torch.Tensor) and rho.dtype == torch.float32 and tuple(rho.shape) == tuple(h.shape[:2])
            and rho.device == h.device):
        raise RuntimeError("neuronspark.pkm_tail: rho must be fp32 (T, B)")
    if rho.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("neuronspark.pkm_tail: rho requires grad; the fused backward gives no gradient to rho (the "
                           "copy-gate strength is fixed)")
    return PkmTailFn.apply(h, ffn, y, rho.unsqueeze(-1))


def pkm_tail_ref(h: torch.Tensor, ffn: torch.Tensor, y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    yg = (y.float() * rho.unsqueeze(-1)).to(y.dtype)
    return h + (ffn + (yg - yg.mean(dim=-1, keepdim=True)))
