"""Residual and value glue of the hippocampal layer (training).

Operators (registry names):

* ``value_residual(v, lam0, gate, carry, glu_add)``: the values' SiLU, the value residual with its mix and the GLU
  tail of the last head, one forward and one backward kernel::

      s    = silu(v)                                                 (bf16)
      lam  = lam0[c] + tanh(float(gate[t, b, c // (C / H)]))         (fp32; H = gate.shape[-1] heads)
      v'   = bf16(float(s) * (1 + lam))           (hippocampal layer 0: carry is None)
      v'   = bf16(float(s) + lam * float(carry))  (later layers: carry = layer 0's s)
      out  = cat([v'[..., :C - d], bf16(float(v'[..., C - d:]) + float(glu_add))], -1)    (d = glu_add.shape[-1])

  ``v`` bf16 (T, B, C), ``lam0`` fp32 (C,), ``gate`` bf16 (T, B, H), ``carry`` bf16 (T, B, C) or None, ``glu_add``
  bf16 (T, B, d).  Returns ``(out, s)`` for hippocampal layer 0 and ``(out, None)`` for the later layers.
* ``centre_rows(r)``: ``r - mean_c(r)`` for the bf16 (T, B, C) output projection of the attention sub-block (bf16
  mean and difference), forward and backward one kernel each.
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
sums of the (T, B, 1) factors formed in registers (fp32 reduction-order class).  ``value_residual`` is fp32
arithmetic on the widened operands with one bf16 rounding where the torch chain holds a bf16 tensor (the SiLU output,
the mixed value before and after the GLU addend, each gradient), products and sums as separate fp32 operations (no
FMA contraction); one program is one head's channels at VRES_TILE /
head-width positions, so the gate logit's gradient is a row sum in registers and the static mix's gradient per-program
column sums reduced by one torch sum (fp32 reduction-order class); the mix (T, B, C) is not materialised.  ``read_gain_residual``'s factor gradient takes the bf16 gradient of ``rg`` that autograd
delivers for the fp32 add.

The read gain's and the PKM tail's launchers are ``torch.library`` custom ops (``neuronspark::hippo_*``) with fake
implementations of the same layouts; the value residual's and the row centring's kernels are launched by their
autograd Functions.
Pure-torch references ``*_ref`` (same parameters): the torch lines of the SiLU, the mix, the value residual and GLU
tail, the row centring, the read gain and the PKM tail.
"""
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
    "CentreRowsFn", "centre_rows", "centre_rows_ref",
    "ReadGainFn", "read_gain_residual", "read_gain_residual_ref",
    "PkmTailFn", "pkm_tail", "pkm_tail_ref", "PKM_TAIL_MIN_ROWS",
]

# ------------------------------------------------------------------------------------------------ fixed constants
VRES_TILE, VRES_WARPS = 4096, 4            # value_residual (elements per program: one head's channels x positions)
FLAT_BLOCK, FLAT_WARPS = 1024, 4           # flat elementwise kernel (the read-gain forward)
ROW_BR, ROW_WARPS = 4, 8                   # row kernels of the backwards (read gain, PKM tail)
CENTRE_BR, CENTRE_WARPS = 4, 8             # centre_rows forward / backward (rows per program, warps)
PKT_FWDX_ROWS, PKT_FWDX_WARPS = 1, 4       # PKM tail fused forward (exact-order bf16 mean)


# ================================================================================================ kernels
@triton.jit
def _silu16(x):
    """bf16(silu(x)) as an fp32 value."""
    return _g_r16(x / (1.0 + tl.exp(-x)))


@triton.jit
def _vres_lam(GATE, LAM0, t, tm, b, h, c, cm, DV, gs0, gs1, gs2):
    """(tanh of the head's gate logit per position, the mix lam0[c] + tanh) of one head's (BT, BD) tile."""
    th = libdevice.tanh(_g_w(tl.load(GATE + (t * gs0 + b * gs1 + h * gs2), mask=tm, other=0.0)))
    lam = tl.load(LAM0 + (h * DV + c), mask=cm, other=0.0)[None, :] + th[:, None]
    return th, lam


@triton.jit
def _vres_fwd_kernel(V, XC, GATE, LAM0, GV, O, VS, T, B, D, DV, DT,
                     vs0, vs1, vs2, xs0, xs1, xs2, gs0, gs1, gs2, as0, as1, as2,
                     FIRST: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """head h (channels [h DV, (h + 1) DV)), BT positions of one b: s = bf16(silu(v)), v' = bf16(s (1 + lam)) (FIRST;
    s also written to VS) or bf16(s + lam xc), out = v' or bf16(v' + gv) on the last DT channels; O, VS contiguous
    (T, B, D)."""
    b = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2).to(tl.int64)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < DV
    m = tm[:, None] & cm[None, :]
    ch = h * DV + c
    off = (t[:, None] * B + b) * D + ch[None, :]
    s = _silu16(_g_w(tl.load(V + (t[:, None] * vs0 + b * vs1 + ch[None, :] * vs2), mask=m, other=0.0)))
    th, lam = _vres_lam(GATE, LAM0, t, tm, b, h, c, cm, DV, gs0, gs1, gs2)
    if FIRST:
        tl.store(VS + off, _g_n(s), mask=m)
        val = _g_r16(s * (1.0 + lam))
    else:
        xc = _g_w(tl.load(XC + (t[:, None] * xs0 + b * xs1 + ch[None, :] * xs2), mask=m, other=0.0))
        val = _g_r16(s + lam * xc)
    ct = ch - (D - DT)
    tail = m & (ct >= 0)[None, :]
    gv = _g_w(tl.load(GV + (t[:, None] * as0 + b * as1 + ct[None, :] * as2), mask=tail, other=0.0))
    tl.store(O + off, _g_n(tl.where(tail, val + gv, val)), mask=m)


@triton.jit
def _vres_bwd_kernel(G, GC, V, XC, GATE, LAM0, GVO, GXC, GGATE, GGV, PL, T, B, D, DV, DT, H,
                     vs0, vs1, vs2, xs0, xs1, xs2, gs0, gs1, gs2,
                     FIRST: tl.constexpr, HAS_GC: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """head h, BT positions of one b, from out's gradient G (and, FIRST with HAS_GC, the gradient GC of s that the later
    layers' carries accumulated), contiguous bf16 (T, B, D): the GLU addend's gradient (G on the last DT channels), s's
    gradient ds = G (later layers) or bf16(GC + bf16(G (1 + lam))) (FIRST), v's gradient bf16(ds silu'(v)), the
    carry's bf16(G lam), the gate logit's bf16(sum_c(dlam) (1 - tanh^2)) and PL[prog, channel] = sum_t dlam with
    dlam = G xc (FIRST: G s)."""
    b = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2).to(tl.int64)
    prog = (tl.program_id(0) * B + tl.program_id(1)).to(tl.int64)
    t = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    tm = t < T
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < DV
    m = tm[:, None] & cm[None, :]
    ch = h * DV + c
    off = (t[:, None] * B + b) * D + ch[None, :]
    g16 = tl.load(G + off, mask=m, other=0.0)
    g = _g_w(g16)
    ct = ch - (D - DT)
    tl.store(GGV + (t[:, None] * B + b) * DT + ct[None, :], g16, mask=m & (ct >= 0)[None, :])
    x = _g_w(tl.load(V + (t[:, None] * vs0 + b * vs1 + ch[None, :] * vs2), mask=m, other=0.0))
    sg = 1.0 / (1.0 + tl.exp(-x))
    th, lam = _vres_lam(GATE, LAM0, t, tm, b, h, c, cm, DV, gs0, gs1, gs2)
    if FIRST:
        ds = _g_r16(g * (1.0 + lam))
        if HAS_GC:
            ds = _g_r16(_g_w(tl.load(GC + off, mask=m, other=0.0)) + ds)
        dlam = g * _silu16(x)
    else:
        xc = _g_w(tl.load(XC + (t[:, None] * xs0 + b * xs1 + ch[None, :] * xs2), mask=m, other=0.0))
        ds = g
        tl.store(GXC + off, _g_n(g * lam), mask=m)
        dlam = g * xc
    tl.store(GVO + off, _g_n(ds * (sg * (1.0 + x * (1.0 - sg)))), mask=m)
    dlam = tl.where(m, dlam, 0.0)
    tl.store(GGATE + ((t * B + b) * H + h), _g_n(tl.sum(dlam, axis=1) * (1.0 - th * th)), mask=tm)
    tl.store(PL + (prog * D + ch), tl.sum(dlam, axis=0), mask=cm)


@triton.jit
def _centre_fwd_kernel(R, O, NR, D, INVB, BR: tl.constexpr, BD: tl.constexpr):
    """o = bf16(r - bf16(mean_c r)) over BR contiguous rows (the mean an fp32 sum times the fp32 factor 1 / D)."""
    row = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    c = tl.arange(0, BD).to(tl.int64)
    m = (row < NR)[:, None] & (c < D)[None, :]
    off = row[:, None] * D + c[None, :]
    x = _g_w(tl.load(R + off, mask=m, other=0.0))
    mean = _g_r16(_g_mul(tl.sum(x, axis=1), _g_bits(INVB)))
    tl.store(O + off, _g_n(_g_sub(x, tl.broadcast_to(mean[:, None], (BR, BD)))), mask=m)


@triton.jit
def _centre_bwd_kernel(G, O, NR, D, INVB, BR: tl.constexpr, BD: tl.constexpr):
    """r's gradient bf16(g + bf16(bf16(sum_c -g) (1 / D))) over BR contiguous rows (SubBackward0's reduced other
    gradient, MeanBackward1's factor, the accumulation with the direct gradient)."""
    row = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    c = tl.arange(0, BD).to(tl.int64)
    m = (row < NR)[:, None] & (c < D)[None, :]
    off = row[:, None] * D + c[None, :]
    g = _g_w(tl.load(G + off, mask=m, other=0.0))
    s = _g_r16(tl.sum(_g_neg(g), axis=1))
    mm = _g_r16(_g_mul(s, _g_bits(INVB)))
    tl.store(O + off, _g_n(_g_add(g, tl.broadcast_to(mm[:, None], (BR, BD)))), mask=m)


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
def _vres_dims(v: torch.Tensor, gate: torch.Tensor):
    T, B, D = (int(x) for x in v.shape)
    H = int(gate.shape[-1])
    bd = _pow2(D // H)
    return T, B, D, H, D // H, max(1, VRES_TILE // bd), bd


class ValueResidualFn(torch.autograd.Function):
    """The values' SiLU, the value residual and the GLU tail (see the module docstring); ``xc`` None = hippocampal
    layer 0, which also returns the SiLU output (the later layers' carry)."""

    @staticmethod
    def forward(ctx, v, lam0, gate, xc, gv):
        first = xc is None
        T, B, D, H, DV, bt, bd = _vres_dims(v, gate)
        out = torch.empty((T, B, D), dtype=torch.bfloat16, device=v.device)
        vs = torch.empty((T, B, D), dtype=torch.bfloat16, device=v.device) if first else out
        x = v if first else xc
        _vres_fwd_kernel[(triton.cdiv(T, bt), B, H)](
            v, x, gate, lam0, gv, out, vs, T, B, D, DV, int(gv.shape[-1]), *_s3(v), *_s3(x), *_s3(gate), *_s3(gv),
            FIRST=first, BT=bt, BD=bd, num_warps=VRES_WARPS, enable_fp_fusion=False)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(v, lam0, gate, xc)
        ctx.dt = int(gv.shape[-1])
        return (out, vs) if first else out

    @staticmethod
    def backward(ctx, g, gc=None):
        v, lam0, gate, xc = ctx.saved_tensors
        first = xc is None
        T, B, D, H, DV, bt, bd = _vres_dims(v, gate)
        dt = ctx.dt
        dev = v.device
        if g is None:                                    # only the SiLU output is read: v's gradient through the SiLU
            if gc is None:
                return None, None, None, None, None
            x = v.float()
            sg = torch.sigmoid(x)
            return (gc.float() * (sg * (1.0 + x * (1.0 - sg)))).to(torch.bfloat16), None, None, None, None
        g = g.contiguous()
        has_gc = gc is not None
        gc = gc.contiguous() if has_gc else g
        gvo = torch.empty((T, B, D), dtype=torch.bfloat16, device=dev)
        gxc = gvo if first else torch.empty((T, B, D), dtype=torch.bfloat16, device=dev)
        ggate = torch.empty((T, B, H), dtype=torch.bfloat16, device=dev)
        ggv = torch.empty((T, B, dt), dtype=torch.bfloat16, device=dev)
        npt = triton.cdiv(T, bt)
        pl = torch.empty((npt * B, D), dtype=torch.float32, device=dev)
        x = v if first else xc
        _vres_bwd_kernel[(npt, B, H)](
            g, gc, v, x, gate, lam0, gvo, gxc, ggate, ggv, pl, T, B, D, DV, dt, H, *_s3(v), *_s3(x), *_s3(gate),
            FIRST=first, HAS_GC=has_gc, BT=bt, BD=bd, num_warps=VRES_WARPS, enable_fp_fusion=False)
        return gvo, pl.sum(0), ggate, (None if first else gxc), ggv


def value_residual(v: torch.Tensor, lam0: torch.Tensor, gate: torch.Tensor, carry: Optional[torch.Tensor],
                   glu_add: torch.Tensor):
    """``(v', s)``: the value stream of a hippocampal layer after the SiLU, the value residual and the GLU addend on
    the last ``glu_add.shape[-1]`` channels (see the module docstring), contiguous bf16 (T, B, C), and for hippocampal
    layer 0 (``carry`` None) ``s = silu(v)``, the later layers' carry (else None).  ``v`` bf16 (T, B, C) (any
    strides), ``lam0`` fp32 (C,), ``gate`` bf16 (T, B, H) gate logits with H dividing C, ``carry`` bf16 (T, B, C) or
    None, ``glu_add`` bf16 (T, B, d).  Training regime."""
    require_cuda(v, "value_residual")
    if not (v.dtype == torch.bfloat16 and v.dim() == 3):
        raise RuntimeError("neuronspark.value_residual: v must be a 3-D bf16 CUDA tensor")
    T, B, D = (int(x) for x in v.shape)
    first = carry is None
    if not first and not (isinstance(carry, torch.Tensor) and carry.dtype == torch.bfloat16 and carry.shape == v.shape
                          and carry.device == v.device):
        raise RuntimeError("neuronspark.value_residual: the carry must be bf16 of v's shape")
    if not (isinstance(lam0, torch.Tensor) and lam0.dtype == torch.float32 and lam0.device == v.device
            and tuple(lam0.shape) == (D,)):
        raise RuntimeError(f"neuronspark.value_residual: lam0 must be fp32 ({D},), got "
                           f"{getattr(lam0, 'dtype', None)} {tuple(getattr(lam0, 'shape', ()))}")
    H = int(gate.shape[-1]) if isinstance(gate, torch.Tensor) and gate.dim() == 3 else 0
    if not (H > 0 and gate.dtype == torch.bfloat16 and tuple(gate.shape) == (T, B, H) and D % H == 0
            and gate.device == v.device and _pow2(D // H) <= VRES_TILE):
        raise RuntimeError(f"neuronspark.value_residual: gate must be bf16 ({T}, {B}, H) with H dividing {D} and heads "
                           f"of at most {VRES_TILE} channels, got {getattr(gate, 'dtype', None)} "
                           f"{tuple(getattr(gate, 'shape', ()))}")
    dt = int(glu_add.shape[-1]) if isinstance(glu_add, torch.Tensor) and glu_add.dim() == 3 else -1
    if not (isinstance(glu_add, torch.Tensor) and glu_add.dtype == torch.bfloat16
            and tuple(glu_add.shape) == (T, B, dt) and glu_add.device == v.device and 0 < dt <= D):
        raise RuntimeError("neuronspark.value_residual: glu_add must be bf16 (T, B, d) with 0 < d <= C")
    if v.numel() >= 2 ** 31 or min(v.stride()) < 0:
        raise RuntimeError("neuronspark.value_residual: tensor too large or negatively strided")
    res = ValueResidualFn.apply(v, lam0.contiguous(), gate, carry, glu_add)
    return res if first else (res, None)


def value_residual_ref(v: torch.Tensor, lam0: torch.Tensor, gate: torch.Tensor, carry: Optional[torch.Tensor],
                       glu_add: torch.Tensor):
    D = int(v.shape[-1])
    dt = int(glu_add.shape[-1])
    s = F.silu(v)
    lam = lam0.view(1, 1, -1) + torch.tanh(gate.to(torch.float32)).repeat_interleave(D // int(gate.shape[-1]), dim=-1)
    if carry is None:
        vp = (s.float() * (1.0 + lam)).to(v.dtype)
    else:
        vp = (s.float() + lam * carry.float()).to(v.dtype)
    out = torch.cat([vp[..., :D - dt], (vp[..., D - dt:].float() + glu_add.float()).to(v.dtype)], dim=-1)
    return out, (s if carry is None else None)


# ================================================================================================ centre_rows
def _centre_launch(kernel, x: torch.Tensor, last):
    T, B, D = (int(v) for v in x.shape)
    out = torch.empty((T, B, D), dtype=torch.bfloat16, device=x.device)
    kernel[(triton.cdiv(T * B, CENTRE_BR),)](x, out, T * B, D, last, BR=CENTRE_BR, BD=_pow2(D),
                                              num_warps=CENTRE_WARPS, enable_fp_fusion=False)
    return out


class CentreRowsFn(torch.autograd.Function):
    """r - mean_c(r) for a contiguous bf16 (T, B, C) r."""

    @staticmethod
    def forward(ctx, r):
        return _centre_launch(_centre_fwd_kernel, r, _invb(int(r.shape[-1])))

    @staticmethod
    def backward(ctx, g):
        return _centre_launch(_centre_bwd_kernel, g.contiguous(), _invb(int(g.shape[-1])))


def centre_rows(r: torch.Tensor) -> torch.Tensor:
    """``r - r.mean(-1, keepdim=True)`` for a bf16 (T, B, C) CUDA ``r`` (the attention sub-block's output projection)
    -> contiguous bf16.  Training regime."""
    require_cuda(r, "centre_rows")
    if not (r.dtype == torch.bfloat16 and r.dim() == 3 and 0 < r.numel() < 2 ** 31):
        raise RuntimeError(f"neuronspark.centre_rows: r must be a 3-D bf16 CUDA tensor, got {r.dtype} "
                           f"{tuple(r.shape)}")
    return CentreRowsFn.apply(r.contiguous())


def centre_rows_ref(r: torch.Tensor) -> torch.Tensor:
    return r - r.mean(dim=-1, keepdim=True)


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
