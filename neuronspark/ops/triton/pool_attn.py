"""The hippocampal layers' pool attention (training): softmax over ``scale * q k^T + mask``, times the tagged values.

Operator (registry name): ``pool_attention(q, k, v, mask, scale, kshared)``.

* ``kshared=False`` (one pool head): ``q``, ``k`` (T, B, dk) fp32 unit rows (the normalised, gained query and the
  normalised key of the head), ``v`` (T, B, dv) bf16 tagged values, ``mask`` (B, 1, T, T) bf16 (``pool_bias_mask``);
  returns (T, B, dv) bf16 (a view; the module widens it to fp32).
* ``kshared=True`` (the extra queries of the last head): ``q`` (T, B, M, dk) fp32 -- formed as
  ``torch.stack(queries, 0).permute(1, 2, 0, 3)`` -- with the head's ``k`` (T, B, dk), ``v`` (T, B, dv) and ``mask``
  shared by the M queries; returns (T, B, M, dv) bf16 (a view; the module reshapes it to (T, B, M dv)).

``scale`` is the dendritic match gain sqrt(dk) = 16.

Numerics contract
-----------------
The casts of the bf16 autocast regime are made explicitly at the same points (q / k fp32 -> bf16 round to nearest
even, into a contiguous layout; v and the mask bf16 as given), then:

* forward: ``S = q16 k16^T`` by a cuBLAS bf16 GEMM with fp32 accumulation and output (the bf16 x bf16 products are
  exact), keys zero-padded to TP = max(64, pow2(T)); a row kernel forms ``x = scale S + float(mask16)`` (columns >= T:
  -inf), ``P = exp(x - max) / sum`` (libdevice expf, IEEE division; a row without a finite entry gives P = 0, as
  ``_safe_softmax``), and stores P as two bf16 pieces ``p1 + p2`` plus the exact remainder; ``O = (p1 + p2) V16`` by one
  GEMM over the stacked pieces, the piece products summed smallest first in fp32 and rounded to bf16;
* backward: ``dP = dO16 V16^T``; a row kernel re-forms the exact fp32 P from the stored pieces, ``Dd = sum_j P dP``,
  ``dS = P (dP - Dd)`` (stored as two bf16 pieces), and the mask gradient ``bf16(dS)`` (with M query heads sharing the
  mask: bf16 of the fp32 sum over the heads in head order); ``dV``, ``dQ = scale dS K16``, ``dK^T = scale Q16^T dS``
  by stacked GEMMs (fp32 accumulation, the pieces summed smallest first), each cast to bf16 in the layout the math
  backend gives (dq / dv contiguous, dk with T fastest, dmask contiguous).

This is the fp32 reduction-order class against a float64 evaluation of the same operation (no TF32 operand rounding,
unlike the TF32 math backend of ``F.scaled_dot_product_attention``).  The pieces, the remainder and the products are
the fixed training configuration (two pieces, exact remainder, stacked GEMMs, strided operands where valid).

Launchers are ``torch.library`` custom ops (``neuronspark::pool_attn_*``) with fake implementations; the
reference :func:`pool_attention_ref` evaluates the same expression on the bf16-rounded operands in float64, with the
gradients rounded at the same points (``kshared``: per query head, before the expand's sum over the M heads).
"""
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda

__all__ = ["PoolAttnFn", "Cast16Fn", "pool_attention", "pool_attention_ref"]

# ------------------------------------------------------------------------------------------------ fixed constants
NSPLIT = 2          # bf16 pieces of P / dS in the GEMMs
_NR = 1             # the exact remainder of P after the two pieces is stored too (the backward forms the exact P)
TP_MIN = 64         # keys are zero-padded to max(64, pow2(T))
PAD_BT = 16         # rows per program of the key / value pad kernel
# row kernels: rows per program r = max(1, min(8, 4096 // TP)), warps 8 when TP r >= 4096 else 4 (automatic tiling)
# piece-sum kernel: BC = min(256, pow2(C)) columns, BR = max(1, 4096 // BC) rows, 4 warps


# ================================================================================================ kernels
@triton.jit
def _exp(x):
    return libdevice.exp(x)


@triton.jit
def _gpad_kernel(X, Y, sx_b, sx_t, sx_d, T, TP, D: tl.constexpr, BT: tl.constexpr):
    """Y[n, t, :] = X[n, t, :] (bf16) for t < T and 0 for T <= t < TP; Y contiguous (N, TP, D), X any strides."""
    pt = tl.program_id(0)
    n = tl.program_id(1).to(tl.int64)
    t = pt * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    t64 = t.to(tl.int64)[:, None]
    x = tl.load(X + n * sx_b + t64 * sx_t + d[None, :] * sx_d, mask=(t < T)[:, None], other=0.0)
    tl.store(Y + n * TP * D + t64 * D + d[None, :], x.to(tl.bfloat16), mask=(t < TP)[:, None])


@triton.jit
def _cast16_kernel(X, Y, sx_b, sx_h, sx_t, sx_d, H, T, D: tl.constexpr, BT: tl.constexpr):
    """Y[b, h, t, :] = bf16(X[b, h, t, :]) (round to nearest even: autocast's cast), X fp32 with any strides, Y
    contiguous (B, H, T, D)."""
    pt = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    b = bh // H
    h = bh % H
    t = pt * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    m = (t < T)[:, None]
    x = tl.load(X + b * sx_b + h * sx_h + t.to(tl.int64)[:, None] * sx_t + d[None, :] * sx_d, mask=m, other=0.0)
    tl.store(Y + bh * T * D + t.to(tl.int64)[:, None] * D + d[None, :], x.to(tl.bfloat16), mask=m)


@triton.jit
def _psum_kernel(X, Y, sx_n, sx_r, sx_i, sy_n, sy_r, R, C, NS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """Y[n, r, c] = bf16(((X_{NS-1} + X_{NS-2}) + ...) + X_0) with X_i = X[n, r, c] at n * sx_n + r * sx_r + i * sx_i + c
    (the NS piece products of one stacked GEMM, the smallest piece first, fp32 adds), r < R, c < C; Y contiguous
    rows (sy_r, unit column stride)."""
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    n = tl.program_id(2).to(tl.int64)
    r = pr * BR + tl.arange(0, BR)
    c = pc * BC + tl.arange(0, BC)
    m = (r < R)[:, None] & (c < C)[None, :]
    base = X + n * sx_n + r.to(tl.int64)[:, None] * sx_r + c[None, :]
    acc = tl.load(base + (NS - 1) * sx_i, mask=m, other=0.0)
    for i in tl.static_range(NS - 1):
        acc = acc + tl.load(base + (NS - 2 - i) * sx_i, mask=m, other=0.0)
    yo = n * sy_n + r.to(tl.int64)[:, None] * sy_r + c[None, :]
    tl.store(Y + yo, acc.to(tl.bfloat16), mask=m)


@triton.jit
def _gsplit(PP, base, p, TP: tl.constexpr, NS: tl.constexpr, rmask):
    """Store fp32 p as NS bf16 pieces at PP + base + i * TP (piece i = the round-to-nearest bf16 of what pieces
    0..i-1 leave; the remainders are exact in fp32); returns the remainder after the NS pieces (fp32, exact)."""
    r = p
    for i in tl.static_range(NS):
        a = r.to(tl.bfloat16)
        tl.store(PP + base + i * TP, a, mask=rmask)
        r = r - a.to(tl.float32)
    return r


@triton.jit
def _gfwd_kernel(S, MSK, PP, PR, ss_bh, ss_t, sm_b, sm_t, H, T, scale,
                 TP: tl.constexpr, R: tl.constexpr, NS: tl.constexpr, NR: tl.constexpr):
    """R rows t of batch row b, each head h < H in turn: x = scale * S[b H + h, t, :] + float(mask[b, 0, t, :])
    (columns >= T: -inf), P = exp(x - max) / sum (a row with no finite entry: P = 0, as _safe_softmax), stored as
    NS bf16 pieces PP[b H + h, t, i, :] (PP contiguous (B H, T, NS, TP); columns >= T exact zeros).  The mask row
    block is loaded once for the H heads.  NR (NS = 2): the remainder after the two pieces as bf16 (exact: at most 8
    significant bits) to PR[b H + h, t, :] (contiguous (B H, T, TP))."""
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    rows = pid * R + tl.arange(0, R)
    cols = tl.arange(0, TP)
    rv = rows < T
    valid = rv[:, None] & (cols < T)[None, :]
    r64 = rows.to(tl.int64)[:, None]
    mk = tl.load(MSK + b * sm_b + r64 * sm_t + cols[None, :], mask=valid, other=float("-inf")).to(tl.float32)
    for h in range(H):
        bh = b * H + h
        sv = tl.load(S + bh * ss_bh + r64 * ss_t + cols[None, :], mask=valid, other=0.0)
        x = sv * scale + mk
        m = tl.max(x, 1)
        m_u = tl.where(m == float("-inf"), 0.0, m)
        e = _exp(x - m_u[:, None])
        l = tl.sum(e, 1)
        p = tl.div_rn(e, tl.where(l == 0.0, 1.0, l)[:, None])     # IEEE division (div.rn), as the warp softmax
        rem = _gsplit(PP, bh * (T * NS * TP) + r64 * (NS * TP) + cols[None, :], p, TP, NS, rv[:, None])
        if NR:
            tl.store(PR + bh * (T * TP) + r64 * TP + cols[None, :], rem.to(tl.bfloat16), mask=rv[:, None])


@triton.jit
def _gbwd_head(PP, PR, DPm, DSP, bh, r64, cols, rv, valid, sd_bh, sd_t, T, TP: tl.constexpr, NS: tl.constexpr,
               NR: tl.constexpr):
    """One head of _gbwd_kernel: P = the forward's pieces summed (smallest first, fp32; NR: the remainder PR first,
    then the NS pieces), dP, Dd = sum_j P dP, dS = P (dP - Dd) stored as NS bf16 pieces; returns dS (fp32)."""
    base = bh * (T * NS * TP) + r64 * (NS * TP) + cols[None, :]
    if NR:
        p = tl.load(PR + bh * (T * TP) + r64 * TP + cols[None, :], mask=rv[:, None], other=0.0).to(tl.float32)
        for i in tl.static_range(NS):
            p = p + tl.load(PP + base + (NS - 1 - i) * TP, mask=rv[:, None], other=0.0).to(tl.float32)
    else:
        p = tl.load(PP + base + (NS - 1) * TP, mask=rv[:, None], other=0.0).to(tl.float32)
        for i in tl.static_range(NS - 1):
            p = p + tl.load(PP + base + (NS - 2 - i) * TP, mask=rv[:, None], other=0.0).to(tl.float32)
    dp = tl.load(DPm + bh * sd_bh + r64 * sd_t + cols[None, :], mask=valid, other=0.0)
    dd = tl.sum(p * dp, 1)
    ds = p * (dp - dd[:, None])
    _gsplit(DSP, base, ds, TP, NS, rv[:, None])
    return ds


@triton.jit
def _gbwd_kernel(PP, PR, DPm, DSP, DM, sd_bh, sd_t, sdm_b, sdm_t, H, T,
                 TP: tl.constexpr, R: tl.constexpr, NS: tl.constexpr, NR: tl.constexpr):
    """R rows t of batch row b, each head h < H in turn (_gbwd_head); the mask gradient DM[b, 0, t, j < T] = bf16(dS)
    for one head, bf16 of the fp32 sum over the heads in head order for H > 1 (DM contiguous rows)."""
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    rows = pid * R + tl.arange(0, R)
    cols = tl.arange(0, TP)
    rv = rows < T
    valid = rv[:, None] & (cols < T)[None, :]
    r64 = rows.to(tl.int64)[:, None]
    acc = _gbwd_head(PP, PR, DPm, DSP, b * H, r64, cols, rv, valid, sd_bh, sd_t, T, TP, NS, NR)
    for h in range(1, H):
        acc = acc + _gbwd_head(PP, PR, DPm, DSP, b * H + h, r64, cols, rv, valid, sd_bh, sd_t, T, TP, NS, NR)
    tl.store(DM + b * sdm_b + r64 * sdm_t + cols[None, :], acc.to(tl.bfloat16), mask=valid)


# ================================================================================================ host helpers
def _viewable(t: torch.Tensor, shape) -> bool:
    try:
        t.view(shape)
        return True
    except RuntimeError:
        return False


def _as_bmm(t: torch.Tensor, shape) -> torch.Tensor:
    """``t`` viewed as ``shape`` (N, R, C) when the view exists and is a valid cuBLAS batched operand (C unit stride,
    row stride >= C); otherwise a contiguous copy in that shape."""
    v = t.view(shape) if t.is_contiguous() else None
    if v is None:
        try:
            v = t.view(shape)
        except RuntimeError:
            v = None
    if v is not None and v.stride(2) == 1 and v.stride(1) >= max(1, int(v.shape[2])):
        return v
    return t.contiguous().view(shape)


def _gemm_cfg(TP: int) -> Tuple[int, int]:
    """(rows per program, warps) of the row-wise kernels for a padded row length TP (every row is reduced by one
    program whatever the setting; the tiling sets the order of the fp32 row sums)."""
    r = max(1, min(8, 4096 // TP))
    w = 8 if TP * r >= 4096 else 4
    return r, w


def _gpad(x: torch.Tensor, TP: int) -> torch.Tensor:
    """(N, T, d) bf16, any strides -> contiguous (N, TP, d) bf16 with zero rows T..TP-1 (one kernel)."""
    N, T, d = (int(s) for s in x.shape)
    y = torch.empty((N, TP, d), dtype=torch.bfloat16, device=x.device)
    _gpad_kernel[(triton.cdiv(TP, PAD_BT), N)](x, y, x.stride(0), x.stride(1), x.stride(2), T, TP, D=d, BT=PAD_BT,
                                              num_warps=4)
    return y


def _psum(x: torch.Tensor, N: int, R: int, C: int, sx_n: int, sx_r: int, sx_i: int, NS: int, y_shape):
    """bf16 (N, R, C) contiguous = the NS piece products of a stacked GEMM output summed smallest first."""
    y = torch.empty(y_shape, dtype=torch.bfloat16, device=x.device)
    BC = min(256, triton.next_power_of_2(C))
    BR = max(1, 4096 // BC)
    _psum_kernel[(triton.cdiv(R, BR), triton.cdiv(C, BC), N)](x, y, sx_n, sx_r, sx_i, R * C, C, R, C, NS=NS, BR=BR,
                                                              BC=BC, num_warps=4)
    return y


def _mm(out: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """out (fp32) = alpha * a @ b, one cuBLAS product (bf16 operands, fp32 accumulation and output)."""
    return torch.baddbmm(out, a, b, out_dtype=torch.float32, beta=0, alpha=alpha, out=out)


def _gemm_softmax(qg, kp, mask, B, H, T, TP, scale, NS):
    """S = q k^T (fp32, (B H, T, TP)), the probability pieces PP (bf16, (B H, T, NS, TP)) and the remainder PR (bf16,
    (B H, T, TP))."""
    dev = qg.device
    S = torch.empty((kp.shape[0], qg.shape[1], TP), dtype=torch.float32, device=dev)
    _mm(S, qg, kp.transpose(1, 2), 1.0)
    PP = torch.empty((B * H, T, NS, TP), dtype=torch.bfloat16, device=dev)
    PR = torch.empty((B * H, T, TP), dtype=torch.bfloat16, device=dev)
    R, W = _gemm_cfg(TP)
    _gfwd_kernel[(triton.cdiv(T, R), B)](
        S, mask, PP, PR, T * TP, TP, mask.stride(0), mask.stride(2), H, T, float(scale), TP=TP, R=R, NS=NS,
        NR=_NR, num_warps=W)
    return PP, PR


def _fwd_shapes(B, H, T, D, DV, ks, vs):
    TP = max(TP_MIN, triton.next_power_of_2(T))
    BH = B * H
    return TP, (B if ks else BH, TP, D), (B if vs else BH, TP, DV)


# ================================================================================================ custom ops
@torch.library.custom_op("neuronspark::pool_attn_cast16", mutates_args=(), schema="(Tensor x) -> Tensor")
def _cast16_op(x):
    B, H, T, D = (int(s) for s in x.shape)
    y = torch.empty((B, H, T, D), dtype=torch.bfloat16, device=x.device)
    BT = max(16, min(128, 8192 // D))
    _cast16_kernel[(triton.cdiv(T, BT), B * H)](x, y, x.stride(0), x.stride(1), x.stride(2), x.stride(3), H, T, D=D,
                                                BT=BT, num_warps=4)
    return y


@_cast16_op.register_fake
def _cast16_fake(x):
    return torch.empty(tuple(x.shape), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("neuronspark::pool_attn_fwd", mutates_args=(),
                         schema="(Tensor qc, Tensor k, Tensor v, Tensor mask, float scale, bool ks, bool vs, int B, "
                                "int H) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")
def _attn_fwd(qc, k, v, mask, scale, ks, vs, B, H):
    """(out (B, H, T, DV) bf16, kp, vp, PP, PR): ``qc`` the (B H, T, D) bf16 query operand, ``k`` / ``v`` the cast
    (B, H, T, .) key / value (head-shared when ks / vs), ``mask`` (B, 1, T, T) bf16."""
    T, D = int(qc.shape[1]), int(qc.shape[2])
    DV = int(v.shape[3])
    NS = NSPLIT
    dev = qc.device
    TP, _, _ = _fwd_shapes(B, H, T, D, DV, ks, vs)
    BH = B * H
    kp = _gpad(k[:, 0] if ks else k.reshape(BH, T, D), TP)            # (B or BH, TP, D)
    vp = _gpad(v[:, 0] if vs else v.reshape(BH, T, DV), TP)           # (B or BH, TP, DV)
    qg = qc.view(B, H * T, D) if ks else qc
    PP, PR = _gemm_softmax(qg, kp, mask, B, H, T, TP, scale, NS)
    pg = PP.view(B, H * T, NS, TP) if vs else PP
    # one GEMM over the pieces stacked along the rows (row (t, i) = p_i[t]); then their sum, smallest first, cast
    Nv, Mv = pg.shape[0], pg.shape[1]
    ostk = torch.empty((Nv, Mv * NS, DV), dtype=torch.float32, device=dev)
    _mm(ostk, pg.reshape(Nv, Mv * NS, TP), vp, 1.0)
    out = _psum(ostk, Nv, Mv, DV, Mv * NS * DV, NS * DV, DV, NS, (B, H, T, DV))
    return out, kp, vp, PP, PR


@_attn_fwd.register_fake
def _attn_fwd_fake(qc, k, v, mask, scale, ks, vs, B, H):
    T, D = int(qc.shape[1]), int(qc.shape[2])
    DV = int(v.shape[3])
    TP, kp_shape, vp_shape = _fwd_shapes(B, H, T, D, DV, ks, vs)
    dev = qc.device
    return (torch.empty((B, H, T, DV), dtype=torch.bfloat16, device=dev),
            torch.empty(kp_shape, dtype=torch.bfloat16, device=dev),
            torch.empty(vp_shape, dtype=torch.bfloat16, device=dev),
            torch.empty((B * H, T, NSPLIT, TP), dtype=torch.bfloat16, device=dev),
            torch.empty((B * H, T, TP), dtype=torch.bfloat16, device=dev))


def _bwd_alloc_dk(B, H, D, T, dev):
    return torch.empty((B * H, D, T), dtype=torch.bfloat16, device=dev)


@torch.library.custom_op("neuronspark::pool_attn_bwd", mutates_args=(),
                         schema="(Tensor qc, Tensor kp, Tensor vp, Tensor PP, Tensor PR, Tensor dout, int B, int H, "
                                "bool ks, bool vs, float scale) -> (Tensor, Tensor, Tensor, Tensor)")
def _attn_bwd(qc, kp, vp, PP, PR, dout, B, H, ks, vs, scale):
    """(dq (B, H, T, D) bf16, dk (B, H, T, D) bf16 with T fastest, dv (B, H, T, DV) bf16, dmask (B, 1, T, T) bf16)."""
    T, D = int(qc.shape[1]), int(qc.shape[2])
    DV = int(vp.shape[2])
    TP = int(kp.shape[1])
    NS = NSPLIT
    BH = B * H
    dev = qc.device
    do = _as_bmm(dout, (BH, T, DV))
    if vs and not _viewable(do, (B, H * T, DV)):
        do = dout.contiguous().view(BH, T, DV)            # the head-shared value path needs the (B, H T, DV) view
    dP = torch.empty((BH, T, TP), dtype=torch.float32, device=dev)
    if vs:
        _mm(dP.view(B, H * T, TP), do.view(B, H * T, DV), vp.transpose(1, 2), 1.0)
    else:
        _mm(dP, do, vp.transpose(1, 2), 1.0)
    DSP = torch.empty((BH, T, NS, TP), dtype=torch.bfloat16, device=dev)
    dm = torch.empty((B, 1, T, T), dtype=torch.bfloat16, device=dev)
    R, W = _gemm_cfg(TP)
    _gbwd_kernel[(triton.cdiv(T, R), B)](
        PP, PR, dP, DSP, dm, T * TP, TP, dm.stride(0), dm.stride(2), H, T, TP=TP, R=R, NS=NS, NR=_NR, num_warps=W)
    del dP
    # dV: rows i * TP + j of (P pieces viewed (T, NS * TP))^T @ dO
    st = torch.empty((BH, NS * TP, DV), dtype=torch.float32, device=dev)
    _mm(st, PP.view(BH, T, NS * TP).transpose(1, 2), do, 1.0)
    dv = _psum(st, BH, T, DV, NS * TP * DV, DV, TP * DV, NS, (B, H, T, DV))
    del st
    # dQ: rows (t, i) of the dS pieces @ k, alpha = scale
    Nk, Mk = (B, H * T) if ks else (BH, T)
    st = torch.empty((Nk, Mk * NS, D), dtype=torch.float32, device=dev)
    _mm(st, DSP.view(Nk, Mk * NS, TP), kp, scale)
    dq = _psum(st, Nk, Mk, D, Mk * NS * D, NS * D, D, NS, (B, H, T, D))
    del st
    # dK^T: q^T @ (the dS pieces side by side along the columns: column i * TP + j), alpha = scale
    st = torch.empty((BH, D, NS * TP), dtype=torch.float32, device=dev)
    _mm(st, qc.transpose(1, 2), DSP.view(BH, T, NS * TP), scale)
    dkt = _psum(st, BH, D, T, D * NS * TP, NS * TP, TP, NS, (BH, D, T))
    del st
    dk = dkt.view(B, H, D, T).transpose(2, 3)
    return dq, dk, dv, dm


@_attn_bwd.register_fake
def _attn_bwd_fake(qc, kp, vp, PP, PR, dout, B, H, ks, vs, scale):
    T, D = int(qc.shape[1]), int(qc.shape[2])
    DV = int(vp.shape[2])
    dev = qc.device
    dk = _bwd_alloc_dk(B, H, D, T, dev).view(B, H, D, T).transpose(2, 3)
    return (torch.empty((B, H, T, D), dtype=torch.bfloat16, device=dev), dk,
            torch.empty((B, H, T, DV), dtype=torch.bfloat16, device=dev),
            torch.empty((B, 1, T, T), dtype=torch.bfloat16, device=dev))


# ================================================================================================ autograd Functions
class PoolAttnFn(torch.autograd.Function):
    """The pool attention on the bf16 operands (q / k / v (B, H, T, .), mask (B, 1, T, T)), as the module docstring
    describes.  ``kshared``: the key is broadcast over the H query heads (its cast copy's heads are equal by
    construction; the products read head 0); a value with head stride 0 is detected here."""

    @staticmethod
    def forward(ctx, q, k, v, mask, scale, kshared):
        B, H, T, D = (int(x) for x in q.shape)
        if tuple(k.shape) != (B, H, T, D) or tuple(v.shape[:3]) != (B, H, T) or tuple(mask.shape) != (B, 1, T, T):
            raise RuntimeError(f"neuronspark.pool_attention: shapes q {tuple(q.shape)} k {tuple(k.shape)} "
                               f"v {tuple(v.shape)} mask {tuple(mask.shape)}")
        if not (q.dtype == k.dtype == v.dtype == mask.dtype == torch.bfloat16) or mask.stride(3) != 1:
            raise RuntimeError("neuronspark.pool_attention: bf16 operands and a mask with unit last stride expected")
        ks = bool(kshared) and H > 1
        vs = H > 1 and v.stride(1) == 0
        qc = _as_bmm(q, (B * H, T, D))
        out, kp, vp, PP, PR = _attn_fwd(qc, k, v, mask, float(scale), ks, vs, B, H)
        ctx.save_for_backward(qc, kp, vp, PP, PR)
        ctx.meta = (B, H, T, int(v.shape[3]), ks, vs, float(scale))
        return out

    @staticmethod
    def backward(ctx, dout):
        qc, kp, vp, PP, PR = ctx.saved_tensors
        B, H, T, DV, ks, vs, scale = ctx.meta
        if dout.dtype != torch.bfloat16 or tuple(dout.shape) != (B, H, T, DV):
            raise RuntimeError(f"neuronspark.pool_attention backward: bf16 {(B, H, T, DV)} output gradient expected, "
                               f"got {dout.dtype} {tuple(dout.shape)}")
        dq, dk, dv, dm = _attn_bwd(qc, kp, vp, PP, PR, dout, B, H, ks, vs, scale)
        return dq, dk, dv, dm, None, None


class Cast16Fn(torch.autograd.Function):
    """autocast's fp32 -> bf16 cast of q / k into a contiguous layout as one kernel (round to nearest even, as
    ``t.to(torch.bfloat16)``); backward the gradient widened to fp32 in its own layout (= ToCopyBackward)."""

    @staticmethod
    def forward(ctx, x):
        return _cast16_op(x)

    @staticmethod
    def backward(ctx, g):
        return g.to(torch.float32)


def _cast16_ok(t: torch.Tensor) -> bool:
    """The fused cast applies to a 4-D fp32 tensor that is not broadcast (no zero stride) with a power-of-two last dim."""
    return (t.dtype == torch.float32 and t.dim() == 4 and all(st != 0 for st in t.stride())
            and int(t.shape[3]) in (16, 32, 64, 128, 256))


def _cast(t: torch.Tensor, dtype: torch.dtype, contiguous: bool) -> torch.Tensor:
    """autocast's cast (``t.to(dtype)`` when t is floating and not already dtype); ``contiguous`` asks for a contiguous
    layout of the cast copy (a layout change of the same values: the copy's gradient keeps that layout)."""
    if t.dtype == dtype:
        return t if (not contiguous or t.stride(-1) == 1) else t.contiguous()
    if contiguous and t.stride(-1) != 1:
        return t.to(dtype=dtype, memory_format=torch.contiguous_format)
    return t.to(dtype)


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor, scale: float,
          kshared: bool) -> torch.Tensor:
    """The bf16-autocast casts of the attention's operands, then PoolAttnFn: (B, H, T, DV) bf16."""
    for name, t in (("query", query), ("key", key), ("value", value), ("attn_mask", attn_mask)):
        if t.dtype not in (torch.float32, torch.bfloat16):
            raise RuntimeError(f"neuronspark.pool_attention: {name} dtype {t.dtype} (fp32 or bf16 expected)")
    q = Cast16Fn.apply(query) if _cast16_ok(query) else _cast(query, torch.bfloat16, True)
    # the key broadcast over the query heads (stride 0; the extra-query read): its bf16 cast is a materialised copy
    # whose heads are equal by construction, so the products read one head
    ks = key.dim() == 4 and key.shape[1] > 1 and key.stride(1) == 0
    if ks != bool(kshared and key.shape[1] > 1):
        raise RuntimeError("neuronspark.pool_attention: kshared does not match the key's layout")
    k = Cast16Fn.apply(key) if _cast16_ok(key) else _cast(key, torch.bfloat16, True)
    v = _cast(value, torch.bfloat16, True)
    m = _cast(attn_mask, torch.bfloat16, True)
    return PoolAttnFn.apply(q, k, v, m, float(scale), ks)


# ================================================================================================ public
def _check_inputs(q, k, v, mask, kshared):
    for n, t in (("q", q), ("k", k), ("v", v), ("mask", mask)):
        require_cuda(t, "pool_attention")
    if kshared:
        if q.dim() != 4:
            raise RuntimeError(f"neuronspark.pool_attention: with kshared q must be (T, B, M, dk), got {tuple(q.shape)}")
        T, B, M, dk = (int(s) for s in q.shape)
    else:
        if q.dim() != 3:
            raise RuntimeError(f"neuronspark.pool_attention: q must be (T, B, dk), got {tuple(q.shape)}")
        T, B, dk = (int(s) for s in q.shape)
    if tuple(k.shape) != (T, B, dk) or k.dtype != q.dtype:
        raise RuntimeError(f"neuronspark.pool_attention: k must be ({T}, {B}, {dk}) of q's dtype, got {k.dtype} "
                           f"{tuple(k.shape)}")
    if v.dim() != 3 or tuple(v.shape[:2]) != (T, B) or v.dtype != torch.bfloat16:
        raise RuntimeError(f"neuronspark.pool_attention: v must be bf16 ({T}, {B}, dv), got {v.dtype} {tuple(v.shape)}")
    if tuple(mask.shape) != (B, 1, T, T) or mask.dtype != torch.bfloat16:
        raise RuntimeError(f"neuronspark.pool_attention: mask must be bf16 ({B}, 1, {T}, {T}), got {mask.dtype} "
                           f"{tuple(mask.shape)}")
    if q.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"neuronspark.pool_attention: q dtype {q.dtype} (fp32 or bf16 expected)")


def pool_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float,
                   kshared: bool) -> torch.Tensor:
    """The pool attention of one head (``kshared=False``) or of the extra queries sharing the head's keys / values
    (``kshared=True``); see the module docstring for shapes.  Returns bf16 (T, B, dv) or (T, B, M, dv) views.
    Training regime (the casts of the bf16 autocast are explicit; the autocast state is not consulted)."""
    _check_inputs(q, k, v, mask, kshared)
    if kshared:
        M = int(q.shape[2])
        _q = q.permute(1, 2, 0, 3)                                      # (B, M, T, dk)
        _k = k.permute(1, 0, 2).unsqueeze(1).expand(-1, M, -1, -1)
        _v = v.permute(1, 0, 2).unsqueeze(1).expand(-1, M, -1, -1)
        out = _sdpa(_q, _k.to(_q.dtype), _v, mask, scale, M > 1)        # (B, M, T, dv)
        return out.permute(2, 0, 1, 3)                                  # (T, B, M, dv)
    qh = q.permute(1, 0, 2).unsqueeze(1)                                # (B, 1, T, dk)
    kh = k.permute(1, 0, 2).unsqueeze(1)
    vh = v.permute(1, 0, 2).unsqueeze(1)                                # (B, 1, T, dv)
    out = _sdpa(qh, kh, vh, mask, scale, False)
    return out.squeeze(1).permute(1, 0, 2)                              # (T, B, dv)


def pool_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float,
                       kshared: bool) -> torch.Tensor:
    """The same expression in float64 on the bf16-rounded operands: ``P = safe_softmax(scale q16 k16^T + mask16)``,
    ``out = bf16(P v16)``; differentiable (float64 autograd, gradients delivered in the inputs' dtypes).

    ``kshared``: the key and the value are expanded over the M query heads before their casts (an ``expand`` of the
    shared key / value, then the bf16 cast of the expanded key), so each query head's
    key / value gradient is rounded on its own (key: bf16, then widened to fp32; value: bf16) and the expand's backward
    then sums the M heads (key in fp32, value in bf16).  The mask stays (B, 1, T, T): its gradient is the sum over the
    heads before its one bf16 rounding, as the attention forms it."""
    if kshared:
        nq = int(q.shape[2])
        qh = q.permute(1, 2, 0, 3)                                      # (B, M, T, dk)
        kh = k.permute(1, 0, 2).unsqueeze(1).expand(-1, nq, -1, -1)     # (B, M, T, dk): one head, M views
        vh = v.permute(1, 0, 2).unsqueeze(1).expand(-1, nq, -1, -1)     # (B, M, T, dv)
    else:
        qh = q.permute(1, 0, 2).unsqueeze(1)                            # (B, 1, T, dk)
        kh = k.permute(1, 0, 2).unsqueeze(1)
        vh = v.permute(1, 0, 2).unsqueeze(1)
    q64 = qh.to(torch.bfloat16).double()
    k64 = kh.to(torch.bfloat16).double()
    v64 = vh.to(torch.bfloat16).double()
    m64 = mask.to(torch.bfloat16).double()
    s = float(scale) * torch.matmul(q64, k64.transpose(-1, -2)) + m64
    mx = s.detach().amax(dim=-1, keepdim=True)
    mx = torch.where(torch.isinf(mx) & (mx < 0), torch.zeros_like(mx), mx)
    e = torch.exp(s - mx)
    den = e.sum(dim=-1, keepdim=True)
    p = e / torch.where(den == 0, torch.ones_like(den), den)
    out = torch.matmul(p, v64).to(torch.bfloat16)
    if kshared:
        return out.permute(2, 0, 1, 3)
    return out.squeeze(1).permute(1, 0, 2)
