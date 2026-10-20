"""SD gated RMSNorm + soma read-out: the SSD output's gated norm (unit weight), the bipolar soft-threshold soma with its
sigmoid surrogate, the axonal gain gamma and the bf16 rounding of W_out's input.  Training forward / backward.

Pieces:

* ``gated_norm_soma_fwd(x, weight, eps, z, theta, gamma) -> (xo, rstd, n)`` (training forward, internal to
  :class:`neuronspark.ops.sd_combined.SDChunkScanSomaFn`): ``g = x * (z * sigmoid(z))`` (fp32), ``rstd = 1 /
  sqrt(sum_N g^2 / N + eps)`` per row (one row sum of the 9 chunk lanes, left to right), ``n = bf16((g * rstd) * w)``
  (stored), ``S = max(n - theta, 0) - max(-n - theta, 0)`` and ``xo = bf16(S * gamma)`` with IEEE single-rounding
  PTX (sub.rn / max.f32 / mul.rn / cvt.rn.bf16).  x, z (M, N) bf16 rows with unit last stride (z the strided slice of
  the projection), weight (N,) fp32 (the unit gated-norm weight), theta / gamma (N,) fp32; N = 9 x 1024.
* ``gated_norm_soma_bwd(dxs, n, x, weight, rstd, z, dz, theta, gamma, alpha) -> (dy, dz, dtheta, dgamma)`` (training
  backward): the soma backward on the stored n -- ``gp = 1 / (1 + exp2(-(a alpha) log2 e))`` and ``gm`` likewise
  (approximate exp2, div.full, FMA), ``grad_n = bf16(g gamma (gp + gm))``, the theta / gamma gradients as fp32 partial
  sums per 128-row program summed over programs in 32-row blocks -- then the gated RMSNorm backward on grad_n (dz
  written into the given buffer, dy (M, N) bf16).  Its two stages are also callable on their own:
  ``soma_bwd(dxs, n, theta, gamma, alpha) -> (grad_n, dtheta, dgamma)`` and ``gated_rmsnorm_bwd(grad_n, x, weight,
  rstd, z, dz) -> (dy, dz)`` (the same launches).
* ``sd_soma_readout_ref(n, theta, gamma)``: the read-out ``bf16(S(n, theta) * gamma)`` in torch, the reference of the
  inference read-outs (the prefill one is fused into :mod:`neuronspark.ops.triton.sd_gated_norm`, the decode one into
  :mod:`neuronspark.ops.triton.sd_step`).

Numerics: the gated norm forward / backward are fp32-reduction-order class vs mamba_ssm's ``layernorm_gated`` (chunk
lanes instead of one 16384-lane row); the soma forward is the torch chain's arithmetic exactly given n; the soma
backward is fp32-rounding class (approximate exp2 / division, reordered parameter sums).  Checked with the
per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the references below.

Launch constants: forward R 1 row / program, 16 warps, CH 1024 (NCH 9), n stored for the backward; soma backward ROWS 128, BM 2, BC 1024, 4 warps, STAGES 4, sum BR 32 BC 64 4 warps; gated norm backward
R 1, 8 warps, CH 1024 (two passes).  The PTX helpers ``_sr_*`` are :mod:`neuronspark.ops.triton._ptx`'s.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _sr_bf16_to_f32, _sr_f32_to_bf16, _sr_mul_rn, _sr_release

__all__ = [
    "SOMA_FWD_CFG", "SOMA_BWD_CFG", "SOMA_SUM_CFG", "NORM_BWD_CFG",
    "gated_norm_soma_fwd", "gated_norm_soma_bwd", "soma_bwd", "gated_rmsnorm_bwd",
    "gated_norm_soma_fwd_ref", "gated_norm_soma_bwd_ref", "soma_bwd_ref", "gated_rmsnorm_bwd_ref", "sd_soma_readout_ref",
    "BipolarRelease", "bipolar_release",
]

# ------------------------------------------------------------------------------------------------ launch constants
SOMA_FWD_CFG = (1, 16, 1024)            # rows / program, num_warps, CH (chunk lanes; N = 9 CH)
SOMA_FWD_NCH = 9                        # the forward kernel handles rows of 9 chunks
SOMA_BWD_CFG = (128, 2, 1024, 4, 4)     # ROWS / program, BM rows / iteration, BC columns, num_warps, STAGES
SOMA_SUM_CFG = (32, 64, 4)              # BR, BC, num_warps of the partial-sum reduction
NORM_BWD_CFG = (1, 8, 1024)             # rows / program, num_warps, CH


# ================================================================================================================
# Forward kernel _gated_norm_soma_fwd_k9: the gated norm + soma forward for rows of NCH = 9 chunk lanes, the chunks written out.
# ================================================================================================================
@triton.jit
def _gated_norm_soma_fwd_k9(X, XO, Y, W, Z, TH, GA, Rstd, sx, sxo, sy, sz, M, N, eps, R: tl.constexpr, CH: tl.constexpr, STORE_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rv = rows < M
    rm = rv[:, None]
    cols = tl.arange(0, CH)[None, :]
    px = X + rows[:, None] * sx + cols
    pz = Z + rows[:, None] * sz + cols
    x0 = tl.load(px, mask=rm, other=0.)
    x1 = tl.load(px + 1 * CH, mask=rm, other=0.)
    x2 = tl.load(px + 2 * CH, mask=rm, other=0.)
    x3 = tl.load(px + 3 * CH, mask=rm, other=0.)
    x4 = tl.load(px + 4 * CH, mask=rm, other=0.)
    x5 = tl.load(px + 5 * CH, mask=rm, other=0.)
    x6 = tl.load(px + 6 * CH, mask=rm, other=0.)
    x7 = tl.load(px + 7 * CH, mask=rm, other=0.)
    x8 = tl.load(px + 8 * CH, mask=rm, other=0.)
    z0 = tl.load(pz, mask=rm, other=0.).to(tl.float32)
    z1 = tl.load(pz + 1 * CH, mask=rm, other=0.).to(tl.float32)
    z2 = tl.load(pz + 2 * CH, mask=rm, other=0.).to(tl.float32)
    z3 = tl.load(pz + 3 * CH, mask=rm, other=0.).to(tl.float32)
    z4 = tl.load(pz + 4 * CH, mask=rm, other=0.).to(tl.float32)
    z5 = tl.load(pz + 5 * CH, mask=rm, other=0.).to(tl.float32)
    z6 = tl.load(pz + 6 * CH, mask=rm, other=0.).to(tl.float32)
    z7 = tl.load(pz + 7 * CH, mask=rm, other=0.).to(tl.float32)
    z8 = tl.load(pz + 8 * CH, mask=rm, other=0.).to(tl.float32)
    w0 = tl.load(W + cols).to(tl.float32)
    w1 = tl.load(W + 1 * CH + cols).to(tl.float32)
    w2 = tl.load(W + 2 * CH + cols).to(tl.float32)
    w3 = tl.load(W + 3 * CH + cols).to(tl.float32)
    w4 = tl.load(W + 4 * CH + cols).to(tl.float32)
    w5 = tl.load(W + 5 * CH + cols).to(tl.float32)
    w6 = tl.load(W + 6 * CH + cols).to(tl.float32)
    w7 = tl.load(W + 7 * CH + cols).to(tl.float32)
    w8 = tl.load(W + 8 * CH + cols).to(tl.float32)
    g0 = x0.to(tl.float32) * (z0 * tl.sigmoid(z0))
    g1 = x1.to(tl.float32) * (z1 * tl.sigmoid(z1))
    g2 = x2.to(tl.float32) * (z2 * tl.sigmoid(z2))
    g3 = x3.to(tl.float32) * (z3 * tl.sigmoid(z3))
    g4 = x4.to(tl.float32) * (z4 * tl.sigmoid(z4))
    g5 = x5.to(tl.float32) * (z5 * tl.sigmoid(z5))
    g6 = x6.to(tl.float32) * (z6 * tl.sigmoid(z6))
    g7 = x7.to(tl.float32) * (z7 * tl.sigmoid(z7))
    g8 = x8.to(tl.float32) * (z8 * tl.sigmoid(z8))
    var = tl.sum(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4 + g5 * g5 + g6 * g6 + g7 * g7 + g8 * g8, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=rv)
    r2 = rstd[:, None]
    pxo = XO + rows[:, None] * sxo + cols
    py = Y + rows[:, None] * sy + cols
    th0 = tl.load(TH + cols)
    ga0 = tl.load(GA + cols)
    y0 = (g0 * r2) * w0
    if STORE_N:
        tl.store(py, y0, mask=rm)
    v0 = _sr_bf16_to_f32(_sr_f32_to_bf16(y0))
    s0, a0, b0 = _sr_release(v0, th0)
    tl.store(pxo, _sr_f32_to_bf16(_sr_mul_rn(s0, ga0)), mask=rm)
    th1 = tl.load(TH + 1 * CH + cols)
    ga1 = tl.load(GA + 1 * CH + cols)
    y1 = (g1 * r2) * w1
    if STORE_N:
        tl.store(py + 1 * CH, y1, mask=rm)
    v1 = _sr_bf16_to_f32(_sr_f32_to_bf16(y1))
    s1, a1, b1 = _sr_release(v1, th1)
    tl.store(pxo + 1 * CH, _sr_f32_to_bf16(_sr_mul_rn(s1, ga1)), mask=rm)
    th2 = tl.load(TH + 2 * CH + cols)
    ga2 = tl.load(GA + 2 * CH + cols)
    y2 = (g2 * r2) * w2
    if STORE_N:
        tl.store(py + 2 * CH, y2, mask=rm)
    v2 = _sr_bf16_to_f32(_sr_f32_to_bf16(y2))
    s2, a2, b2 = _sr_release(v2, th2)
    tl.store(pxo + 2 * CH, _sr_f32_to_bf16(_sr_mul_rn(s2, ga2)), mask=rm)
    th3 = tl.load(TH + 3 * CH + cols)
    ga3 = tl.load(GA + 3 * CH + cols)
    y3 = (g3 * r2) * w3
    if STORE_N:
        tl.store(py + 3 * CH, y3, mask=rm)
    v3 = _sr_bf16_to_f32(_sr_f32_to_bf16(y3))
    s3, a3, b3 = _sr_release(v3, th3)
    tl.store(pxo + 3 * CH, _sr_f32_to_bf16(_sr_mul_rn(s3, ga3)), mask=rm)
    th4 = tl.load(TH + 4 * CH + cols)
    ga4 = tl.load(GA + 4 * CH + cols)
    y4 = (g4 * r2) * w4
    if STORE_N:
        tl.store(py + 4 * CH, y4, mask=rm)
    v4 = _sr_bf16_to_f32(_sr_f32_to_bf16(y4))
    s4, a4, b4 = _sr_release(v4, th4)
    tl.store(pxo + 4 * CH, _sr_f32_to_bf16(_sr_mul_rn(s4, ga4)), mask=rm)
    th5 = tl.load(TH + 5 * CH + cols)
    ga5 = tl.load(GA + 5 * CH + cols)
    y5 = (g5 * r2) * w5
    if STORE_N:
        tl.store(py + 5 * CH, y5, mask=rm)
    v5 = _sr_bf16_to_f32(_sr_f32_to_bf16(y5))
    s5, a5, b5 = _sr_release(v5, th5)
    tl.store(pxo + 5 * CH, _sr_f32_to_bf16(_sr_mul_rn(s5, ga5)), mask=rm)
    th6 = tl.load(TH + 6 * CH + cols)
    ga6 = tl.load(GA + 6 * CH + cols)
    y6 = (g6 * r2) * w6
    if STORE_N:
        tl.store(py + 6 * CH, y6, mask=rm)
    v6 = _sr_bf16_to_f32(_sr_f32_to_bf16(y6))
    s6, a6, b6 = _sr_release(v6, th6)
    tl.store(pxo + 6 * CH, _sr_f32_to_bf16(_sr_mul_rn(s6, ga6)), mask=rm)
    th7 = tl.load(TH + 7 * CH + cols)
    ga7 = tl.load(GA + 7 * CH + cols)
    y7 = (g7 * r2) * w7
    if STORE_N:
        tl.store(py + 7 * CH, y7, mask=rm)
    v7 = _sr_bf16_to_f32(_sr_f32_to_bf16(y7))
    s7, a7, b7 = _sr_release(v7, th7)
    tl.store(pxo + 7 * CH, _sr_f32_to_bf16(_sr_mul_rn(s7, ga7)), mask=rm)
    th8 = tl.load(TH + 8 * CH + cols)
    ga8 = tl.load(GA + 8 * CH + cols)
    y8 = (g8 * r2) * w8
    if STORE_N:
        tl.store(py + 8 * CH, y8, mask=rm)
    v8 = _sr_bf16_to_f32(_sr_f32_to_bf16(y8))
    s8, a8, b8 = _sr_release(v8, th8)
    tl.store(pxo + 8 * CH, _sr_f32_to_bf16(_sr_mul_rn(s8, ga8)), mask=rm)


# ================================================================================================================
# end of the forward kernel
# ================================================================================================================


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _soma_fast_bwd_p(DX, N, TH, GA, GN, QTH, QGA, M, C, alpha,
                     ROWS: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, STAGES: tl.constexpr):
    """Soma backward over ROWS rows x BC columns: grad_n = bf16(g gamma (gp + gn)), the theta / gamma partial sums of
    the program's rows in fp32; the row loop as tl.range(.., num_stages=STAGES) (loads of later row pairs issued ahead
    by the software pipeliner)."""
    r = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    cm = cols < C
    th = tl.load(TH + cols, mask=cm, other=0.0)[None, :]
    ga = tl.load(GA + cols, mask=cm, other=0.0)[None, :]
    at = tl.zeros((BM, BC), tl.float32)
    ag = tl.zeros((BM, BC), tl.float32)
    for i in tl.range(0, ROWS, BM, num_stages=STAGES):
        rows = r * ROWS + i + tl.arange(0, BM)
        mask = (rows < M)[:, None] & cm[None, :]
        off = rows.to(tl.int64)[:, None] * C + cols[None, :]
        g = tl.load(DX + off, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(N + off, mask=mask, other=0.0).to(tl.float32)
        a, b = v - th, -v - th
        s = tl.maximum(a, 0.0) - tl.maximum(b, 0.0)
        gs = g * ga
        gp = 1.0 / (1.0 + tl.exp2(-(a * alpha) * 1.4426950408889634))
        gm = 1.0 / (1.0 + tl.exp2(-(b * alpha) * 1.4426950408889634))
        tl.store(GN + off, (gs * (gp + gm)).to(tl.bfloat16), mask=mask)
        at += tl.where(mask, -gs * (gp - gm), 0.0)
        ag += tl.where(mask, g * s, 0.0)
    off = r.to(tl.int64) * C + cols
    tl.store(QTH + off, tl.sum(at, axis=0), mask=cm)
    tl.store(QGA + off, tl.sum(ag, axis=0), mask=cm)


@triton.jit
def _soma_fast_sum(QTH, QGA, OTH, OGA, R, C,
                   BR: tl.constexpr, BC: tl.constexpr):
    cols = tl.program_id(0) * BC + tl.arange(0, BC)
    cm = cols < C
    at = tl.zeros((BC,), tl.float32)
    ag = tl.zeros((BC,), tl.float32)
    for start in range(0, R, BR):
        rows = start + tl.arange(0, BR)
        mask = (rows < R)[:, None] & cm[None, :]
        off = rows.to(tl.int64)[:, None] * C + cols[None, :]
        at += tl.sum(tl.load(QTH + off, mask=mask, other=0.0), axis=0)
        ag += tl.sum(tl.load(QGA + off, mask=mask, other=0.0), axis=0)
    tl.store(OTH + cols, at, mask=cm)
    tl.store(OGA + cols, ag, mask=cm)


@triton.jit
def _gated_norm_bwd_kernel(X, W, Z, DY, DX, DZ, Rstd, sx, sz, sdy, sdx, sdz, M, N, NCH, R: tl.constexpr, CH: tl.constexpr):
    """Gated RMSNorm backward (norm_before_gate False, RMS, one group, no weight gradient), R rows per program, two
    passes over the NCH chunks of CH columns: pass 1 acc += h * wdy (per lane, chunk order), c1 = one row sum / N;
    pass 2 reloads x / z / dy and stores dz and dx."""
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rv = rows < M
    rm = rv[:, None]
    cols = tl.arange(0, CH)[None, :]
    px = X + rows[:, None] * sx + cols
    pz = Z + rows[:, None] * sz + cols
    pdy = DY + rows[:, None] * sdy + cols
    r2 = tl.load(Rstd + rows, mask=rv, other=0.)[:, None]
    acc = tl.zeros((R, CH), dtype=tl.float32)
    for c in range(NCH):
        x = tl.load(px + c * CH, mask=rm, other=0.).to(tl.float32)
        z = tl.load(pz + c * CH, mask=rm, other=0.).to(tl.float32)
        d = tl.load(pdy + c * CH, mask=rm, other=0.).to(tl.float32)
        w = tl.load(W + c * CH + cols).to(tl.float32)
        s = tl.sigmoid(z)
        h = (x * z * s) * r2
        wd = w * d
        acc += h * wd
    c1 = (tl.sum(acc, axis=1) / N)[:, None]
    pdz = DZ + rows[:, None] * sdz + cols
    pdx = DX + rows[:, None] * sdx + cols
    for c in range(NCH):
        x = tl.load(px + c * CH, mask=rm, other=0.).to(tl.float32)
        z = tl.load(pz + c * CH, mask=rm, other=0.).to(tl.float32)
        d = tl.load(pdy + c * CH, mask=rm, other=0.).to(tl.float32)
        w = tl.load(W + c * CH + cols).to(tl.float32)
        s = tl.sigmoid(z)
        h = (x * z * s) * r2
        wd = w * d
        g = (wd - h * c1) * r2
        tl.store(pdz + c * CH, g * x * s * (1 + z * (1 - s)), mask=rm)
        tl.store(pdx + c * CH, g * (z * s), mask=rm)


# ------------------------------------------------------------------------------------------------ launchers
def _soma_operands(x, weight, z, theta, gamma, what):
    M, N = x.shape
    if (x.stride(-1) != 1 or z is None or z.stride(-1) != 1 or tuple(z.shape) != (M, N) or tuple(weight.shape) != (N,)
            or weight.stride(-1) != 1 or x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16):
        raise RuntimeError(f"neuronspark.{what}: gated norm operands outside the supported form (bf16 rows, unit last "
                           "stride)")
    for name, t in (("theta", theta), ("gamma", gamma)):
        if (t.dtype != torch.float32 or t.dim() != 1 or int(t.shape[0]) != N or not t.is_contiguous()
                or t.device != x.device):
            raise RuntimeError(f"neuronspark.{what}: {name} must be a contiguous float32 ({N},) tensor on {x.device}, got "
                               f"{t.dtype} {tuple(t.shape)} {t.device}")
    return M, N


def gated_norm_soma_fwd(x, weight, eps, z, theta, gamma):
    """(x_soma (M, N) bf16, rstd (M,) fp32, n (M, N) bf16): see the module docstring."""
    require_cuda(x, "gated_norm_soma_fwd")
    M, N = _soma_operands(x, weight, z, theta, gamma, "gated_norm_soma_fwd")
    R, nw, CH = SOMA_FWD_CFG
    if N != SOMA_FWD_NCH * CH:
        raise RuntimeError(f"neuronspark.gated_norm_soma_fwd: the row width must be {SOMA_FWD_NCH} x {CH}, got N {N}")
    xo = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    n = torch.empty((M, N), dtype=x.dtype, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device.index):
        _gated_norm_soma_fwd_k9[(triton.cdiv(M, R),)](
            x, xo, n, weight, z, theta, gamma, rstd, x.stride(0), xo.stride(0), n.stride(0), z.stride(0), M, N,
            float(eps), R=R, CH=CH, STORE_N=True, num_warps=nw)
    return xo, rstd, n


def _soma_bwd(dx, n, theta, gamma, alpha, gn, gtheta, ggamma):
    """grad_n (gn, preallocated (M, C) bf16) and the theta / gamma gradients (preallocated (C,) fp32) of the soma
    read-out for the incoming x_soma gradient ``dx`` (M, C) bf16 contiguous and the stored n."""
    m, c = n.shape
    rows, bm, bc, warps, stages = SOMA_BWD_CFG
    br, sbc, swarps = SOMA_SUM_CFG
    r = triton.cdiv(m, rows)
    parts = torch.empty((2, r, c), device=n.device, dtype=torch.float32)
    _soma_fast_bwd_p[(r, triton.cdiv(c, bc))](
        dx, n, theta, gamma, gn, parts[0], parts[1], m, c, float(alpha),
        ROWS=rows, BM=bm, BC=bc, STAGES=stages, num_warps=warps, enable_fp_fusion=True)
    _soma_fast_sum[(triton.cdiv(c, sbc),)](
        parts[0], parts[1], gtheta, ggamma, r, c,
        BR=br, BC=sbc, num_warps=swarps, enable_fp_fusion=True)


def _gated_rmsnorm_bwd(dy, x, weight, rstd, z, dz):
    """(dx, dz) of mamba_ssm ``_layer_norm_bwd(dy, x, weight, None, eps, None, rstd, z, group_size=N,
    norm_before_gate=False, is_rms_norm=True, recompute_output=False, dz=dz)`` without the weight gradient; dz is
    written into the given buffer."""
    M, N = x.shape
    if (x.stride(-1) != 1 or dy.stride(-1) != 1 or tuple(dy.shape) != (M, N) or z is None or z.stride(-1) != 1
            or tuple(z.shape) != (M, N) or tuple(weight.shape) != (N,) or weight.stride(-1) != 1 or dz is None
            or tuple(dz.shape) != (M, N) or dz.stride(-1) != 1 or tuple(rstd.shape) != (M,)):
        raise RuntimeError("neuronspark.gated_norm_soma_bwd: gated norm backward operands outside the supported form")
    R, nw, CH = NORM_BWD_CFG
    nch = N // CH
    if N != nch * CH or nch < 1:
        raise RuntimeError(f"neuronspark.gated_norm_soma_bwd: N = {N} is not NCH x CH with CH = {CH}")
    dx = torch.empty_like(x)
    with torch.cuda.device(x.device.index):
        _gated_norm_bwd_kernel[(triton.cdiv(M, R),)](x, weight, z, dy, dx, dz, rstd, x.stride(0), z.stride(0), dy.stride(0),
                                              dx.stride(0), dz.stride(0), M, N, nch, R=R, CH=CH, num_warps=nw)
    return dx, dz


def gated_norm_soma_bwd(dxs, n, x, weight, rstd, z, dz, theta, gamma, alpha):
    """(dy (M, N) bf16 = the gradient at the gated norm's input x, dz (written into the given (M, N) buffer), dtheta
    (N,) fp32, dgamma (N,) fp32) for the incoming gradient dxs (M, N) bf16 of x_soma: the soma backward on the stored
    n, then the gated RMSNorm backward on grad_n (see the module docstring)."""
    require_cuda(x, "gated_norm_soma_bwd")
    M, N = _soma_operands(x, weight, z, theta, gamma, "gated_norm_soma_bwd")
    if tuple(dxs.shape) != (M, N) or not dxs.is_contiguous() or tuple(n.shape) != (M, N) or not n.is_contiguous():
        raise RuntimeError("neuronspark.gated_norm_soma_bwd: dxs / n must be contiguous (M, N) rows")
    if isinstance(alpha, bool) or not float(alpha) > 0.0:
        raise RuntimeError(f"neuronspark.gated_norm_soma_bwd: the surrogate slope must be > 0, got {alpha!r}")
    gn = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    gth = torch.empty((N,), dtype=torch.float32, device=x.device)
    gga = torch.empty((N,), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device.index):
        _soma_bwd(dxs, n, theta, gamma, float(alpha), gn, gth, gga)
    dy, dz = _gated_rmsnorm_bwd(gn, x, weight, rstd, z, dz)
    return dy, dz, gth, gga


def soma_bwd(dxs, n, theta, gamma, alpha):
    """(grad_n (M, N) bf16, dtheta (N,) fp32, dgamma (N,) fp32): the first stage of :func:`gated_norm_soma_bwd` (the
    soma backward on the stored n, the same launch).  dxs / n (M, N) bf16 contiguous; theta / gamma (N,) fp32."""
    require_cuda(n, "soma_bwd")
    if (n.dim() != 2 or n.dtype != torch.bfloat16 or not n.is_contiguous() or tuple(dxs.shape) != tuple(n.shape)
            or dxs.dtype != torch.bfloat16 or not dxs.is_contiguous() or dxs.device != n.device):
        raise RuntimeError("neuronspark.soma_bwd: dxs / n must be contiguous bf16 (M, N) rows on one device")
    M, N = (int(v) for v in n.shape)
    for name, t in (("theta", theta), ("gamma", gamma)):
        if (t.dtype != torch.float32 or t.dim() != 1 or int(t.shape[0]) != N or not t.is_contiguous()
                or t.device != n.device):
            raise RuntimeError(f"neuronspark.soma_bwd: {name} must be a contiguous float32 ({N},) tensor on {n.device}")
    if isinstance(alpha, bool) or not float(alpha) > 0.0:
        raise RuntimeError(f"neuronspark.soma_bwd: the surrogate slope must be > 0, got {alpha!r}")
    gn = torch.empty((M, N), dtype=torch.bfloat16, device=n.device)
    gth = torch.empty((N,), dtype=torch.float32, device=n.device)
    gga = torch.empty((N,), dtype=torch.float32, device=n.device)
    with torch.cuda.device(n.device.index):
        _soma_bwd(dxs, n, theta, gamma, float(alpha), gn, gth, gga)
    return gn, gth, gga


def gated_rmsnorm_bwd(grad_n, x, weight, rstd, z, dz):
    """(dy (M, N) bf16, dz written into the given (M, N) buffer): the second stage of :func:`gated_norm_soma_bwd` (the
    gated RMSNorm backward on grad_n with the stored rstd, the same launch)."""
    require_cuda(x, "gated_rmsnorm_bwd")
    return _gated_rmsnorm_bwd(grad_n, x, weight, rstd, z, dz)


# ------------------------------------------------------------------------------------------------ references
class BipolarRelease(torch.autograd.Function):
    """Bipolar supra-threshold release ``S = max(v - theta, 0) - max(-v - theta, 0)`` with the sigmoid surrogate
    ``dS/dv = sigmoid(alpha (v - theta)) + sigmoid(alpha (-v - theta))``, ``dS/dtheta = -(sigmoid(alpha (v - theta)) -
    sigmoid(alpha (-v - theta)))`` (alpha > 0)."""

    @staticmethod
    def forward(ctx, v, theta, alpha):
        ctx.save_for_backward(v, theta)
        ctx.alpha = float(alpha)
        return torch.clamp_min(v - theta, 0.0) - torch.clamp_min(-v - theta, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        v, theta = ctx.saved_tensors
        gate_pos = torch.sigmoid((v - theta) * ctx.alpha)
        gate_neg = torch.sigmoid((-v - theta) * ctx.alpha)
        return grad_output * (gate_pos + gate_neg), -grad_output * (gate_pos - gate_neg), None


def bipolar_release(v: torch.Tensor, theta: torch.Tensor, alpha: float) -> torch.Tensor:
    return BipolarRelease.apply(v, theta, alpha)


def _gated_norm_ref(x, weight, eps, z):
    """fp32 mamba_ssm layernorm_gated (norm_before_gate False, RMS): g = x silu(z), n = g rsqrt(mean(g^2) + eps) w."""
    zf = z.float()
    g = x.float() * (zf * torch.sigmoid(zf))
    rstd = torch.rsqrt(g.square().mean(dim=-1, keepdim=True) + float(eps))
    return (g * rstd) * weight.float(), rstd.squeeze(-1)


def gated_norm_soma_fwd_ref(x, weight, eps, z, theta, gamma):
    """Pure torch: (x_soma bf16, rstd fp32, n bf16) with n = bf16(gated RMSNorm), x_soma = bf16(S(n, theta) gamma)."""
    y, rstd = _gated_norm_ref(x, weight, eps, z)
    n = y.to(torch.bfloat16)
    s = torch.clamp_min(n.float() - theta, 0.0) - torch.clamp_min(-n.float() - theta, 0.0)
    return (s * gamma).to(torch.bfloat16), rstd, n


_LOG2E = 1.4426950408889634


def soma_bwd_ref(dxs, n, theta, gamma, alpha):
    """Pure torch: the soma read-out backward on the stored n with the sigmoid surrogate, per element in fp32 as the
    kernel evaluates it: a = n - theta, b = -n - theta, S = max(a, 0) - max(b, 0), gs = dxs gamma,
    gp = 1 / (1 + exp2(-(a alpha) log2 e)), gm likewise on b, grad_n = bf16(gs (gp + gm)); dtheta = the row sum of
    -gs (gp - gm), dgamma = the row sum of dxs S (float64 sums).  Returns (grad_n bf16, dtheta fp32, dgamma fp32)."""
    g = dxs.float()
    v = n.float()
    th = theta.float()[None, :]
    a = v - th
    b = -v - th
    s = torch.clamp_min(a, 0.0) - torch.clamp_min(b, 0.0)
    gs = g * gamma.float()[None, :]
    gp = 1.0 / (1.0 + torch.exp2(-(a * float(alpha)) * _LOG2E))
    gm = 1.0 / (1.0 + torch.exp2(-(b * float(alpha)) * _LOG2E))
    grad_n = (gs * (gp + gm)).to(torch.bfloat16)
    dtheta = (-gs.double() * (gp - gm).double()).sum(0)
    dgamma = (g.double() * s.double()).sum(0)
    return grad_n, dtheta.float(), dgamma.float()


def gated_rmsnorm_bwd_ref(grad_n, x, weight, rstd, z):
    """Pure torch: the gated RMSNorm backward (norm_before_gate False, RMS, one group of N, no weight gradient) on
    grad_n with the stored rstd, the kernel's expressions in fp32: s = sigmoid(z), h = (x z s) rstd, wd = w grad_n,
    c1 = sum_N(h wd) / N (float64 sum), g = (wd - h c1) rstd, dz = g x s (1 + z (1 - s)), dy = g (z s).  Returns (dy bf16, dz bf16)."""
    N = int(x.shape[-1])
    d = grad_n.float()
    xf = x.float()
    zf = z.float()
    r2 = rstd.float()[:, None]
    s = torch.sigmoid(zf)
    h = (xf * zf * s) * r2
    wd = weight.float()[None, :] * d
    c1 = ((h.double() * wd.double()).sum(-1, keepdim=True) / N).float()
    g = (wd - h * c1) * r2
    dz = g * xf * s * (1.0 + zf * (1.0 - s))
    dy = g * (zf * s)
    return dy.to(torch.bfloat16), dz.to(torch.bfloat16)


def gated_norm_soma_bwd_ref(dxs, n, x, weight, rstd, z, dz, theta, gamma, alpha):
    """Pure torch: the two-stage backward (the soma backward on the stored n, then the gated RMSNorm backward on
    grad_n with the stored rstd) by :func:`soma_bwd_ref` and :func:`gated_rmsnorm_bwd_ref`.  The dz buffer is not
    used.  Returns (dy bf16, dz bf16, dtheta fp32, dgamma fp32)."""
    grad_n, dtheta, dgamma = soma_bwd_ref(dxs, n, theta, gamma, alpha)
    dy, dzr = gated_rmsnorm_bwd_ref(grad_n, x, weight, rstd, z)
    return dy, dzr, dtheta, dgamma


def sd_soma_readout_ref(n: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Pure torch: ``v = n.float(); S = clamp_min(v - theta, 0) - clamp_min(-v - theta, 0); bf16(S * gamma)``."""
    v = n.float()
    s = torch.clamp_min(v - theta, 0.0) - torch.clamp_min(-v - theta, 0.0)
    return (s * gamma).to(torch.bfloat16)
