"""SD chunk scan kernels: the chunk-scan forward with TMA loads (inference prefill calls that end in a partial chunk;
the arithmetic reference of the Gluon forward) and three pieces of the training backward (dx / ddt / dD, the
intra-chunk dCB and ddA, the ddA assembly).

Internal pieces (not registry ops), all for the SD call of the SSD scan (ngroups 1, D of shape (nheads,), no z, no
seq_idx, dstate <= 128, headdim a power of two):

* ``chunk_scan_fwd_tma(cb, x, dt, dA_cumsum, C, states, D) -> out`` (b, l, h, P) in x's dtype: mamba_ssm
  ``_chunk_scan_fwd`` with descriptor (TMA) loads of C[m] (1, BM, N), states_h (P, N), CB[m, k] (BM, BK) and x_h[k]
  (1, BK, P); one K loop with the stock causal mask.  out[m] = exp(dA_m) C_m states^T + sum_{k <= m}
  bf16(CB[m, k] exp(min(dA_m - dA_k, 0)) dt_k) x_k + D x_m, rounded once to x's dtype.
* ``dx_tma(x, dt, dA_cumsum, B, CB, dout, dstates, D, dx) -> (dx, ddt, dD)``: mamba_ssm
  ``_chunk_scan_chunk_state_bwd_dx_kernel`` (deterministic tile workspaces) with descriptor loads of B[m], dstates_h,
  CB[k, m] and dout_h[k]; one K loop with the causal / chunk-end mask applied under a uniform branch only on the k
  tiles that need it; the decay as ``ex2.approx.ftz`` (results below 2^-126 flush to zero).  ddt (b, h, c, l) and dD
  (h,) fp32 from the workspaces by ``_fin_dx_kernel``.
* ``dda_tma(x, dt, dA_cumsum, dout, cb) -> (dCB partials, V)``: per (lower tile I >= J, b c, head split) the
  intra-chunk products g_h = (dout_h[I] x_h[J]^T) dt_J exp(min(dA_I - dA_J, 0)); dCB partials = sum over the split's
  heads (masked to rows >= cols), V = the per-head row / column sums of g_h * CB (rows > cols) for the assembly; the
  next head's dt / dA rows are loaded one iteration ahead.
* ``dda_asm(v) -> ddA_intra`` (b, h, c, chunk) fp32: ddA[k] = sum_{m >= k} sum_{n < k} T[m, n] from the V grid
  (mamba_ssm ``_chunk_scan_bwd_ddAcs_stable``'s quantity, summation order changed).

Numerics vs the stock mamba_ssm kernels: the same per-element expressions, dot shapes and K order (the stock bf16
rounding points kept); the reduction orders of ddt / dD / ddA partial sums and the FTZ decay of dx are fp32-rounding
class.  Checked with the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the references below.
``dda_tma`` builds its lower-tile table on the first call (a documented first-call cache keyed by (NB, device)); that
call must not be inside a CUDA-graph capture (it raises).

Launch constants: forward (BM 64, BK 32, 4 warps, 3 stages; the prefill configuration); dx (BM 64, BK 64, 4 warps,
2 stages, masked tiles only, FTZ decay); dda (BS 32, NB 8, 2 warps, 3 stages, 24 heads per program, lower tiles only,
next-head prefetch, no FTZ); dda assembly 1 warp.  TMA descriptors: sm_90+.

The forward and dx kernels are mamba_ssm 2.3.2.post1's ``ssd_chunk_scan.py`` ``_chunk_scan_fwd_kernel`` /
``_chunk_scan_chunk_state_bwd_dx_kernel`` (Apache-2.0, see NOTICE) with the descriptor loads and masks stated in
their docstrings; the epilogues are the stock code.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _ex2ftz
from neuronspark.ops.triton.mamba import CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N
from neuronspark.ops.triton.sd_states import _chunked, _desc3, _ex2ftz_ref, _exp_f32

__all__ = [
    "FWD_CFG", "DX_CFG", "DDA_CFG", "DDA_ASM_WARPS",
    "chunk_scan_fwd_tma", "dx_tma", "dda_tma", "dda_asm",
    "chunk_scan_fwd_tma_ref", "dx_tma_ref", "dda_ref", "ssd_forward_ref", "softplus_ref", "chunk_cumsum_fwd_ref",
]

# ------------------------------------------------------------------------------------------------ launch constants
FWD_CFG = (64, 32, 4, 3)            # BM, BK, num_warps, num_stages
DX_CFG = (64, 64, 4, 2)             # BM, BK, num_warps, num_stages
DDA_CFG = (32, 8, 2, 3, 24)         # BS, NB (chunk = BS NB), num_warps, num_stages, heads per program
DDA_ASM_WARPS = 1

# (NB, device index) -> int32 table of the lower tiles I * NB + J (J <= I): a documented first-call cache
_TIJ: Dict[Tuple[int, int], torch.Tensor] = {}


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _dda_kernel(XD, DD, DT, DA, CB, V, DCBP, nchunks, nheads, hpp,
                stb, sth, stc, sab, sah, sac, scb, scc, scm,
                svb, svh, svc, sqb, sqc, sqs, sqm, TIJ,
                P: tl.constexpr, BS: tl.constexpr, NB: tl.constexpr):
    """program = (lower tile t of TIJ (t -> I * NB + J), b * nchunks + c, head split).  A program with I > J also
    stores the zero tile (J, I) of its split (the upper tiles get no program).  dout_h[I] / x_h[J] are (1, BS, P) TMA
    boxes (rows t >= seqlen read as zeros); the next head's dt row and dA rows / cols are loaded one iteration ahead
    (the last head re-reads its own)."""
    t = tl.program_id(0)
    bc = tl.program_id(1)
    sp = tl.program_id(2)
    b = bc // nchunks
    c = bc - b * nchunks
    t = tl.load(TIJ + t)
    I = t // NB
    J = t - I * NB
    r = tl.arange(0, BS)
    rows = I * BS + r
    cols = J * BS + r
    qb = DCBP + b * sqb + c * sqc + sp * sqs + rows[:, None] * sqm + cols[None, :]
    if I > J:
        tl.store(DCBP + b * sqb + c * sqc + sp * sqs + cols[:, None] * sqm + rows[None, :],
                 tl.zeros((BS, BS), dtype=tl.float32))
    if J > I:
        tl.store(qb, tl.zeros((BS, BS), dtype=tl.float32))
    else:
        h0 = sp * hpp
        h1 = tl.minimum(h0 + hpp, nheads)
        cb = tl.load(CB + b * scb + c * scc + rows[:, None] * scm + cols[None, :])
        cb = tl.where(rows[:, None] > cols[None, :], cb, 0.0)
        vb = V + b * svb + c * svc + (I * NB + J) * (2 * BS)
        dacc = tl.zeros((BS, BS), dtype=tl.float32)
        tr = c * (BS * NB) + I * BS
        tcl = c * (BS * NB) + J * BS
        dt_c = tl.load(DT + b * stb + h0 * sth + c * stc + cols)
        ab0 = DA + b * sab + h0 * sah + c * sac
        da_r = tl.load(ab0 + rows)
        da_c = tl.load(ab0 + cols)
        for h in range(h0, h1):
            dI = DD.load([b, tr, h * P]).reshape(BS, P)
            xJ = XD.load([b, tcl, h * P]).reshape(BS, P)
            hn = tl.minimum(h + 1, h1 - 1)
            dt_n = tl.load(DT + b * stb + hn * sth + c * stc + cols)
            abn = DA + b * sab + hn * sah + c * sac
            da_rn = tl.load(abn + rows)
            da_cn = tl.load(abn + cols)
            g = tl.dot(dI, tl.trans(xJ))
            g *= dt_c[None, :]
            g *= tl.exp(tl.minimum(da_r[:, None] - da_c[None, :], 0.0))
            dacc += g
            tt = g * cb
            tl.store(vb + h * svh + r, tl.sum(tt, axis=1))
            tl.store(vb + h * svh + BS + r, tl.sum(tt, axis=0))
            dt_c = dt_n
            da_r = da_rn
            da_c = da_cn
        tl.store(qb, tl.where(rows[:, None] >= cols[None, :], dacc, 0.0))


@triton.jit
def _dda_asm_kernel(V, OUT, svb, svh, svc, sob, soh, soc, BS: tl.constexpr, NB: tl.constexpr):
    """Per (head, chunk, batch), V layout tile = I * NB + J: every load issued first (the (NB, NB, BS) row-sum and
    column-sum grids of the lower tiles), one (NB, BS) store; for k in block K,
        ddA[k] = suffix(RS_K)[k] - suffix(cs(K, K))[k] + exclusive_prefix(sum_{I > K} cs(I, K))[k]
                 + sum_{I > K > J} total(I, J),
    RS_K = sum_{J <= K} rs(K, J), cs(K, K) the diagonal tile's column sums, suffix(v)[k] = sum_{j >= k} v[j];
    = sum_{m >= k} sum_{n < k} T[m, n] (the intra-chunk ddA)."""
    h = tl.program_id(0)
    c = tl.program_id(1)
    b = tl.program_id(2)
    r = tl.arange(0, BS)
    kb = tl.arange(0, NB)
    i3 = kb[:, None, None]
    j3 = kb[None, :, None]
    low = (j3 <= i3) & (r[None, None, :] >= 0)
    gb = V + b * svb + h * svh + c * svc + (i3 * NB + j3) * (2 * BS) + r[None, None, :]
    R3 = tl.load(gb, mask=low, other=0.)                       # rs(I, J)[m]
    C3 = tl.load(gb + BS, mask=low, other=0.)                  # cs(I, J)[n]
    o2 = kb[:, None] * BS + r[None, :]
    RS = tl.sum(R3, axis=1)                                    # (K = I, m): rows of block K over all n < m
    csd = tl.sum(tl.where(i3 == j3, C3, 0.), axis=0)           # (K = J, n): the diagonal tile's column sums
    CSb = tl.sum(tl.where(i3 > j3, C3, 0.), axis=0)            # (K = J, n): column sums of the tiles below
    tot = tl.sum(R3, axis=2)                                   # (I, J) tile totals
    k3 = kb[:, None, None]
    strad = tl.sum(tl.sum(tl.where((j3 > k3) & (k3 > kb[None, None, :]), tot[None, :, :], 0.), axis=2), axis=1)
    o = (tl.cumsum(RS, axis=1, reverse=True) - tl.cumsum(csd, axis=1, reverse=True)
         + (tl.cumsum(CSb, axis=1) - CSb) + strad[:, None])
    tl.store(OUT + b * sob + h * soh + c * soc + o2, o)


@triton.jit
def _dx_kernel(X, CBD, DOD, DOUT, DT, DA, D, BD, DSD, DX, DDT, DDW,
               chunk_size, seqlen, batch, nchunks,
               sxb, sxl, sxh, sob, sol, soh, stb, sth, stc, sab, sah, sac,
               sdxb, sdxl, sdxh, sTb, sTc, sTh, sTl, sWb, sWc, sWh, sWm,
               H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    """mamba_ssm _chunk_scan_chunk_state_bwd_dx_kernel (SD call) with descriptor loads of B[m] (1, BM, N), dstates_h
    (P, N), CB[k, m] (BK, BM, transposed in registers to the stock's (m, k) tile) and dout_h[k] (1, BK, P); one K loop
    with the (causal & chunk-end) mask under a uniform branch only on the k tiles that need it (the diagonal block's
    and a partial chunk's last; elsewhere it is all-true); the decay's exp as ex2.approx.ftz.  The epilogue (dt, D dout,
    dx store, x, dD / ddt partials) is the stock code."""
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, P)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dab = DA + pid_b * sab + pid_c * sac + pid_h * sah
    dA_cs_m = tl.load(dab + offs_m, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dA_cs_last = tl.load(dab + (chunk_size - 1)).to(tl.float32)
    scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    t0 = pid_c * chunk_size
    b = BD.load([pid_b, t0 + pid_m * BM, 0]).reshape(BM, N)
    ds = DSD.load([((pid_b * nchunks + pid_c) * H + pid_h) * P, 0])
    acc = tl.dot(b, tl.trans(ds)) * scale[:, None]
    offs_k = tl.arange(0, BK)
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BM
    row0 = (pid_b * nchunks + pid_c) * chunk_size
    K_DE = K_MIN + BM
    for k in range(K_MIN, K_MAX, BK):
        k = tl.multiple_of(k, BK)
        cb = tl.trans(CBD.load([row0 + k, pid_m * BM]))    # (BM, BK) = CB[k, m]^T
        dout = DOD.load([pid_b, t0 + k, pid_h * P]).reshape(BK, P)
        dA_cs_k = tl.load(dab + k + offs_k, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        cb *= _ex2ftz(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0) * 1.4426950408889634)
        need = (k < K_DE) | (k + BK > K_MAX)                  # uniform: the diagonal block / a partial chunk end
        if need:
            mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
            cb = tl.where(mask, cb, 0.0)
        acc += tl.dot(cb.to(DOUT.dtype.element_ty), dout)
    # epilogue: the stock code
    dt_m = tl.load(DT + pid_b * stb + pid_c * stc + pid_h * sth + offs_m, mask=offs_m < chunk_size_limit,
                   other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dxp = DX + pid_b * sdxb + pid_c * chunk_size * sdxl + pid_h * sdxh
    dx_ptrs = dxp + (offs_m[:, None] * sdxl + offs_n[None, :])
    dop = DOUT + pid_b * sob + pid_c * chunk_size * sol + pid_h * soh
    dout_res = tl.load(dop + (offs_m[:, None] * sol + offs_n[None, :]),
                       mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < P), other=0.0).to(tl.float32)
    Dh = tl.load(D + pid_h).to(tl.float32)
    dx += dout_res * Dh
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < P))
    xp = X + pid_b * sxb + pid_c * chunk_size * sxl + pid_h * sxh
    x = tl.load(xp + (offs_m[:, None] * sxl + offs_n[None, :]),
                mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < P), other=0.0).to(tl.float32)
    dD = tl.sum(dout_res * x)
    tl.store(DDW + pid_b * sWb + pid_c * sWc + pid_h * sWh + pid_m * sWm, dD)
    ddt = tl.sum(acc * x, axis=1)
    tl.store(DDT + pid_b * sTb + pid_c * sTc + pid_h * sTh + offs_m * sTl, ddt, mask=offs_m < chunk_size)


@triton.jit
def _fin_dx_kernel(WT, DDT, WD, DD, R, NT, NVM, BCH, NVN, H, DH,
                   wds0, wds1, wds2, wds3, wds4,
                   NTP: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
    """programs < cdiv(R, BR): DDT[r] = sum_{t < NT} WT[r * DH + t] (WT (R, DH) contiguous: the ddt workspace, NT of its
    DH tiles written); programs >= that: DD[h] = sum over m < NVM, (b, c) < BCH, n < NVN of the dD workspace
    (pid_m_tiles, batch, nchunks, nheads, dD_hdim) (strides wds*; (b, c) contiguous: offset (b nchunks + c) wds2)."""
    pid = tl.program_id(0)
    nb = tl.cdiv(R, BR)
    if pid < nb:
        r = (pid * BR + tl.arange(0, BR)).to(tl.int64)
        t = tl.arange(0, NTP)
        m = (r < R)[:, None] & (t < NT)[None, :]
        v = tl.load(WT + r[:, None] * DH + t[None, :], mask=m, other=0.0)
        tl.store(DDT + r, tl.sum(v, axis=1), mask=r < R)
    else:
        h = (pid - nb).to(tl.int64)
        # dD workspace (pid_m_tiles, batch, nchunks, nheads, dD_hdim): sum over m < NVM, (b, c) < BCH, n < NVN
        acc = tl.zeros([BK], dtype=tl.float32)
        n_items = NVM * BCH * NVN
        for k0 in range(0, n_items, BK):
            k = k0 + tl.arange(0, BK)
            km = k < n_items
            n = k % NVN
            q = k // NVN
            bc = q % BCH
            mm = q // BCH
            off = mm.to(tl.int64) * wds0 + bc.to(tl.int64) * wds2 + h * wds3 + n.to(tl.int64) * wds4
            acc += tl.load(WD + off, mask=km, other=0.0)
        tl.store(DD + h, tl.sum(acc, axis=0))


@triton.jit
def _fwd_kernel(CBD, XD, X, OUT, DT, DA, CD, SD, D,
                chunk_size, seqlen, batch, nchunks,
                sxb, sxl, sxh, sob, sol, soh, stb, sth, stc, sab, sah, sac,
                H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr):
    """mamba_ssm _chunk_scan_fwd_kernel (SD call: causal, D (nheads,), no z / seq_idx, dstate <= 128, BLOCK_SIZE_N =
    headdim) with descriptor loads of C[m] (1, BM, N), states_h (P, N), CB[m, k] (BM, BK) and x_h[k] (1, BK, P); one K
    loop with the stock causal mask.  The D residual and the output store are the stock code."""
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, P)
    dab = DA + pid_b * sab + pid_c * sac + pid_h * sah
    dtb = DT + pid_b * stb + pid_c * stc + pid_h * sth
    dA_cs_m = tl.load(dab + offs_m, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    t0 = pid_c * chunk_size
    cm = CD.load([pid_b, t0 + pid_m * BM, 0]).reshape(BM, N)
    stt = SD.load([((pid_b * nchunks + pid_c) * H + pid_h) * P, 0])
    scale_m = tl.exp(dA_cs_m)
    acc = tl.dot(cm, tl.trans(stt)) * scale_m[:, None]
    offs_k = tl.arange(0, BK)
    row = (pid_b * nchunks + pid_c) * chunk_size + pid_m * BM
    K_MAX = min((pid_m + 1) * BM, chunk_size_limit)
    for k in range(0, K_MAX, BK):                             # the reference's causal mask on every k tile
        cb = CBD.load([row, k])
        dA_cs_k = tl.load(dab + k + offs_k, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_k[None, :]), 0.0))
        dt_k = tl.load(dtb + k + offs_k, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        cb = tl.where(offs_m[:, None] >= k + offs_k[None, :], cb, 0.0)
        cb = cb.to(X.dtype.element_ty)
        xk = XD.load([pid_b, t0 + k, pid_h * P]).reshape(BK, P)
        acc += tl.dot(cb, xk)
    Dh = tl.load(D + pid_h).to(tl.float32)
    xp = X + pid_b * sxb + pid_c * chunk_size * sxl + pid_h * sxh
    x_residual = tl.load(xp + (offs_m[:, None] * sxl + offs_n[None, :]),
                         mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < P), other=0.0).to(tl.float32)
    acc += x_residual * Dh
    op = OUT + pid_b * sob + pid_c * chunk_size * sol + pid_h * soh
    tl.store(op + (sol * offs_m[:, None] + offs_n[None, :]), acc,
             mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < P))


# ------------------------------------------------------------------------------------------------ launchers
def _p2(n):
    return int(triton.next_power_of_2(max(int(n), 1)))


def dda_tma(x, dt, dA_cumsum, dout, cb):
    """(dCB head-split partials (batch, nchunks, nsplit, 1, chunk, chunk) fp32, V (batch, nheads, nchunks,
    NB * NB * 2 BS) fp32 row / column sums for ``dda_asm``).  x / dout (b, l, h, P) bf16 with a contiguous (h, P)
    part; dt / dA_cumsum (b, h, nc, chunk) fp32 with unit last stride; cb (b, nc, 1, chunk, chunk) fp32."""
    require_cuda(x, "dda_tma")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    BS, NB, nw, st, hcfg = DDA_CFG
    if BS * NB != chunk or dout.shape != x.shape or tuple(cb.shape) != (batch, nchunks, 1, chunk, chunk):
        raise RuntimeError(f"neuronspark.dda_tma: shapes x {tuple(x.shape)} dout {tuple(dout.shape)} cb {tuple(cb.shape)} "
                           f"chunk {chunk} BS {BS} NB {NB}")
    if cb.dtype != torch.float32 or dt.stride(-1) != 1 or dA_cumsum.stride(-1) != 1 or cb.stride(-1) != 1:
        raise RuntimeError("neuronspark.dda_tma: fp32 CB and unit last strides of dt / dA_cumsum / CB expected")
    hpp = min(max(hcfg, 1), nheads)
    nsplit = math.ceil(nheads / hpp)
    v = torch.empty((batch, nheads, nchunks, NB * NB * 2 * BS), dtype=torch.float32, device=x.device)
    dcbp = torch.empty((batch, nchunks, nsplit, 1, chunk, chunk), dtype=torch.float32, device=x.device)
    xd = _desc3(x, BS)
    dd = _desc3(dout, BS)
    key = (NB, x.device.index)
    tij = _TIJ.get(key)
    if tij is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("neuronspark.dda_tma: the lower-tile table is built on the first (eager) call; first call "
                               "inside a CUDA-graph capture")
        tij = torch.tensor([i * NB + j for i in range(NB) for j in range(i + 1)], dtype=torch.int32, device=x.device)
        _TIJ[key] = tij
    grid = (NB * (NB + 1) // 2, batch * nchunks, nsplit)
    with torch.cuda.device(x.device.index):
        _dda_kernel[grid](
            xd, dd, dt, dA_cumsum, cb, v, dcbp, nchunks, nheads, hpp,
            dt.stride(0), dt.stride(1), dt.stride(2), dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            cb.stride(0), cb.stride(1), cb.stride(3), v.stride(0), v.stride(1), v.stride(2),
            dcbp.stride(0), dcbp.stride(1), dcbp.stride(2), dcbp.stride(4), tij,
            P=headdim, BS=BS, NB=NB, num_warps=nw, num_stages=st)
    return dcbp, v


def dda_asm(v):
    """The intra-chunk ddA (batch, nheads, nchunks, chunk) fp32 from the V grid of ``dda_tma``."""
    BS, NB = DDA_CFG[0], DDA_CFG[1]
    batch, nheads, nchunks, _ = v.shape
    out = torch.empty((batch, nheads, nchunks, BS * NB), dtype=torch.float32, device=v.device)
    with torch.cuda.device(v.device.index):
        _dda_asm_kernel[(nheads, nchunks, batch)](
            v, out, v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1), out.stride(2),
            BS=BS, NB=NB, num_warps=DDA_ASM_WARPS)
    return out


def dx_tma(x, dt, dA_cumsum, B, CB, dout, dstates, D, dx):
    """(dx (written into ``dx`` when given), ddt (b, h, c, l) fp32, dD (nheads,) in D's dtype) of the stock
    ``_chunk_scan_chunk_state_bwd_dx`` (see the module docstring)."""
    require_cuda(x, "dx_tma")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    BM, BK, nw, st = DX_CFG
    if ngroups != 1 or dstate > 128 or headdim not in (16, 32, 64, 128):
        raise RuntimeError(f"neuronspark.dx_tma: ngroups {ngroups} dstate {dstate} headdim {headdim} (SD call: 1 / <= 128 "
                           "/ pow2)")
    if B.shape != (batch, seqlen, ngroups, dstate):
        raise RuntimeError("neuronspark.dx_tma: B shape")
    if CB.shape != (batch, nchunks, ngroups, chunk_size, chunk_size) or CB.dtype != torch.float32:
        raise RuntimeError("neuronspark.dx_tma: CB must be fp32 (batch, nchunks, 1, chunk, chunk)")
    if dt.shape != (batch, nheads, nchunks, chunk_size) or dA_cumsum.shape != dt.shape:
        raise RuntimeError("neuronspark.dx_tma: dt / dA_cumsum shapes")
    if dout.shape != x.shape or dstates.shape != (batch, nchunks, nheads, headdim, dstate):
        raise RuntimeError("neuronspark.dx_tma: dout / dstates shapes")
    if dstates.dtype != B.dtype or dout.dtype != x.dtype:
        raise RuntimeError(f"neuronspark.dx_tma: dstates {dstates.dtype} / B {B.dtype}, dout {dout.dtype} / x {x.dtype} "
                           "must agree")
    if D is None or D.dim() != 1 or D.stride(-1) != 1:
        raise RuntimeError("neuronspark.dx_tma: D must be (nheads,) contiguous (the SD mixer's call)")
    if (chunk_size % BM) or (BM % BK) or B.stride(-1) != 1 or not CB.is_contiguous() or not dstates.is_contiguous():
        raise RuntimeError(f"neuronspark.dx_tma: chunk {chunk_size} BM {BM} BK {BK} or operand layouts not supported")
    if dt.stride(-1) != 1 or dA_cumsum.stride(-1) != 1 or x.stride(-1) != 1 or dout.stride(-1) != 1:
        raise RuntimeError("neuronspark.dx_tma: unit last strides of dt / dA_cumsum / x / dout expected")
    pid_m_tiles = triton.cdiv(chunk_size, 32)
    tile_count = math.ceil(headdim / CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
    nvm = triton.cdiv(chunk_size, BM)
    if nvm > pid_m_tiles:
        raise RuntimeError(f"neuronspark.dx_tma: {nvm} m tiles, the dD workspace holds {pid_m_tiles}")
    dD_ws = torch.empty(pid_m_tiles, batch, nchunks, nheads, tile_count, device=D.device, dtype=torch.float32)
    if dx is None:
        dx = torch.empty_like(x)
    elif dx.shape != x.shape or dx.stride(-1) != 1:
        raise RuntimeError("neuronspark.dx_tma: dx must have x's shape and a unit last stride")
    ddt_ws = torch.empty(batch, nheads, nchunks, chunk_size, tile_count, device=dout.device, dtype=torch.float32)
    bd = TensorDescriptor(B, [batch, seqlen, dstate], [B.stride(0), B.stride(1), 1], [1, BM, dstate])
    dsd = TensorDescriptor(dstates, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, dstate])
    cbd = TensorDescriptor(CB, [batch * nchunks * chunk_size, chunk_size], [chunk_size, 1], [BK, BM])
    dod = _desc3(dout, BK)
    grid = (nvm, batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _dx_kernel[grid](
            x, cbd, dod, dout, dt, dA_cumsum, D, bd, dsd, dx, ddt_ws, dD_ws,
            chunk_size, seqlen, batch, nchunks,
            x.stride(0), x.stride(1), x.stride(2), dout.stride(0), dout.stride(1), dout.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2), dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            ddt_ws.stride(0), ddt_ws.stride(2), ddt_ws.stride(1), ddt_ws.stride(3),
            dD_ws.stride(1), dD_ws.stride(2), dD_ws.stride(3), dD_ws.stride(0),
            H=nheads, P=headdim, N=dstate, BM=BM, BK=BK, num_warps=nw, num_stages=st)
    R = batch * nheads * nchunks * chunk_size
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    dD32 = torch.empty(nheads, device=D.device, dtype=torch.float32)
    BR = 256
    _fin_dx_kernel[(triton.cdiv(R, BR) + nheads,)](
        ddt_ws, ddt, dD_ws, dD32, R, 1, nvm, batch * nchunks, 1, nheads, tile_count,
        *dD_ws.stride(), NTP=_p2(tile_count), BR=BR, BK=1024, num_warps=4)
    dD = dD32 if D.dtype == torch.float32 else dD32.to(D.dtype)
    return dx, ddt, dD


def chunk_scan_fwd_tma(cb, x, dt, dA_cumsum, C, states, D):
    """out (batch, seqlen, nheads, headdim) in x's dtype = mamba_ssm ``_chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states,
    D=D, z=None, seq_idx=None)[0]`` for the SD call (see _fwd_kernel).  cb (b, nc, 1, chunk, chunk) fp32 contiguous;
    x (b, l, h, P) with a contiguous (h, P) part; dt / dA_cumsum (b, h, nc, chunk) with unit last stride; C (b, l, 1, N)
    with unit last stride; states (b, nc, h, P, N) contiguous in C's dtype; D (h,) contiguous.  Partial last chunks are
    supported (rows past seqlen are the descriptors' zero fill and are not stored)."""
    require_cuda(x, "chunk_scan_fwd_tma")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    BM, BK, nw, st = FWD_CFG
    if ngroups != 1 or dstate > 128 or headdim not in (16, 32, 64, 128):
        raise RuntimeError(f"neuronspark.chunk_scan_fwd_tma: ngroups {ngroups} dstate {dstate} headdim {headdim} (SD call: "
                           "1 / <= 128 / pow2)")
    if C.shape != (batch, seqlen, ngroups, dstate):
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: C shape")
    if cb.shape != (batch, nchunks, ngroups, chunk_size, chunk_size) or cb.dtype != torch.float32:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: cb must be fp32 (batch, nchunks, 1, chunk, chunk)")
    if dt.shape != (batch, nheads, nchunks, chunk_size) or dA_cumsum.shape != dt.shape:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: dt / dA_cumsum shapes")
    if states.shape != (batch, nchunks, nheads, headdim, dstate):
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: states shape")
    if states.dtype != C.dtype or D is None or D.dim() != 1 or D.stride(-1) != 1:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: states in C's dtype and D (nheads,) contiguous expected (the "
                           "SD call)")
    if (chunk_size % BM) or (BM % BK) or C.stride(-1) != 1 or not cb.is_contiguous() or not states.is_contiguous():
        raise RuntimeError(f"neuronspark.chunk_scan_fwd_tma: chunk {chunk_size} BM {BM} BK {BK} or operand layouts not "
                           "supported")
    if dt.stride(-1) != 1 or dA_cumsum.stride(-1) != 1 or x.stride(-1) != 1:
        raise RuntimeError("neuronspark.chunk_scan_fwd_tma: unit last strides of dt / dA_cumsum / x expected")
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    cd = TensorDescriptor(C, [batch, seqlen, dstate], [C.stride(0), C.stride(1), 1], [1, BM, dstate])
    sd = TensorDescriptor(states, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, dstate])
    cbd = TensorDescriptor(cb, [batch * nchunks * chunk_size, chunk_size], [chunk_size, 1], [BM, BK])
    xd = _desc3(x, BK)
    grid = (triton.cdiv(chunk_size, BM), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _fwd_kernel[grid](
            cbd, xd, x, out, dt, dA_cumsum, cd, sd, D,
            chunk_size, seqlen, batch, nchunks,
            x.stride(0), x.stride(1), x.stride(2), out.stride(0), out.stride(1), out.stride(2),
            dt.stride(0), dt.stride(1), dt.stride(2), dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            H=nheads, P=headdim, N=dstate, BM=BM, BK=BK, num_warps=nw, num_stages=st)
    return out


# ------------------------------------------------------------------------------------------------ references
# Pure torch.  The piece references (``chunk_scan_fwd_tma_ref``, ``dx_tma_ref``, ``dda_ref``) are the stock mamba_ssm
# math in float64 between the kernels' rounding points, which they reproduce: the masked fp32 operand tile
# that a kernel rounds to bf16 before its dot is evaluated in fp32 with the kernel's expression order (the decay
# ``tl.exp`` as ``exp2(t * log2 e)``, see ``sd_states._exp_f32``) and rounded the same way.  ``ssd_forward_ref`` is the
# float32 differentiable whole-forward math without the kernels' internal roundings.
def _tril(chunk: int, device) -> torch.Tensor:
    i = torch.arange(chunk, device=device)
    return i[:, None] >= i[None, :]


def chunk_scan_fwd_tma_ref(cb, x, dt, dA_cumsum, C, states, D):
    """Reference of ``chunk_scan_fwd_tma`` (mamba_ssm ssd_chunk_scan.py ``_chunk_scan_fwd_kernel``; ``_fwd_kernel``'s
    expressions): out[m] = exp(dA_m) C_m states_c^T
    + sum_{k <= m} P[m, k] x_k + D x_m with the dot operand P = bf16(where(m >= k, (CB[m, k] exp(min(dA_m - dA_k, 0)))
    dt_k, 0)) formed in fp32 as the kernel forms it (``cb *= exp(..)``, ``cb *= dt_k``, mask, cast to x's dtype);
    float64 sums; the result in x's dtype."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)                                   # (b, c, l, h, p)
    Cc = _chunked(C[:, :, 0], nchunks, chunk)                          # (b, c, l, n)
    dA = dA_cumsum.float()                                             # (b, h, c, l)
    y_off = torch.einsum("bcln,bchpn,bhcl->bclhp", Cc, states.double(), _exp_f32(dA).double())
    seg = torch.clamp_max(dA[..., :, None] - dA[..., None, :], 0.0)    # (b, h, c, m, k) fp32: min(dA_m - dA_k, 0)
    Pm = cb[:, :, 0].float()[:, :, None] * _exp_f32(seg).permute(0, 2, 1, 3, 4)      # (b, c, h, m, k) fp32
    Pm = Pm * dt.float().permute(0, 2, 1, 3)[:, :, :, None, :]                        # * dt_k
    Pm = torch.where(_tril(chunk, x.device), Pm, torch.zeros_like(Pm)).to(x.dtype).double()
    y = y_off + torch.einsum("bchls,bcshp->bclhp", Pm, xc) + D.double()[None, None, None, :, None] * xc
    return y.reshape(batch, nchunks * chunk, nheads, headdim)[:, :seqlen].to(x.dtype)


def dx_tma_ref(x, dt, dA_cumsum, B, CB, dout, dstates, D, dx):
    """Reference of ``dx_tma`` (mamba_ssm ``_chunk_scan_chunk_state_bwd_dx`` math; ``_dx_kernel``'s expressions with
    the FTZ decay): acc[m] = exp(min(dA_last - dA_m, 0)) B_m dstates^T +
    sum_{k >= m} K[k, m] dout_k with the dot operand K = bf16(where(k >= m, CB[k, m] ex2ftz(min(dA_k - dA_m, 0)
    log2 e), 0)) formed in fp32 as the kernel forms it (cast to dout's dtype); dx = acc dt_m + D dout_m;
    ddt[m] = sum_p acc x_m; dD = sum dout x; float64 sums.  Returns (dx in x's dtype, ddt (b, h, c, l) fp32, dD (h,)
    fp32); ``dx`` (the output buffer argument of the kernel launcher) is ignored."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)
    do = _chunked(dout, nchunks, chunk)
    Bc = _chunked(B[:, :, 0], nchunks, chunk)
    dA = dA_cumsum.float()
    sc = _exp_f32(torch.clamp_max(dA[..., -1:] - dA, 0.0)).double()   # (b, h, c, l)
    acc = torch.einsum("bcln,bchpn,bhcl->bclhp", Bc, dstates.double(), sc)
    seg = torch.clamp_max(dA[..., :, None] - dA[..., None, :], 0.0)    # (b, h, c, k, m) fp32: min(dA_k - dA_m, 0)
    K = CB[:, :, 0].float()[:, :, None] * _ex2ftz_ref(seg).permute(0, 2, 1, 3, 4)    # (b, c, h, k, m) fp32
    K = torch.where(_tril(chunk, x.device), K, torch.zeros_like(K)).to(dout.dtype).double()
    acc = acc + torch.einsum("bchkm,bckhp->bcmhp", K, do)
    dtc = dt.double().permute(0, 2, 3, 1)                              # (b, c, l, h)
    dxr = acc * dtc[..., None] + do * D.double()[None, None, None, :, None]
    ddt = (acc * xc).sum(-1).permute(0, 3, 1, 2)                       # (b, h, c, l)
    dD = (do * xc).sum((0, 1, 2, 4))
    dxr = dxr.reshape(batch, nchunks * chunk, nheads, headdim)[:, :seqlen]
    return dxr.to(x.dtype), ddt.float(), dD.float()


def dda_ref(x, dt, dA_cumsum, dout, cb):
    """float64 reference of ``dda_tma`` + ``dda_asm`` (mamba_ssm ``_chunk_scan_bwd_dcb`` and
    ``_chunk_scan_bwd_ddAcs_stable`` math): g_h[m, n] = (dout_h[m] . x_h[n]) dt_n exp(min(dA_m - dA_n, 0));
    dCB[m, n] = sum_h g_h[m, n] for n <= m (0 above); ddA_intra[k] = sum_{m >= k} sum_{n < k} g_h[m, n] CB[m, n] (n < m).
    Returns (dCB (b, c, 1, chunk, chunk) fp32, ddA_intra (b, h, c, chunk) fp32)."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    xc = _chunked(x, nchunks, chunk)
    do = _chunked(dout, nchunks, chunk)
    dA = dA_cumsum.double()
    tril = _tril(chunk, x.device)
    g = torch.einsum("bcmhp,bcnhp->bhcmn", do, xc)
    g = g * dt.double()[..., None, :] * torch.exp(torch.clamp_max(dA[..., :, None] - dA[..., None, :], 0.0))
    dcb = (g * tril).sum(1)[:, :, None]                                # (b, c, 1, m, n)
    strict = tril & ~torch.eye(chunk, dtype=torch.bool, device=x.device)
    G = g * cb[:, :, 0].double()[:, None] * strict                     # (b, h, c, m, n)
    Q = torch.cumsum(G, dim=-1) - G                                    # Q[m, k] = sum_{n < k} G[m, n]
    dda = (Q * tril).sum(-2)                                           # sum over m >= k
    return dcb.float(), dda.float()


def softplus_ref(v: torch.Tensor) -> torch.Tensor:
    """The SSD kernels' softplus (Triton >= 3 form of mamba_ssm ops/triton/softplus.py): ``where(v <= 20,
    log(exp(v) + 1), v)``; the exponential is taken of ``min(v, 20)`` so that the unselected branch stays finite under
    autograd (the selected values are unchanged)."""
    return torch.where(v <= 20.0, torch.log(torch.exp(torch.clamp_max(v, 20.0)) + 1.0), v)


def chunk_cumsum_fwd_ref(dt_raw, A, dt_bias, chunk_size):
    """Reference of the vendored ``_chunk_cumsum_fwd`` for the SD call (dt_softplus, dt_limit (0, inf);
    ssd_chunk_state.py ``_chunk_cumsum_fwd_kernel``): dt = max(softplus(dt_raw + dt_bias), 0) in fp32 (0 past seqlen),
    dA_cumsum = the per-chunk cumulative sum of dt A (float64 sums).  dt_raw (b, l, h), A / dt_bias (h,); returns
    (dA_cumsum, dt) (b, h, nchunks, chunk) fp32, the kernel's output order."""
    batch, seqlen, nheads = dt_raw.shape
    nchunks = math.ceil(seqlen / chunk_size)
    L = nchunks * chunk_size
    dt = torch.clamp_min(softplus_ref(dt_raw.float() + dt_bias.float()), 0.0)          # (b, l, h)
    if L != seqlen:
        dt = F.pad(dt, (0, 0, 0, L - seqlen))
    dt = dt.reshape(batch, nchunks, chunk_size, nheads).permute(0, 3, 1, 2).contiguous()   # (b, h, c, chunk)
    dA_cs = torch.cumsum(dt.double() * A.float().double()[None, :, None, None], dim=-1)
    return dA_cs.float(), dt


def ssd_forward_ref(x, dt_raw, A, B, C, chunk_size, D, dt_bias):
    """Pure-torch float32 SSD forward of the SD call (differentiable; mamba_ssm ``_mamba_chunk_scan_combined_fwd`` math,
    ssd_combined.py 343-392): dt = softplus(dt_raw + dt_bias) (kernel form), dA = dt A per step, chunked: intra-chunk
    y_diag[l] = sum_{s <= l} (C_l . B_s) exp(dA_cs[l] - dA_cs[s]) dt_s x_s, chunk states and their passing, the
    inter-chunk term exp(dA_cs[l]) C_l s_c^T, and D x.  x (b, l, h, P), dt_raw (b, l, h), A / D / dt_bias (h,),
    B / C (b, l, 1, N); returns y (b, l, h, P) in x's dtype."""
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    L = nchunks * chunk_size

    def pad(t):
        if int(t.shape[1]) == L:
            return t
        return F.pad(t, [0, 0] * (t.dim() - 2) + [0, L - int(t.shape[1])])

    dt = softplus_ref(dt_raw.float() + dt_bias.float())                # (b, l, h)
    dt = pad(dt).reshape(batch, nchunks, chunk_size, nheads)           # padded steps: dt 0
    xc = pad(x.float()).reshape(batch, nchunks, chunk_size, nheads, headdim)
    Bc = pad(B[:, :, 0].float()).reshape(batch, nchunks, chunk_size, -1)
    Cc = pad(C[:, :, 0].float()).reshape(batch, nchunks, chunk_size, -1)
    dA = dt * A.float()                                                # (b, c, l, h)
    dA_cs = torch.cumsum(dA, dim=2).permute(0, 3, 1, 2)                # (b, h, c, l)
    CB = torch.einsum("bcln,bcsn->bcls", Cc, Bc)
    tril = _tril(chunk_size, x.device)
    decay = torch.exp(torch.clamp_max(dA_cs[..., :, None] - dA_cs[..., None, :], 0.0)) * tril     # (b, h, c, l, s)
    M = CB[:, :, None] * decay.permute(0, 2, 1, 3, 4) * dt.permute(0, 1, 3, 2)[:, :, :, None, :]  # (b, c, h, l, s)
    y = torch.einsum("bchls,bcshp->bclhp", M, xc)
    w = torch.exp(torch.clamp_max(dA_cs[..., -1:] - dA_cs, 0.0)) * dt.permute(0, 3, 1, 2)       # (b, h, c, s)
    S = torch.einsum("bhcs,bcshp,bcsn->bchpn", w, xc, Bc)
    s = torch.zeros_like(S[:, 0])
    prev = []
    for c in range(nchunks):
        prev.append(s)
        s = torch.exp(dA_cs[:, :, c, -1])[..., None, None] * s + S[:, c]
    prev = torch.stack(prev, dim=1)                                    # (b, c, h, p, n)
    y = y + torch.einsum("bcln,bchpn,bhcl->bclhp", Cc, prev, torch.exp(dA_cs))
    y = y + D.float()[None, None, None, :, None] * xc
    return y.reshape(batch, L, nheads, headdim)[:, :seqlen].to(x.dtype)
