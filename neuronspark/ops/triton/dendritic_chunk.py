"""Per-chunk pieces of the dendritic layer's chunked scan: the write-strength transform with its in-chunk decay sums,
and the chunk-local overlap matrix of the read and write directions, each with its backward.

The membrane recurrence of a dendritic-layer head, ``S_t = a_t S_{t-1} + dt_t x_t B_t^T`` with ``a_t = exp(dt_t A)``, is
evaluated on chunks of ``cs`` tokens.  Two quantities are formed per chunk before the scan kernels run:

* the write strength and its decay sums::

      v_t    = f32(dt_raw_t) + dt_bias
      dt_t   = clamp(softplus(v_t), lo, hi)      softplus(v) = log(exp(v) + 1) for v <= 20, v above
      dA_cs  = sum_{s <= t in the chunk} dt_s A   (fp32)

  ``dt_t = 0`` for positions past the sequence end; both outputs (b, h, nc, cs) fp32 are written for the whole chunk.
* the overlap of the read direction C and the write direction B inside each chunk::

      CB[b, c, g, i, j] = sum_k bf16(C[b, c cs + i, g, k]) bf16(B[b, c cs + j, g, k])     (fp32 accumulation)

  zero where i or j is past the sequence end; the whole (cs, cs) block is formed.

Backward pieces::

    chunk_dt_cumsum_bwd: g_t = ddA_t A + ddt_out_t (zero where the clamp was active),
                         ddt_raw_t = g_t sigmoid(v_t) (v_t <= 20) or g_t,
                         per (b, c, h): sum_t ddA_t dt_t and sum_t ddt_raw_t (the dA / ddt_bias partials)
    chunk_cb_bwd:        out[c cs + m, k] = sum_i bf16(dout[b, c, g, i, m]) bf16(a[c cs + i, g, k]) + f32(residual)

``ddA`` is the gradient with respect to dt_t A at each position (the caller folds the in-chunk sums of dA_cs into it).
With ``a = C`` and ``dout = dCB`` the second line is dB; with ``a = B`` and ``dout = dCB^T`` (a transposed view) it is
dC.

References: :func:`chunk_dt_cumsum_ref`, :func:`chunk_cb_ref`, :func:`chunk_dt_cumsum_bwd_ref`, :func:`chunk_cb_bwd_ref`
(torch, float64 accumulation over the same bf16 operands).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["CUMSUM_FWD_CFG", "CUMSUM_BWD_CFG", "CB_CFG", "CB_BWD_CFG", "SMALL_CALL_CHUNKS",
           "chunk_dt_cumsum", "chunk_dt_cumsum_bwd", "chunk_cb", "chunk_cb_bwd",
           "chunk_dt_cumsum_ref", "chunk_dt_cumsum_bwd_ref", "chunk_cb_ref", "chunk_cb_bwd_ref"]

# ================================================================================================ fixed constants
#: calls with fewer (batch x nchunks) than this take the "small" launch configurations (more, smaller programs)
SMALL_CALL_CHUNKS = 16
#: write-strength forward / backward: {"large" / "small": (heads per program, warps)} -- one chunk of a head block per
#: program ((cs, heads) tile forward, (heads, cs) backward)
CUMSUM_FWD_CFG = {"large": (8, 4), "small": (4, 8)}
CUMSUM_BWD_CFG = {"large": (4, 4), "small": (4, 8)}
#: overlap forward: {"large" / "small": ((rows, cols, k) tile, warps, pipeline stages)}
CB_CFG = {"large": ((64, 32, 32), 2, 2), "small": ((32, 64, 64), 8, 3)}
#: overlap backward by dout's orientation: {"rows" (dout read as stored, the dB call) / "cols" (a transposed view, the
#: dC call): ((output rows, k columns, summed positions) tile, warps, pipeline stages)}
CB_BWD_CFG = {"rows": ((128, 64, 64), 4, 3), "cols": ((32, 128, 32), 2, 3)}
SOFTPLUS_LINEAR_ABOVE = 20.0


# ================================================================================================ kernels
@triton.jit
def _softplus_f32(v):
    return tl.where(v <= 20.0, tl.log(tl.exp(v) + 1.0), v)


@triton.jit
def _dt_cumsum_fwd_kernel(DT, BIAS, A, DT_OUT, DA_CS, T, H, NC, CS,
                          s_dt_b, s_dt_t, s_dt_h, lo, hi,
                          HAS_BIAS: tl.constexpr, SOFTPLUS: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr):
    # one program: batch b, chunk c, heads [h0, h0 + BH); the tile is time-major (BT, BH).  The program's base offsets
    # are 64-bit scalars, the offsets inside the tile 32-bit.
    b = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2) * BH + tl.arange(0, BH)
    t = tl.arange(0, BT)
    hm = h < H
    n_valid = tl.minimum(CS, T - c * CS)
    live = (t[:, None] < n_valid) & hm[None, :]
    v = tl.load(DT + (b * s_dt_b + c * CS * s_dt_t) + (t[:, None] * s_dt_t + h[None, :] * s_dt_h), mask=live,
                other=0.0).to(tl.float32)
    if HAS_BIAS:
        v = v + tl.load(BIAS + h, mask=hm, other=0.0).to(tl.float32)[None, :]
    if SOFTPLUS:
        v = _softplus_f32(v)
    v = tl.minimum(tl.maximum(v, lo), hi)
    dt = tl.where(live, v, 0.0)
    a = tl.load(A + h, mask=hm, other=0.0).to(tl.float32)
    da_cs = tl.cumsum(dt * a[None, :], axis=0)
    # outputs (b, h, nc, cs): position t of head h at ((b H + h) NC + c) CS + t
    base = (b * H * NC + c) * CS
    out = h[None, :] * (NC * CS) + t[:, None]
    keep = (t[:, None] < CS) & hm[None, :]
    tl.store(DT_OUT + base + out, dt, mask=keep)
    tl.store(DA_CS + base + out, da_cs, mask=keep)


@triton.jit
def _dt_cumsum_bwd_kernel(DDA, DDT_OUT, DT, BIAS, A, DDT, WA, WB, T, H, NC, CS,
                          s_ga_b, s_ga_h, s_ga_c, s_ga_t, s_go_b, s_go_h, s_go_c, s_go_t,
                          s_dt_b, s_dt_t, s_dt_h, s_ddt_b, s_ddt_t, s_ddt_h, lo, hi,
                          HAS_BIAS: tl.constexpr, SOFTPLUS: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr):
    # statement order of mamba_ssm's _chunk_cumsum_bwd_kernel (Apache-2.0, see NOTICE), with this file's arguments and
    # stored per-(batch, chunk) partials.  A head-major (BH, BT) tile: the program's bases are 64-bit scalar offsets,
    # every pointer tensor is formed before the first load, and g = ddA A + ddt_out is masked in its own layout before
    # the clamp mask (formed from dt) is applied, so the loads of ddt_out, A, dt and dt_bias are all in flight together
    b = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    DDA += b * s_ga_b + c.to(tl.int64) * s_ga_c
    DDT_OUT += b * s_go_b + c.to(tl.int64) * s_go_c
    DT += b * s_dt_b + c.to(tl.int64) * CS * s_dt_t
    DDT += b * s_ddt_b + c.to(tl.int64) * CS * s_ddt_t
    h = tl.program_id(2) * BH + tl.arange(0, BH)
    t = tl.arange(0, BT)
    p_ga = DDA + (h[:, None] * s_ga_h + t[None, :] * s_ga_t)
    p_go = DDT_OUT + (h[:, None] * s_go_h + t[None, :] * s_go_t)
    p_dt = DT + (h[:, None] * s_dt_h + t[None, :] * s_dt_t)
    p_ddt = DDT + (h[:, None] * s_ddt_h + t[None, :] * s_ddt_t)
    n_valid = tl.minimum(CS, T - c * CS)
    dda = tl.load(p_ga, mask=(h[:, None] < H) & (t[None, :] < n_valid), other=0.0).to(tl.float32)
    g = tl.load(p_go, mask=(h[:, None] < H) & (t[None, :] < n_valid), other=0.0).to(tl.float32)
    a = tl.load(A + h, mask=h < H, other=0.0).to(tl.float32)
    g = dda * a[:, None] + g
    v = tl.load(p_dt, mask=(h[:, None] < H) & (t[None, :] < n_valid), other=0.0).to(tl.float32)
    if HAS_BIAS:
        v += tl.load(BIAS + h, mask=h < H, other=0.0).to(tl.float32)[:, None]
    if SOFTPLUS:
        v_in = v
        v = tl.where(v <= 20.0, tl.log(tl.exp(v) + 1.0), v)
    clamped = (v < lo) | (v > hi)
    v = tl.minimum(tl.maximum(v, lo), hi)
    v = tl.where((h[:, None] < H) & (t[None, :] < n_valid), v, 0.0)
    g = tl.where((h[:, None] < H) & (t[None, :] < n_valid), g, 0.0)
    g = tl.where(clamped, 0.0, g)
    if SOFTPLUS:
        g = tl.where(v_in <= 20.0, g * tl.sigmoid(v_in), g)
    tl.store(p_ddt, g, mask=(h[:, None] < H) & (t[None, :] < n_valid))
    ws = (b * NC + c) * H + h
    tl.store(WA + ws, tl.sum(dda * v, axis=1), mask=h < H)
    tl.store(WB + ws, tl.sum(g, axis=1), mask=h < H)


@triton.jit
def _cb_fwd_kernel(C, B, CB, T, G, NC, CS, K,
                   s_c_b, s_c_t, s_c_g, s_b_b, s_b_t, s_b_g,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NSTAGES: tl.constexpr):
    # grid (cs / BM * cs / BN, NC * G, batch): one (BM, BN) tile of one chunk's overlap block
    n_tn = tl.cdiv(CS, BN)
    ti = tl.program_id(0) // n_tn
    tj = tl.program_id(0) % n_tn
    c = (tl.program_id(1) // G).to(tl.int64)
    g = (tl.program_id(1) % G).to(tl.int64)
    b = tl.program_id(2).to(tl.int64)
    n_valid = tl.minimum(CS, T - c * CS)
    i = ti * BM + tl.arange(0, BM)
    j = tj * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    c_row = C + b * s_c_b + g * s_c_g + (c * CS + i)[:, None].to(tl.int64) * s_c_t
    b_row = B + b * s_b_b + g * s_b_g + (c * CS + j)[None, :].to(tl.int64) * s_b_t
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in tl.range(0, K, BK, num_stages=NSTAGES):
        km = (k0 + kk) < K
        ct = tl.load(c_row + (k0 + kk)[None, :], mask=(i[:, None] < n_valid) & km[None, :], other=0.0)
        bt = tl.load(b_row + (k0 + kk)[:, None], mask=km[:, None] & (j[None, :] < n_valid), other=0.0)
        acc = tl.dot(ct.to(tl.bfloat16), bt.to(tl.bfloat16), acc)
    out = ((b * NC + c) * G + g) * CS * CS + i[:, None].to(tl.int64) * CS + j[None, :]
    tl.store(CB + out, acc, mask=(i[:, None] < CS) & (j[None, :] < CS))


@triton.jit
def _cb_bwd_kernel(AA, DOUT, RES, OUT, T, G, NC, CS, K,
                   s_a_b, s_a_t, s_a_g, s_d_b, s_d_c, s_d_g, s_d_i, s_d_m,
                   s_r_b, s_r_t, s_r_g, s_o_b, s_o_t, s_o_g,
                   HAS_RES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BS: tl.constexpr,
                   NSTAGES: tl.constexpr):
    # grid (cs / BM * K / BN, NC * G, batch): out rows c cs + m (m in the tile), columns k in the tile; int64 base
    # offsets per (batch, chunk, group), int32 offsets inside the chunk
    n_tn = tl.cdiv(K, BN)
    tm = tl.program_id(0) // n_tn
    tk = tl.program_id(0) % n_tn
    c = tl.program_id(1) // G
    g = tl.program_id(1) % G
    b = tl.program_id(2).to(tl.int64)
    t0 = (c * CS).to(tl.int64)
    n_valid = tl.minimum(CS, T - c * CS)
    m = tm * BM + tl.arange(0, BM)
    k = tk * BN + tl.arange(0, BN)
    ss = tl.arange(0, BS)
    km = k < K
    d_base = DOUT + b * s_d_b + c.to(tl.int64) * s_d_c + g.to(tl.int64) * s_d_g
    a_base = AA + b * s_a_b + t0 * s_a_t + g.to(tl.int64) * s_a_g
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for i0 in tl.range(0, CS, BS, num_stages=NSTAGES):
        i = i0 + ss
        im = i < n_valid
        # dout^T tile (BM over m, BS over i) and a tile (BS over positions i, BN over k)
        dt_ = tl.load(d_base + m[:, None] * s_d_m + i[None, :] * s_d_i, mask=(m[:, None] < CS) & im[None, :], other=0.0)
        at = tl.load(a_base + i[:, None] * s_a_t + k[None, :], mask=im[:, None] & km[None, :], other=0.0)
        acc = tl.dot(dt_.to(tl.bfloat16), at.to(tl.bfloat16), acc)
    keep = (m[:, None] < n_valid) & km[None, :]
    if HAS_RES:
        acc += tl.load(RES + b * s_r_b + t0 * s_r_t + g.to(tl.int64) * s_r_g + m[:, None] * s_r_t + k[None, :],
                       mask=keep, other=0.0).to(tl.float32)
    tl.store(OUT + b * s_o_b + t0 * s_o_t + g.to(tl.int64) * s_o_g + m[:, None] * s_o_t + k[None, :],
             acc.to(OUT.dtype.element_ty), mask=keep)


# ================================================================================================ launchers
def _p2(n: int) -> int:
    return int(triton.next_power_of_2(max(int(n), 1)))


def _size_class(batch: int, nchunks: int) -> str:
    return "small" if batch * nchunks < SMALL_CALL_CHUNKS else "large"


def _check_dt(dt, A, dt_bias, name):
    if dt.dim() != 3:
        raise RuntimeError(f"neuronspark.{name}: dt must be (batch, seqlen, nheads), got {tuple(dt.shape)}")
    nheads = int(dt.shape[2])
    if A.shape != (nheads,):
        raise RuntimeError(f"neuronspark.{name}: A must be ({nheads},), got {tuple(A.shape)}")
    if dt_bias is not None and dt_bias.shape != (nheads,):
        raise RuntimeError(f"neuronspark.{name}: dt_bias must be ({nheads},), got {tuple(dt_bias.shape)}")


def chunk_dt_cumsum(dt: torch.Tensor, A: torch.Tensor, chunk_size: int, dt_bias: Optional[torch.Tensor] = None,
                    softplus: bool = True, dt_limit: Tuple[float, float] = (0.0, float("inf"))
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """dt (batch, seqlen, nheads) any float dtype and strides, A / dt_bias (nheads,) -> (dA_cs, dt_out), each
    (batch, nheads, nchunks, chunk_size) fp32 contiguous (module docstring)."""
    require_cuda(dt, "chunk_dt_cumsum")
    _check_dt(dt, A, dt_bias, "chunk_dt_cumsum")
    batch, seqlen, nheads = (int(s) for s in dt.shape)
    cs = int(chunk_size)
    nc = -(-seqlen // cs)
    dt_out = torch.empty(batch, nheads, nc, cs, device=dt.device, dtype=torch.float32)
    da_cs = torch.empty_like(dt_out)
    bias = dt_bias if dt_bias is not None else A
    bh, nw = CUMSUM_FWD_CFG[_size_class(batch, nc)]
    grid = (batch, nc, triton.cdiv(nheads, bh))
    _dt_cumsum_fwd_kernel[grid](dt, bias, A, dt_out, da_cs, seqlen, nheads, nc, cs,
                                dt.stride(0), dt.stride(1), dt.stride(2), float(dt_limit[0]), float(dt_limit[1]),
                                HAS_BIAS=dt_bias is not None, SOFTPLUS=bool(softplus), BT=_p2(cs), BH=bh, num_warps=nw)
    return da_cs, dt_out


def chunk_dt_cumsum_bwd(ddA: torch.Tensor, ddt_out: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
                        dt_bias: Optional[torch.Tensor], ddt: torch.Tensor, ws_a: torch.Tensor,
                        ws_b: torch.Tensor, softplus: bool = True,
                        dt_limit: Tuple[float, float] = (0.0, float("inf"))) -> None:
    """Backward of :func:`chunk_dt_cumsum` given ddA (gradient w.r.t. dt A per position) and ddt_out, both
    (batch, nheads, nchunks, chunk_size) float, any strides: writes ddt (dt's shape; any strides) and the per-(batch,
    chunk, head) partials ws_a = sum_t ddA dt, ws_b = sum_t ddt (each (batch, nchunks, nheads) fp32 contiguous; every
    slot is written)."""
    require_cuda(dt, "chunk_dt_cumsum_bwd")
    _check_dt(dt, A, dt_bias, "chunk_dt_cumsum_bwd")
    batch, seqlen, nheads = (int(s) for s in dt.shape)
    _, _, nc, cs = (int(s) for s in ddA.shape)
    want = (batch, nheads, nc, cs)
    for name, tsr in (("ddA", ddA), ("ddt_out", ddt_out)):
        if tuple(tsr.shape) != want:
            raise RuntimeError(f"neuronspark.chunk_dt_cumsum_bwd: {name} must be {want}, got {tuple(tsr.shape)}")
    for name, tsr in (("ws_a", ws_a), ("ws_b", ws_b)):
        if tuple(tsr.shape) != (batch, nc, nheads) or not tsr.is_contiguous() or tsr.dtype != torch.float32:
            raise RuntimeError(f"neuronspark.chunk_dt_cumsum_bwd: {name} must be {(batch, nc, nheads)} fp32 contiguous")
    if ddt.shape != dt.shape:
        raise RuntimeError("neuronspark.chunk_dt_cumsum_bwd: ddt must have dt's shape")
    bias = dt_bias if dt_bias is not None else A
    bh, nw = CUMSUM_BWD_CFG[_size_class(batch, nc)]
    grid = (batch, nc, triton.cdiv(nheads, bh))
    _dt_cumsum_bwd_kernel[grid](ddA, ddt_out, dt, bias, A, ddt, ws_a, ws_b, seqlen, nheads, nc, cs,
                                *ddA.stride(), *ddt_out.stride(),
                                dt.stride(0), dt.stride(1), dt.stride(2), ddt.stride(0), ddt.stride(1), ddt.stride(2),
                                float(dt_limit[0]), float(dt_limit[1]),
                                HAS_BIAS=dt_bias is not None, SOFTPLUS=bool(softplus), BT=_p2(cs), BH=bh, num_warps=nw)


def _check_rows(name, x):
    if x.dim() != 4 or x.stride(-1) != 1 or x.dtype != torch.bfloat16:
        raise RuntimeError(f"neuronspark.{name}: operands must be (batch, seqlen, ngroups, k) bf16 with a unit "
                           f"last stride, got {tuple(x.shape)} {x.dtype} strides {tuple(x.stride())}")


def chunk_cb(C: torch.Tensor, B: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """C, B (batch, seqlen, ngroups, k) bf16 with a unit last stride -> CB (batch, nchunks, ngroups, cs, cs)
    fp32 contiguous (module docstring)."""
    require_cuda(C, "chunk_cb")
    _check_rows("chunk_cb", C)
    _check_rows("chunk_cb", B)
    if B.shape != C.shape:
        raise RuntimeError(f"neuronspark.chunk_cb: B {tuple(B.shape)} and C {tuple(C.shape)} differ")
    batch, seqlen, ngroups, k = (int(s) for s in C.shape)
    cs = int(chunk_size)
    nc = -(-seqlen // cs)
    cb = torch.empty(batch, nc, ngroups, cs, cs, device=C.device, dtype=torch.float32)
    (bm, bn, bk), nw, ns = CB_CFG[_size_class(batch, nc)]
    grid = (triton.cdiv(cs, bm) * triton.cdiv(cs, bn), nc * ngroups, batch)
    _cb_fwd_kernel[grid](C, B, cb, seqlen, ngroups, nc, cs, k,
                         C.stride(0), C.stride(1), C.stride(2), B.stride(0), B.stride(1), B.stride(2),
                         BM=bm, BN=bn, BK=bk, NSTAGES=ns, num_warps=nw)
    return cb


def chunk_cb_bwd(a: torch.Tensor, dout: torch.Tensor, residual: Optional[torch.Tensor] = None,
                 out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """a (batch, seqlen, ngroups, k) bf16, dout (batch, nchunks, ngroups, cs, cs) any float dtype and strides
    (a transposed view is read as it is), residual (a's shape, any float dtype, unit last stride) or None ->
    out = per chunk bf16(dout)^T bf16(a) + residual in out's dtype (out: a's shape, unit last stride; allocated like a
    when None)."""
    require_cuda(a, "chunk_cb_bwd")
    _check_rows("chunk_cb_bwd", a)
    batch, seqlen, ngroups, k = (int(s) for s in a.shape)
    if dout.dim() != 5 or dout.shape[0] != batch or dout.shape[2] != ngroups or dout.shape[3] != dout.shape[4]:
        raise RuntimeError(f"neuronspark.chunk_cb_bwd: dout must be (batch, nchunks, ngroups, cs, cs), got "
                           f"{tuple(dout.shape)}")
    nc, cs = int(dout.shape[1]), int(dout.shape[-1])
    if nc != -(-seqlen // cs):
        raise RuntimeError(f"neuronspark.chunk_cb_bwd: {nc} chunks of {cs} do not cover seqlen {seqlen}")
    if residual is not None and (residual.shape != a.shape or residual.stride(-1) != 1):
        raise RuntimeError("neuronspark.chunk_cb_bwd: residual must have a's shape and a unit last stride")
    if out is None:
        out = torch.empty_like(a)
    elif out.shape != a.shape or out.stride(-1) != 1:
        raise RuntimeError("neuronspark.chunk_cb_bwd: out must have a's shape and a unit last stride")
    res = residual if residual is not None else out
    (bm, bn, bs), nw, ns = CB_BWD_CFG["rows" if dout.stride(-1) == 1 else "cols"]
    grid = (triton.cdiv(cs, bm) * triton.cdiv(k, bn), nc * ngroups, batch)
    _cb_bwd_kernel[grid](a, dout, res, out, seqlen, ngroups, nc, cs, k,
                         a.stride(0), a.stride(1), a.stride(2),
                         dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4),
                         res.stride(0), res.stride(1), res.stride(2), out.stride(0), out.stride(1), out.stride(2),
                         HAS_RES=residual is not None, BM=bm, BN=bn, BS=bs, NSTAGES=ns, num_warps=nw)
    return out


# ================================================================================================ references
def _softplus_ref(v: torch.Tensor) -> torch.Tensor:
    return torch.where(v <= SOFTPLUS_LINEAR_ABOVE, torch.log(torch.exp(v) + 1.0), v)


def _chunked(x: torch.Tensor, cs: int) -> torch.Tensor:
    """(batch, seqlen, ...) -> (batch, nchunks, cs, ...), zero-padded past seqlen"""
    seqlen = x.shape[1]
    pad = (-seqlen) % cs
    if pad:
        x = torch.cat([x, x.new_zeros((x.shape[0], pad) + tuple(x.shape[2:]))], dim=1)
    return x.unflatten(1, (-1, cs))


def chunk_dt_cumsum_ref(dt, A, chunk_size, dt_bias=None, softplus=True, dt_limit=(0.0, float("inf"))):
    """torch reference of :func:`chunk_dt_cumsum` (fp32 elementwise as the kernel, the in-chunk sums in float64)."""
    batch, seqlen, nheads = dt.shape
    cs = int(chunk_size)
    v = dt.float() + (dt_bias.float() if dt_bias is not None else 0.0)
    s = _softplus_ref(v) if softplus else v
    s = s.clamp(min=float(dt_limit[0]), max=float(dt_limit[1]))
    d = _chunked(s, cs)                                                              # (b, nc, cs, h), 0 past seqlen
    da_cs = torch.cumsum(d.double() * A.double(), dim=2).float()
    return da_cs.permute(0, 3, 1, 2).contiguous(), d.permute(0, 3, 1, 2).contiguous()


def chunk_dt_cumsum_bwd_ref(ddA, ddt_out, dt, A, dt_bias=None, softplus=True, dt_limit=(0.0, float("inf"))):
    """torch reference of :func:`chunk_dt_cumsum_bwd`: (ddt (dt's shape, fp32), ws_a, ws_b (batch, nchunks, nheads))."""
    batch, seqlen, nheads = dt.shape
    cs = int(ddA.shape[-1])
    v = _chunked(dt.float() + (dt_bias.float() if dt_bias is not None else 0.0), cs)  # (b, nc, cs, h)
    live = _chunked(torch.ones_like(dt, dtype=torch.bool), cs)
    s = _softplus_ref(v) if softplus else v
    clamped = (s < float(dt_limit[0])) | (s > float(dt_limit[1]))
    s = torch.where(live, s.clamp(min=float(dt_limit[0]), max=float(dt_limit[1])), torch.zeros_like(s))
    dda = ddA.permute(0, 2, 3, 1).double() * live
    g = (ddt_out.permute(0, 2, 3, 1).double() * live + dda * A.double())
    g = torch.where(live & ~clamped, g, torch.zeros_like(g))
    if softplus:
        g = torch.where(v <= SOFTPLUS_LINEAR_ABOVE, g * torch.sigmoid(v.double()), g)
    ddt = g.flatten(1, 2)[:, :seqlen].float()
    return ddt, (dda * s.double()).sum(dim=2).float(), g.sum(dim=2).float()


def chunk_cb_ref(C, B, chunk_size):
    """torch reference of :func:`chunk_cb`: the bf16 operands multiplied in float64."""
    cs = int(chunk_size)
    cc = _chunked(C.to(torch.bfloat16).double(), cs)                                # (b, nc, cs, g, k)
    bb = _chunked(B.to(torch.bfloat16).double(), cs)
    return torch.einsum("bcigk,bcjgk->bcgij", cc, bb).float()


def chunk_cb_bwd_ref(a, dout, residual=None, out_dtype=None):
    """torch reference of :func:`chunk_cb_bwd`: bf16(dout)^T bf16(a) in float64, + residual, one rounding."""
    seqlen = a.shape[1]
    cs = int(dout.shape[-1])
    aa = _chunked(a.to(torch.bfloat16).double(), cs)                                # (b, nc, cs, g, k)
    dd = dout.to(torch.bfloat16).double()                                            # (b, nc, g, i, m)
    r = torch.einsum("bcgim,bcigk->bcmgk", dd, aa).flatten(1, 2)[:, :seqlen]
    if residual is not None:
        r = r + residual.double()
    return r.to(out_dtype or a.dtype)
