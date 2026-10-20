"""dendritic-layer SSD backward driver: the store-not-recompute backward of the SSD scan of the dendritic-layer call
(training).

``bwd_store_glue(dout, x, dt_raw, A, B, C, chunk_size, D, dt_bias, dx, ddt, dB, dC, dA_cumsum, dt_sp, CB,
states_passed) -> (dx, ddt, dA, dB, dC, dD, ddt_bias)`` is mamba_ssm ``_mamba_chunk_scan_combined_bwd`` for the
dendritic-layer call (z None, no initial / final states, no seq_idx, dt_softplus, dt_limit (0, inf), ngroups 1, D
(nheads,)) on the tensors the forward kept (dA_cumsum, softplus'd dt, CB, the fp32 passed states) instead of recomputing
them. Inputs: dout (b, l, h, P) bf16 (the gated norm's input gradient), x (b, l, h, P) bf16 / B, C (b, l, 1, N) bf16
views of the conv output, dt_raw (b, l, h) bf16 view of the projection, A / D / dt_bias (h,) fp32, dx / ddt / dB / dC
the output buffers (views of the projection and conv-input gradients, written in place), dA_cumsum / dt_sp (b, h, nc,
chunk) fp32, CB (b, nc, 1, chunk, chunk) fp32, states_passed (b, nc, h, P, N) fp32 (the state entering each chunk).
Returns dx, ddt, dB, dC (the given buffers), dA, dD, ddt_bias (h,) fp32. Whole chunks only (l % chunk == 0, the training
regime; the chunk-scan input backward is a whole-chunk Gluon kernel): otherwise it raises.

The sequence (each step's order and operands are fixed):
  1. ``gluon.dendritic_dstates.dstates_train``: dstates (bf16), the state-passing ddA workspace, the passed states
  (bf16); 2. ``gluon.dendritic_chunk_scan_dx.chunk_scan_dx_gluon``: dx, ddt (fp32 per position), dD; 3.
  ``dendritic_states.db_tma`` / ``dc_tma``: dB / dC head-split partials and the ddA_next / ddA_prev workspaces; 4.
  ``gluon.dendritic_dda.dda_gluon``: dCB head-split partials and the V grid of the intra-chunk ddA; 5. ``splits_sum``:
  dB, dC, dCB = sums over the head-split axis (one launch); 6. vendored ``_bmm_chunk_bwd`` twice: dB += C dCB, dC += B
  dCB^T (dot operands bf16: the fp32 dCB is rounded to bf16
     for the dot, fp32 accumulation + residual, bf16 store into the given buffers);
  7. ``dendritic_scan.dda_asm``: the intra-chunk ddA;
  8. ``ddA_combine``: ddA = intra + (cumsum(ddA_next) + reverse_cumsum(ddA_prev + the state-passing block sum at the
     chunk's last position)) per (b, h, c) row;
  9. ``chunk_cumsum_bwd``: the vendored ``_chunk_cumsum_bwd_kernel`` (deterministic workspaces) + ``_fin_cs_kernel``:
     ddt into the given buffer (bf16), dA and ddt_bias fp32 sums over (b, c).

Numerics: every elementwise operation is mamba_ssm's; the ORDER of the fp32 sums changes (register tree sums /
tl.cumsum instead of ATen's reduce and scan kernels, no zero padding added): fp32 reduction-order class, checked with
the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against :func:`bwd_store_glue_ref`.

Launch constants: ``_splits_kernel`` BLOCK 1024, 4 warps; ``_ddA_kernel`` BR = min(16, 4096 / pow2(chunk)), 4 warps;
``_fin_cs_kernel`` BRW 64, BH = min(64, pow2(nheads)), 4 warps; vendored kernels per
:mod:`neuronspark.ops.triton.mamba` (``CHUNK_CUMSUM_BWD_TRAIN``, ``BMM_CHUNK_BWD_TRAIN``), launched directly with
these fixed configurations (no autotuner).  The dx finish kernel lives with its only caller in
:mod:`neuronspark.ops.triton.dendritic_scan`.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton.mamba import (
    BMM_CHUNK_BWD_TRAIN,
    CHUNK_CUMSUM_BWD_TRAIN,
    _bmm_chunk_bwd,
    _chunk_cumsum_bwd_kernel,
)
from neuronspark.ops.gluon.dendritic_chunk_scan_dx import chunk_scan_dx_gluon
from neuronspark.ops.gluon.dendritic_dda import dda_gluon
from neuronspark.ops.triton.dendritic_scan import chunk_scan_dx_ref, dda_asm, dda_ref, softplus_ref
from neuronspark.ops.gluon.dendritic_dstates import dstates_train, dstates_train_ref
from neuronspark.ops.triton.dendritic_states import _chunked, db_tma, db_tma_ref, dc_tma, dc_tma_ref

__all__ = ["bwd_store_glue", "bwd_store_glue_ref", "splits_sum", "ddA_combine", "chunk_cumsum_bwd"]


# ------------------------------------------------------------------------------------------------ kernels
@triton.jit
def _splits_kernel(A0, O0, N0, IN0, S0, A1, O1, N1, IN1, S1, A2, O2, N2, IN2, S2, NB0, NB1,
                   BLOCK: tl.constexpr):
    """three segments: O[o * IN + i] = sum_{s < S} A[(o * S + s) * IN + i] (A (outer, S, inner) contiguous)."""
    pid = tl.program_id(0)
    if pid < NB0:
        A, O, N, IN, S, p = A0, O0, N0, IN0, S0, pid
    elif pid < NB0 + NB1:
        A, O, N, IN, S, p = A1, O1, N1, IN1, S1, pid - NB0
    else:
        A, O, N, IN, S, p = A2, O2, N2, IN2, S2, pid - NB0 - NB1
    e = (p * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < N
    o = e // IN
    i = e - o * IN
    base = o * S * IN + i
    acc = tl.load(A + base, mask=m, other=0.0)
    for s in range(1, S):
        acc = acc + tl.load(A + base + s * IN, mask=m, other=0.0)
    tl.store(O + e, acc, mask=m)


@triton.jit
def _ddA_kernel(WST, WDB, WDC, WSP, OUT, R, L, NVST, NTDB, NTDC, NVSP,
                st_r, st_m, db_r, db_l, dc_r, dc_l, sp_r,
                BR: tl.constexpr, BL: tl.constexpr, NTP: tl.constexpr, NSP: tl.constexpr):
    """per row r = (b, h, c) and l < L: OUT = st + (cumsum(nxt) + revcumsum(prev)): st = the intra-chunk workspace's
    valid blocks summed, nxt = the db workspace's written tiles (0 at l = 0), prev = the dC workspace's tiles with the
    state-passing block sum added at l = L - 1."""
    r = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    rm = r < R
    l = tl.arange(0, BL)
    lm = l < L
    m2 = rm[:, None] & lm[None, :]
    # st: ddAcs_stable workspace (R, nblk, L): the valid m blocks summed
    st = tl.zeros([BR, BL], dtype=tl.float32)
    for k in range(0, NVST):
        st += tl.load(WST + r[:, None] * st_r + k * st_m + l[None, :], mask=m2, other=0.0)
    # nxt: db workspace (R, L, tiles); slot l = 0 is never written (zero by construction)
    nxt = tl.zeros([BR, BL], dtype=tl.float32)
    prev = tl.zeros([BR, BL], dtype=tl.float32)
    for t in range(0, NTP):
        nxt += tl.load(WDB + r[:, None] * db_r + l[None, :] * db_l + t, mask=m2 & (l[None, :] >= 1) & (t < NTDB),
                       other=0.0)
        prev += tl.load(WDC + r[:, None] * dc_r + l[None, :] * dc_l + t, mask=m2 & (t < NTDC), other=0.0)
    k = tl.arange(0, NSP)
    spv = tl.sum(tl.load(WSP + r[:, None] * sp_r + k[None, :], mask=rm[:, None] & (k < NVSP)[None, :], other=0.0),
                 axis=1)
    prev = tl.where(l[None, :] == L - 1, prev + spv[:, None], prev)
    nxt_c = tl.cumsum(nxt, axis=1)
    prev_c = tl.cumsum(prev, axis=1, reverse=True)
    tl.store(OUT + r[:, None] * L + l[None, :], st + (nxt_c + prev_c), mask=m2)


@triton.jit
def _fin_cs_kernel(WA, WB, DA, DB, NR, H, HAS_B: tl.constexpr, BRW: tl.constexpr, BH: tl.constexpr):
    """DA[h] = sum_{rows < NR} WA[row, h] (WA (batch, nchunks, nheads) contiguous: NR = batch * nchunks); DB alike."""
    h = (tl.program_id(0) * BH + tl.arange(0, BH)).to(tl.int64)
    hm = h < H
    acc_a = tl.zeros([BRW, BH], dtype=tl.float32)
    acc_b = tl.zeros([BRW, BH], dtype=tl.float32)
    for r0 in range(0, NR, BRW):
        rr = (r0 + tl.arange(0, BRW)).to(tl.int64)
        m = (rr < NR)[:, None] & hm[None, :]
        acc_a += tl.load(WA + rr[:, None] * H + h[None, :], mask=m, other=0.0)
        if HAS_B:
            acc_b += tl.load(WB + rr[:, None] * H + h[None, :], mask=m, other=0.0)
    tl.store(DA + h, tl.sum(acc_a, axis=0), mask=hm)
    if HAS_B:
        tl.store(DB + h, tl.sum(acc_b, axis=0), mask=hm)


# ------------------------------------------------------------------------------------------------ launchers
def _p2(n):
    return int(triton.next_power_of_2(max(int(n), 1)))


def splits_sum(*parts_dims):
    """[(parts, outer_dims)]: per tensor A (outer..., S, inner...) contiguous -> A.sum(len(outer_dims)) (one launch)."""
    segs = []
    for a, k in parts_dims:
        if not a.is_contiguous() or a.dtype != torch.float32:
            raise RuntimeError("neuronspark.dendritic_glue.splits_sum: contiguous fp32 parts expected")
        shp = tuple(a.shape)
        outer = math.prod(shp[:k])
        S = shp[k]
        inner = math.prod(shp[k + 1:])
        out = torch.empty(shp[:k] + shp[k + 1:], device=a.device, dtype=torch.float32)
        segs.append((a, out, outer * inner, inner, S))
    while len(segs) < 3:
        segs.append((segs[0][0], segs[0][1], 0, 1, 1))
    BLOCK = 1024
    nbs = [triton.cdiv(n, BLOCK) for _, _, n, _, _ in segs]
    (a0, o0, n0, i0, s0), (a1, o1, n1, i1, s1), (a2, o2, n2, i2, s2) = segs
    _splits_kernel[(sum(nbs),)](a0, o0, n0, i0, s0, a1, o1, n1, i1, s1, a2, o2, n2, i2, s2, nbs[0], nbs[1],
                                BLOCK=BLOCK, num_warps=4)
    return [s[1] for s in segs[:len(parts_dims)]]


def ddA_combine(ws_st, nv_st, ws_db, nt_db, ws_dc, nt_dc, ws_sp, nv_sp):
    """ddA (b, h, c, L) = st + (cumsum(nxt) + revcumsum(prev)) (see _ddA_kernel); one launch."""
    b, h, c, nblk, L = ws_st.shape
    R = b * h * c
    for w in (ws_st, ws_db, ws_dc, ws_sp):
        if not w.is_contiguous():
            raise RuntimeError("neuronspark.dendritic_glue.ddA_combine: contiguous workspaces expected")
    out = torch.empty(b, h, c, L, device=ws_st.device, dtype=torch.float32)
    BL = _p2(L)
    BR = max(1, min(16, 4096 // BL))
    NTP = max(_p2(ws_db.shape[-1]), _p2(ws_dc.shape[-1]))
    _ddA_kernel[(triton.cdiv(R, BR),)](ws_st, ws_db, ws_dc, ws_sp, out, R, L, nv_st, nt_db, nt_dc, nv_sp,
                                       nblk * L, L, L * ws_db.shape[-1], ws_db.shape[-1], L * ws_dc.shape[-1],
                                       ws_dc.shape[-1], ws_sp.shape[-1],
                                       BR=BR, BL=BL, NTP=NTP, NSP=_p2(ws_sp.shape[-1]), num_warps=4)
    return out


def chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias, ddt):
    """mamba_ssm ``_chunk_cumsum_bwd`` (dt_softplus True, dt_limit (0, inf)) with per-(batch, chunk) workspaces and one
    finalize launch for dA / ddt_bias: (ddt (written into ``ddt`` when given), dA fp32 (h,), ddt_bias fp32 (h,)).
    The kernel stores every workspace slot (deterministic reduction), so the workspaces are not zero-filled."""
    batch, seqlen, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    if ddA.shape != (batch, nheads, nchunks, chunk_size) or ddt_out.shape != (batch, nheads, nchunks, chunk_size):
        raise RuntimeError("neuronspark.dendritic_glue.chunk_cumsum_bwd: ddA / ddt_out shapes")
    if A.shape != (nheads,) or dt_bias is None or dt_bias.shape != (nheads,):
        raise RuntimeError("neuronspark.dendritic_glue.chunk_cumsum_bwd: A / dt_bias must be (nheads,)")
    ws_b = torch.empty(batch, nchunks, nheads, device=dt.device, dtype=torch.float32)
    ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
    if ddt is not None:
        if ddt.shape != dt.shape:
            raise RuntimeError("neuronspark.dendritic_glue.chunk_cumsum_bwd: ddt must have dt's shape")
    else:
        ddt = torch.empty_like(dt)
    dA = torch.empty_like(A, dtype=torch.float32)
    ws_a = torch.empty(batch, nchunks, nheads, device=dt.device, dtype=torch.float32)
    meta, num_warps, num_stages = CHUNK_CUMSUM_BWD_TRAIN
    grid = (batch, nchunks, triton.cdiv(nheads, meta["BLOCK_SIZE_H"]))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_bwd_kernel[grid](
            ddA, ddt_out, dt, A, dt_bias, ddt, ws_a, ws_b,
            batch, seqlen, nheads, chunk_size,
            0.0, float("inf"),
            ddA.stride(0), ddA.stride(2), ddA.stride(1), ddA.stride(3),
            ddt_out.stride(0), ddt_out.stride(2), ddt_out.stride(1), ddt_out.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0),
            ddt.stride(0), ddt.stride(1), ddt.stride(2),
            ws_a.stride(0), ws_a.stride(1), ws_a.stride(-1),
            ws_b.stride(0), ws_b.stride(1), ws_b.stride(-1),
            True,
            HAS_DT_BIAS=True,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
            DETERMINISTIC_REDUCTION=True,
            **meta, num_warps=num_warps, num_stages=num_stages)
    BH = min(64, _p2(nheads))
    _fin_cs_kernel[(triton.cdiv(nheads, BH),)](ws_a, ws_b, dA, ddt_bias, batch * nchunks, nheads, HAS_B=True,
                                              BRW=64, BH=BH, num_warps=4)
    return ddt, dA, ddt_bias


# ------------------------------------------------------------------------------------------------ the backward body
def bwd_store_glue(dout, x, dt_raw, A, B, C, chunk_size, D, dt_bias, dx, ddt, dB, dC, dA_cumsum, dt_sp, CB,
                   states_passed):
    """The SSD backward of the dendritic-layer call on the kept forward tensors (see the module docstring)."""
    require_cuda(dout, "bwd_store_glue")
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    _, _, ngroups, dstate = B.shape
    if ngroups != 1:
        raise RuntimeError("neuronspark.bwd_store_glue: the dendritic-layer call has ngroups 1")
    headdim = int(x.shape[-1])
    dB_given, dC_given, ddt_given = dB, dC, ddt
    dt = dt_sp
    # dstates GEMM + both state-passing passes
    dstates, ws_sp, nv_sp, states = dstates_train(states_passed, dA_cumsum, dout, C, x.dtype)
    states = states.unflatten(-1, (headdim, dstate))
    dstates = dstates.unflatten(-1, (headdim, dstate))
    dx, ddt, dD = chunk_scan_dx_gluon(x, dt, dA_cumsum, B, CB, dout, dstates, D, dx)
    dB_p, ws_db, nt_db = db_tma(x, dt, dA_cumsum, dstates, B, ngroups)
    dC_p, ws_dc, nt_dc = dc_tma(states.to(x.dtype), dA_cumsum, dout, C, ngroups)
    dcb_p, ddA_v = dda_gluon(x, dt, dA_cumsum, dout, CB)
    dB, dC, dCB = splits_sum((dB_p, 2), (dC_p, 2), (dcb_p, 2))
    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given, config=BMM_CHUNK_BWD_TRAIN)
    _bmm_chunk_bwd(B, dCB.transpose(-1, -2), residual=dC, out=dC_given, config=BMM_CHUNK_BWD_TRAIN)
    # intra-chunk ddA (b, h, c, L) as a one-block workspace
    ws_st, nv_st = dda_asm(ddA_v).unsqueeze(3), 1
    ddA = ddA_combine(ws_st, nv_st, ws_db, nt_db, ws_dc, nt_dc, ws_sp, nv_sp)
    ddt_given, dA, ddt_bias = chunk_cumsum_bwd(ddA, ddt, dt_raw, A, dt_bias, ddt_given)
    return dx, ddt_given, dA, dB_given, dC_given, dD, ddt_bias


def bwd_store_glue_ref(dout, x, dt_raw, A, B, C, chunk_size, D, dt_bias, dx, ddt, dB, dC, dA_cumsum, dt_sp, CB,
                       states_passed):
    """Pure torch: the SSD backward of the dendritic-layer call on the kept forward tensors (the sequence of
    :func:`bwd_store_glue`), each step by its piece reference -- the stock mamba_ssm math in float64 between the
    kernels' rounding points, which are reproduced:

      1. ``dstates_train_ref``: dstates (bf16, from bf16(dout exp(dA)) C sums), the state-passing ddA block sums, the
         passed states (bf16);
      2. ``chunk_scan_dx_ref``: dx, ddt_out, dD (the dot operand bf16(CB[k, m] decay) of the dx kernel);
      3. ``db_tma_ref`` / ``dc_tma_ref``: dB / dC summed over heads, ddA_next / ddA_prev;
      4. ``dda_ref``: dCB and the intra-chunk ddA;
      5. ``_bmm_chunk_bwd`` (ssd_bmm.py ``_bmm_chunk_bwd_kernel``): dB += sum_l bf16(dCB[l, s]) C_l, dC += sum_s
         bf16(dCB[l, s]) B_s (bf16 dot operands, the fp32 head sums as the residual), rounded to B's / C's dtype;
      6. ``ddA_combine``: ddA = intra + cumsum(ddA_next) + reverse_cumsum(ddA_prev + the block sum at l = L - 1);
      7. ``_chunk_cumsum_bwd_kernel`` (ssd_chunk_state.py): ddt = (ddA A + ddt_out) sigmoid(dt_raw + dt_bias) where
         dt_raw + dt_bias <= 20 (rounded to dt_raw's dtype), dA = sum ddA softplus(dt_raw + dt_bias), ddt_bias = sum
         ddt (before the rounding).

    The output buffers (dx, ddt, dB, dC) are not used.  Returns (dx, ddt, dA, dB, dC, dD, ddt_bias) in the kernel's
    dtypes (dx / ddt / dB / dC in the input dtypes, dA / dD / ddt_bias fp32)."""
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dA_cumsum.shape
    dstate = int(B.shape[-1])
    L = nchunks * chunk
    # 1. dstates + both state-passing passes
    dstates, ws_sp, nv_sp, states = dstates_train_ref(states_passed, dA_cumsum, dout, C, x.dtype)
    dstates = dstates.unflatten(-1, (headdim, dstate))
    states = states.unflatten(-1, (headdim, dstate))
    # 2. dx / ddt_out / dD
    dx_r, ddt_out, dD = chunk_scan_dx_ref(x, dt_sp, dA_cumsum, B, CB, dout, dstates, D, None)
    # 3. dB / dC head sums and the ddA_next / ddA_prev workspaces
    dB_h, ddA_next = db_tma_ref(x, dt_sp, dA_cumsum, dstates, B, 1)
    dC_h, ddA_prev = dc_tma_ref(states, dA_cumsum, dout, C, 1)
    # 4. dCB and the intra-chunk ddA
    dCB, ddA_intra = dda_ref(x, dt_sp, dA_cumsum, dout, CB)
    # 5. dB += C^T dCB, dC += dCB B (bf16 dot operands, fp32 residual)
    dcb = dCB[:, :, 0].to(B.dtype).double()                                            # (b, c, l, s)
    Bc = _chunked(B[:, :, 0], nchunks, chunk)                                          # (b, c, l, n)
    Cc = _chunked(C[:, :, 0], nchunks, chunk)
    dB_r = torch.einsum("bcls,bcln->bcsn", dcb, Cc).reshape(batch, L, 1, dstate)[:, :seqlen] + dB_h.double()
    dC_r = torch.einsum("bcls,bcsn->bcln", dcb, Bc).reshape(batch, L, 1, dstate)[:, :seqlen] + dC_h.double()
    # 6. ddA per (b, h, c, l)
    prev = ddA_prev.double().clone()
    prev[..., chunk - 1] += ws_sp[..., :nv_sp].double().sum(-1)
    ddA = (ddA_intra.double() + torch.cumsum(ddA_next.double(), dim=-1)
           + torch.flip(torch.cumsum(torch.flip(prev, dims=(-1,)), dim=-1), dims=(-1,)))
    # 7. the chunk-cumsum backward (positions < seqlen)

    def per_step(t):                                                                   # (b, h, c, chunk) -> (b, l, h)
        return t.reshape(batch, nheads, L)[:, :, :seqlen].permute(0, 2, 1)

    ddA_l = per_step(ddA)
    dt_pre = dt_raw.float() + dt_bias.float()                                          # (b, l, h) fp32
    dt_re = torch.clamp_min(softplus_ref(dt_pre), 0.0).double()
    ddt_l = ddA_l * A.double() + per_step(ddt_out.double())
    ddt_l = torch.where(dt_pre <= 20.0, ddt_l * torch.sigmoid(dt_pre.double()), ddt_l)
    dA_r = (ddA_l * dt_re).sum((0, 1))
    ddt_bias_r = ddt_l.sum((0, 1))
    return (dx_r, ddt_l.to(dt_raw.dtype), dA_r.float(), dB_r.to(B.dtype), dC_r.to(C.dtype), dD.float(),
            ddt_bias_r.float())
