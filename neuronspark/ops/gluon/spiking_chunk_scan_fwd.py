"""spiking-layer chunk-scan forward of the training regime, written in Gluon (Triton 3.5.1's explicit-layout dialect)
for Blackwell GPUs: a persistent, warp-specialised kernel (TMA loads with mbarrier rings, asynchronous tcgen05 MMAs into
tensor memory).

``chunk_scan_fwd_gluon(cb, x, dt, dA_cumsum, C, states, D) -> out`` computes what mamba_ssm ``_chunk_scan_fwd`` computes
for the spiking-layer call (causal, D of shape (nheads,), no z, no seq_idx, ngroups 1):

    out[m] = sum_{k <= m} bf16(CB[m, k] exp(min(dA_m - dA_k, 0)) dt_k) x_k  +  exp(dA_m) (C_m . states_c^T)  +  D x_m

rounded once to bf16.  Inputs: cb (b, nc, 1, chunk, chunk) fp32 contiguous; x (b, l, h, P) bf16 with strides
(l h' P', h' P', P, 1) (the conv output view, row stride >= h P); dt / dA_cumsum (b, h, nc, chunk) fp32 contiguous;
C (b, l, 1, N) bf16 with unit last stride and batch stride l * row stride; states (b, nc, h, P, N) bf16 contiguous;
D (h,) fp32 contiguous.  Output (b, l, h, P) bf16 contiguous.

Work decomposition: items (b c m-tile, head pair) are split into G = #SM contiguous ranges of near-equal cost
(cost = W0 + the item's K steps) on the first call per (batch, nchunks, device) -- a documented first-call cache; that
call must be eager (it raises inside a CUDA-graph capture).  Per CTA: one load warp (the item's C row block and CB row
block, the NH heads' states tiles, the K-step x tiles), one stage warp (each head's dA / dt K rows and dA_m column into a
DD-deep ring), one MMA warp (the inter-chunk products C . states^T into their own TMEM accumulators, the intra-chunk
P x products), two compute partitions of NWC warps (one per head: the bf16 P tiles of each K step, the causal mask on
the diagonal K step), and the default 4-warp epilogue (out = acc_intra + exp(dA_m) acc_inter + x D, bf16 store).

Numerics: the per-element P expression and its bf16 rounding point are the stock kernel's; the decay exponent is
``ex2.approx.ftz`` of ``min(dA_m - dA_k, 0) log2 e`` (results below 2^-126 flush to zero); the inter-chunk term is
accumulated separately and added in the epilogue (fp32 order changes).  Checked with the per-operator tolerance rule
of :mod:`neuronspark.utils.testing` against the TMA chunk-scan forward
(:func:`neuronspark.ops.triton.spiking_scan.chunk_scan_fwd_tma`, the stock kernel's arithmetic).  Full chunks only
(seqlen % chunk == 0); otherwise it raises.

Launch constants: BM = BK = 64, NH 2 heads per item (one compute partition each), x ring R 2, accumulator buffers NB 2, NWC 4 warps per compute partition, work-split cost W0 3.0, registers 48 (load /
stage / MMA warps) and 128 (compute partitions), dA / dt ring depth DD 2, no K-step operand prefetch, P ring RP 3,
states ring SR 1, compute layout CW 4 columns per thread group; persistent grid = #SM.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_32x32b_reg_layout, tcgen05_mma, tcgen05_commit, mbarrier, tma,
    fence_async_shared)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from neuronspark.ops.gluon._ptx import _ex2ftz, _tc_fence
from neuronspark.ops.triton import num_sms, require_cuda, require_sm100

__all__ = ["BM", "BK", "NH", "R", "NB", "NWC", "W0", "REGS_W", "REGS_C", "DD", "RP", "SR", "CW",
           "chunk_scan_fwd_gluon", "chunk_scan_fwd_gluon_ref"]

# ------------------------------------------------------------------------------------------------ launch constants
BM = 64          # rows of an item's m tile
BK = 64          # K step (== BM)
NH = 2           # heads per item: one compute partition each
R = 2            # x ring slots (K steps in flight)
NB = 2           # accumulator buffers (items in flight between MMA and epilogue)
NWC = 4          # warps per compute partition
W0 = 3.0         # fixed per-item cost of the work split
REGS_W = 48      # registers of the load / stage / MMA warps
REGS_C = 128     # registers of the compute partitions
DD = 2           # dA / dt ring depth
RP = 3           # P ring slots
SR = 1           # states ring (items)
CW = 4           # compute layout columns per thread group

# (batch, nchunks, MT, HG, G, device index) -> int32 (G + 1,) item ranges per CTA: a documented first-call cache
_WORK_SPLIT: Dict[Tuple[int, int, int, int, int, int], torch.Tensor] = {}


# ------------------------------------------------------------------------------------------------ partitions
@gluon.jit
def _epilogue_partition(cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                        acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
                        cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
                        BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                        HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                        NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                        RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr):
    """Default partition (4 warps): per item, out = acc_intra + exp(dA_m) acc_inter + x D for its NH heads (dA_m and x
    read from global memory), bf16 store, then the accumulator buffer is released to the MMA warp."""
    AL: gl.constexpr = get_tmem_32x32b_reg_layout(BM, P, [BM, P], 4)
    pid = gl.program_id(0)
    i0 = gl.load(starts + pid)
    i1 = gl.load(starts + pid + 1)
    offs_n = gl.arange(0, P, layout=gl.SliceLayout(0, AL))
    offs_r = gl.arange(0, BM, layout=gl.SliceLayout(1, AL))
    for it in range(i0, i1):
        loc = it - i0
        a = loc % NB
        bcm = it // HG
        hg = it - bcm * HG
        m = bcm % MT
        bc = bcm // MT
        c = bc // batch
        b = bc - c * batch
        h0 = hg * NH
        rows = (b * seqlen + c * CS + m * BM + offs_r).to(gl.int64)
        xp = X + rows[:, None] * sxl + h0 * P + offs_n[None, :]
        xr0 = gl.load(xp)
        if NH > 1:
            xr1 = gl.load(xp + P)
        dap = DAP + ((b * H + h0) * nchunks + c) * CS + m * BM + offs_r
        da0 = gl.load(dap)
        if NH > 1:
            da1 = gl.load(dap + nchunks * CS)
        mbarrier.wait(done_bar.index(a), (loc // NB) & 1)
        _tc_fence(1, 4)
        for hh in gl.static_range(NH):
            if hh == 0:
                dav = da0
                xr = xr0.to(gl.float32)
            elif hh == 1:
                dav = da1
                xr = xr1.to(gl.float32)
            else:
                dav = gl.load(dap + hh * nchunks * CS)
                xr = gl.load(xp + hh * P).to(gl.float32)
            scale = gl.expand_dims(gl.exp(dav), 1)                            # (BM, 1) exp(dA_m)
            o = acc_i.index(a * NH + hh).load(AL) + acc_c.index(a * NH + hh).load(AL) * scale
            Dh = gl.load(D + h0 + hh)
            o = o + xr * Dh
            gl.store(OUT + rows[:, None] * sol + (h0 + hh) * P + offs_n[None, :], o.to(gl.bfloat16))
        _tc_fence(0, 4)
        gl.thread_barrier()
        mbarrier.arrive(acc_free.index(a))


@gluon.jit
def _load_partition(cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                    acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
                    cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
                    BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                    HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                    NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                    RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr):
    """Load warp: per (b c m) the C row block and the CB row block (resident while the item's heads run), per item the
    NH states tiles (ring of SR items), per K step the NH x tiles (ring of R steps)."""
    pid = gl.program_id(0)
    i0 = gl.load(starts + pid)
    i1 = gl.load(starts + pid + 1)
    prev = -1
    nres = 0
    tstep = 0
    for it in range(i0, i1):
        loc = it - i0
        bcm = it // HG
        hg = it - bcm * HG
        m = bcm % MT
        bc = bcm // MT
        c = bc // batch
        b = bc - c * batch
        h0 = hg * NH
        n_k = (m + 1) * (BM // BK)
        xrow0 = b * seqlen + c * CS
        if bcm != prev:
            mbarrier.wait(res_free, (nres & 1) ^ 1)
            mbarrier.expect(res_bar, BM * N * 2 + (CS // BK) * BM * BK * 4)
            tma.async_copy_global_to_shared(cd, [xrow0 + m * BM, 0], res_bar, c_s)
            cbrow = (b * nchunks + c) * CS + m * BM
            for j in gl.static_range(CS // BK):
                tma.async_copy_global_to_shared(cbd, [cbrow, j * BK], res_bar, cbr_s.index(j))
            nres += 1
            prev = bcm
        for hh in gl.static_range(NH):                                  # the item's NH states tiles (ring of SR items)
            u = loc * NH + hh
            mbarrier.wait(st_free.index(u % (SR * NH)), ((u // (SR * NH)) & 1) ^ 1)
            mbarrier.expect(st_bar.index(u % (SR * NH)), P * N * 2)
            tma.async_copy_global_to_shared(sd, [((b * nchunks + c) * H + h0 + hh) * P, 0], st_bar.index(u % (SR * NH)),
                                            st_s.index(u % (SR * NH)))
        for kt in range(n_k):
            ks = tstep % R
            mbarrier.wait(x_free.index(ks), ((tstep // R) & 1) ^ 1)
            mbarrier.expect(x_bar.index(ks), NH * BK * P * 2)
            for hh in gl.static_range(NH):
                tma.async_copy_global_to_shared(xd, [xrow0 + kt * BK, (h0 + hh) * P], x_bar.index(ks), x_s.index(ks * NH + hh))
            tstep += 1


@gluon.jit
def _stage_partition(cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                     acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
                     cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
                     BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                     HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                     NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                     RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr):
    # TMA the NH heads' dA / dt (1, BK) tiles and the dA_m row into the DD-deep ring, then the dA_m column into dam_s (1 warp)
    NKT: gl.constexpr = CS // BK
    LS: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 1], [1, 0])
    LC: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 1], [0, 1])
    pid = gl.program_id(0)
    i0 = gl.load(starts + pid)
    i1 = gl.load(starts + pid + 1)
    for it in range(i0, i1):
        loc = it - i0
        a = loc % DD
        bcm = it // HG
        hg = it - bcm * HG
        m = bcm % MT
        bc = bcm // MT
        c = bc // batch
        b = bc - c * batch
        h0 = hg * NH
        mbarrier.wait(dad_free.index(a), ((loc // DD) & 1) ^ 1)
        mbarrier.expect(dad_tma.index(a), NH * (2 * CS * 4 + BM * 4))
        for hh in gl.static_range(NH):
            r = (b * H + h0 + hh) * nchunks + c
            for k2 in gl.static_range(NKT):
                tma.async_copy_global_to_shared(dakd, [r * NKT + k2, 0], dad_tma.index(a), db_s.index((a * NH + hh) * NKT + k2))
                tma.async_copy_global_to_shared(dtkd, [r * NKT + k2, 0], dad_tma.index(a), dtt_s.index((a * NH + hh) * NKT + k2))
            tma.async_copy_global_to_shared(damd, [r, m * BM], dad_tma.index(a), dar_s.index(a * NH + hh))
        mbarrier.wait(dad_tma.index(a), (loc // DD) & 1)
        for hh in gl.static_range(NH):
            row = dar_s.index(a * NH + hh).load(LS)
            dam_s.index(a * NH + hh).store(gl.convert_layout(gl.permute(row, [1, 0]), LC))
        gl.thread_barrier()
        mbarrier.arrive(dad_bar.index(a))


@gluon.jit
def _mma_partition(cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                   acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
                   cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
                   BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                   HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                   NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                   RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr):
    """MMA warp: per item the NH inter-chunk products C . states^T (acc_c), then per K step the NH intra-chunk products
    P x (acc_i), each head's MMA gated by its own compute partition's P tile."""
    pid = gl.program_id(0)
    i0 = gl.load(starts + pid)
    i1 = gl.load(starts + pid + 1)
    prev = -1
    nres = 0
    tstep = 0
    for it in range(i0, i1):
        loc = it - i0
        a = loc % NB
        bcm = it // HG
        m = bcm % MT
        n_k = (m + 1) * (BM // BK)
        if bcm != prev:
            mbarrier.wait(res_bar, nres & 1)
            nres += 1
            prev = bcm
        mbarrier.wait(acc_free.index(a), ((loc // NB) & 1) ^ 1)
        _tc_fence(1, 1)
        for hh in gl.static_range(NH):
            u = loc * NH + hh
            mbarrier.wait(st_bar.index(u % (SR * NH)), (u // (SR * NH)) & 1)
            tcgen05_mma(c_s, st_s.index(u % (SR * NH)).permute((1, 0)), acc_c.index(a * NH + hh), use_acc=False,
                        mbarriers=[st_free.index(u % (SR * NH))])
        for kt in range(n_k):
            ks = tstep % R
            kp = tstep % RP
            mbarrier.wait(x_bar.index(ks), (tstep // R) & 1)
            for hh in gl.static_range(NH):
                mbarrier.wait(p_ready.index(kp * NH + hh), (tstep // RP) & 1)
                tcgen05_mma(p_s.index(kp * NH + hh), x_s.index(ks * NH + hh), acc_i.index(a * NH + hh), use_acc=kt > 0,
                            mbarriers=[p_free.index(kp * NH + hh), x_free.index(ks)], mbarrier_preds=[True, hh == NH - 1])
            tstep += 1
        tcgen05_commit(done_bar.index(a))
        last = (it + 1 == i1) | ((it + 1) // HG != bcm)
        if last:
            tcgen05_commit(res_free)


@gluon.jit
def _kstep(p_s, p_ready, p_free, tstep, cbv, am, bkv, dtv, offs_m, kt,
           BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, NWC: gl.constexpr, HH: gl.constexpr,
           MASK: gl.constexpr, CW: gl.constexpr = 8):
    """One K step of compute partition HH: P = bf16(cb exp(min(dA_m - dA_k, 0)) dt_k) (masked on the diagonal step)
    into the P ring slot of this step, then signal the MMA warp."""
    L: gl.constexpr = gl.BlockedLayout([BM // (4 * NWC), CW], [4, 8], [NWC, 1], [1, 0])
    ks = tstep % R
    mbarrier.wait(p_free.index(ks * NH + HH), ((tstep // R) & 1) ^ 1)
    p = cbv * _ex2ftz(gl.minimum(am - bkv, 0.0) * 1.4426950408889634)       # the stock expression: its bf16 rounding points of P
    p = p * dtv
    if MASK:
        offs_k = kt * BK + gl.arange(0, BK, layout=gl.SliceLayout(0, L))
        p = gl.where(offs_m[:, None] >= offs_k[None, :], p, 0.0)
    p_s.index(ks * NH + HH).store(p.to(gl.bfloat16))
    fence_async_shared()
    gl.thread_barrier()
    mbarrier.arrive(p_ready.index(ks * NH + HH))
    return tstep + 1


@gluon.jit
def _compute_body(cbr_s, p_s, db_s, dam_s, dtt_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                  BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, RP: gl.constexpr, CS: gl.constexpr,
                  HG: gl.constexpr, MT: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr, HH: gl.constexpr,
                  CW: gl.constexpr = 8):
    L: gl.constexpr = gl.BlockedLayout([BM // (4 * NWC), CW], [4, 8], [NWC, 1], [1, 0])
    NKT: gl.constexpr = CS // BK
    pid = gl.program_id(0)
    i0 = gl.load(starts + pid)
    i1 = gl.load(starts + pid + 1)
    prev = -1
    nres = 0
    tstep = 0
    for it in range(i0, i1):
        loc = it - i0
        a = loc % DD
        bcm = it // HG
        m = bcm % MT
        n_k = m + 1
        if bcm != prev:
            mbarrier.wait(res_bar, nres & 1)
            nres += 1
            prev = bcm
        mbarrier.wait(dad_bar.index(a), (loc // DD) & 1)
        i = a * NH + HH
        am = dam_s.index(i).load(L)                                          # (BM, 1) dA_m
        offs_m = m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, L))
        for kt in range(0, n_k - 1):
            cbv = cbr_s.index(kt).load(L)
            bkv = db_s.index(i * NKT + kt).load(L)
            dtv = dtt_s.index(i * NKT + kt).load(L)
            tstep = _kstep(p_s, p_ready, p_free, tstep, cbv, am, bkv, dtv, offs_m, kt, BM, BK, NH, RP, NWC, HH, False, CW)
        kl = n_k - 1
        cbv = cbr_s.index(kl).load(L)
        bkv = db_s.index(i * NKT + kl).load(L)
        dtv = dtt_s.index(i * NKT + kl).load(L)
        tstep = _kstep(p_s, p_ready, p_free, tstep, cbv, am, bkv, dtv, offs_m, kl, BM, BK, NH, RP, NWC, HH, True, CW)
        mbarrier.arrive(dad_free.index(a))
        last = (it + 1 == i1) | ((it + 1) // HG != bcm)
        if last:
            mbarrier.arrive(res_free)


@gluon.jit
def _compute0_partition(cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                        acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
                        cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
                        BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                        HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                        NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                        RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr):
    _compute_body(cbr_s, p_s, db_s, dam_s, dtt_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                  BM, BK, NH, RP, CS, HG, MT, NWC, DD, 0, CW)


@gluon.jit
def _compute1_partition(cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                        acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
                        cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
                        BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                        HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                        NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                        RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr):
    _compute_body(cbr_s, p_s, db_s, dam_s, dtt_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
                  BM, BK, NH, RP, CS, HG, MT, NWC, DD, 1, CW)


# ------------------------------------------------------------------------------------------------ kernel
@gluon.jit
def _chunk_scan_fwd_kernel(cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, starts, seqlen, nchunks, batch, sxl, sol,
                           H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BM: gl.constexpr,
                           BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, NB: gl.constexpr, NWC: gl.constexpr,
                           DD: gl.constexpr, REGS_W: gl.constexpr, REGS_C: gl.constexpr, RP: gl.constexpr = 2,
                           SR: gl.constexpr = 2, CW: gl.constexpr = 8):
    MT: gl.constexpr = CS // BM
    HG: gl.constexpr = H // NH
    NKT: gl.constexpr = CS // BK
    TL: gl.constexpr = TensorMemoryLayout(block=(BM, P), unpacked=True)
    PL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BK], gl.bfloat16)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    SL2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    cbr_s = gl.allocate_shared_memory(gl.float32, [NKT, BM, BK], cbd.layout)
    c_s = gl.allocate_shared_memory(gl.bfloat16, [BM, N], cd.layout)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [R * NH, BK, P], xd.layout)
    p_s = gl.allocate_shared_memory(gl.bfloat16, [RP * NH, BM, BK], PL)
    st_s = gl.allocate_shared_memory(gl.bfloat16, [SR * NH, P, N], sd.layout)
    db_s = gl.allocate_shared_memory(gl.float32, [DD * NH * NKT, 1, BK], dakd.layout)
    dam_s = gl.allocate_shared_memory(gl.float32, [DD * NH, BM, 1], SL2)
    dtt_s = gl.allocate_shared_memory(gl.float32, [DD * NH * NKT, 1, BK], dtkd.layout)
    dar_s = gl.allocate_shared_memory(gl.float32, [DD * NH, 1, BM], damd.layout)
    res_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
    res_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    dad_tma = gl.allocate_shared_memory(gl.int64, [DD, 1], BL)
    dad_bar = gl.allocate_shared_memory(gl.int64, [DD, 1], BL)
    dad_free = gl.allocate_shared_memory(gl.int64, [DD, 1], BL)
    done_bar = gl.allocate_shared_memory(gl.int64, [NB, 1], BL)
    acc_free = gl.allocate_shared_memory(gl.int64, [NB, 1], BL)
    p_ready = gl.allocate_shared_memory(gl.int64, [RP * NH, 1], BL)
    p_free = gl.allocate_shared_memory(gl.int64, [RP * NH, 1], BL)
    x_bar = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    x_free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    st_bar = gl.allocate_shared_memory(gl.int64, [SR * NH, 1], BL)
    st_free = gl.allocate_shared_memory(gl.int64, [SR * NH, 1], BL)
    mbarrier.init(res_bar, count=1)
    mbarrier.init(res_free, count=NH + 1)                      # the NH compute partitions + the MMA warp's commit
    for i in gl.static_range(DD):
        mbarrier.init(dad_tma.index(i), count=1)
        mbarrier.init(dad_bar.index(i), count=1)
        mbarrier.init(dad_free.index(i), count=NH)             # the NH compute partitions
    for i in gl.static_range(NB):
        mbarrier.init(done_bar.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=1)
    for i in gl.static_range(RP * NH):
        mbarrier.init(p_ready.index(i), count=1)
        mbarrier.init(p_free.index(i), count=1)
    for i in gl.static_range(R):
        mbarrier.init(x_bar.index(i), count=1)
        mbarrier.init(x_free.index(i), count=1)
    for i in gl.static_range(SR * NH):
        mbarrier.init(st_bar.index(i), count=1)
        mbarrier.init(st_free.index(i), count=1)
    fence_async_shared()
    acc_i = allocate_tensor_memory(gl.float32, [NB * NH, BM, P], TL)
    acc_c = allocate_tensor_memory(gl.float32, [NB * NH, BM, P], TL)
    gl.warp_specialize(
            (cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
             acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
             cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
             BM, BK, NH, R, CS, HG, MT, P, N, H, NB, NWC, DD, RP, SR, CW),
            _epilogue_partition,
            (cbr_s, p_s, db_s, dam_s, dar_s, res_bar, res_free, dad_bar, dad_free, p_ready, p_free, starts,
             acc_i, acc_c, c_s, st_s, x_s, st_bar, st_free, x_bar, x_free, done_bar, acc_free, dad_tma, dtt_s,
             cbd, xd, cd, sd, dakd, dtkd, damd, X, OUT, D, DAP, seqlen, sxl, sol, nchunks, batch,
             BM, BK, NH, R, CS, HG, MT, P, N, H, NB, NWC, DD, RP, SR, CW),
            [_load_partition, _stage_partition, _mma_partition, _compute0_partition, _compute1_partition],
            [1, 1, 1, NWC, NWC],
            [REGS_W, REGS_W, REGS_W, REGS_C, REGS_C])


# ------------------------------------------------------------------------------------------------ launcher
def _work_split(batch: int, nchunks: int, MT: int, HG: int, G: int, dev: torch.device) -> torch.Tensor:
    """(G + 1,) int32 boundaries of contiguous item ranges of near-equal cost (cost = W0 + the m tile's K steps)."""
    key = (batch, nchunks, MT, HG, G, dev.index)
    starts = _WORK_SPLIT.get(key)
    if starts is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("neuronspark.chunk_scan_fwd_gluon: the work split is built on the first (eager) call; first "
                               "call inside a CUDA-graph capture")
        n_items = batch * nchunks * MT * HG
        cost = [W0 + (((i // HG) % MT) + 1) for i in range(n_items)]
        tot = sum(cost)
        bounds, acc_c, p_ = [0], 0.0, 1
        for i, cc in enumerate(cost):
            acc_c += cc
            while p_ < G and acc_c >= tot * p_ / G:
                bounds.append(i + 1)
                p_ += 1
        while len(bounds) < G + 1:
            bounds.append(n_items)
        bounds[-1] = n_items
        starts = torch.tensor(bounds, dtype=torch.int32, device=dev)
        _WORK_SPLIT[key] = starts
    return starts


def chunk_scan_fwd_gluon(cb, x, dt, dA_cumsum, C, states, D):
    """The spiking-layer chunk-scan forward (see the module docstring); out (b, l, h, P) bf16."""
    require_cuda(x, "chunk_scan_fwd_gluon")
    require_sm100(x.device, "chunk_scan_fwd_gluon")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    _, _, ngroups, dstate = C.shape
    if (ngroups != 1 or seqlen % chunk or chunk % BM or BM != BK or BM != 64 or NH != 2 or nheads % NH
            or 2 * NB * NH * headdim > 512 or NWC not in (2, 4, 8) or BM % (4 * NWC) or DD < 2 or SR not in (1, 2)):
        raise RuntimeError(f"neuronspark.chunk_scan_fwd_gluon: ngroups {ngroups} seqlen {seqlen} chunk {chunk} (full "
                           f"chunks only) BM {BM} BK {BK} NH {NH} nheads {nheads} NB {NB} headdim {headdim}")
    if x.dtype != torch.bfloat16 or C.dtype != torch.bfloat16:
        raise RuntimeError("neuronspark.chunk_scan_fwd_gluon: bf16 x and C expected")
    if x.stride(3) != 1 or x.stride(2) != headdim or x.stride(0) != seqlen * x.stride(1):
        raise RuntimeError(f"neuronspark.chunk_scan_fwd_gluon: x strides {tuple(x.stride())}")
    if C.stride(3) != 1 or C.stride(0) != seqlen * C.stride(1):
        raise RuntimeError(f"neuronspark.chunk_scan_fwd_gluon: C strides {tuple(C.stride())}")
    if not cb.is_contiguous() or cb.dtype != torch.float32 or not states.is_contiguous() or states.dtype != torch.bfloat16:
        raise RuntimeError("neuronspark.chunk_scan_fwd_gluon: contiguous fp32 CB and bf16 states expected")
    if not dA_cumsum.is_contiguous() or not dt.is_contiguous() or D.dim() != 1 or D.stride(0) != 1:
        raise RuntimeError("neuronspark.chunk_scan_fwd_gluon: contiguous dt / dA_cumsum (b, h, c, l) and D (nheads,) "
                           "expected")
    dev = x.device
    MT, HG = chunk // BM, nheads // NH
    G = num_sms(dev)
    starts = _work_split(batch, nchunks, MT, HG, G, dev)
    out = torch.empty(batch, seqlen, nheads, headdim, device=dev, dtype=x.dtype)
    f32, b16 = gl.float32, gl.bfloat16
    rows = batch * nheads * nchunks
    cbd = TensorDescriptor(cb, [batch * nchunks * chunk, chunk], [chunk, 1], [BM, BK], gl.NVMMASharedLayout.get_default_for([BM, BK], f32))
    xd = TensorDescriptor(x, [batch * seqlen, nheads * headdim], [x.stride(1), 1], [BK, headdim],
                          gl.NVMMASharedLayout.get_default_for([BK, headdim], b16))
    cd = TensorDescriptor(C, [batch * seqlen, dstate], [C.stride(1), 1], [BM, dstate], gl.NVMMASharedLayout.get_default_for([BM, dstate], b16))
    sd = TensorDescriptor(states, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, dstate],
                          gl.NVMMASharedLayout.get_default_for([headdim, dstate], b16))
    dakd = TensorDescriptor(dA_cumsum, [rows * (chunk // BK), BK], [BK, 1], [1, BK], gl.NVMMASharedLayout.get_default_for([1, BK], f32))
    dtkd = TensorDescriptor(dt, [rows * (chunk // BK), BK], [BK, 1], [1, BK], gl.NVMMASharedLayout.get_default_for([1, BK], f32))
    damd = TensorDescriptor(dA_cumsum, [rows, chunk], [chunk, 1], [1, BM], gl.NVMMASharedLayout.get_default_for([1, BM], f32))
    with torch.cuda.device(dev.index):
        _chunk_scan_fwd_kernel[(G,)](
            cbd, xd, cd, sd, dakd, dtkd, damd, x, out, D, dA_cumsum, starts, seqlen, nchunks, batch, x.stride(1),
            out.stride(1),
            H=nheads, P=headdim, N=dstate, CS=chunk, BM=BM, BK=BK, NH=NH, R=R, NB=NB, NWC=NWC, DD=DD, REGS_W=REGS_W,
            REGS_C=REGS_C, RP=RP, SR=SR, CW=CW, num_warps=4)
    return out


def chunk_scan_fwd_gluon_ref(cb, x, dt, dA_cumsum, C, states, D):
    """Pure-torch reference: :func:`neuronspark.ops.triton.spiking_scan.chunk_scan_fwd_tma_ref` (the stock math with its
    bf16
    rounding point of the P operand reproduced, float64 sums; the decay without the flush below 2^-126)."""
    from neuronspark.ops.triton.spiking_scan import chunk_scan_fwd_tma_ref
    return chunk_scan_fwd_tma_ref(cb, x, dt, dA_cumsum, C, states, D)
