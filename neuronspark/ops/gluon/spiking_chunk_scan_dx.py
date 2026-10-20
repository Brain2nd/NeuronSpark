"""spiking-layer chunk-scan backward to the inputs (dx, ddt, dD) of the training regime, written in Gluon (Triton
3.5.1's explicit-layout dialect) for Blackwell GPUs: a persistent, warp-specialised kernel (TMA loads with mbarrier
rings, asynchronous tcgen05 MMAs into tensor memory), the transposed counterpart of
:mod:`neuronspark.ops.gluon.spiking_chunk_scan_fwd`.

``chunk_scan_dx_gluon(x, dt, dA_cumsum, B, CB, dout, dstates, D) -> (dx, ddt, dD)`` computes what mamba_ssm
``_chunk_scan_chunk_state_bwd_dx`` computes for the spiking-layer call (D of shape (nheads,), no seq_idx, ngroups 1):

    acc[m] = sum_{k >= m} bf16(CB[k, m] exp(min(dA_k - dA_m, 0))) dout_k  +  exp(min(dA_last - dA_m, 0)) (B_m . dstates_c^T)
    dx[m]  = acc[m] dt_m + D dout_m          (rounded once to bf16)
    ddt[m] = sum_p acc[m, p] x[m, p]         dD = sum_{b, l, p} dout x

Inputs: x / dout (b, l, h, P) bf16 with strides (l h' P', h' P', P, 1); dt / dA_cumsum (b, h, nc, chunk) fp32
contiguous; B (b, l, 1, N) bf16 with unit last stride and batch stride l * row stride; CB (b, nc, 1, chunk, chunk) fp32
contiguous; dstates (b, nc, h, P, N) bf16 contiguous; D (h,) fp32 contiguous.  Outputs: dx (b, l, h, P) bf16 contiguous,
ddt (b, h, nc, chunk) fp32 contiguous, dD (h,) in D's dtype.

Work decomposition: items (b c m-tile, head pair) as in the forward, split into G = #SM contiguous ranges of near-equal
cost (cost = W0 + the item's K steps, chunk / BK - m) on the first call per (batch, nchunks, device) -- a documented
first-call cache; that call must be eager (it raises inside a CUDA-graph capture).  Per CTA: one load warp (per b c m the
B row block and the CB column block m, i.e. the (BK, BM) tiles CB[k, m]; per item the NH heads' dstates tiles; per K
step the NH dout tiles), one stage warp (each head's dA_cumsum row of the chunk as (1, BK) tiles, and the same tiles
transposed to (BK, 1) columns, into a DD-deep ring), one MMA warp (the NH products B . dstates^T into their own TMEM
accumulators, then per K step K^T dout with K^T the (BK, BM) tile read as the MMA's MN-major A operand), one compute
partition of NWC warps (both heads' bf16 tiles K^T[k, m] of each K step from one read of the CB tile, the causal mask on the
diagonal step), and
two epilogue partitions of NWE warps, one per head (the default partition and a worker: acc = acc_intra +
exp(min(dA_last - dA_m, 0)) acc_inter; dx, the ddt row sums and the dD row partials sum_p dout x in EH column slices of
the accumulator; the accumulator buffer released before the stores).  dD is one reduction of the row partials after the kernel.

Numerics: the per-element K expression and its bf16 rounding point are the accepted TMA kernel's
(the Triton TMA kernel it replaced: ``ex2.approx.ftz`` of ``min(dA_k - dA_m, 0) log2 e``, results below 2^-126 flushed to
zero); the inter-chunk term is accumulated separately and added in the epilogue, and the ddt / dD sums run in this
kernel's order (fp32 order changes).  Checked with the per-operator tolerance rule of :mod:`neuronspark.utils.testing`
against the float64 reference :func:`neuronspark.ops.triton.spiking_scan.chunk_scan_dx_ref`.  Full chunks only (seqlen %
chunk == 0); otherwise it raises.

Launch constants: BM = BK = 64, NH 2 heads per item, dout ring R 2, accumulator buffers NB 2, NWC 8 compute warps, work-split
cost W0 3.0, registers 48 (load / stage / MMA warps), 112 (compute) and 96 (the worker epilogue), dA ring depth DD 2, K^T ring
RP 2, dstates ring SR 1, compute layout CW 4 columns per thread group, two epilogue partitions of NWE 4 warps working in EH 2
column slices of the accumulator; persistent grid = #SM.
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

__all__ = ["BM", "BK", "NH", "R", "NB", "NWC", "W0", "REGS_W", "REGS_C", "REGS_E", "DD", "RP", "SR", "CW", "NWE", "EH",
           "chunk_scan_dx_gluon", "chunk_scan_dx_gluon_ref"]

# ------------------------------------------------------------------------------------------------ launch constants
BM = 64          # rows of an item's m tile
BK = 64          # K step (== BM)
NH = 2           # heads per item: one compute partition each
R = 2            # dout ring slots (K steps in flight)
NB = 2           # accumulator buffers (items in flight between MMA and epilogue)
NWC = 8          # warps of the compute partition (both heads)
W0 = 3.0         # fixed per-item cost of the work split
REGS_W = 48      # registers of the load / stage / MMA warps
REGS_C = 112     # registers of the compute partition
DD = 2           # dA ring depth (items)
RP = 2           # K^T ring slots
SR = 1           # dstates ring (items)
CW = 4           # compute layout columns per thread group
NWE = 4          # warps per epilogue partition (one partition per head: the default one and a worker)
REGS_E = 96      # registers of the second epilogue partition
EH = 2           # epilogue column slices of the (BM, P) accumulator (registers of one slice at a time)

# (batch, nchunks, MT, HG, G, device index) -> int32 (G + 1,) item ranges per CTA: a documented first-call cache
_WORK_SPLIT: Dict[Tuple[int, int, int, int, int, int], torch.Tensor] = {}


# ------------------------------------------------------------------------------------------------ partitions
@gluon.jit
def _epi_rows(it, i1, X, DOUT, DT, DA, D, seqlen, nchunks, batch, sxl, sdl, offs_r, offs_n,
              BM: gl.constexpr, NH: gl.constexpr, CS: gl.constexpr, HG: gl.constexpr, MT: gl.constexpr,
              P: gl.constexpr, H: gl.constexpr, HE: gl.constexpr):
    """Item ``it``'s row operands for head HE (the first column slice of dout / x; dA_m, dA_last, dt_m, D_h); masked past
    the CTA's last item."""
    ok = it < i1
    bcm = it // HG
    hg = it - bcm * HG
    m = bcm % MT
    bc = bcm // MT
    c = bc // batch
    b = bc - c * batch
    h = hg * NH + HE
    rows = (b * seqlen + c * CS + m * BM + offs_r).to(gl.int64)
    r = (b * H + h) * nchunks + c                           # row of (b, h, c) in the (b h c, chunk) views
    lrow = m * BM + offs_r
    rm = (offs_r < BM) & ok
    dov = gl.load(DOUT + rows[:, None] * sdl + h * P + offs_n[None, :], mask=rm[:, None], other=0.0)
    xv = gl.load(X + rows[:, None] * sxl + h * P + offs_n[None, :], mask=rm[:, None], other=0.0)
    dav = gl.load(DA + r * CS + lrow, mask=rm, other=0.0)
    dal = gl.load(DA + r * CS + (CS - 1) + offs_r * 0, mask=rm, other=0.0)
    dtv = gl.load(DT + r * CS + lrow, mask=rm, other=0.0)
    dh = gl.load(D + h + offs_r * 0, mask=rm, other=0.0)
    return dov, xv, dav, dal, dtv, dh


@gluon.jit
def _epilogue_body(acc_i, acc_c, done_bar, acc_free, starts, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch,
                   sxl, sdl, sdxl, BM: gl.constexpr, NH: gl.constexpr, CS: gl.constexpr, HG: gl.constexpr,
                   MT: gl.constexpr, P: gl.constexpr, H: gl.constexpr, NB: gl.constexpr, HE: gl.constexpr,
                   NWE: gl.constexpr, EH: gl.constexpr):
    """Epilogue of head HE of every item (NWE warps), in EH column slices of the accumulator (the registers of one slice at
    a time): acc = acc_intra + exp(min(dA_last - dA_m, 0)) acc_inter; dx = acc dt_m + D dout_m (bf16 store), ddt[m] =
    sum_p acc x_m and the dD row partial sum_p dout_m x_m (row sums only: no reduction across the rows); the accumulator
    buffer is released after the last slice's tensor-memory loads.  (Loading the next item's row operands during this one
    measured slower: gdx_probe4, 443-492 vs 380 us.)"""
    PS: gl.constexpr = P // EH
    AL: gl.constexpr = get_tmem_32x32b_reg_layout(BM, PS, [BM, PS], NWE)
    pid = gl.program_id(0)
    i0 = gl.load(starts + pid)
    i1 = gl.load(starts + pid + 1)
    offs_n = gl.arange(0, PS, layout=gl.SliceLayout(0, AL))
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
        h = hg * NH + HE
        rows = (b * seqlen + c * CS + m * BM + offs_r).to(gl.int64)
        r = (b * H + h) * nchunks + c
        lrow = m * BM + offs_r
        cdo, cx, dav, dal, dtv, dh = _epi_rows(it, i1, X, DOUT, DT, DA, D, seqlen, nchunks, batch, sxl, sdl, offs_r, offs_n,
                                               BM, NH, CS, HG, MT, P, H, HE)
        scale = gl.expand_dims(gl.exp(gl.minimum(dal - dav, 0.0)), 1)                     # (BM, 1)
        dtc = gl.expand_dims(dtv, 1)
        dhc = gl.expand_dims(dh, 1)
        mbarrier.wait(done_bar.index(a), (loc // NB) & 1)
        _tc_fence(1, NWE)
        ddt = gl.zeros([BM], gl.float32, layout=gl.SliceLayout(1, AL))
        ddw = gl.zeros([BM], gl.float32, layout=gl.SliceLayout(1, AL))
        for e in gl.static_range(EH):
            acc = (acc_i.index(a * NH + HE).slice(e * PS, PS).load(AL)
                   + acc_c.index(a * NH + HE).slice(e * PS, PS).load(AL) * scale)
            if e == EH - 1:
                _tc_fence(0, NWE)
                gl.thread_barrier()
                mbarrier.arrive(acc_free.index(a))
            dof = cdo.to(gl.float32)
            xf = cx.to(gl.float32)
            if e + 1 < EH:                                              # the next slice's row operands ahead
                cdo = gl.load(DOUT + rows[:, None] * sdl + h * P + (e + 1) * PS + offs_n[None, :])
                cx = gl.load(X + rows[:, None] * sxl + h * P + (e + 1) * PS + offs_n[None, :])
            dx = acc * dtc + dof * dhc
            gl.store(DX + rows[:, None] * sdxl + h * P + e * PS + offs_n[None, :], dx.to(gl.bfloat16))
            ddt += gl.sum(acc * xf, axis=1)
            ddw += gl.sum(dof * xf, axis=1)
        gl.store(DDT + r * CS + lrow, ddt)
        gl.store(DDW + r * CS + lrow, ddw)


@gluon.jit
def _epilogue_partition(cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
                        acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
                        cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
                        BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                        HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                        NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                        RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr, NWE: gl.constexpr, EH: gl.constexpr):
    """Default partition (NWE warps): the epilogue of each item's first head."""
    _epilogue_body(acc_i, acc_c, done_bar, acc_free, starts, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch,
                   sxl, sdl, sdxl, BM, NH, CS, HG, MT, P, H, NB, 0, NWE, EH)


@gluon.jit
def _epilogue1_partition(cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
                        acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
                        cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
                        BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                        HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                        NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                        RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr, NWE: gl.constexpr, EH: gl.constexpr):
    """Worker partition (NWE warps): the epilogue of each item's second head."""
    _epilogue_body(acc_i, acc_c, done_bar, acc_free, starts, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch,
                   sxl, sdl, sdxl, BM, NH, CS, HG, MT, P, H, NB, 1, NWE, EH)


@gluon.jit
def _load_partition(cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
                    acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
                    cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
                    BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                    HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                    NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                    RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr, NWE: gl.constexpr, EH: gl.constexpr):
    """Load warp: per (b c m) the B row block and the CB column block m (all chunk / BK tiles CB[k, m]; resident while the
    item's heads run), per item the NH dstates tiles (ring of SR items), per K step k >= m the NH dout tiles (ring of R)."""
    NKT: gl.constexpr = CS // BK
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
        xrow0 = b * seqlen + c * CS
        if bcm != prev:
            mbarrier.wait(res_free, (nres & 1) ^ 1)
            mbarrier.expect(res_bar, BM * N * 2 + NKT * BK * BM * 4)
            tma.async_copy_global_to_shared(bd, [xrow0 + m * BM, 0], res_bar, b_s)
            cbrow = (b * nchunks + c) * CS
            for j in gl.static_range(NKT):
                tma.async_copy_global_to_shared(cbd, [cbrow + j * BK, m * BM], res_bar, cbr_s.index(j))
            nres += 1
            prev = bcm
        for hh in gl.static_range(NH):                                  # the item's NH dstates tiles (ring of SR items)
            u = loc * NH + hh
            mbarrier.wait(ds_free.index(u % (SR * NH)), ((u // (SR * NH)) & 1) ^ 1)
            mbarrier.expect(ds_bar.index(u % (SR * NH)), P * N * 2)
            tma.async_copy_global_to_shared(dsd, [((b * nchunks + c) * H + h0 + hh) * P, 0], ds_bar.index(u % (SR * NH)),
                                            ds_s.index(u % (SR * NH)))
        for kt in range(m, NKT):
            ks = tstep % R
            mbarrier.wait(do_free.index(ks), ((tstep // R) & 1) ^ 1)
            mbarrier.expect(do_bar.index(ks), NH * BK * P * 2)
            for hh in gl.static_range(NH):
                tma.async_copy_global_to_shared(dod, [xrow0 + kt * BK, (h0 + hh) * P], do_bar.index(ks),
                                                do_s.index(ks * NH + hh))
            tstep += 1


@gluon.jit
def _stage_partition(cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
                     acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
                     cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
                     BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                     HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                     NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                     RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr, NWE: gl.constexpr, EH: gl.constexpr):
    """Stage warp: TMA the NH heads' dA_cumsum chunk rows as (1, BK) tiles into the DD-deep ring (dA_m rows), then the
    tiles k >= m transposed to (BK, 1) columns (dA_k)."""
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
        mbarrier.expect(dad_tma.index(a), NH * CS * 4)
        for hh in gl.static_range(NH):
            r = (b * H + h0 + hh) * nchunks + c
            for k2 in gl.static_range(NKT):
                tma.async_copy_global_to_shared(dakd, [r * NKT + k2, 0], dad_tma.index(a), dar_s.index((a * NH + hh) * NKT + k2))
        mbarrier.wait(dad_tma.index(a), (loc // DD) & 1)
        for hh in gl.static_range(NH):
            for k2 in gl.static_range(NKT):
                if k2 >= m:
                    j = (a * NH + hh) * NKT + k2
                    row = dar_s.index(j).load(LS)
                    dac_s.index(j).store(gl.convert_layout(gl.permute(row, [1, 0]), LC))
        gl.thread_barrier()
        mbarrier.arrive(dad_bar.index(a))


@gluon.jit
def _mma_partition(cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
                   acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
                   cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
                   BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                   HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                   NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                   RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr, NWE: gl.constexpr, EH: gl.constexpr):
    """MMA warp: per item the NH inter-chunk products B . dstates^T (acc_c), then per K step the NH intra-chunk products
    K dout (acc_i) with K the transposed view of the (BK, BM) K^T tile, each head's MMA gated by its compute partition."""
    NKT: gl.constexpr = CS // BK
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
        if bcm != prev:
            mbarrier.wait(res_bar, nres & 1)
            nres += 1
            prev = bcm
        mbarrier.wait(acc_free.index(a), ((loc // NB) & 1) ^ 1)
        _tc_fence(1, 1)
        for hh in gl.static_range(NH):
            u = loc * NH + hh
            mbarrier.wait(ds_bar.index(u % (SR * NH)), (u // (SR * NH)) & 1)
            tcgen05_mma(b_s, ds_s.index(u % (SR * NH)).permute((1, 0)), acc_c.index(a * NH + hh), use_acc=False,
                        mbarriers=[ds_free.index(u % (SR * NH))])
        for kt in range(m, NKT):
            ks = tstep % R
            kp = tstep % RP
            mbarrier.wait(do_bar.index(ks), (tstep // R) & 1)
            for hh in gl.static_range(NH):
                mbarrier.wait(p_ready.index(kp * NH + hh), (tstep // RP) & 1)
                tcgen05_mma(p_s.index(kp * NH + hh).permute((1, 0)), do_s.index(ks * NH + hh), acc_i.index(a * NH + hh),
                            use_acc=kt > m, mbarriers=[p_free.index(kp * NH + hh), do_free.index(ks)],
                            mbarrier_preds=[True, hh == NH - 1])
            tstep += 1
        tcgen05_commit(done_bar.index(a))
        last = (it + 1 == i1) | ((it + 1) // HG != bcm)
        if last:
            tcgen05_commit(res_free)


@gluon.jit
def _kstep(p_s, p_ready, p_free, tstep, cbv, am0, bk0, am1, bk1, offs_m, kt,
           BK: gl.constexpr, NH: gl.constexpr, RP: gl.constexpr, NWC: gl.constexpr, MASK: gl.constexpr,
           CW: gl.constexpr):
    """One K step of the compute partition, both heads of the item from one read of the CB tile: K^T_h = bf16(CB[k, m]
    exp(min(dA_hk - dA_hm, 0))) (masked to k >= m on the diagonal step) into the ring slot of this step, then signal the
    MMA warp (one arrival per head's slot)."""
    L: gl.constexpr = gl.BlockedLayout([BK // 16, CW], [4, 8], [4, NWC // 4], [1, 0])
    ks = tstep % RP
    mbarrier.wait(p_free.index(ks * NH), ((tstep // RP) & 1) ^ 1)
    mbarrier.wait(p_free.index(ks * NH + 1), ((tstep // RP) & 1) ^ 1)
    p0 = cbv * _ex2ftz(gl.minimum(bk0 - am0, 0.0) * 1.4426950408889634)   # the accepted expression: its bf16 rounding point
    p1 = cbv * _ex2ftz(gl.minimum(bk1 - am1, 0.0) * 1.4426950408889634)
    if MASK:
        offs_k = kt * BK + gl.arange(0, BK, layout=gl.SliceLayout(1, L))
        keep = offs_k[:, None] >= offs_m[None, :]
        p0 = gl.where(keep, p0, 0.0)
        p1 = gl.where(keep, p1, 0.0)
    p_s.index(ks * NH).store(p0.to(gl.bfloat16))
    p_s.index(ks * NH + 1).store(p1.to(gl.bfloat16))
    fence_async_shared()
    gl.thread_barrier()
    mbarrier.arrive(p_ready.index(ks * NH))
    mbarrier.arrive(p_ready.index(ks * NH + 1))
    return tstep + 1


@gluon.jit
def _compute_partition(cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
                       acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
                       cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
                       BM: gl.constexpr, BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, CS: gl.constexpr,
                       HG: gl.constexpr, MT: gl.constexpr, P: gl.constexpr, N: gl.constexpr, H: gl.constexpr,
                       NB: gl.constexpr, NWC: gl.constexpr, DD: gl.constexpr,
                       RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr, NWE: gl.constexpr, EH: gl.constexpr):
    """Compute partition (NWC warps over the (BK, BM) tile: 4 along the rows, NWC / 4 along the columns): per item and K step
    k >= m the item's two heads' K^T tiles from one load of the resident CB tile (NH == 2, the launcher's contract)."""
    L: gl.constexpr = gl.BlockedLayout([BK // 16, CW], [4, 8], [4, NWC // 4], [1, 0])        # over (BK, BM)
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
        if bcm != prev:
            mbarrier.wait(res_bar, nres & 1)
            nres += 1
            prev = bcm
        mbarrier.wait(dad_bar.index(a), (loc // DD) & 1)
        i0h = a * NH * NKT
        i1h = i0h + NKT
        am0 = dar_s.index(i0h + m).load(L)                                   # (1, BM) dA_m of each head
        am1 = dar_s.index(i1h + m).load(L)
        offs_m = m * BM + gl.arange(0, BM, layout=gl.SliceLayout(0, L))
        cbv = cbr_s.index(m).load(L)                                         # the diagonal step: k tile == m tile
        tstep = _kstep(p_s, p_ready, p_free, tstep, cbv, am0, dac_s.index(i0h + m).load(L), am1,
                       dac_s.index(i1h + m).load(L), offs_m, m, BK, NH, RP, NWC, True, CW)
        for kt in range(m + 1, NKT):
            cbv = cbr_s.index(kt).load(L)
            tstep = _kstep(p_s, p_ready, p_free, tstep, cbv, am0, dac_s.index(i0h + kt).load(L), am1,
                           dac_s.index(i1h + kt).load(L), offs_m, kt, BK, NH, RP, NWC, False, CW)
        mbarrier.arrive(dad_free.index(a))
        last = (it + 1 == i1) | ((it + 1) // HG != bcm)
        if last:
            mbarrier.arrive(res_free)


# ------------------------------------------------------------------------------------------------ kernel
@gluon.jit
def _chunk_scan_dx_kernel(cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, starts, seqlen, nchunks, batch,
                          sxl, sdl, sdxl,
                          H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BM: gl.constexpr,
                          BK: gl.constexpr, NH: gl.constexpr, R: gl.constexpr, NB: gl.constexpr, NWC: gl.constexpr,
                          DD: gl.constexpr, REGS_W: gl.constexpr, REGS_C: gl.constexpr, REGS_E: gl.constexpr,
                          RP: gl.constexpr, SR: gl.constexpr, CW: gl.constexpr, NWE: gl.constexpr, EH: gl.constexpr):
    MT: gl.constexpr = CS // BM
    HG: gl.constexpr = H // NH
    NKT: gl.constexpr = CS // BK
    TL: gl.constexpr = TensorMemoryLayout(block=(BM, P), unpacked=True)
    PL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BK, BM], gl.bfloat16)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    SL2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    cbr_s = gl.allocate_shared_memory(gl.float32, [NKT, BK, BM], cbd.layout)
    b_s = gl.allocate_shared_memory(gl.bfloat16, [BM, N], bd.layout)
    do_s = gl.allocate_shared_memory(gl.bfloat16, [R * NH, BK, P], dod.layout)
    p_s = gl.allocate_shared_memory(gl.bfloat16, [RP * NH, BK, BM], PL)
    ds_s = gl.allocate_shared_memory(gl.bfloat16, [SR * NH, P, N], dsd.layout)
    dar_s = gl.allocate_shared_memory(gl.float32, [DD * NH * NKT, 1, BK], dakd.layout)
    dac_s = gl.allocate_shared_memory(gl.float32, [DD * NH * NKT, BK, 1], SL2)
    res_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
    res_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    dad_tma = gl.allocate_shared_memory(gl.int64, [DD, 1], BL)
    dad_bar = gl.allocate_shared_memory(gl.int64, [DD, 1], BL)
    dad_free = gl.allocate_shared_memory(gl.int64, [DD, 1], BL)
    done_bar = gl.allocate_shared_memory(gl.int64, [NB, 1], BL)
    acc_free = gl.allocate_shared_memory(gl.int64, [NB, 1], BL)
    p_ready = gl.allocate_shared_memory(gl.int64, [RP * NH, 1], BL)
    p_free = gl.allocate_shared_memory(gl.int64, [RP * NH, 1], BL)
    do_bar = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    do_free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    ds_bar = gl.allocate_shared_memory(gl.int64, [SR * NH, 1], BL)
    ds_free = gl.allocate_shared_memory(gl.int64, [SR * NH, 1], BL)
    mbarrier.init(res_bar, count=1)
    mbarrier.init(res_free, count=2)                           # the compute partition + the MMA warp's commit
    for i in gl.static_range(DD):
        mbarrier.init(dad_tma.index(i), count=1)
        mbarrier.init(dad_bar.index(i), count=1)
        mbarrier.init(dad_free.index(i), count=1)              # the compute partition
    for i in gl.static_range(NB):
        mbarrier.init(done_bar.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=NH)             # the NH epilogue partitions
    for i in gl.static_range(RP * NH):
        mbarrier.init(p_ready.index(i), count=1)
        mbarrier.init(p_free.index(i), count=1)
    for i in gl.static_range(R):
        mbarrier.init(do_bar.index(i), count=1)
        mbarrier.init(do_free.index(i), count=1)
    for i in gl.static_range(SR * NH):
        mbarrier.init(ds_bar.index(i), count=1)
        mbarrier.init(ds_free.index(i), count=1)
    fence_async_shared()
    acc_i = allocate_tensor_memory(gl.float32, [NB * NH, BM, P], TL)
    acc_c = allocate_tensor_memory(gl.float32, [NB * NH, BM, P], TL)
    gl.warp_specialize(
            (cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
             acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
             cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
             BM, BK, NH, R, CS, HG, MT, P, N, H, NB, NWC, DD, RP, SR, CW, NWE, EH),
            _epilogue_partition,
            (cbr_s, p_s, dar_s, dac_s, res_bar, res_free, dad_tma, dad_bar, dad_free, p_ready, p_free, starts,
             acc_i, acc_c, b_s, ds_s, do_s, ds_bar, ds_free, do_bar, do_free, done_bar, acc_free,
             cbd, dod, bd, dsd, dakd, X, DOUT, DT, DA, D, DX, DDT, DDW, seqlen, nchunks, batch, sxl, sdl, sdxl,
             BM, BK, NH, R, CS, HG, MT, P, N, H, NB, NWC, DD, RP, SR, CW, NWE, EH),
            [_load_partition, _stage_partition, _mma_partition, _compute_partition, _epilogue1_partition],
            [1, 1, 1, NWC, NWE],
            [REGS_W, REGS_W, REGS_W, REGS_C, REGS_E])


# ------------------------------------------------------------------------------------------------ launcher
def _work_split(batch: int, nchunks: int, MT: int, HG: int, G: int, dev: torch.device) -> torch.Tensor:
    """(G + 1,) int32 boundaries of contiguous item ranges of near-equal cost (cost = W0 + the m tile's K steps)."""
    key = (batch, nchunks, MT, HG, G, dev.index)
    starts = _WORK_SPLIT.get(key)
    if starts is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("neuronspark.chunk_scan_dx_gluon: the work split is built on the first (eager) call; first "
                               "call inside a CUDA-graph capture")
        n_items = batch * nchunks * MT * HG
        cost = [W0 + (MT - ((i // HG) % MT)) for i in range(n_items)]
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


def chunk_scan_dx_gluon(x, dt, dA_cumsum, B, CB, dout, dstates, D, dx=None):
    """(dx (written into ``dx`` when given), ddt (b, h, c, chunk) fp32, dD (h,) in D's dtype) of the spiking-layer chunk
    scan's input
    backward (see the module docstring)."""
    require_cuda(x, "chunk_scan_dx_gluon")
    require_sm100(x.device, "chunk_scan_dx_gluon")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    _, _, ngroups, dstate = B.shape
    if (ngroups != 1 or seqlen % chunk or chunk % BM or BM != BK or BM != 64 or NH != 2 or nheads % NH
            or 2 * NB * NH * headdim > 512 or NWC not in (4, 8, 16) or BK % 16 or DD < 2 or SR not in (1, 2)):
        raise RuntimeError(f"neuronspark.chunk_scan_dx_gluon: ngroups {ngroups} seqlen {seqlen} chunk {chunk} (full "
                           f"chunks only) BM {BM} BK {BK} NH {NH} nheads {nheads} NB {NB} headdim {headdim}")
    if x.dtype != torch.bfloat16 or dout.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise RuntimeError("neuronspark.chunk_scan_dx_gluon: bf16 x, dout and B expected")
    for t, name in ((x, "x"), (dout, "dout")):
        if t.shape != (batch, seqlen, nheads, headdim) or t.stride(3) != 1 or t.stride(2) != headdim \
                or t.stride(0) != seqlen * t.stride(1):
            raise RuntimeError(f"neuronspark.chunk_scan_dx_gluon: {name} shape / strides {tuple(t.shape)} {tuple(t.stride())}")
    if B.stride(3) != 1 or B.stride(0) != seqlen * B.stride(1):
        raise RuntimeError(f"neuronspark.chunk_scan_dx_gluon: B strides {tuple(B.stride())}")
    if CB.shape != (batch, nchunks, 1, chunk, chunk) or not CB.is_contiguous() or CB.dtype != torch.float32:
        raise RuntimeError("neuronspark.chunk_scan_dx_gluon: contiguous fp32 CB (b, nc, 1, chunk, chunk) expected")
    if (dstates.shape != (batch, nchunks, nheads, headdim, dstate) or not dstates.is_contiguous()
            or dstates.dtype != torch.bfloat16):
        raise RuntimeError("neuronspark.chunk_scan_dx_gluon: contiguous bf16 dstates (b, nc, h, P, N) expected")
    if (not dA_cumsum.is_contiguous() or not dt.is_contiguous() or dt.dtype != torch.float32
            or dA_cumsum.dtype != torch.float32 or D.dim() != 1 or D.stride(0) != 1):
        raise RuntimeError("neuronspark.chunk_scan_dx_gluon: contiguous fp32 dt / dA_cumsum (b, h, c, l) and D (nheads,) "
                           "expected")
    dev = x.device
    if dx is None:
        dx = torch.empty(batch, seqlen, nheads, headdim, device=dev, dtype=x.dtype)
    elif dx.shape != x.shape or dx.dtype != x.dtype or dx.stride(3) != 1 or dx.stride(2) != headdim \
            or dx.stride(0) != seqlen * dx.stride(1):
        raise RuntimeError(f"neuronspark.chunk_scan_dx_gluon: dx buffer {tuple(dx.shape)} {tuple(dx.stride())}")
    MT, HG = chunk // BM, nheads // NH
    G = num_sms(dev)
    starts = _work_split(batch, nchunks, MT, HG, G, dev)
    ddt = torch.empty(batch, nheads, nchunks, chunk, device=dev, dtype=torch.float32)
    ddw = torch.empty(batch, nheads, nchunks, chunk, device=dev, dtype=torch.float32)
    f32, b16 = gl.float32, gl.bfloat16
    rows = batch * nheads * nchunks
    cbd = TensorDescriptor(CB, [batch * nchunks * chunk, chunk], [chunk, 1], [BK, BM], gl.NVMMASharedLayout.get_default_for([BK, BM], f32))
    dod = TensorDescriptor(dout, [batch * seqlen, nheads * headdim], [dout.stride(1), 1], [BK, headdim],
                           gl.NVMMASharedLayout.get_default_for([BK, headdim], b16))
    bd = TensorDescriptor(B, [batch * seqlen, dstate], [B.stride(1), 1], [BM, dstate], gl.NVMMASharedLayout.get_default_for([BM, dstate], b16))
    dsd = TensorDescriptor(dstates, [batch * nchunks * nheads * headdim, dstate], [dstate, 1], [headdim, dstate],
                           gl.NVMMASharedLayout.get_default_for([headdim, dstate], b16))
    dakd = TensorDescriptor(dA_cumsum, [rows * (chunk // BK), BK], [BK, 1], [1, BK], gl.NVMMASharedLayout.get_default_for([1, BK], f32))
    with torch.cuda.device(dev.index):
        _chunk_scan_dx_kernel[(G,)](
            cbd, dod, bd, dsd, dakd, x, dout, dt, dA_cumsum, D, dx, ddt, ddw, starts, seqlen, nchunks, batch,
            x.stride(1), dout.stride(1), dx.stride(1),
            H=nheads, P=headdim, N=dstate, CS=chunk, BM=BM, BK=BK, NH=NH, R=R, NB=NB, NWC=NWC, DD=DD, REGS_W=REGS_W,
            REGS_C=REGS_C, REGS_E=REGS_E, RP=RP, SR=SR, CW=CW, NWE=NWE, EH=EH, num_warps=NWE)
    dD32 = ddw.sum(dim=(0, 2, 3))
    return dx, ddt, (dD32 if D.dtype == torch.float32 else dD32.to(D.dtype))


def chunk_scan_dx_gluon_ref(x, dt, dA_cumsum, B, CB, dout, dstates, D):
    """Pure-torch reference: :func:`neuronspark.ops.triton.spiking_scan.chunk_scan_dx_ref` (the bf16 rounding point of
    the K
    operand and the FTZ decay reproduced, float64 sums)."""
    from neuronspark.ops.triton.spiking_scan import chunk_scan_dx_ref
    return chunk_scan_dx_ref(x, dt, dA_cumsum, B, CB, dout, dstates, D, None)
