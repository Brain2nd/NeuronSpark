"""dendritic-layer intra-chunk dCB / ddA pieces of the SSD scan's training backward, written in Gluon for Blackwell
GPUs: the per-head products dout_h x_h^T of a chunk's lower 128 x 128 tiles on the tensor cores, their decay scaling,
the head-split dCB sums and the 32 x 32 sub-tile row / column sums of the CB-weighted products (the input of the ddA
assembly).

For chunk c of batch row b and head h (chunk length CS 256, ``dA = dA_cumsum[b, h, c]``, ``dt = dt[b, h, c]``)::

    g_h[m, n]  = (dout_h[m] . x_h[n]) dt[n] exp(min(dA[m] - dA[n], 0))                    (fp32)
    dCB_s[m, n] = sum_{h in split s} g_h[m, n]   for n <= m, 0 above                      (fp32)
    T_h[m, n]  = g_h[m, n] CB[m, n]              for n < m                                  (fp32)
    V_h        = per 32 x 32 sub-tile (I, J) of the chunk: rs[m] = sum_n T_h[m, n], cs[n] = sum_m T_h[m, n]

-- the dCB of the dendritic-layer call (ngroups 1) and the pieces that
:func:`neuronspark.ops.triton.dendritic_scan.dda_asm` assembles into the intra-chunk ddA. Returns ``(dCB head-split
partials (b, nc, NSPLIT, 1, CS, CS) fp32, V (b, h, nc, 8 * 8 * 64) fp32)``: the split partials are summed by the caller
(``dendritic_glue.splits_sum``), V is read by ``dda_asm`` (the upper sub-tiles of V are written as zeros and never
read).

Work items: (b c, head split s of HPS heads, lower tile (I, J) in {(0, 0), (1, 0), (1, 1)} of 128 rows / columns), the
three tiles of one (b c, s) on consecutive CTAs (their dout / x rows shared through L2).  Per CTA (persistent, items in
round-robin): a load warp streams each head's (128, P) dout rows of tile row I and x rows of tile column J through an
R-deep mbarrier ring (TMA, 3-D (batch, seqlen, h P) descriptors); an MMA warp issues g = dout_I x_J^T (128 x 128 x P)
into one of two TMEM buffers; two compute partitions of 8 warps (one per 64-column half of the tile; thread = tile row,
32 columns per thread) scale g, accumulate dCB in TMEM (fp32, over the split's heads in order), form T against the
item's CB tile (loaded once per item into TMEM, zero on and above the diagonal) and write the V sums: rs in thread, cs
by a warp transpose-reduce (five butterfly exchanges, one column per lane).  The dCB tile leaves through shared-memory
staging and TMA stores; the tile (0, 1) of a split is written as zeros by the (1, 0) item.

The decay without a per-element exponent: dA is non-increasing along the chunk, so for the rows m of a 32-row block q
(first row r_q) and n <= m, exp(dA_m - dA_n) = R_m C_q[n] with R_m = exp(dA_m - dA[r_q]) <= 1 (one exponent per thread)
and C_q[n] = exp(min(dA[r_q] - dA_n, 80)), <= 1 below the block and <= exp(D_q) on its diagonal block (D_q = dA[r_q] -
dA[r_q + 31]); dt_n is folded into the per-head table C (4 x 64 per partition, one exponent per thread, through shared
memory).  A diagonal tile with some D_q > 80 takes the per-element exponent for that head.

Numerics: the reference's expressions (g dt exp(min(dA_m - dA_n, 0)); dCB summed over the split's heads in head order
from zero) with the exponent as the product R_m C_q[n] on the factorised path; the dot's fp32 accumulation is the tensor
cores', the V sums are this kernel's order (the per-operator rule class of :mod:`neuronspark.utils.testing` against
:func:`neuronspark.ops.triton.dendritic_scan.dda_ref`). Full chunks only (seqlen % 256 == 0, the training regime); P 64.

Launch constants: BM 128 (tile rows / columns), R 4 ring slots, NSPLIT 6 head splits (24 heads each at H 144), NWG 8
warps per compute partition, registers 24 (load / MMA warps), 120 (the worker compute partition; the default one gets
104); grid = #SM persistent CTAs."""
from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_32x32b_reg_layout, tcgen05_mma, mbarrier, tma,
    fence_async_shared)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from neuronspark.ops.gluon._ptx import _tc_fence
from neuronspark.ops.triton import num_sms, require_cuda, require_sm100

__all__ = ["BM", "R", "NSPLIT", "NWG", "REGS_W", "REGS_C", "dda_gluon"]

# ------------------------------------------------------------------------------------------------ launch constants
BM = 128         # tile rows / columns (a chunk of 256 has the lower tiles (0, 0), (1, 0), (1, 1))
R = 4            # ring slots (heads in flight)
NSPLIT = 6       # head splits per (b, c) (the dCB partials' split axis)
NWG = 8          # warps per compute partition (thread = tile row; two warps per row block, 32 columns each)
REGS_W = 24      # registers of the load / MMA warps
REGS_C = 120     # registers of the worker compute partition (the default partition gets 224 - REGS_C)
CS = 256         # the chunk length the kernel covers
SUB = 32         # sub-tile edge of the V grid (dda_asm's BS)


# the launch constants as compile-time values of the kernels
_BM = gl.constexpr(BM)
_R = gl.constexpr(R)
_NSPLIT = gl.constexpr(NSPLIT)
_CS = gl.constexpr(CS)
_SUB = gl.constexpr(SUB)


# ------------------------------------------------------------------------------------------------ helpers
@gluon.jit
def _shfl_bfly(v, m):
    return gl.inline_asm_elementwise("shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r", [v, m],
                                     dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _xr_step(v, S: gl.constexpr, K: gl.constexpr):
    """One reduce-scatter step over the lanes of a warp: (rows, K) -> (rows, K / 2) by even / odd columns; lanes with
    bit S (of the row index = the lane) keep the odd columns and receive their partner's odd ones (the partner keeps
    the even)."""
    ev, od = gl.split(gl.reshape(v, [_BM, K // 2, 2]))
    rr = gl.arange(0, _BM, layout=gl.SliceLayout(1, ev.type.layout))
    bit = ((rr & S) != 0)[:, None]
    send = gl.where(bit, ev, od)
    keep = gl.where(bit, od, ev)
    return keep + _shfl_bfly(send, gl.full([_BM, K // 2], S, gl.int32, layout=ev.type.layout))


@gluon.jit
def _col_sums(t):
    """(BM, 64) -> (BM, 2): per 32-column half the sums over the 32 rows of each warp, column (row index % 32) of the
    half per lane."""
    t = _xr_step(t, 1, 2 * _SUB)
    t = _xr_step(t, 2, _SUB)
    t = _xr_step(t, 4, _SUB // 2)
    t = _xr_step(t, 8, _SUB // 4)
    return _xr_step(t, 16, _SUB // 8)


@gluon.jit
def _item_coords(it, nchunks):
    t = it % 3
    rest = it // 3
    s = rest % _NSPLIT
    bc = rest // _NSPLIT
    return bc // nchunks, bc % nchunks, s, (t + 1) // 2, t // 2, t


# ------------------------------------------------------------------------------------------------ partitions
@gluon.jit
def _compute_body(do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
                  dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks,
                  H: gl.constexpr, P: gl.constexpr, HPS: gl.constexpr, NWG: gl.constexpr, J0: gl.constexpr):
    """One compute partition (NWG warps, thread = tile row): the 64 columns 32 J0 .. 32 J0 + 63 of every head's g.

    The decay: dA is non-increasing along the chunk, so for rows m of the 32-row block q (first row r_q) and n <= m,
    exp(dA_m - dA_n) = R_m C_q[n] with R_m = exp(dA_m - dA[r_q]) <= 1 and C_q[n] = exp(min(dA[r_q] - dA_n, 80)) (<= 1
    below the block; <= exp(D_q) on its diagonal block, D_q = dA[r_q] - dA[r_q + 31]).  The products equal the
    per-element exponent's while every D_q of a diagonal tile is <= 80; otherwise the head's tile takes the per-element
    exponent.  dt_n is folded into the table C (4 x 64 per head, one exponent per thread, through shared memory)."""
    W: gl.constexpr = 2 * _SUB                                         # columns of the partition
    L: gl.constexpr = get_tmem_32x32b_reg_layout(_BM, W, [_BM, W], NWG)
    L32: gl.constexpr = get_tmem_32x32b_reg_layout(_BM, _SUB, [_BM, _SUB], NWG)
    LR: gl.constexpr = gl.SliceLayout(1, L)
    LC: gl.constexpr = gl.SliceLayout(0, L)
    KT: gl.constexpr = W * 4 // NWG                                    # columns per thread
    LT: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, NWG // 4], [0, 1])        # the (4, W) table
    L3: gl.constexpr = gl.BlockedLayout([1, 1, KT], [1, 32, 1], [4, 1, NWG // 4], [0, 1, 2])
    TS3: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    TS2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    r = gl.arange(0, _BM, layout=LR)
    cl = gl.arange(0, W, layout=LC)
    tq = gl.arange(0, 4, layout=gl.SliceLayout(1, LT))
    tcol = gl.arange(0, W, layout=gl.SliceLayout(0, LT))
    G = gl.num_programs(0)
    tc = 0                                                             # heads consumed (g / table buffer index)
    for it in range(gl.program_id(0), n_items, G):
        b, c, s, I, J, t = _item_coords(it, nchunks)
        rows = I * _BM + r                                             # chunk rows of the tile
        cols = J * _BM + J0 * _SUB + cl                                # chunk columns of the partition
        tn = J * _BM + J0 * _SUB + tcol
        # the item's CB tile, zero on and above the diagonal (T is strictly lower)
        cbase = (b * nchunks + c) * _CS
        v = gl.load(CB + (cbase + rows)[:, None] * _CS + cols[None, :])
        cb_t.slice(J0 * _SUB, W).store(gl.where(rows[:, None] > cols[None, :], v, 0.0))
        h0 = s * HPS
        nh = gl.minimum(HPS, H - h0)
        for hh in range(nh):
            h = h0 + hh
            a = tc % 2
            dbase = ((b * H + h) * nchunks + c) * _CS
            dam = gl.load(DA + dbase + rows)
            ref = gl.load(DA + dbase + I * _BM + (r // _SUB) * _SUB)
            rf = gl.exp(dam - ref)                                     # R_m
            tref = gl.load(DA + dbase + I * _BM + tq * _SUB)
            tda = gl.load(DA + dbase + tn)
            tdt = gl.load(DT + dbase + tn)
            ct = tdt[None, :] * gl.exp(gl.minimum(tref[:, None] - tda[None, :], 80.0))     # C_q[n] dt_n
            dmax = gl.load(DA + dbase + I * _BM) - gl.load(DA + dbase + I * _BM + _SUB - 1)
            for q in gl.static_range(1, 4):
                dmax = gl.maximum(dmax, gl.load(DA + dbase + I * _BM + q * _SUB)
                                  - gl.load(DA + dbase + I * _BM + q * _SUB + _SUB - 1))
            exact = (I == J) & (dmax > 80.0)                           # a diagonal tile with a steep block
            tb = ct_s.index((J0 // 2) * 2 + a)                         # this partition's table of the head's parity
            tb._reinterpret(gl.float32, [4, W], TS2).store(ct)
            gl.thread_barrier()
            cw3 = tb._reinterpret(gl.float32, [4, 1, W], TS3).load(L3)
            cw3, _ = gl.broadcast(cw3, gl.zeros([4, 32, W], gl.float32, layout=L3))
            cw = gl.convert_layout(gl.reshape(cw3, [_BM, W]), L)
            mbarrier.wait(g_done.index(a), (tc // 2) & 1)
            _tc_fence(1, NWG)
            g = g_t.index(a).slice(J0 * _SUB, W).load(L)
            _tc_fence(0, NWG)
            gl.thread_barrier()
            mbarrier.arrive(g_free.index(a))
            if exact:
                dan = gl.load(DA + dbase + cols)
                dtn = gl.load(DT + dbase + cols)
                f = (g * dtn[None, :]) * gl.exp(gl.minimum(dam[:, None] - dan[None, :], 0.0))
            else:
                f = (g * cw) * rf[:, None]
            if hh == 0:
                acc = f
            else:
                acc = dcb_t.slice(J0 * _SUB, W).load(L) + f
            dcb_t.slice(J0 * _SUB, W).store(acc)
            tt = f * cb_t.slice(J0 * _SUB, W).load(L)
            vb = V + ((b * H + h) * nchunks + c) * (8 * 8 * 2 * _SUB)
            rsum = gl.sum(gl.reshape(tt, [_BM, 2, _SUB]), axis=2)            # (BM, 2): per half, in thread
            ro = gl.arange(0, _BM, layout=gl.SliceLayout(1, rsum.type.layout))[:, None]
            ho = gl.arange(0, 2, layout=gl.SliceLayout(0, rsum.type.layout))[None, :]
            gl.store(vb + ((I * 4 + ro // _SUB) * 8 + J * 4 + J0 + ho) * (2 * _SUB) + ro % _SUB, rsum)
            csum = _col_sums(tt)                                       # (BM, 2): column (row % 32) of each half
            co = gl.arange(0, _BM, layout=gl.SliceLayout(1, csum.type.layout))[:, None]
            hc = gl.arange(0, 2, layout=gl.SliceLayout(0, csum.type.layout))[None, :]
            gl.store(vb + ((I * 4 + co // _SUB) * 8 + J * 4 + J0 + hc) * (2 * _SUB) + _SUB + co % _SUB, csum)
            tc += 1
        # the split's dCB tile (rows >= cols), through shared staging and TMA stores; (1, 0) also zeroes tile (0, 1)
        orow = ((b * nchunks + c) * _NSPLIT + s) * _CS
        tma.store_wait(0)
        r32 = gl.arange(0, _BM, layout=gl.SliceLayout(1, L32))
        c32 = gl.arange(0, _SUB, layout=gl.SliceLayout(0, L32))
        for jj in gl.static_range(2):
            cols32 = J * _BM + (J0 + jj) * _SUB + c32
            d = dcb_t.slice((J0 + jj) * _SUB, _SUB).load(L32)
            dcb_s.index(J0 + jj).store(gl.where((I * _BM + r32)[:, None] >= cols32[None, :], d, 0.0))
        fence_async_shared()
        gl.thread_barrier()
        for jj in gl.static_range(2):
            tma.async_copy_shared_to_global(dcbd, [orow + I * _BM, J * _BM + (J0 + jj) * _SUB], dcb_s.index(J0 + jj))
        if t == 1:
            zr = orow + r
            gl.store(DCBP + zr[:, None] * _CS + (_BM + J0 * _SUB + cl)[None, :], gl.zeros([_BM, W], gl.float32, layout=L))
    tma.store_wait(0)


@gluon.jit
def _compute0_partition(do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
                        dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks,
                        H: gl.constexpr, P: gl.constexpr, HPS: gl.constexpr, NWG: gl.constexpr):
    _compute_body(do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
                  dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks, H, P, HPS, NWG, 0)


@gluon.jit
def _compute1_partition(do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
                        dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks,
                        H: gl.constexpr, P: gl.constexpr, HPS: gl.constexpr, NWG: gl.constexpr):
    _compute_body(do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
                  dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks, H, P, HPS, NWG, 2)


@gluon.jit
def _load_partition(do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
                    dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks,
                    H: gl.constexpr, P: gl.constexpr, HPS: gl.constexpr, NWG: gl.constexpr):
    """Load warp: per item and head the (BM, P) dout rows of tile row I and x rows of tile column J."""
    G = gl.num_programs(0)
    tc = 0
    for it in range(gl.program_id(0), n_items, G):
        b, c, s, I, J, t = _item_coords(it, nchunks)
        h0 = s * HPS
        nh = gl.minimum(HPS, H - h0)
        for hh in range(nh):
            h = h0 + hh
            slot = tc % _R
            mbarrier.wait(free.index(slot), ((tc // _R) & 1) ^ 1)
            mbarrier.expect(ready.index(slot), 2 * _BM * P * 2)
            tma.async_copy_global_to_shared(dod, [b, c * _CS + I * _BM, h * P], ready.index(slot),
                                            do_s.index(slot)._reinterpret(gl.bfloat16, [1, _BM, P], dod.layout))
            tma.async_copy_global_to_shared(xd, [b, c * _CS + J * _BM, h * P], ready.index(slot),
                                            x_s.index(slot)._reinterpret(gl.bfloat16, [1, _BM, P], xd.layout))
            tc += 1


@gluon.jit
def _mma_partition(do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
                   dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks,
                   H: gl.constexpr, P: gl.constexpr, HPS: gl.constexpr, NWG: gl.constexpr):
    """MMA warp: g (BM, BM) = dout_I (BM, P) . x_J^T (P, BM) per head into the TMEM buffer of the head's parity."""
    G = gl.num_programs(0)
    tc = 0
    for it in range(gl.program_id(0), n_items, G):
        b, c, s, I, J, t = _item_coords(it, nchunks)
        h0 = s * HPS
        nh = gl.minimum(HPS, H - h0)
        for hh in range(nh):
            slot = tc % _R
            a = tc % 2
            mbarrier.wait(ready.index(slot), (tc // _R) & 1)
            mbarrier.wait(g_free.index(a), ((tc // 2) & 1) ^ 1)        # both compute partitions released it
            _tc_fence(1, 1)
            tcgen05_mma(do_s.index(slot), x_s.index(slot).permute((1, 0)), g_t.index(a), use_acc=False,
                        mbarriers=[free.index(slot), g_done.index(a)])
            tc += 1


# ------------------------------------------------------------------------------------------------ kernel
@gluon.jit
def _dda_kernel(dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks,
                H: gl.constexpr, P: gl.constexpr, HPS: gl.constexpr, NWG: gl.constexpr, REGS_W: gl.constexpr,
                REGS_C: gl.constexpr):
    TL: gl.constexpr = TensorMemoryLayout(block=(_BM, _BM), unpacked=True)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    XL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([_BM, P], gl.bfloat16)
    do_s = gl.allocate_shared_memory(gl.bfloat16, [_R, _BM, P], XL)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [_R, _BM, P], XL)
    dcb_s = gl.allocate_shared_memory(gl.float32, [4, _BM, _SUB], dcbd.layout)
    ct_s = gl.allocate_shared_memory(gl.float32, [4, 4, 2 * _SUB], gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0]))
    ready = gl.allocate_shared_memory(gl.int64, [_R, 1], BL)
    free = gl.allocate_shared_memory(gl.int64, [_R, 1], BL)
    g_done = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    g_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    for i in gl.static_range(_R):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(g_done.index(i), count=1)
        mbarrier.init(g_free.index(i), count=2)                        # both compute partitions release a buffer
    fence_async_shared()
    g_t = allocate_tensor_memory(gl.float32, [2, _BM, _BM], TL)
    cb_t = allocate_tensor_memory(gl.float32, [_BM, _BM], TL)
    dcb_t = allocate_tensor_memory(gl.float32, [_BM, _BM], TL)
    gl.warp_specialize(
        (do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
         dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks, H, P, HPS, NWG),
        _compute0_partition,
        (do_s, x_s, ready, free, g_done, g_free, g_t, cb_t, dcb_t, dcb_s, ct_s,
         dod, xd, dcbd, DA, DT, CB, V, DCBP, n_items, nchunks, H, P, HPS, NWG),
        [_compute1_partition, _load_partition, _mma_partition],           # (the compute warps warpgroup-aligned)
        [NWG, 1, 1],
        [REGS_C, REGS_W, REGS_W])


# ------------------------------------------------------------------------------------------------ launcher
def dda_gluon(x, dt, dA_cumsum, dout, cb):
    """(dCB head-split partials (b, nc, NSPLIT, 1, CS, CS) fp32, V (b, h, nc, 8 * 8 * 64) fp32) for x / dout (b, l, h, P)
    bf16 with a contiguous (h, P) part and row / batch strides that are multiples of 8, dt / dA_cumsum (b, h, nc, CS) fp32
    contiguous, cb (b, nc, 1, CS, CS) fp32 contiguous (see the module docstring)."""
    require_cuda(x, "dda_gluon")
    require_sm100(x.device, "dda_gluon")
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    if (headdim != 64 or chunk != CS or seqlen != nchunks * CS or dout.shape != x.shape
            or tuple(dA_cumsum.shape) != tuple(dt.shape) or tuple(dt.shape[:2]) != (batch, nheads)
            or tuple(cb.shape) != (batch, nchunks, 1, CS, CS)):
        raise RuntimeError(f"neuronspark.dda_gluon: shapes x {tuple(x.shape)} dout {tuple(dout.shape)} dt {tuple(dt.shape)} "
                           f"cb {tuple(cb.shape)} (P 64, full chunks of {CS})")
    if x.dtype != torch.bfloat16 or dout.dtype != torch.bfloat16:
        raise RuntimeError("neuronspark.dda_gluon: bf16 x / dout expected")
    for name, t in (("x", x), ("dout", dout)):
        if t.stride(3) != 1 or t.stride(2) != headdim or t.stride(1) % 8 or t.stride(0) % 8:
            raise RuntimeError(f"neuronspark.dda_gluon: {name} strides {tuple(t.stride())}")
    if (dt.dtype != torch.float32 or dA_cumsum.dtype != torch.float32 or cb.dtype != torch.float32
            or not dt.is_contiguous() or not dA_cumsum.is_contiguous() or not cb.is_contiguous()):
        raise RuntimeError("neuronspark.dda_gluon: contiguous fp32 dt / dA_cumsum / cb expected")
    dev = x.device
    hps = -(-nheads // NSPLIT)
    v = torch.empty((batch, nheads, nchunks, 8 * 8 * 2 * SUB), dtype=torch.float32, device=dev)
    dcbp = torch.empty((batch, nchunks, NSPLIT, 1, CS, CS), dtype=torch.float32, device=dev)
    b16 = gl.bfloat16
    xl3 = gl.NVMMASharedLayout.get_default_for([1, BM, headdim], b16)
    xd = TensorDescriptor(x, [batch, seqlen, nheads * headdim], [x.stride(0), x.stride(1), 1], [1, BM, headdim], xl3)
    dod = TensorDescriptor(dout, [batch, seqlen, nheads * headdim], [dout.stride(0), dout.stride(1), 1],
                           [1, BM, headdim], xl3)
    rows = batch * nchunks * NSPLIT * CS
    dcbd = TensorDescriptor(dcbp.view(rows, CS), [rows, CS], [CS, 1], [BM, SUB],
                            gl.NVMMASharedLayout.get_default_for([BM, SUB], gl.float32))
    n_items = batch * nchunks * NSPLIT * 3
    G = min(num_sms(dev), n_items)
    with torch.cuda.device(dev.index):
        _dda_kernel[(G,)](dod, xd, dcbd, dA_cumsum, dt, cb, v, dcbp, n_items, nchunks,
                          H=nheads, P=headdim, HPS=hps, NWG=NWG, REGS_W=REGS_W, REGS_C=REGS_C, num_warps=NWG)
    return dcbp, v
