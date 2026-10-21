"""Backward of the hippocampal layers' fused pool read (:mod:`neuronspark.ops.pool_read`), written in Gluon (Triton
3.5.1's explicit-layout dialect) for Blackwell GPUs: warp-specialised kernels (TMA loads with mbarrier rings,
asynchronous tcgen05 MMAs into tensor memory), each with one load warp, one MMA warp and a default 4-warp compute
partition holding one tile row per thread; P and dS enter their products as two bf16 pieces.

* ``pool_read_dv_gluon`` -> dgv (T, B, H, dv) bf16: one CTA per (key tile of BNK rows, b, key head), longest first; per
  step (a query tile of BMQ_V rows of one query head of the key head) S^T = K Q^T (one step ahead), P^T =
  exp(scale S^T + bias^T - lse) (the tile's bias from the id windows, the clock difference and the bf16 rounding; the
  causal mask on the diagonal steps), dV += P^T dO;
* ``pool_read_dk_gluon`` -> (dk fp32, dcum fp32, the beta-slot sums): the same decomposition (BMQ_K query rows per
  step) with dP^T = V dO^T, dS^T = P^T (dP^T - D), dK += dS^T Q; the bias is formed once per query tile for the key
  head's query heads; each thread sums its key row's dS (the clock's column sum) and the matched dS per order (the
  beta sums, written per row and reduced afterwards);
* ``pool_read_dq_gluon`` -> dq fp32: one CTA per (query tile of BM_Q rows, b, query head), longest first; per key tile
  of BN_Q rows S = Q K^T and dP = dO V^T (one tile ahead), dS = P (dP - D), dQ += dS K.

Checked with the per-operator tolerance rule against :func:`neuronspark.ops.pool_read.pool_read_ref`.  No compiler
scratch inside the partitions (every reduction runs along a thread's own row).

Launch constants: BNK 128 key rows, BMQ_V = BMQ_K = 32 query rows per step; BM_Q 128 query rows, BN_Q 32 key rows per
step; rings R_V 3 (value gradient; 10 % faster than 2 at 4K) and R 2 (the others: shared memory); registers 48 for
the load / MMA warps.
"""
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_32x32b_reg_layout, tcgen05_mma, tcgen05_commit, mbarrier, tma,
    fence_async_shared)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from neuronspark.ops.gluon._ptx import _tc_fence
from neuronspark.ops.gluon.pool_read_tile import LOG2E, _row, _window_term
from neuronspark.ops.triton import require_cuda, require_sm100

__all__ = ["BNK", "BMQ_V", "BMQ_K", "BM_Q", "BN_Q", "R", "R_V", "REGS_W", "pool_read_dv_gluon", "pool_read_dk_gluon",
           "pool_read_dq_gluon"]

# ------------------------------------------------------------------------------------------------ launch constants
BNK = 128        # key rows per CTA (one per thread of the compute partition)
BMQ_V = 32       # query rows per step of the value-gradient kernel
BMQ_K = 32       # query rows per step of the key-gradient kernel
BM_Q = 128       # query rows per CTA of the query-gradient kernel
BN_Q = 32        # key rows per step of the query-gradient kernel
R = 2            # ring slots (key-gradient and query-gradient kernels)
R_V = 3          # ring slots of the value-gradient kernel
REGS_W = 48      # registers of the load and MMA warps


# ================================================================================================ value gradient
@gluon.jit
def _kv_item(T, B, H, HQ, BNK: gl.constexpr, BMQ: gl.constexpr):
    """(key tile, b, key head, query heads of the key head, first query tile, steps) of this CTA, longest first."""
    pid = gl.program_id(0)
    nbh = B * H
    kt = pid // nbh
    bh = pid % nbh
    b = bh // H
    h = H - 1 - bh % H
    nq = 1 + (HQ - H) * (h == H - 1).to(gl.int32)
    qt0 = (kt * BNK) // BMQ
    steps = (gl.cdiv(T, BMQ) - qt0) * nq
    return kt, b, h, nq, qt0, steps


@gluon.jit
def _step_head(s, nq, h, H):
    """(query tile offset, query head) of step s: the key head's query heads h, then H .. HQ - 1."""
    r = s % nq
    return s // nq, gl.where(r == 0, h, H - 1 + r)


@gluon.jit
def _dv_compute(k_s, q_s, do_s, lse_s, cq_s, qt_s, p1_s, p2_s, st_tm, dv_tm, k_bar, ring_bar, ring_free, s_ready,
                s_free, p_ready, p_free, done, kd, qd, dod, ld, cd, td, KTOK, CUMT, SLOTS, DGV, T, B, H, HQ, scale,
                DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
                NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Default partition (4 warps, one key row per thread): P^T into shared memory, then the dV epilogue."""
    L: gl.constexpr = get_tmem_32x32b_reg_layout(BNK, BMQ, [BNK, BMQ], 4)
    LR: gl.constexpr = gl.SliceLayout(1, L)
    LC: gl.constexpr = gl.SliceLayout(0, L)
    kt, b, h, nq, qt0, steps = _kv_item(T, B, H, HQ, BNK, BMQ)
    j = kt * BNK + gl.arange(0, BNK, layout=LR)
    jm = j < T
    j64 = j.to(gl.int64)
    cum_k = gl.load(CUMT + (b * H + h).to(gl.int64) * T + j64, mask=jm, other=0.0)
    krow = KTOK + (b.to(gl.int64) * T + j64) * NMAX
    k0 = gl.load(krow, mask=jm, other=-2)
    k1 = k0
    k2 = k0
    k3 = k0
    if NMAX > 1:
        k1 = gl.load(krow + 1, mask=jm, other=-2)
    if NMAX > 2:
        k2 = gl.load(krow + 2, mask=jm, other=-2)
    if NMAX > 3:
        k3 = gl.load(krow + 3, mask=jm, other=-2)
    beta1 = gl.load(SLOTS + h * NBS + 1)
    beta2 = gl.load(SLOTS + h * NBS + 2)
    beta3 = gl.load(SLOTS + h * NBS + 3)
    beta4 = gl.load(SLOTS + h * NBS + 4)
    for s in range(steps):
        sb = s % 2
        sl = s % R
        dq, _i = _step_head(s, nq, h, H)
        qt = qt0 + dq
        mbarrier.wait(s_ready.index(sb), (s // 2) & 1)
        _tc_fence(1, 4)
        st = st_tm.index(sb).load(L)
        _tc_fence(0, 4)
        gl.thread_barrier()
        mbarrier.arrive(s_free.index(sb))
        mbarrier.wait(ring_bar.index(sl), (s // R) & 1)
        tt = qt * BMQ + gl.arange(0, BMQ, layout=LC)
        lse = _row(lse_s.index(sl), BMQ, gl.float32, LC)
        cum_q = _row(cq_s.index(sl), BMQ, gl.float32, LC)
        q0 = _row(qt_s.index(sl * NMAX), BMQ, gl.int32, LC)
        q1 = q0
        q2 = q0
        q3 = q0
        if NMAX > 1:
            q1 = _row(qt_s.index(sl * NMAX + 1), BMQ, gl.int32, LC)
        if NMAX > 2:
            q2 = _row(qt_s.index(sl * NMAX + 2), BMQ, gl.int32, LC)
        if NMAX > 3:
            q3 = _row(qt_s.index(sl * NMAX + 3), BMQ, gl.int32, LC)
        x, _m1, _m2, _m3, _m4 = _window_term(k0, k1, k2, k3, q0, q1, q2, q3, beta1, beta2, beta3, beta4, BNK, BMQ,
                                             L, ORDMASK, NMAX)
        diag = qt * BMQ < (kt + 1) * BNK
        if diag:
            x = gl.where(j[:, None] <= tt[None, :] - 1, x, 0.0)
        bias = ((cum_q[None, :] - cum_k[:, None]) + x).to(gl.bfloat16).to(gl.float32)
        p = gl.exp2(((st * scale + bias) - lse[None, :]) * LOG2E)
        if diag | ((qt + 1) * BMQ > T):
            p = gl.where((j[:, None] <= tt[None, :]) & (tt[None, :] < T), p, 0.0)
        p1 = p.to(gl.bfloat16)
        p2 = (p - p1.to(gl.float32)).to(gl.bfloat16)
        if s > 0:
            mbarrier.wait(p_free, (s - 1) & 1)
        p1_s.store(p1)
        p2_s.store(p2)
        fence_async_shared()
        gl.thread_barrier()
        mbarrier.arrive(p_ready)
    # epilogue: dV in bf16
    mbarrier.wait(done, 0)
    _tc_fence(1, 4)
    LO: gl.constexpr = get_tmem_32x32b_reg_layout(BNK, 64, [BNK, 64], 4)
    jo = kt * BNK + gl.arange(0, BNK, layout=gl.SliceLayout(1, LO))
    orow = ((jo.to(gl.int64) * B + b) * H + h) * DV
    for c in gl.static_range(DV // 64):
        g = dv_tm.slice(c * 64, 64).load(LO)
        ptr = orow[:, None] + (c * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, LO)))[None, :]
        gl.store(DGV + ptr, g.to(gl.bfloat16), mask=(jo < T)[:, None])


@gluon.jit
def _dv_load(k_s, q_s, do_s, lse_s, cq_s, qt_s, p1_s, p2_s, st_tm, dv_tm, k_bar, ring_bar, ring_free, s_ready, s_free,
             p_ready, p_free, done, kd, qd, dod, ld, cd, td, KTOK, CUMT, SLOTS, DGV, T, B, H, HQ, scale,
             DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
             NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Load warp: the key tile, then per step the query head's queries, output gradients, log-sum-exp, the clocks and
    the query id windows of the query tile."""
    kt, b, h, nq, qt0, steps = _kv_item(T, B, H, HQ, BNK, BMQ)
    mbarrier.expect(k_bar, BNK * DK * 2)
    tma.async_copy_global_to_shared(kd, [kt * BNK, (b * H + h) * DK], k_bar, k_s)
    for s in range(steps):
        sl = s % R
        dq, i = _step_head(s, nq, h, H)
        t0 = (qt0 + dq) * BMQ
        mbarrier.wait(ring_free.index(sl), ((s // R) & 1) ^ 1)
        mbarrier.expect(ring_bar.index(sl), BMQ * (DK + DV) * 2 + 2 * BMQ * 4 + NMAX * BMQ * 4)
        tma.async_copy_global_to_shared(qd, [t0, (b * HQ + i) * DK], ring_bar.index(sl), q_s.index(sl))
        tma.async_copy_global_to_shared(dod, [t0, (b * HQ + i) * DV], ring_bar.index(sl), do_s.index(sl))
        tma.async_copy_global_to_shared(ld, [b * HQ + i, t0], ring_bar.index(sl), lse_s.index(sl))
        tma.async_copy_global_to_shared(cd, [b * H + h, t0], ring_bar.index(sl), cq_s.index(sl))
        for n in gl.static_range(NMAX):
            tma.async_copy_global_to_shared(td, [b * NMAX + n, t0], ring_bar.index(sl), qt_s.index(sl * NMAX + n))


@gluon.jit
def _dv_mma(k_s, q_s, do_s, lse_s, cq_s, qt_s, p1_s, p2_s, st_tm, dv_tm, k_bar, ring_bar, ring_free, s_ready, s_free,
            p_ready, p_free, done, kd, qd, dod, ld, cd, td, KTOK, CUMT, SLOTS, DGV, T, B, H, HQ, scale,
            DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
            NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """MMA warp: S^T = K Q^T one step ahead, then dV += P2^T dO and dV += P1^T dO once the step's P^T is stored."""
    kt, b, h, nq, qt0, steps = _kv_item(T, B, H, HQ, BNK, BMQ)
    mbarrier.wait(k_bar, 0)
    mbarrier.wait(ring_bar.index(0), 0)
    tcgen05_mma(k_s, q_s.index(0).permute((1, 0)), st_tm.index(0), use_acc=False, mbarriers=[s_ready.index(0)])
    for s in range(steps):
        if s + 1 < steps:
            sl1 = (s + 1) % R
            sb1 = (s + 1) % 2
            mbarrier.wait(ring_bar.index(sl1), ((s + 1) // R) & 1)
            mbarrier.wait(s_free.index(sb1), (((s + 1) // 2) & 1) ^ 1)
            tcgen05_mma(k_s, q_s.index(sl1).permute((1, 0)), st_tm.index(sb1), use_acc=False,
                        mbarriers=[s_ready.index(sb1)])
        mbarrier.wait(p_ready, s & 1)
        _tc_fence(1, 1)
        sl = s % R
        tcgen05_mma(p2_s, do_s.index(sl), dv_tm, use_acc=s > 0)
        tcgen05_mma(p1_s, do_s.index(sl), dv_tm, use_acc=True, mbarriers=[p_free, ring_free.index(sl)])
    tcgen05_commit(done)


@gluon.jit
def _dv_kernel(kd, qd, dod, ld, cd, td, KTOK, CUMT, SLOTS, DGV, T, B, H, HQ, scale,
               DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
               NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr, REGS_W: gl.constexpr):
    PL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BNK, BMQ], gl.bfloat16)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    k_s = gl.allocate_shared_memory(gl.bfloat16, [BNK, DK], kd.layout)
    q_s = gl.allocate_shared_memory(gl.bfloat16, [R, BMQ, DK], qd.layout)
    do_s = gl.allocate_shared_memory(gl.bfloat16, [R, BMQ, DV], dod.layout)
    lse_s = gl.allocate_shared_memory(gl.float32, [R, 1, BMQ], ld.layout)
    cq_s = gl.allocate_shared_memory(gl.float32, [R, 1, BMQ], cd.layout)
    qt_s = gl.allocate_shared_memory(gl.int32, [R * NMAX, 1, BMQ], td.layout)
    p1_s = gl.allocate_shared_memory(gl.bfloat16, [BNK, BMQ], PL)
    p2_s = gl.allocate_shared_memory(gl.bfloat16, [BNK, BMQ], PL)
    k_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
    ring_bar = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    ring_free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    s_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    p_ready = gl.allocate_shared_memory(gl.int64, [1], BL)
    p_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    done = gl.allocate_shared_memory(gl.int64, [1], BL)
    mbarrier.init(k_bar, count=1)
    for k in gl.static_range(R):
        mbarrier.init(ring_bar.index(k), count=1)
        mbarrier.init(ring_free.index(k), count=1)
    for k in gl.static_range(2):
        mbarrier.init(s_ready.index(k), count=1)
        mbarrier.init(s_free.index(k), count=1)
    mbarrier.init(p_ready, count=1)
    mbarrier.init(p_free, count=1)
    mbarrier.init(done, count=1)
    fence_async_shared()
    st_tm = allocate_tensor_memory(gl.float32, [2, BNK, BMQ], TensorMemoryLayout(block=(BNK, BMQ), unpacked=True))
    dv_tm = allocate_tensor_memory(gl.float32, [BNK, DV], TensorMemoryLayout(block=(BNK, DV), unpacked=True))
    # the argument tuples are written out in the call: a tuple bound to a name turns the constexprs into tensors
    gl.warp_specialize(
        (k_s, q_s, do_s, lse_s, cq_s, qt_s, p1_s, p2_s, st_tm, dv_tm, k_bar, ring_bar, ring_free, s_ready, s_free,
         p_ready, p_free, done, kd, qd, dod, ld, cd, td, KTOK, CUMT, SLOTS, DGV, T, B, H, HQ, scale,
         DK, DV, BNK, BMQ, R, NMAX, NBS, ORDMASK),
        _dv_compute,
        (k_s, q_s, do_s, lse_s, cq_s, qt_s, p1_s, p2_s, st_tm, dv_tm, k_bar, ring_bar, ring_free, s_ready, s_free,
         p_ready, p_free, done, kd, qd, dod, ld, cd, td, KTOK, CUMT, SLOTS, DGV, T, B, H, HQ, scale,
         DK, DV, BNK, BMQ, R, NMAX, NBS, ORDMASK),
        [_dv_load, _dv_mma], [1, 1], [REGS_W, REGS_W])


# ================================================================================================ key gradient
@gluon.jit
def _dk_compute(k_s, v_s, q_s, do_s, lse_s, dl_s, cq_s, qt_s, d1_s, d2_s, st_tm, dp_tm, dk_tm, kv_bar, ring_bar,
                ring_free, s_ready, s_free, p_ready, p_free, done, kd, vd, qd, dod, ld, dd, cd, td, KTOK, CUMT, SLOTS,
                DK_, DCUM, DBETA, T, B, H, HQ, scale,
                DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
                NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Default partition (4 warps, one key row per thread): dS^T into shared memory, the clock's column sums and the
    beta sums of the row, then the dK epilogue."""
    L: gl.constexpr = get_tmem_32x32b_reg_layout(BNK, BMQ, [BNK, BMQ], 4)
    LR: gl.constexpr = gl.SliceLayout(1, L)
    LC: gl.constexpr = gl.SliceLayout(0, L)
    kt, b, h, nq, qt0, steps = _kv_item(T, B, H, HQ, BNK, BMQ)
    j = kt * BNK + gl.arange(0, BNK, layout=LR)
    jm = j < T
    j64 = j.to(gl.int64)
    cum_k = gl.load(CUMT + (b * H + h).to(gl.int64) * T + j64, mask=jm, other=0.0)
    krow = KTOK + (b.to(gl.int64) * T + j64) * NMAX
    k0 = gl.load(krow, mask=jm, other=-2)
    k1 = k0
    k2 = k0
    k3 = k0
    if NMAX > 1:
        k1 = gl.load(krow + 1, mask=jm, other=-2)
    if NMAX > 2:
        k2 = gl.load(krow + 2, mask=jm, other=-2)
    if NMAX > 3:
        k3 = gl.load(krow + 3, mask=jm, other=-2)
    beta1 = gl.load(SLOTS + h * NBS + 1)
    beta2 = gl.load(SLOTS + h * NBS + 2)
    beta3 = gl.load(SLOTS + h * NBS + 3)
    beta4 = gl.load(SLOTS + h * NBS + 4)
    col = gl.zeros([BNK], gl.float32, layout=LR)
    gb1 = gl.zeros([BNK], gl.float32, layout=LR)
    gb2 = gl.zeros([BNK], gl.float32, layout=LR)
    gb3 = gl.zeros([BNK], gl.float32, layout=LR)
    gb4 = gl.zeros([BNK], gl.float32, layout=LR)
    bias = gl.zeros([BNK, BMQ], gl.float32, layout=L)
    none = bias < -1.0
    m1 = none
    m2 = none
    m3 = none
    m4 = none
    for s in range(steps):
        sb = s % 2
        sl = s % R
        dq, _i = _step_head(s, nq, h, H)
        qt = qt0 + dq
        mbarrier.wait(s_ready.index(sb), (s // 2) & 1)
        _tc_fence(1, 4)
        st = st_tm.index(sb).load(L)
        dpt = dp_tm.index(sb).load(L)
        _tc_fence(0, 4)
        gl.thread_barrier()
        mbarrier.arrive(s_free.index(sb))
        mbarrier.wait(ring_bar.index(sl), (s // R) & 1)
        tt = qt * BMQ + gl.arange(0, BMQ, layout=LC)
        lse = _row(lse_s.index(sl), BMQ, gl.float32, LC)
        dlt = _row(dl_s.index(sl), BMQ, gl.float32, LC)
        diag = qt * BMQ < (kt + 1) * BNK
        if s % nq == 0:
            # the query tile's bias and order masks, shared by the key head's query heads
            cum_q = _row(cq_s.index(sl), BMQ, gl.float32, LC)
            q0 = _row(qt_s.index(sl * NMAX), BMQ, gl.int32, LC)
            q1 = q0
            q2 = q0
            q3 = q0
            if NMAX > 1:
                q1 = _row(qt_s.index(sl * NMAX + 1), BMQ, gl.int32, LC)
            if NMAX > 2:
                q2 = _row(qt_s.index(sl * NMAX + 2), BMQ, gl.int32, LC)
            if NMAX > 3:
                q3 = _row(qt_s.index(sl * NMAX + 3), BMQ, gl.int32, LC)
            x, m1, m2, m3, m4 = _window_term(k0, k1, k2, k3, q0, q1, q2, q3, beta1, beta2, beta3, beta4, BNK, BMQ,
                                             L, ORDMASK, NMAX)
            if diag:
                past = j[:, None] <= tt[None, :] - 1
                x = gl.where(past, x, 0.0)
                m1 = m1 & past
                m2 = m2 & past
                m3 = m3 & past
                m4 = m4 & past
            bias = ((cum_q[None, :] - cum_k[:, None]) + x).to(gl.bfloat16).to(gl.float32)
        p = gl.exp2(((st * scale + bias) - lse[None, :]) * LOG2E)
        if diag | ((qt + 1) * BMQ > T):
            p = gl.where((j[:, None] <= tt[None, :]) & (tt[None, :] < T), p, 0.0)
        ds = p * (dpt - dlt[None, :])
        col = col + gl.sum(ds, axis=1)
        if (ORDMASK >> 1) & 1:
            gb1 = gb1 + gl.sum(gl.where(m1, ds, 0.0), axis=1)
        if (ORDMASK >> 2) & 1:
            gb2 = gb2 + gl.sum(gl.where(m2, ds, 0.0), axis=1)
        if (ORDMASK >> 3) & 1:
            gb3 = gb3 + gl.sum(gl.where(m3, ds, 0.0), axis=1)
        if (ORDMASK >> 4) & 1:
            gb4 = gb4 + gl.sum(gl.where(m4, ds, 0.0), axis=1)
        d1 = ds.to(gl.bfloat16)
        d2 = (ds - d1.to(gl.float32)).to(gl.bfloat16)
        if s > 0:
            mbarrier.wait(p_free, (s - 1) & 1)
        d1_s.store(d1)
        d2_s.store(d2)
        fence_async_shared()
        gl.thread_barrier()
        mbarrier.arrive(p_ready)
    # the clock gradient (minus the column sums) and the row's beta sums
    row = (b * H + h).to(gl.int64) * T + j64
    gl.store(DCUM + (j64 * B + b) * H + h, -col, mask=jm)
    pb = DBETA + row * NBS
    gl.store(pb, gl.zeros([BNK], gl.float32, layout=LR), mask=jm)
    gl.store(pb + 1, gb1, mask=jm)
    gl.store(pb + 2, gb2, mask=jm)
    gl.store(pb + 3, gb3, mask=jm)
    gl.store(pb + 4, gb4, mask=jm)
    # epilogue: dK = scale dS^T Q in fp32
    mbarrier.wait(done, 0)
    _tc_fence(1, 4)
    LO: gl.constexpr = get_tmem_32x32b_reg_layout(BNK, 64, [BNK, 64], 4)
    jo = kt * BNK + gl.arange(0, BNK, layout=gl.SliceLayout(1, LO))
    orow = ((jo.to(gl.int64) * B + b) * H + h) * DK
    for c in gl.static_range(DK // 64):
        g = dk_tm.slice(c * 64, 64).load(LO) * scale
        ptr = orow[:, None] + (c * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, LO)))[None, :]
        gl.store(DK_ + ptr, g, mask=(jo < T)[:, None])


@gluon.jit
def _dk_load(k_s, v_s, q_s, do_s, lse_s, dl_s, cq_s, qt_s, d1_s, d2_s, st_tm, dp_tm, dk_tm, kv_bar, ring_bar,
             ring_free, s_ready, s_free, p_ready, p_free, done, kd, vd, qd, dod, ld, dd, cd, td, KTOK, CUMT, SLOTS,
             DK_, DCUM, DBETA, T, B, H, HQ, scale,
             DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
             NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Load warp: the key and value tiles, then per step the query head's queries, output gradients, log-sum-exp and
    D, the clocks and the query id windows of the query tile."""
    kt, b, h, nq, qt0, steps = _kv_item(T, B, H, HQ, BNK, BMQ)
    mbarrier.expect(kv_bar, BNK * (DK + DV) * 2)
    tma.async_copy_global_to_shared(kd, [kt * BNK, (b * H + h) * DK], kv_bar, k_s)
    tma.async_copy_global_to_shared(vd, [kt * BNK, (b * H + h) * DV], kv_bar, v_s)
    for s in range(steps):
        sl = s % R
        dq, i = _step_head(s, nq, h, H)
        t0 = (qt0 + dq) * BMQ
        mbarrier.wait(ring_free.index(sl), ((s // R) & 1) ^ 1)
        mbarrier.expect(ring_bar.index(sl), BMQ * (DK + DV) * 2 + 3 * BMQ * 4 + NMAX * BMQ * 4)
        tma.async_copy_global_to_shared(qd, [t0, (b * HQ + i) * DK], ring_bar.index(sl), q_s.index(sl))
        tma.async_copy_global_to_shared(dod, [t0, (b * HQ + i) * DV], ring_bar.index(sl), do_s.index(sl))
        tma.async_copy_global_to_shared(ld, [b * HQ + i, t0], ring_bar.index(sl), lse_s.index(sl))
        tma.async_copy_global_to_shared(dd, [b * HQ + i, t0], ring_bar.index(sl), dl_s.index(sl))
        tma.async_copy_global_to_shared(cd, [b * H + h, t0], ring_bar.index(sl), cq_s.index(sl))
        for n in gl.static_range(NMAX):
            tma.async_copy_global_to_shared(td, [b * NMAX + n, t0], ring_bar.index(sl), qt_s.index(sl * NMAX + n))


@gluon.jit
def _dk_mma(k_s, v_s, q_s, do_s, lse_s, dl_s, cq_s, qt_s, d1_s, d2_s, st_tm, dp_tm, dk_tm, kv_bar, ring_bar,
            ring_free, s_ready, s_free, p_ready, p_free, done, kd, vd, qd, dod, ld, dd, cd, td, KTOK, CUMT, SLOTS,
            DK_, DCUM, DBETA, T, B, H, HQ, scale,
            DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
            NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """MMA warp: S^T = K Q^T and dP^T = V dO^T one step ahead, then dK += dS2^T Q and dK += dS1^T Q once the step's
    dS^T is stored."""
    kt, b, h, nq, qt0, steps = _kv_item(T, B, H, HQ, BNK, BMQ)
    mbarrier.wait(kv_bar, 0)
    mbarrier.wait(ring_bar.index(0), 0)
    tcgen05_mma(k_s, q_s.index(0).permute((1, 0)), st_tm.index(0), use_acc=False)
    tcgen05_mma(v_s, do_s.index(0).permute((1, 0)), dp_tm.index(0), use_acc=False, mbarriers=[s_ready.index(0)])
    for s in range(steps):
        if s + 1 < steps:
            sl1 = (s + 1) % R
            sb1 = (s + 1) % 2
            mbarrier.wait(ring_bar.index(sl1), ((s + 1) // R) & 1)
            mbarrier.wait(s_free.index(sb1), (((s + 1) // 2) & 1) ^ 1)
            tcgen05_mma(k_s, q_s.index(sl1).permute((1, 0)), st_tm.index(sb1), use_acc=False)
            tcgen05_mma(v_s, do_s.index(sl1).permute((1, 0)), dp_tm.index(sb1), use_acc=False,
                        mbarriers=[s_ready.index(sb1)])
        mbarrier.wait(p_ready, s & 1)
        _tc_fence(1, 1)
        sl = s % R
        tcgen05_mma(d2_s, q_s.index(sl), dk_tm, use_acc=s > 0)
        tcgen05_mma(d1_s, q_s.index(sl), dk_tm, use_acc=True, mbarriers=[p_free, ring_free.index(sl)])
    tcgen05_commit(done)


@gluon.jit
def _dk_kernel(kd, vd, qd, dod, ld, dd, cd, td, KTOK, CUMT, SLOTS, DK_, DCUM, DBETA, T, B, H, HQ, scale,
               DK: gl.constexpr, DV: gl.constexpr, BNK: gl.constexpr, BMQ: gl.constexpr, R: gl.constexpr,
               NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr, REGS_W: gl.constexpr):
    PL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BNK, BMQ], gl.bfloat16)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    k_s = gl.allocate_shared_memory(gl.bfloat16, [BNK, DK], kd.layout)
    v_s = gl.allocate_shared_memory(gl.bfloat16, [BNK, DV], vd.layout)
    q_s = gl.allocate_shared_memory(gl.bfloat16, [R, BMQ, DK], qd.layout)
    do_s = gl.allocate_shared_memory(gl.bfloat16, [R, BMQ, DV], dod.layout)
    lse_s = gl.allocate_shared_memory(gl.float32, [R, 1, BMQ], ld.layout)
    dl_s = gl.allocate_shared_memory(gl.float32, [R, 1, BMQ], dd.layout)
    cq_s = gl.allocate_shared_memory(gl.float32, [R, 1, BMQ], cd.layout)
    qt_s = gl.allocate_shared_memory(gl.int32, [R * NMAX, 1, BMQ], td.layout)
    d1_s = gl.allocate_shared_memory(gl.bfloat16, [BNK, BMQ], PL)
    d2_s = gl.allocate_shared_memory(gl.bfloat16, [BNK, BMQ], PL)
    kv_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
    ring_bar = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    ring_free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    s_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    p_ready = gl.allocate_shared_memory(gl.int64, [1], BL)
    p_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    done = gl.allocate_shared_memory(gl.int64, [1], BL)
    mbarrier.init(kv_bar, count=1)
    for k in gl.static_range(R):
        mbarrier.init(ring_bar.index(k), count=1)
        mbarrier.init(ring_free.index(k), count=1)
    for k in gl.static_range(2):
        mbarrier.init(s_ready.index(k), count=1)
        mbarrier.init(s_free.index(k), count=1)
    mbarrier.init(p_ready, count=1)
    mbarrier.init(p_free, count=1)
    mbarrier.init(done, count=1)
    fence_async_shared()
    TL: gl.constexpr = TensorMemoryLayout(block=(BNK, BMQ), unpacked=True)
    st_tm = allocate_tensor_memory(gl.float32, [2, BNK, BMQ], TL)
    dp_tm = allocate_tensor_memory(gl.float32, [2, BNK, BMQ], TL)
    dk_tm = allocate_tensor_memory(gl.float32, [BNK, DK], TensorMemoryLayout(block=(BNK, DK), unpacked=True))
    # the argument tuples are written out in the call: a tuple bound to a name turns the constexprs into tensors
    gl.warp_specialize(
        (k_s, v_s, q_s, do_s, lse_s, dl_s, cq_s, qt_s, d1_s, d2_s, st_tm, dp_tm, dk_tm, kv_bar, ring_bar, ring_free,
         s_ready, s_free, p_ready, p_free, done, kd, vd, qd, dod, ld, dd, cd, td, KTOK, CUMT, SLOTS, DK_, DCUM, DBETA,
         T, B, H, HQ, scale, DK, DV, BNK, BMQ, R, NMAX, NBS, ORDMASK),
        _dk_compute,
        (k_s, v_s, q_s, do_s, lse_s, dl_s, cq_s, qt_s, d1_s, d2_s, st_tm, dp_tm, dk_tm, kv_bar, ring_bar, ring_free,
         s_ready, s_free, p_ready, p_free, done, kd, vd, qd, dod, ld, dd, cd, td, KTOK, CUMT, SLOTS, DK_, DCUM, DBETA,
         T, B, H, HQ, scale, DK, DV, BNK, BMQ, R, NMAX, NBS, ORDMASK),
        [_dk_load, _dk_mma], [1, 1], [REGS_W, REGS_W])



# ================================================================================================ query gradient
@gluon.jit
def _q_item(T, B, HQ, H, BM: gl.constexpr, BN: gl.constexpr):
    """(query tile, b, query head, key head, key tiles) of this CTA, longest first."""
    pid = gl.program_id(0)
    nbi = B * HQ
    mt = gl.cdiv(T, BM) - 1 - pid // nbi
    bi = pid % nbi
    b = bi // HQ
    i = bi % HQ
    h = gl.minimum(i, H - 1)
    return mt, b, i, h, gl.cdiv(gl.minimum((mt + 1) * BM, T), BN)


@gluon.jit
def _dq_compute(q_s, do_s, k_s, v_s, ck_s, kt_s, d1_s, d2_s, s_tm, dp_tm, dq_tm, q_bar, ring_bar, ring_free, s_ready,
                s_free, p_ready, p_free, done, qd, dod, kd, vd, cd, td, QTOK, CUMT, SLOTS, LSE, DELTA, DQ, T, B, H, HQ,
                scale, DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
                NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Default partition (4 warps, one query row per thread): dS into shared memory, then the dQ epilogue."""
    L: gl.constexpr = get_tmem_32x32b_reg_layout(BM, BN, [BM, BN], 4)
    LR: gl.constexpr = gl.SliceLayout(1, L)
    LC: gl.constexpr = gl.SliceLayout(0, L)
    mt, b, i, h, n_kv = _q_item(T, B, HQ, H, BM, BN)
    t = mt * BM + gl.arange(0, BM, layout=LR)
    tm = t < T
    t64 = t.to(gl.int64)
    cum_t = gl.load(CUMT + (b * H + h).to(gl.int64) * T + t64, mask=tm, other=0.0)
    lse = gl.load(LSE + (b * HQ + i).to(gl.int64) * T + t64, mask=tm, other=0.0)
    dlt = gl.load(DELTA + (b * HQ + i).to(gl.int64) * T + t64, mask=tm, other=0.0)
    qrow = QTOK + (b.to(gl.int64) * T + t64) * NMAX
    q0 = gl.load(qrow, mask=tm, other=-1)
    q1 = q0
    q2 = q0
    q3 = q0
    if NMAX > 1:
        q1 = gl.load(qrow + 1, mask=tm, other=-1)
    if NMAX > 2:
        q2 = gl.load(qrow + 2, mask=tm, other=-1)
    if NMAX > 3:
        q3 = gl.load(qrow + 3, mask=tm, other=-1)
    beta1 = gl.load(SLOTS + h * NBS + 1)
    beta2 = gl.load(SLOTS + h * NBS + 2)
    beta3 = gl.load(SLOTS + h * NBS + 3)
    beta4 = gl.load(SLOTS + h * NBS + 4)
    for j in range(n_kv):
        sb = j % 2
        sl = j % R
        mbarrier.wait(s_ready.index(sb), (j // 2) & 1)
        _tc_fence(1, 4)
        s = s_tm.index(sb).load(L)
        dp = dp_tm.index(sb).load(L)
        _tc_fence(0, 4)
        gl.thread_barrier()
        mbarrier.arrive(s_free.index(sb))
        mbarrier.wait(ring_bar.index(sl), (j // R) & 1)
        cum_k = _row(ck_s.index(sl), BN, gl.float32, LC)
        jj = j * BN + gl.arange(0, BN, layout=LC)
        k0 = _row(kt_s.index(sl * NMAX), BN, gl.int32, LC)
        k1 = k0
        k2 = k0
        k3 = k0
        if NMAX > 1:
            k1 = _row(kt_s.index(sl * NMAX + 1), BN, gl.int32, LC)
        if NMAX > 2:
            k2 = _row(kt_s.index(sl * NMAX + 2), BN, gl.int32, LC)
        if NMAX > 3:
            k3 = _row(kt_s.index(sl * NMAX + 3), BN, gl.int32, LC)
        x, _m1, _m2, _m3, _m4 = _window_term(q0, q1, q2, q3, k0, k1, k2, k3, beta1, beta2, beta3, beta4, BM, BN, L,
                                             ORDMASK, NMAX)
        diag = (j + 1) * BN > mt * BM
        if diag:
            x = gl.where(jj[None, :] <= t[:, None] - 1, x, 0.0)
        bias = ((cum_t[:, None] - cum_k[None, :]) + x).to(gl.bfloat16).to(gl.float32)
        p = gl.exp2(((s * scale + bias) - lse[:, None]) * LOG2E)
        if diag:
            p = gl.where((jj[None, :] <= t[:, None]) & (jj[None, :] < T), p, 0.0)
        ds = p * (dp - dlt[:, None])
        d1 = ds.to(gl.bfloat16)
        d2 = (ds - d1.to(gl.float32)).to(gl.bfloat16)
        if j > 0:
            mbarrier.wait(p_free, (j - 1) & 1)
        d1_s.store(d1)
        d2_s.store(d2)
        fence_async_shared()
        gl.thread_barrier()
        mbarrier.arrive(p_ready)
    # epilogue: dQ = scale dS K in fp32
    mbarrier.wait(done, 0)
    _tc_fence(1, 4)
    LO: gl.constexpr = get_tmem_32x32b_reg_layout(BM, 64, [BM, 64], 4)
    to = mt * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, LO))
    orow = ((to.to(gl.int64) * B + b) * HQ + i) * DK
    for c in gl.static_range(DK // 64):
        g = dq_tm.slice(c * 64, 64).load(LO) * scale
        ptr = orow[:, None] + (c * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, LO)))[None, :]
        gl.store(DQ + ptr, g, mask=(to < T)[:, None])


@gluon.jit
def _dq_load(q_s, do_s, k_s, v_s, ck_s, kt_s, d1_s, d2_s, s_tm, dp_tm, dq_tm, q_bar, ring_bar, ring_free, s_ready,
             s_free, p_ready, p_free, done, qd, dod, kd, vd, cd, td, QTOK, CUMT, SLOTS, LSE, DELTA, DQ, T, B, H, HQ,
             scale, DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
             NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Load warp: the query and output-gradient tiles, then per key tile the keys, values, clocks and key id windows."""
    mt, b, i, h, n_kv = _q_item(T, B, HQ, H, BM, BN)
    mbarrier.expect(q_bar, BM * (DK + DV) * 2)
    tma.async_copy_global_to_shared(qd, [mt * BM, (b * HQ + i) * DK], q_bar, q_s)
    tma.async_copy_global_to_shared(dod, [mt * BM, (b * HQ + i) * DV], q_bar, do_s)
    for j in range(n_kv):
        sl = j % R
        mbarrier.wait(ring_free.index(sl), ((j // R) & 1) ^ 1)
        mbarrier.expect(ring_bar.index(sl), BN * (DK + DV) * 2 + BN * 4 + NMAX * BN * 4)
        tma.async_copy_global_to_shared(kd, [j * BN, (b * H + h) * DK], ring_bar.index(sl), k_s.index(sl))
        tma.async_copy_global_to_shared(vd, [j * BN, (b * H + h) * DV], ring_bar.index(sl), v_s.index(sl))
        tma.async_copy_global_to_shared(cd, [b * H + h, j * BN], ring_bar.index(sl), ck_s.index(sl))
        for n in gl.static_range(NMAX):
            tma.async_copy_global_to_shared(td, [b * NMAX + n, j * BN], ring_bar.index(sl), kt_s.index(sl * NMAX + n))


@gluon.jit
def _dq_mma(q_s, do_s, k_s, v_s, ck_s, kt_s, d1_s, d2_s, s_tm, dp_tm, dq_tm, q_bar, ring_bar, ring_free, s_ready,
            s_free, p_ready, p_free, done, qd, dod, kd, vd, cd, td, QTOK, CUMT, SLOTS, LSE, DELTA, DQ, T, B, H, HQ,
            scale, DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
            NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """MMA warp: S = Q K^T and dP = dO V^T one key tile ahead, then dQ += dS2 K and dQ += dS1 K once the tile's dS is
    stored."""
    mt, b, i, h, n_kv = _q_item(T, B, HQ, H, BM, BN)
    mbarrier.wait(q_bar, 0)
    mbarrier.wait(ring_bar.index(0), 0)
    tcgen05_mma(q_s, k_s.index(0).permute((1, 0)), s_tm.index(0), use_acc=False)
    tcgen05_mma(do_s, v_s.index(0).permute((1, 0)), dp_tm.index(0), use_acc=False, mbarriers=[s_ready.index(0)])
    for j in range(n_kv):
        if j + 1 < n_kv:
            sl1 = (j + 1) % R
            sb1 = (j + 1) % 2
            mbarrier.wait(ring_bar.index(sl1), ((j + 1) // R) & 1)
            mbarrier.wait(s_free.index(sb1), (((j + 1) // 2) & 1) ^ 1)
            tcgen05_mma(q_s, k_s.index(sl1).permute((1, 0)), s_tm.index(sb1), use_acc=False)
            tcgen05_mma(do_s, v_s.index(sl1).permute((1, 0)), dp_tm.index(sb1), use_acc=False,
                        mbarriers=[s_ready.index(sb1)])
        mbarrier.wait(p_ready, j & 1)
        _tc_fence(1, 1)
        sl = j % R
        tcgen05_mma(d2_s, k_s.index(sl), dq_tm, use_acc=j > 0)
        tcgen05_mma(d1_s, k_s.index(sl), dq_tm, use_acc=True, mbarriers=[p_free, ring_free.index(sl)])
    tcgen05_commit(done)


@gluon.jit
def _dq_kernel(qd, dod, kd, vd, cd, td, QTOK, CUMT, SLOTS, LSE, DELTA, DQ, T, B, H, HQ, scale,
               DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
               NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr, REGS_W: gl.constexpr):
    PL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    q_s = gl.allocate_shared_memory(gl.bfloat16, [BM, DK], qd.layout)
    do_s = gl.allocate_shared_memory(gl.bfloat16, [BM, DV], dod.layout)
    k_s = gl.allocate_shared_memory(gl.bfloat16, [R, BN, DK], kd.layout)
    v_s = gl.allocate_shared_memory(gl.bfloat16, [R, BN, DV], vd.layout)
    ck_s = gl.allocate_shared_memory(gl.float32, [R, 1, BN], cd.layout)
    kt_s = gl.allocate_shared_memory(gl.int32, [R * NMAX, 1, BN], td.layout)
    d1_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BN], PL)
    d2_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BN], PL)
    q_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
    ring_bar = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    ring_free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    s_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    p_ready = gl.allocate_shared_memory(gl.int64, [1], BL)
    p_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    done = gl.allocate_shared_memory(gl.int64, [1], BL)
    mbarrier.init(q_bar, count=1)
    for k in gl.static_range(R):
        mbarrier.init(ring_bar.index(k), count=1)
        mbarrier.init(ring_free.index(k), count=1)
    for k in gl.static_range(2):
        mbarrier.init(s_ready.index(k), count=1)
        mbarrier.init(s_free.index(k), count=1)
    mbarrier.init(p_ready, count=1)
    mbarrier.init(p_free, count=1)
    mbarrier.init(done, count=1)
    fence_async_shared()
    TL: gl.constexpr = TensorMemoryLayout(block=(BM, BN), unpacked=True)
    s_tm = allocate_tensor_memory(gl.float32, [2, BM, BN], TL)
    dp_tm = allocate_tensor_memory(gl.float32, [2, BM, BN], TL)
    dq_tm = allocate_tensor_memory(gl.float32, [BM, DK], TensorMemoryLayout(block=(BM, DK), unpacked=True))
    # the argument tuples are written out in the call: a tuple bound to a name turns the constexprs into tensors
    gl.warp_specialize(
        (q_s, do_s, k_s, v_s, ck_s, kt_s, d1_s, d2_s, s_tm, dp_tm, dq_tm, q_bar, ring_bar, ring_free, s_ready, s_free,
         p_ready, p_free, done, qd, dod, kd, vd, cd, td, QTOK, CUMT, SLOTS, LSE, DELTA, DQ, T, B, H, HQ, scale,
         DK, DV, BM, BN, R, NMAX, NBS, ORDMASK),
        _dq_compute,
        (q_s, do_s, k_s, v_s, ck_s, kt_s, d1_s, d2_s, s_tm, dp_tm, dq_tm, q_bar, ring_bar, ring_free, s_ready, s_free,
         p_ready, p_free, done, qd, dod, kd, vd, cd, td, QTOK, CUMT, SLOTS, LSE, DELTA, DQ, T, B, H, HQ, scale,
         DK, DV, BM, BN, R, NMAX, NBS, ORDMASK),
        [_dq_load, _dq_mma], [1, 1], [REGS_W, REGS_W])



# ================================================================================================ launchers
def _row_desc(x: torch.Tensor, block: int) -> TensorDescriptor:
    """One-row boxes (1, block) of a 2-D fp32 / int32 tensor, linear in shared memory."""
    rows, cols = (int(s) for s in x.shape)
    return TensorDescriptor(x, [rows, cols], [cols, 1], [1, block],
                            gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2))


def _tile_desc(x: torch.Tensor, block_rows: int) -> TensorDescriptor:
    """(block_rows, d) tiles of the (T, B, heads, d) bf16 tensor x: head i of batch row b at column (b heads + i) d."""
    T, B, n, d = (int(s) for s in x.shape)
    return TensorDescriptor(x, [T, B * n * d], [B * n * d, 1], [block_rows, d],
                            gl.NVMMASharedLayout.get_default_for([block_rows, d], gl.bfloat16))


def pool_read_dv_gluon(q16: torch.Tensor, k16: torch.Tensor, do16: torch.Tensor, lse: torch.Tensor,
                       cum_t: torch.Tensor, qtok_t: torch.Tensor, ktok: torch.Tensor, slots: torch.Tensor,
                       ordmask: int, scale: float, dv: int) -> torch.Tensor:
    """The tagged values' gradient (T, B, H, dv) bf16 of the pool read (see the module docstring): q16 (T, B, HQ, dk)
    / k16 (T, B, H, dk) bf16, do16 (T, B, HQ, dv) bf16, lse (B, HQ, T) fp32, cum_t (B H, T) fp32, qtok_t (B n_max, T)
    int32, ktok (B, T, n_max) int32, slots (H, NB) fp32."""
    require_cuda(q16, "pool_read_dv_gluon")
    require_sm100(q16.device, "pool_read_dv_gluon")
    T, B, HQ, dk = (int(s) for s in q16.shape)
    H = int(k16.shape[2])
    NMAX = int(ktok.shape[-1])
    for name, x, dt in (("q16", q16, torch.bfloat16), ("k16", k16, torch.bfloat16), ("do16", do16, torch.bfloat16),
                        ("lse", lse, torch.float32), ("cum_t", cum_t, torch.float32), ("qtok_t", qtok_t, torch.int32),
                        ("ktok", ktok, torch.int32), ("slots", slots, torch.float32)):
        if x.dtype != dt or not x.is_contiguous():
            raise RuntimeError(f"neuronspark.pool_read_dv_gluon: {name} must be a contiguous {dt} tensor")
    if (tuple(do16.shape) != (T, B, HQ, dv) or tuple(lse.shape) != (B, HQ, T) or tuple(cum_t.shape) != (B * H, T)
            or tuple(qtok_t.shape) != (B * NMAX, T) or tuple(ktok.shape) != (B, T, NMAX) or T % 4 or dv % 64):
        raise RuntimeError("neuronspark.pool_read_dv_gluon: shapes")
    dgv = torch.empty((T, B, H, dv), dtype=torch.bfloat16, device=q16.device)
    kd = _tile_desc(k16, BNK)
    qd = _tile_desc(q16, BMQ_V)
    dod = _tile_desc(do16, BMQ_V)
    ld = _row_desc(lse.view(B * HQ, T), BMQ_V)
    cd = _row_desc(cum_t, BMQ_V)
    td = _row_desc(qtok_t, BMQ_V)
    grid = (triton.cdiv(T, BNK) * B * H,)
    with torch.cuda.device(q16.device.index):
        _dv_kernel[grid](kd, qd, dod, ld, cd, td, ktok, cum_t, slots, dgv, T, B, H, HQ, float(scale),
                         DK=dk, DV=dv, BNK=BNK, BMQ=BMQ_V, R=R_V, NMAX=NMAX, NBS=int(slots.shape[1]),
                         ORDMASK=int(ordmask), REGS_W=REGS_W, num_warps=4)
    return dgv


def pool_read_dk_gluon(q16: torch.Tensor, k16: torch.Tensor, gv: torch.Tensor, do16: torch.Tensor, lse: torch.Tensor,
                       delta: torch.Tensor, cum_t: torch.Tensor, qtok_t: torch.Tensor, ktok: torch.Tensor,
                       slots: torch.Tensor, ordmask: int, scale: float):
    """(dk (T, B, H, dk) fp32, dcum (T, B, H) fp32, beta-slot sums (H, NB) fp32) of the pool read (see the module
    docstring): as :func:`pool_read_dv_gluon`, with the tagged values gv (T, B, H, dv) bf16 and D (B, HQ, T) fp32."""
    require_cuda(q16, "pool_read_dk_gluon")
    require_sm100(q16.device, "pool_read_dk_gluon")
    T, B, HQ, dk = (int(s) for s in q16.shape)
    H, dv = int(k16.shape[2]), int(gv.shape[-1])
    NMAX = int(ktok.shape[-1])
    NB = int(slots.shape[1])
    for name, x, dt in (("q16", q16, torch.bfloat16), ("k16", k16, torch.bfloat16), ("gv", gv, torch.bfloat16),
                        ("do16", do16, torch.bfloat16), ("lse", lse, torch.float32), ("delta", delta, torch.float32),
                        ("cum_t", cum_t, torch.float32), ("qtok_t", qtok_t, torch.int32), ("ktok", ktok, torch.int32),
                        ("slots", slots, torch.float32)):
        if x.dtype != dt or not x.is_contiguous():
            raise RuntimeError(f"neuronspark.pool_read_dk_gluon: {name} must be a contiguous {dt} tensor")
    if (tuple(do16.shape) != (T, B, HQ, dv) or tuple(lse.shape) != (B, HQ, T) or tuple(delta.shape) != (B, HQ, T)
            or tuple(cum_t.shape) != (B * H, T) or tuple(qtok_t.shape) != (B * NMAX, T)
            or tuple(ktok.shape) != (B, T, NMAX) or T % 4 or dk % 64):
        raise RuntimeError("neuronspark.pool_read_dk_gluon: shapes")
    dev = q16.device
    dkk = torch.empty((T, B, H, dk), dtype=torch.float32, device=dev)
    dcum = torch.empty((T, B, H), dtype=torch.float32, device=dev)
    part = torch.empty((B * H, T, NB), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(T, BNK) * B * H,)
    with torch.cuda.device(dev.index):
        _dk_kernel[grid](_tile_desc(k16, BNK), _tile_desc(gv, BNK), _tile_desc(q16, BMQ_K), _tile_desc(do16, BMQ_K),
                         _row_desc(lse.view(B * HQ, T), BMQ_K), _row_desc(delta.view(B * HQ, T), BMQ_K),
                         _row_desc(cum_t, BMQ_K), _row_desc(qtok_t, BMQ_K), ktok, cum_t, slots, dkk, dcum, part,
                         T, B, H, HQ, float(scale), DK=dk, DV=dv, BNK=BNK, BMQ=BMQ_K, R=R, NMAX=NMAX, NBS=NB,
                         ORDMASK=int(ordmask), REGS_W=REGS_W, num_warps=4)
    return dkk, dcum, part.view(B, H, T, NB).sum(dim=(0, 2))


def pool_read_dq_gluon(q16: torch.Tensor, k16: torch.Tensor, gv: torch.Tensor, do16: torch.Tensor, lse: torch.Tensor,
                       delta: torch.Tensor, cum_t: torch.Tensor, ktok_t: torch.Tensor, qtok: torch.Tensor,
                       slots: torch.Tensor, ordmask: int, scale: float) -> torch.Tensor:
    """The queries' gradient (T, B, HQ, dk) fp32 of the pool read (see the module docstring), with the key id windows
    transposed ktok_t (B n_max, T) int32 and the query id windows qtok (B, T, n_max) int32."""
    require_cuda(q16, "pool_read_dq_gluon")
    require_sm100(q16.device, "pool_read_dq_gluon")
    T, B, HQ, dk = (int(s) for s in q16.shape)
    H, dv = int(k16.shape[2]), int(gv.shape[-1])
    NMAX = int(qtok.shape[-1])
    for name, x, dt in (("q16", q16, torch.bfloat16), ("k16", k16, torch.bfloat16), ("gv", gv, torch.bfloat16),
                        ("do16", do16, torch.bfloat16), ("lse", lse, torch.float32), ("delta", delta, torch.float32),
                        ("cum_t", cum_t, torch.float32), ("ktok_t", ktok_t, torch.int32), ("qtok", qtok, torch.int32),
                        ("slots", slots, torch.float32)):
        if x.dtype != dt or not x.is_contiguous():
            raise RuntimeError(f"neuronspark.pool_read_dq_gluon: {name} must be a contiguous {dt} tensor")
    if (tuple(do16.shape) != (T, B, HQ, dv) or tuple(lse.shape) != (B, HQ, T) or tuple(delta.shape) != (B, HQ, T)
            or tuple(cum_t.shape) != (B * H, T) or tuple(ktok_t.shape) != (B * NMAX, T)
            or tuple(qtok.shape) != (B, T, NMAX) or T % 4 or dk % 64):
        raise RuntimeError("neuronspark.pool_read_dq_gluon: shapes")
    dq = torch.empty((T, B, HQ, dk), dtype=torch.float32, device=q16.device)
    grid = (triton.cdiv(T, BM_Q) * B * HQ,)
    with torch.cuda.device(q16.device.index):
        _dq_kernel[grid](_tile_desc(q16, BM_Q), _tile_desc(do16, BM_Q), _tile_desc(k16, BN_Q), _tile_desc(gv, BN_Q),
                         _row_desc(cum_t, BN_Q), _row_desc(ktok_t, BN_Q), qtok, cum_t, slots, lse, delta, dq,
                         T, B, H, HQ, float(scale), DK=dk, DV=dv, BM=BM_Q, BN=BN_Q, R=R, NMAX=NMAX,
                         NBS=int(slots.shape[1]), ORDMASK=int(ordmask), REGS_W=REGS_W, num_warps=4)
    return dq
