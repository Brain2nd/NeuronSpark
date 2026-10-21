"""Forward of the hippocampal layers' fused pool read (:mod:`neuronspark.ops.pool_read`), written in Gluon
(Triton 3.5.1's explicit-layout dialect) for Blackwell GPUs: a warp-specialised kernel (TMA loads with mbarrier rings,
asynchronous tcgen05 MMAs into tensor memory).

``pool_read_fwd_gluon(q16, k16, gv, cum_t, ktok_t, qtok, slots, ordmask, scale)`` -> (out bf16, out fp32, lse): the
operator's forward (see :mod:`neuronspark.ops.pool_read` for the definition), from the bf16 queries / keys
(T, B, HQ, dk) / (T, B, H, dk), the tagged values (T, B, H, dv) bf16, the clocks transposed (B H, T) fp32, the key id
windows transposed (B n_max, T) int32, the query id windows (B, T, n_max) int32 and the beta slots (H, NB) fp32.

One CTA per (query tile of BM rows, b, query head), issued longest first.  One load warp (the query tile once; per key
tile of BN rows the key and value tiles, the clocks and the key id windows, into an R-deep ring), one MMA warp
(S_j = Q K_j^T into one of two TMEM S buffers, issued one key tile ahead; O += P2 V_j, O += P1 V_j), and the default
4-warp partition, one query row per thread: the bias of the tile (the induction matches of the id windows, the clock
difference, the bf16 rounding), the causal mask on the diagonal tiles, the online softmax with exp2, P split into two
bf16 pieces into shared memory, the rescaling of the TMEM accumulator by every new row maximum (after the previous
tile's P V products), and the epilogue (O / l in bf16 and fp32, the row log-sum-exp).

Numerics: the definition's bias and its rounding point; P enters the value product as two bf16 pieces; ``exp2`` of
the log2e-scaled scores.  Checked with the per-operator tolerance rule against
:func:`neuronspark.ops.pool_read.pool_read_ref`.  No compiler scratch inside the partitions (the
reductions are along each thread's own row).

Launch constants: BM 128 query rows, BN 64 key rows, ring R 2, registers 48 for the load / MMA warps.
"""
from __future__ import annotations

from typing import Tuple

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

__all__ = ["BM", "BN", "R", "REGS_W", "pool_read_fwd_gluon"]

# ------------------------------------------------------------------------------------------------ launch constants
BM = 128         # query rows per CTA (one per thread of the softmax partition)
BN = 64          # key rows per tile
R = 2            # key / value ring slots
REGS_W = 48      # registers of the load and MMA warps


# ------------------------------------------------------------------------------------------------ partitions
@gluon.jit
def _item(T, B, H, HQ, BM: gl.constexpr, BN: gl.constexpr):
    """(query tile, b, query head, key head, key tiles) of this CTA; the longest items first."""
    pid = gl.program_id(0)
    nbi = B * HQ
    mt = gl.cdiv(T, BM) - 1 - pid // nbi
    bi = pid % nbi
    b = bi // HQ
    i = bi % HQ
    h = gl.minimum(i, H - 1)
    n_kv = gl.cdiv(gl.minimum((mt + 1) * BM, T), BN)
    return mt, b, i, h, n_kv


@gluon.jit
def _softmax_partition(q_s, k_s, v_s, c_s, kt_s, p1_s, p2_s, s_tm, o_tm, q_bar, kv_bar, kv_free, s_ready, s_free,
                       p_ready, p_free, o_done, qd, kd, vd, cd, td, QTOK, CUMT, SLOTS, O, O32, LSE, T, B, H, HQ, scale,
                       DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
                       NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Default partition (4 warps, one query row per thread)."""
    L: gl.constexpr = get_tmem_32x32b_reg_layout(BM, BN, [BM, BN], 4)
    LR: gl.constexpr = gl.SliceLayout(1, L)
    LC: gl.constexpr = gl.SliceLayout(0, L)
    mt, b, i, h, n_kv = _item(T, B, H, HQ, BM, BN)
    t = mt * BM + gl.arange(0, BM, layout=LR)
    tm = t < T
    t64 = t.to(gl.int64)
    cum_t = gl.load(CUMT + (b * H + h).to(gl.int64) * T + t64, mask=tm, other=0.0)
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
    m_i = gl.full([BM], float("-inf"), gl.float32, layout=LR)
    l_i = gl.zeros([BM], gl.float32, layout=LR)
    for j in range(n_kv):
        sb = j % 2
        sl = j % R
        mbarrier.wait(s_ready.index(sb), (j // 2) & 1)
        _tc_fence(1, 4)
        s = s_tm.index(sb).load(L)
        _tc_fence(0, 4)
        gl.thread_barrier()
        mbarrier.arrive(s_free.index(sb))
        mbarrier.wait(kv_bar.index(sl), (j // R) & 1)
        cum_j = _row(c_s.index(sl), BN, gl.float32, LC)
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
        bias = ((cum_t[:, None] - cum_j[None, :]) + x).to(gl.bfloat16).to(gl.float32)
        sd = s * scale + bias
        if diag:
            sd = gl.where((jj[None, :] <= t[:, None]) & (jj[None, :] < T), sd, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(sd, axis=1))
        alpha = gl.exp2((m_i - m_new) * LOG2E)
        p = gl.exp2((sd - m_new[:, None]) * LOG2E)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        p1 = p.to(gl.bfloat16)
        p2 = (p - p1.to(gl.float32)).to(gl.bfloat16)
        if j > 0:
            # the previous tile's P V products are done: P may be overwritten, O rescaled to the new maximum
            mbarrier.wait(p_free, (j - 1) & 1)
            _tc_fence(1, 4)
            for c in gl.static_range(DV // BN):
                oc = o_tm.slice(c * BN, BN).load(L)
                o_tm.slice(c * BN, BN).store(oc * alpha[:, None])
            _tc_fence(0, 4)
        p1_s.store(p1)
        p2_s.store(p2)
        fence_async_shared()
        gl.thread_barrier()
        mbarrier.arrive(p_ready)
        m_i = m_new
    # epilogue: O / l (bf16 and fp32) and the row log-sum-exp
    mbarrier.wait(o_done, 0)
    _tc_fence(1, 4)
    inv = 1.0 / l_i
    orow = ((t64 * B + b) * HQ + i) * DV
    for c in gl.static_range(DV // BN):
        oc = o_tm.slice(c * BN, BN).load(L) * inv[:, None]
        ptr = orow[:, None] + (c * BN + gl.arange(0, BN, layout=LC))[None, :]
        gl.store(O + ptr, oc.to(gl.bfloat16), mask=tm[:, None])
        gl.store(O32 + ptr, oc, mask=tm[:, None])
    gl.store(LSE + (b * HQ + i).to(gl.int64) * T + t64, m_i + gl.log(l_i), mask=tm)


@gluon.jit
def _load_partition(q_s, k_s, v_s, c_s, kt_s, p1_s, p2_s, s_tm, o_tm, q_bar, kv_bar, kv_free, s_ready, s_free,
                    p_ready, p_free, o_done, qd, kd, vd, cd, td, QTOK, CUMT, SLOTS, O, O32, LSE, T, B, H, HQ, scale,
                    DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
                    NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """Load warp: the query tile, then per key tile the keys, values, clocks and key id windows into the ring."""
    mt, b, i, h, n_kv = _item(T, B, H, HQ, BM, BN)
    mbarrier.expect(q_bar, BM * DK * 2)
    tma.async_copy_global_to_shared(qd, [mt * BM, (b * HQ + i) * DK], q_bar, q_s)
    for j in range(n_kv):
        sl = j % R
        mbarrier.wait(kv_free.index(sl), ((j // R) & 1) ^ 1)
        mbarrier.expect(kv_bar.index(sl), BN * (DK + DV) * 2 + BN * 4 + NMAX * BN * 4)
        tma.async_copy_global_to_shared(kd, [j * BN, (b * H + h) * DK], kv_bar.index(sl), k_s.index(sl))
        tma.async_copy_global_to_shared(vd, [j * BN, (b * H + h) * DV], kv_bar.index(sl), v_s.index(sl))
        tma.async_copy_global_to_shared(cd, [b * H + h, j * BN], kv_bar.index(sl), c_s.index(sl))
        for n in gl.static_range(NMAX):
            tma.async_copy_global_to_shared(td, [b * NMAX + n, j * BN], kv_bar.index(sl), kt_s.index(sl * NMAX + n))


@gluon.jit
def _mma_partition(q_s, k_s, v_s, c_s, kt_s, p1_s, p2_s, s_tm, o_tm, q_bar, kv_bar, kv_free, s_ready, s_free,
                   p_ready, p_free, o_done, qd, kd, vd, cd, td, QTOK, CUMT, SLOTS, O, O32, LSE, T, B, H, HQ, scale,
                   DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
                   NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr):
    """MMA warp: S_{j+1} = Q K_{j+1}^T one tile ahead, then O += P2 V_j and O += P1 V_j once the tile's P is stored."""
    mt, b, i, h, n_kv = _item(T, B, H, HQ, BM, BN)
    mbarrier.wait(q_bar, 0)
    mbarrier.wait(kv_bar.index(0), 0)
    tcgen05_mma(q_s, k_s.index(0).permute((1, 0)), s_tm.index(0), use_acc=False, mbarriers=[s_ready.index(0)])
    for j in range(n_kv):
        if j + 1 < n_kv:
            sl1 = (j + 1) % R
            sb1 = (j + 1) % 2
            mbarrier.wait(kv_bar.index(sl1), ((j + 1) // R) & 1)
            mbarrier.wait(s_free.index(sb1), (((j + 1) // 2) & 1) ^ 1)
            tcgen05_mma(q_s, k_s.index(sl1).permute((1, 0)), s_tm.index(sb1), use_acc=False,
                        mbarriers=[s_ready.index(sb1)])
        mbarrier.wait(p_ready, j & 1)
        _tc_fence(1, 1)
        sl = j % R
        tcgen05_mma(p2_s, v_s.index(sl), o_tm, use_acc=j > 0)
        tcgen05_mma(p1_s, v_s.index(sl), o_tm, use_acc=True, mbarriers=[p_free, kv_free.index(sl)])
    tcgen05_commit(o_done)


# ------------------------------------------------------------------------------------------------ kernel
@gluon.jit
def _pool_read_fwd_kernel(qd, kd, vd, cd, td, QTOK, CUMT, SLOTS, O, O32, LSE, T, B, H, HQ, scale,
                          DK: gl.constexpr, DV: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr, R: gl.constexpr,
                          NMAX: gl.constexpr, NBS: gl.constexpr, ORDMASK: gl.constexpr, REGS_W: gl.constexpr):
    PL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BM, BN], gl.bfloat16)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    q_s = gl.allocate_shared_memory(gl.bfloat16, [BM, DK], qd.layout)
    k_s = gl.allocate_shared_memory(gl.bfloat16, [R, BN, DK], kd.layout)
    v_s = gl.allocate_shared_memory(gl.bfloat16, [R, BN, DV], vd.layout)
    c_s = gl.allocate_shared_memory(gl.float32, [R, 1, BN], cd.layout)
    kt_s = gl.allocate_shared_memory(gl.int32, [R * NMAX, 1, BN], td.layout)
    p1_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BN], PL)
    p2_s = gl.allocate_shared_memory(gl.bfloat16, [BM, BN], PL)
    q_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
    kv_bar = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    kv_free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    s_ready = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    s_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    p_ready = gl.allocate_shared_memory(gl.int64, [1], BL)
    p_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    o_done = gl.allocate_shared_memory(gl.int64, [1], BL)
    mbarrier.init(q_bar, count=1)
    for k in gl.static_range(R):
        mbarrier.init(kv_bar.index(k), count=1)
        mbarrier.init(kv_free.index(k), count=1)
    for k in gl.static_range(2):
        mbarrier.init(s_ready.index(k), count=1)
        mbarrier.init(s_free.index(k), count=1)
    mbarrier.init(p_ready, count=1)
    mbarrier.init(p_free, count=1)
    mbarrier.init(o_done, count=1)
    fence_async_shared()
    s_tm = allocate_tensor_memory(gl.float32, [2, BM, BN], TensorMemoryLayout(block=(BM, BN), unpacked=True))
    o_tm = allocate_tensor_memory(gl.float32, [BM, DV], TensorMemoryLayout(block=(BM, DV), unpacked=True))
    # the argument tuples are written out in the call: a tuple bound to a name turns the constexprs into tensors
    gl.warp_specialize(
        (q_s, k_s, v_s, c_s, kt_s, p1_s, p2_s, s_tm, o_tm, q_bar, kv_bar, kv_free, s_ready, s_free, p_ready, p_free,
         o_done, qd, kd, vd, cd, td, QTOK, CUMT, SLOTS, O, O32, LSE, T, B, H, HQ, scale,
         DK, DV, BM, BN, R, NMAX, NBS, ORDMASK),
        _softmax_partition,
        (q_s, k_s, v_s, c_s, kt_s, p1_s, p2_s, s_tm, o_tm, q_bar, kv_bar, kv_free, s_ready, s_free, p_ready, p_free,
         o_done, qd, kd, vd, cd, td, QTOK, CUMT, SLOTS, O, O32, LSE, T, B, H, HQ, scale,
         DK, DV, BM, BN, R, NMAX, NBS, ORDMASK),
        [_load_partition, _mma_partition], [1, 1], [REGS_W, REGS_W])


# ------------------------------------------------------------------------------------------------ launcher
def pool_read_fwd_gluon(q16: torch.Tensor, k16: torch.Tensor, gv: torch.Tensor, cum_t: torch.Tensor,
                        ktok_t: torch.Tensor, qtok: torch.Tensor, slots: torch.Tensor, ordmask: int, scale: float
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The pool read's forward (see the module docstring): (out (T, B, HQ, dv) bf16, out32 fp32, lse (B, HQ, T))."""
    require_cuda(q16, "pool_read_fwd_gluon")
    require_sm100(q16.device, "pool_read_fwd_gluon")
    T, B, HQ, dk = (int(s) for s in q16.shape)
    H, dv = int(k16.shape[2]), int(gv.shape[-1])
    NMAX = int(qtok.shape[-1])
    for name, x, dt in (("q16", q16, torch.bfloat16), ("k16", k16, torch.bfloat16), ("gv", gv, torch.bfloat16),
                        ("cum_t", cum_t, torch.float32), ("ktok_t", ktok_t, torch.int32),
                        ("qtok", qtok, torch.int32), ("slots", slots, torch.float32)):
        if x.dtype != dt or not x.is_contiguous():
            raise RuntimeError(f"neuronspark.pool_read_fwd_gluon: {name} must be a contiguous {dt} tensor")
    if (tuple(k16.shape) != (T, B, H, dk) or tuple(gv.shape) != (T, B, H, dv) or tuple(cum_t.shape) != (B * H, T)
            or tuple(ktok_t.shape) != (B * NMAX, T) or tuple(qtok.shape) != (B, T, NMAX) or dk % 64 or dv % BN
            or T % 4 or not 1 <= NMAX <= 4):
        raise RuntimeError(f"neuronspark.pool_read_fwd_gluon: shapes q {tuple(q16.shape)} k {tuple(k16.shape)} "
                           f"gv {tuple(gv.shape)} cum_t {tuple(cum_t.shape)} ktok_t {tuple(ktok_t.shape)} "
                           f"qtok {tuple(qtok.shape)} (T % 4 == 0, dk % 64 == 0, dv % {BN} == 0)")
    dev = q16.device
    out = torch.empty((T, B, HQ, dv), dtype=torch.bfloat16, device=dev)
    out32 = torch.empty((T, B, HQ, dv), dtype=torch.float32, device=dev)
    lse = torch.empty((B, HQ, T), dtype=torch.float32, device=dev)
    b16 = gl.bfloat16
    qd = TensorDescriptor(q16, [T, B * HQ * dk], [B * HQ * dk, 1], [BM, dk],
                          gl.NVMMASharedLayout.get_default_for([BM, dk], b16))
    kd = TensorDescriptor(k16, [T, B * H * dk], [B * H * dk, 1], [BN, dk],
                          gl.NVMMASharedLayout.get_default_for([BN, dk], b16))
    vd = TensorDescriptor(gv, [T, B * H * dv], [B * H * dv, 1], [BN, dv],
                          gl.NVMMASharedLayout.get_default_for([BN, dv], b16))
    row = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=32, rank=2)        # one linear row per box
    cd = TensorDescriptor(cum_t, [B * H, T], [T, 1], [1, BN], row)
    td = TensorDescriptor(ktok_t, [B * NMAX, T], [T, 1], [1, BN], row)
    grid = (triton.cdiv(T, BM) * B * HQ,)
    with torch.cuda.device(dev.index):
        _pool_read_fwd_kernel[grid](qd, kd, vd, cd, td, qtok, cum_t, slots, out, out32, lse, T, B, H, HQ, float(scale),
                                    DK=dk, DV=dv, BM=BM, BN=BN, R=R, NMAX=NMAX, NBS=int(slots.shape[1]),
                                    ORDMASK=int(ordmask), REGS_W=REGS_W, num_warps=4)
    return out, out32, lse
