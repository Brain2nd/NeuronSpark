"""dendritic-layer state gradients of the SSD scan's training backward, written in Gluon for Blackwell GPUs: the
chunk-scan dstates product and the state-passing backward of one (b, h) item walking its chunks in reverse, the gradient
carried on chip.

For an item with the forward's passed states ``s_c`` (fp32, the state entering chunk c; ``s_0 = 0``) and chunk length CS
(``dA_end[c] = dA_cumsum[b, h, c, CS - 1]``), for k = nc - 1 .. 1 (``d = 0`` before the first)::

    conv[k]  = s_k                                                     (the passed state, output dtype)
    ws[k, j] = exp(dA_end[k]) sum_{(p, n) in block j} s_k[p, n] d[p, n]     (blocks of 4 rows of P, fp32)
    A_k      = sum_t bf16(dout[t] exp(dA_cumsum[t]))^T C[t]            (P, N) fp32
    d        = exp(dA_end[k]) d + A_k ;  dstates[k - 1] = d           (output dtype)

with ``dstates[nc - 1] = 0``, ``conv[0] = 0`` and ``ws[0] = 0`` -- mamba_ssm ``_chunk_scan_bwd_dstates`` (the stock dout
scaling and its bf16 rounding point, fp32 accumulation) feeding ``_state_passing_bwd`` (CONVERT_STATES, no dfinal /
dinit), on the forward's stored passed states (``chunk_states_train(.., passed_fp32=True)``: the fp32 values the stock
backward recomputes with ``_state_passing_fwd``).

Per CTA (persistent, items (b, h) in round-robin): a load warp streams the (BK, P) dout and (BK, N) C tiles of each K step
through an R-deep mbarrier ring and each chunk's passed-state tile; a compute partition scales each dout tile in place
(one row factor per token, bf16); an MMA warp issues A_k += dout_tile^T . C_tile (the dout tile as a permuted MN-major
view) into one of two TMEM accumulators; the default partition (4 warps) carries d and stores the outputs through
shared tiles and TMA stores (the 16 workspace values per chunk by plain stores).

Numerics: the stock expressions and rounding points; the fp32 accumulation order of the tensor cores and of the block
sums (the per-operator rule class of :mod:`neuronspark.utils.testing` against :func:`dstates_train_ref`).  A partial last
chunk is supported (its rows past the sequence get a zero scaled dout).

Launch constants: BK 64 tokens per K step, R 4 ring slots, NWC 4 compute warps, registers 40 (load / MMA warps) and
128 (compute); grid = #SM persistent CTAs.
"""
from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_32x32b_reg_layout, tcgen05_mma, tcgen05_commit, mbarrier, tma,
    fence_async_shared)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from neuronspark.ops.gluon._ptx import _tc_fence
from neuronspark.ops.triton import num_sms, require_cuda, require_sm100

__all__ = ["BK", "R", "NWC", "REGS_W", "REGS_C", "WS_ROWS", "dstates_train", "dstates_train_ref"]

# ------------------------------------------------------------------------------------------------ launch constants
BK = 64          # tokens per K step
R = 4            # ring slots (K steps in flight)
NWC = 4          # compute warps (the dout scaling)
REGS_W = 40      # registers of the load / MMA warps
REGS_C = 128     # registers of the compute partition
WS_ROWS = 4      # rows of P per workspace block (512 elements at N 128: the stock state-passing block)
_GL_DT = {torch.bfloat16: gl.bfloat16, torch.float32: gl.float32}


# ------------------------------------------------------------------------------------------------ partitions
@gluon.jit
def _epilogue_partition(do_s, c_s, ready, scaled, free, acc_done, acc_free, acc, s_s, s_bar, s_free, dst_s, cv_s,
                        dod, cd, sd, dstd, cvd, DA, WS, n_items, seqlen, nchunks,
                        H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                        R: gl.constexpr, NWC: gl.constexpr, NBLK: gl.constexpr, WSR: gl.constexpr):
    """Default partition (4 warps): per item the carried gradient d; per chunk the converted state, the workspace block
    sums, the gradient update and the dstates store."""
    AL: gl.constexpr = get_tmem_32x32b_reg_layout(P, N, [P, N], 4)
    LB: gl.constexpr = gl.BlockedLayout([1, NBLK], [32, 1], [4, 1], [1, 0])
    rows = gl.arange(0, P, layout=gl.SliceLayout(1, LB))
    blk = gl.arange(0, NBLK, layout=gl.SliceLayout(0, LB))
    sel = (rows // WSR)[:, None] == blk[None, :]                       # (P, NBLK): row p belongs to block j
    G = gl.num_programs(0)
    cc = 0                                                             # chunks consumed (accumulator index)
    sc_ = 0                                                            # passed-state tiles consumed
    for it in range(gl.program_id(0), n_items, G):
        b = it // H
        h = it - b * H
        d = gl.zeros([P, N], gl.float32, layout=AL)
        # dstates[nc - 1] = 0, conv[0] = 0, ws[0] = 0 (the staging buffers free before and after: once per item)
        tma.store_wait(0)
        dst_s.index(0).store(d.to(dst_s.dtype))
        cv_s.index(0).store(d.to(cv_s.dtype))
        fence_async_shared()
        gl.thread_barrier()
        tma.async_copy_shared_to_global(dstd, [((b * nchunks + nchunks - 1) * H + h) * P, 0], dst_s.index(0))
        tma.async_copy_shared_to_global(cvd, [((b * nchunks) * H + h) * P, 0], cv_s.index(0))
        tma.store_wait(0)
        gl.store(WS + ((b * H + h) * nchunks) * NBLK + blk, gl.zeros([NBLK], gl.float32, layout=gl.SliceLayout(0, LB)))
        for i in range(nchunks - 1):
            k = nchunks - 1 - i
            row = ((b * nchunks + k) * H + h) * P
            da_end = gl.load(DA + ((b * H + h) * nchunks + k) * CS + (CS - 1))
            scale = gl.exp(da_end)
            mbarrier.wait(s_bar, sc_ & 1)
            s = s_s.load(AL)
            sc_ += 1
            u = cc % 2
            tma.store_wait(3)                                          # the buffers' stores of two chunks ago are done
            cv_s.index(u).store(s.to(cv_s.dtype))
            fence_async_shared()
            gl.thread_barrier()
            # s_s is released only after every thread's values of it were consumed (the conv tile above) and the proxy fence:
            # the load warp's next TMA write into s_s must not overtake this partition's reads of it
            mbarrier.arrive(s_free)
            tma.async_copy_shared_to_global(cvd, [row, 0], cv_s.index(u))
            # ws[k, j] = exp(dA_end) sum over block j of s * d (the gradient before this chunk's update)
            rs = gl.convert_layout(gl.sum(s * d, axis=1), gl.SliceLayout(1, LB))
            wsv = gl.sum(gl.where(sel, rs[:, None], 0.0), axis=0) * scale
            gl.store(WS + ((b * H + h) * nchunks + k) * NBLK + blk, wsv)
            mbarrier.wait(acc_done.index(u), (cc // 2) & 1)
            _tc_fence(1, 4)
            ak = acc.index(u).load(AL)
            _tc_fence(0, 4)
            gl.thread_barrier()
            mbarrier.arrive(acc_free.index(u))
            d = scale * d + ak                                         # the stock state-passing backward step
            tma.store_wait(3)
            dst_s.index(u).store(d.to(dst_s.dtype))
            fence_async_shared()
            gl.thread_barrier()
            tma.async_copy_shared_to_global(dstd, [((b * nchunks + k - 1) * H + h) * P, 0], dst_s.index(u))
            cc += 1
    tma.store_wait(0)


@gluon.jit
def _load_partition(do_s, c_s, ready, scaled, free, acc_done, acc_free, acc, s_s, s_bar, s_free, dst_s, cv_s,
                    dod, cd, sd, dstd, cvd, DA, WS, n_items, seqlen, nchunks,
                    H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                    R: gl.constexpr, NWC: gl.constexpr, NBLK: gl.constexpr, WSR: gl.constexpr):
    """Load warp: per chunk (in reverse) its passed-state tile, then per K step the (BK, P) dout tile of head h and the
    (BK, N) C tile into the ring."""
    NKT: gl.constexpr = CS // BK
    G = gl.num_programs(0)
    t = 0
    sc_ = 0
    for it in range(gl.program_id(0), n_items, G):
        b = it // H
        h = it - b * H
        for i in range(nchunks - 1):
            k = nchunks - 1 - i
            mbarrier.wait(s_free, (sc_ & 1) ^ 1)
            mbarrier.expect(s_bar, P * N * 4)
            tma.async_copy_global_to_shared(sd, [((b * nchunks + k) * H + h) * P, 0], s_bar, s_s)
            sc_ += 1
            for kt in gl.static_range(NKT):
                slot = t % R
                mbarrier.wait(free.index(slot), ((t // R) & 1) ^ 1)
                mbarrier.expect(ready.index(slot), BK * P * 2 + BK * N * 2)
                r0 = b * seqlen + k * CS + kt * BK
                tma.async_copy_global_to_shared(dod, [r0, h * P], ready.index(slot), do_s.index(slot))
                tma.async_copy_global_to_shared(cd, [r0, 0], ready.index(slot), c_s.index(slot))
                t += 1


@gluon.jit
def _compute_partition(do_s, c_s, ready, scaled, free, acc_done, acc_free, acc, s_s, s_bar, s_free, dst_s, cv_s,
                       dod, cd, sd, dstd, cvd, DA, WS, n_items, seqlen, nchunks,
                       H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                       R: gl.constexpr, NWC: gl.constexpr, NBLK: gl.constexpr, WSR: gl.constexpr):
    """Compute partition (NWC warps): dout_tile[t, :] *= exp(dA_cumsum[t]), rounded to bf16, in place."""
    NKT: gl.constexpr = CS // BK
    L: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NWC, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(1, L)
    LC: gl.constexpr = gl.BlockedLayout([NKT, 1], [1, 32], [1, NWC], [0, 1])
    offs_kc = gl.arange(0, NKT, layout=gl.SliceLayout(1, LC))[:, None] * BK + gl.arange(0, BK, layout=gl.SliceLayout(0, LC))[None, :]
    selk = gl.arange(0, NKT, layout=gl.SliceLayout(1, LC))[:, None]
    offs_kl = gl.arange(0, BK, layout=LK)
    G = gl.num_programs(0)
    t = 0
    for it in range(gl.program_id(0), n_items, G):
        b = it // H
        h = it - b * H
        for i in range(nchunks - 1):
            k = nchunks - 1 - i
            base = ((b * H + h) * nchunks + k) * CS
            sc = gl.exp(gl.load(DA + base + offs_kc))                  # the stock dout scale of every token of the chunk
            lim = seqlen - k * CS
            for kt in gl.static_range(NKT):
                slot = t % R
                scale = gl.sum(gl.where(selk == kt, sc, 0.0), axis=0)  # row kt (one nonzero term per sum)
                mbarrier.wait(ready.index(slot), (t // R) & 1)
                dv = do_s.index(slot).load(L).to(gl.float32)
                ds = dv * gl.convert_layout(scale, LK)[:, None]
                if kt * BK + BK > lim:                                 # a partial chunk: rows past the sequence weigh 0
                    ds = gl.where((kt * BK + offs_kl < lim)[:, None], ds, 0.0)
                do_s.index(slot).store(ds.to(gl.bfloat16))
                fence_async_shared()
                gl.thread_barrier()
                mbarrier.arrive(scaled.index(slot))
                t += 1


@gluon.jit
def _mma_partition(do_s, c_s, ready, scaled, free, acc_done, acc_free, acc, s_s, s_bar, s_free, dst_s, cv_s,
                   dod, cd, sd, dstd, cvd, DA, WS, n_items, seqlen, nchunks,
                   H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                   R: gl.constexpr, NWC: gl.constexpr, NBLK: gl.constexpr, WSR: gl.constexpr):
    """MMA warp: A_k (P, N) = sum over the chunk's K steps of dout_tile^T (P, BK) . C_tile (BK, N)."""
    NKT: gl.constexpr = CS // BK
    G = gl.num_programs(0)
    t = 0
    cc = 0
    for it in range(gl.program_id(0), n_items, G):
        for i in range(nchunks - 1):
            a = cc % 2
            mbarrier.wait(acc_free.index(a), ((cc // 2) & 1) ^ 1)
            _tc_fence(1, 1)
            for kt in gl.static_range(NKT):
                slot = t % R
                mbarrier.wait(ready.index(slot), (t // R) & 1)
                mbarrier.wait(scaled.index(slot), (t // R) & 1)
                tcgen05_mma(do_s.index(slot).permute((1, 0)), c_s.index(slot), acc.index(a), use_acc=kt > 0,
                            mbarriers=[free.index(slot)])
                t += 1
            tcgen05_commit(acc_done.index(a))
            cc += 1


# ------------------------------------------------------------------------------------------------ kernel
@gluon.jit
def _dstates_kernel(dod, cd, sd, dstd, cvd, DA, WS, n_items, seqlen, nchunks,
                    H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                    R: gl.constexpr, NWC: gl.constexpr, NBLK: gl.constexpr, WSR: gl.constexpr,
                    REGS_W: gl.constexpr, REGS_C: gl.constexpr):
    TL: gl.constexpr = TensorMemoryLayout(block=(P, N), unpacked=True)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    do_s = gl.allocate_shared_memory(gl.bfloat16, [R, BK, P], dod.layout)
    c_s = gl.allocate_shared_memory(gl.bfloat16, [R, BK, N], cd.layout)
    s_s = gl.allocate_shared_memory(gl.float32, [P, N], sd.layout)
    dst_s = gl.allocate_shared_memory(dstd.dtype, [2, P, N], dstd.layout)
    cv_s = gl.allocate_shared_memory(cvd.dtype, [2, P, N], cvd.layout)
    ready = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    scaled = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    acc_done = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    acc_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    s_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
    s_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    for i in gl.static_range(R):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(scaled.index(i), count=1)
        mbarrier.init(free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(acc_done.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=1)
    mbarrier.init(s_bar, count=1)
    mbarrier.init(s_free, count=1)
    fence_async_shared()
    acc = allocate_tensor_memory(gl.float32, [2, P, N], TL)
    gl.warp_specialize(
        (do_s, c_s, ready, scaled, free, acc_done, acc_free, acc, s_s, s_bar, s_free, dst_s, cv_s,
         dod, cd, sd, dstd, cvd, DA, WS, n_items, seqlen, nchunks, H, P, N, CS, BK, R, NWC, NBLK, WSR),
        _epilogue_partition,
        (do_s, c_s, ready, scaled, free, acc_done, acc_free, acc, s_s, s_bar, s_free, dst_s, cv_s,
         dod, cd, sd, dstd, cvd, DA, WS, n_items, seqlen, nchunks, H, P, N, CS, BK, R, NWC, NBLK, WSR),
        [_load_partition, _compute_partition, _mma_partition],
        [1, NWC, 1],
        [REGS_W, REGS_C, REGS_W])


# ------------------------------------------------------------------------------------------------ launcher
def dstates_train(states, dA_cumsum, dout, C, out_dtype):
    """(dstates (b, nc, h, P N) in out_dtype, ws (b, h, nc, P / 4) fp32, n_valid = P / 4, conv (b, nc, h, P N) in
    out_dtype) from the forward's passed states ``states`` (b, nc, h, P, N) fp32 contiguous (``states[0] = 0``), dA_cumsum
    (b, h, nc, chunk) fp32 contiguous, dout (b, l, h, P) bf16 with a contiguous (h, P) part and C (b, l, 1, N) bf16 with
    unit last stride (see the module docstring)."""
    require_cuda(dout, "dstates_train")
    require_sm100(dout.device, "dstates_train")
    batch, nchunks, nheads, headdim, dstate = states.shape
    _, seqlen, _, _ = dout.shape
    chunk = int(dA_cumsum.shape[-1])
    if (headdim != 64 or dstate != 128 or chunk % BK or nchunks < 2 or not (nchunks - 1) * chunk < seqlen <= nchunks * chunk
            or dout.shape != (batch, seqlen, nheads, headdim) or C.shape != (batch, seqlen, 1, dstate)
            or dA_cumsum.shape != (batch, nheads, nchunks, chunk)):
        raise RuntimeError(f"neuronspark.dstates_train: shapes states {tuple(states.shape)} dout {tuple(dout.shape)} C "
                           f"{tuple(C.shape)} dA_cumsum {tuple(dA_cumsum.shape)} (P 64, N 128, >= 2 chunks of a multiple of "
                           f"{BK} tokens covering the sequence)")
    if states.dtype != torch.float32 or not states.is_contiguous():
        raise RuntimeError("neuronspark.dstates_train: contiguous fp32 passed states expected")
    if dout.dtype != torch.bfloat16 or C.dtype != torch.bfloat16 or out_dtype not in _GL_DT:
        raise RuntimeError(f"neuronspark.dstates_train: bf16 dout / C and an fp32 / bf16 out_dtype expected")
    if (dout.stride(3) != 1 or dout.stride(2) != headdim or dout.stride(0) != seqlen * dout.stride(1) or dout.stride(1) % 8
            or C.stride(3) != 1 or C.stride(0) != seqlen * C.stride(1) or C.stride(1) % 8):
        raise RuntimeError(f"neuronspark.dstates_train: dout strides {tuple(dout.stride())} / C strides {tuple(C.stride())}")
    if dA_cumsum.dtype != torch.float32 or not dA_cumsum.is_contiguous():
        raise RuntimeError("neuronspark.dstates_train: contiguous fp32 dA_cumsum expected")
    dev = dout.device
    nblk = headdim // WS_ROWS
    dstates = torch.empty((batch, nchunks, nheads, headdim * dstate), device=dev, dtype=out_dtype)
    conv = torch.empty((batch, nchunks, nheads, headdim * dstate), device=dev, dtype=out_dtype)
    ws = torch.empty((batch, nheads, nchunks, nblk), device=dev, dtype=torch.float32)
    rows = batch * nchunks * nheads * headdim
    b16, f32 = gl.bfloat16, gl.float32
    dod = TensorDescriptor(dout, [batch * seqlen, nheads * headdim], [dout.stride(1), 1], [BK, headdim],
                           gl.NVMMASharedLayout.get_default_for([BK, headdim], b16))
    cd = TensorDescriptor(C, [batch * seqlen, dstate], [C.stride(1), 1], [BK, dstate],
                          gl.NVMMASharedLayout.get_default_for([BK, dstate], b16))
    sd = TensorDescriptor(states, [rows, dstate], [dstate, 1], [headdim, dstate],
                          gl.NVMMASharedLayout.get_default_for([headdim, dstate], f32))
    od = _GL_DT[out_dtype]
    dstd = TensorDescriptor(dstates.view(rows, dstate), [rows, dstate], [dstate, 1], [headdim, dstate],
                            gl.NVMMASharedLayout.get_default_for([headdim, dstate], od))
    cvd = TensorDescriptor(conv.view(rows, dstate), [rows, dstate], [dstate, 1], [headdim, dstate],
                           gl.NVMMASharedLayout.get_default_for([headdim, dstate], od))
    n_items = batch * nheads
    G = min(num_sms(dev), n_items)
    with torch.cuda.device(dev.index):
        _dstates_kernel[(G,)](dod, cd, sd, dstd, cvd, dA_cumsum, ws, n_items, seqlen, nchunks,
                              H=nheads, P=headdim, N=dstate, CS=chunk, BK=BK, R=R, NWC=NWC, NBLK=nblk, WSR=WS_ROWS,
                              REGS_W=REGS_W, REGS_C=REGS_C, num_warps=4)
    return dstates, ws, nblk, conv


def dstates_train_ref(states, dA_cumsum, dout, C, out_dtype):
    """Pure-torch reference (float64 sums; the scaled dout rounded to bf16 at the stock point, every exponential in the
    kernels' fp32 form ``exp2(t log2 e)``): the returns of :func:`dstates_train` (ws of shape (b, h, nc, P / 4))."""
    from neuronspark.ops.triton.dendritic_states import _exp_f32
    b, nc, h, p, n = states.shape
    L = int(dout.shape[1])
    chunk = int(dA_cumsum.shape[-1])
    s64 = states.double()
    end = _exp_f32(dA_cumsum[:, :, :, -1]).double()                    # (b, h, nc): exp(dA_end) in the kernels' fp32 form
    pad = nc * chunk - L
    do = torch.nn.functional.pad(dout, (0, 0, 0, 0, 0, pad)).view(b, nc, chunk, h, p)
    cc = torch.nn.functional.pad(C[:, :, 0], (0, 0, 0, pad)).view(b, nc, chunk, n).double()
    scale_t = _exp_f32(dA_cumsum).permute(0, 2, 3, 1)                  # (b, nc, chunk, h) fp32, the stock factor
    ds = (do.float() * scale_t[..., None]).to(torch.bfloat16).double()  # bf16(dout exp(dA)) as the stock operand
    A = torch.einsum("bctgp,bctn->bcgpn", ds, cc)                      # (b, nc, h, P, N)
    dst = torch.zeros_like(s64)
    conv = torch.zeros_like(s64)
    ws = torch.zeros((b, h, nc, p // WS_ROWS), dtype=torch.float64, device=states.device)
    d = torch.zeros((b, h, p, n), dtype=torch.float64, device=states.device)
    for k in range(nc - 1, 0, -1):
        e = end[:, :, k][..., None, None]                               # (b, h, 1, 1)
        conv[:, k] = s64[:, k]
        ws[:, :, k] = (s64[:, k] * d).reshape(b, h, p // WS_ROWS, WS_ROWS * n).sum(-1) * e[..., 0]
        d = e * d + A[:, k]
        dst[:, k - 1] = d
    return (dst.reshape(b, nc, h, p * n).to(out_dtype), ws.float(), p // WS_ROWS,
            conv.reshape(b, nc, h, p * n).to(out_dtype))
