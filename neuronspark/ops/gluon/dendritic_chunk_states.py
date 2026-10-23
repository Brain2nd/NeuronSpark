"""dendritic-layer chunk states of the SSD scan, written in Gluon for Blackwell GPUs: the chunk-state product and the
state passing of one (b, h) item walking its chunks in order, with the state carried on chip.

For chunk c of an item (chunk length CS, dA_last = dA_cumsum[b, h, c, CS - 1]):

    A_c     = sum_k x[k]^T bf16(B[k] scale[k])        scale[k] = exp(min(dA_last - dA_cumsum[k], 0)) dt[k]   (P, N) fp32
    s_{c+1} = exp(dA_last) s_c + A_c                                                                          (fp32)

the chunk-state body (the bf16 rounding point of the scaled B, fp32 accumulation) and the state-passing step.
Forms:

* :func:`chunk_states_train` (no initial state): ``states_pre[c] = A_c`` fp32 (or, with ``passed_fp32``, the passed
  state ``s_c`` fp32) and ``states[c] = s_c`` in the output dtype (``states[0] = 0``; the final state is not stored);
* :func:`chunk_states_prefill` (the carried state ``s_0``, or zero): ``states[c] = s_c`` in the output dtype and the final
  state ``s_nc`` fp32 -- the same quantities from the initial state.

Per CTA (persistent, items (b, h) in round-robin): a load warp streams the (BK, P) x tiles and the (BK, N) B tiles of
each K step through an R-deep mbarrier ring; a compute partition of NWC warps scales each B tile in place (one row
factor per token, bf16); an MMA warp issues A_c += x_tile^T . B_tile (the x tile as a permuted MN-major view) into one of
two TMEM accumulators; the default partition (4 warps) waits for each chunk's last MMA, reads A_c, carries the state and
stores the outputs.

Numerics: the scale and the passing step are the reference's expressions (``exp``), the scaled B rounded to bf16 at the
reference's point; the fp32 accumulation over the chunk's tokens is the tensor cores' order (the per-operator rule class
of :mod:`neuronspark.utils.testing` against :func:`chunk_states_train_ref` / :func:`chunk_states_prefill_ref`).  A
partial last chunk is supported: its rows past the sequence get a zero scaled B (dA_last is the chunk's last element, as
the passing reads it).  x and B are read through 3-D (batch, seqlen, .) descriptors: any batch stride (a time slice of a
longer buffer qualifies).

Launch constants: BK 64 tokens per K step, R 4 ring slots, NWC 8 compute warps, registers 40 (load / MMA warps) and 128
(compute); grid = #SM persistent CTAs.  The epilogue writes the per-chunk outputs from registers by plain stores (the
TMA store of the bf16 state tile, which the compiler split over the four epilogue warps, faulted with an out-of-range
address when the GPU was time-sliced with another process; the plain stores do not); the compute partition reads each K
step's dA / dt in the tile's row layout.  No partition
uses an operation that needs compiler scratch memory (a layout conversion, a reduction across warps): the allocator
places such scratch over the warp_specialize capture buffer, which partitions that start later still read.  The
initial-state barriers exist only with an initial state: an initialised barrier the kernel no longer uses is dead to the
allocator, which put the capture buffer over it, and on a time-sliced GPU the context switch wrote the barrier back
over the captured tensor-memory address (the MMA warp then issued its MMAs at an address out of range).
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

__all__ = ["BK", "R", "NWC", "REGS_W", "REGS_C", "CTAS_PER_SM", "chunk_states_train", "chunk_states_prefill",
           "chunk_states_train_ref", "chunk_states_prefill_ref"]

# ------------------------------------------------------------------------------------------------ launch constants
BK = 64          # tokens per K step
R = 4            # ring slots (K steps in flight)
NWC = 8          # compute warps (the B scaling)
REGS_W = 40      # registers of the load / MMA warps
REGS_C = 128     # registers of the compute partition
CTAS_PER_SM = 1  # persistent CTAs per SM
_GL_DT = {torch.bfloat16: gl.bfloat16, torch.float32: gl.float32}


# ------------------------------------------------------------------------------------------------ partitions
@gluon.jit
def _epilogue_partition(x_s, b_s, ready, scaled, free, acc_done, acc_free, acc, init_s, init_bar, init_free, xd, bd, initd, DT, DA, STP, PRP, FNP,
                        n_items, seqlen, nchunks,
                        H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                        R: gl.constexpr, NWC: gl.constexpr, HAS_INIT: gl.constexpr, STORE_PRE: gl.constexpr,
                        STORE_FIN: gl.constexpr, INIT_BYTES: gl.constexpr, PASSED: gl.constexpr):
    """Default partition (4 warps): per item the carried state s; per chunk the state entering it and (training) the
    fp32 passed state or the chunk's own state A_c, written from registers by plain stores."""
    AL: gl.constexpr = get_tmem_32x32b_reg_layout(P, N, [P, N], 4)
    rows = gl.arange(0, P, layout=gl.SliceLayout(1, AL))
    cols = gl.arange(0, N, layout=gl.SliceLayout(0, AL))
    G = gl.num_programs(0)
    cc = 0                                                             # chunks consumed by this CTA (accumulator index)
    ni = 0                                                             # items of this CTA
    for it in range(gl.program_id(0), n_items, G):
        b = it // H
        h = it - b * H
        if HAS_INIT:
            mbarrier.wait(init_bar, ni & 1)
            s = init_s.load(AL).to(gl.float32)
        else:
            s = gl.zeros([P, N], gl.float32, layout=AL)
        for c in range(nchunks):
            row = ((b * nchunks + c) * H + h) * P
            da_last = gl.load(DA + ((b * H + h) * nchunks + c) * CS + (CS - 1))
            u = cc % 2
            offs = (row + rows).to(gl.int64)[:, None] * N + cols[None, :]
            gl.store(STP + offs, s.to(STP.dtype.element_ty))           # the state entering chunk c
            if HAS_INIT:
                if c == 0:
                    # init_s is released only after every thread's values of it were consumed (the chunk-0 state above)
                    gl.thread_barrier()
                    mbarrier.arrive(init_free)
            mbarrier.wait(acc_done.index(u), (cc // 2) & 1)
            _tc_fence(1, 4)
            ac = acc.index(u).load(AL)
            _tc_fence(0, 4)
            gl.thread_barrier()
            mbarrier.arrive(acc_free.index(u))
            if STORE_PRE:
                gl.store(PRP + offs, s if PASSED else ac)              # PASSED: the fp32 state entering the chunk
            s = gl.exp(da_last) * s + ac                               # the state-passing step
            cc += 1
        if STORE_FIN:
            gl.store(FNP + ((b * H + h) * P + rows).to(gl.int64)[:, None] * N + cols[None, :], s)
        ni += 1


@gluon.jit
def _load_partition(x_s, b_s, ready, scaled, free, acc_done, acc_free, acc, init_s, init_bar, init_free, xd, bd, initd, DT, DA, STP, PRP, FNP,
                    n_items, seqlen, nchunks,
                    H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                    R: gl.constexpr, NWC: gl.constexpr, HAS_INIT: gl.constexpr, STORE_PRE: gl.constexpr,
                    STORE_FIN: gl.constexpr, INIT_BYTES: gl.constexpr, PASSED: gl.constexpr):
    """Load warp: per item its initial state (prefill form), per K step the (BK, P) x tile of head h and the (BK, N) B
    tile into the ring."""
    NKT: gl.constexpr = CS // BK
    G = gl.num_programs(0)
    t = 0
    ni = 0
    for it in range(gl.program_id(0), n_items, G):
        b = it // H
        h = it - b * H
        if HAS_INIT:
            mbarrier.wait(init_free, (ni & 1) ^ 1)
            mbarrier.expect(init_bar, INIT_BYTES)
            tma.async_copy_global_to_shared(initd, [(b * H + h) * P, 0], init_bar, init_s)
        ni += 1
        for c in range(nchunks):
            for kt in gl.static_range(NKT):
                slot = t % R
                mbarrier.wait(free.index(slot), ((t // R) & 1) ^ 1)
                mbarrier.expect(ready.index(slot), BK * P * 2 + BK * N * 2)
                t0 = c * CS + kt * BK
                # the 2-D ring tiles seen as the descriptors' (1, BK, .) boxes (the same bytes and swizzle)
                tma.async_copy_global_to_shared(xd, [b, t0, h * P], ready.index(slot),
                                                x_s.index(slot)._reinterpret(gl.bfloat16, [1, BK, P], xd.layout))
                tma.async_copy_global_to_shared(bd, [b, t0, 0], ready.index(slot),
                                                b_s.index(slot)._reinterpret(gl.bfloat16, [1, BK, N], bd.layout))
                t += 1


@gluon.jit
def _compute_partition(x_s, b_s, ready, scaled, free, acc_done, acc_free, acc, init_s, init_bar, init_free, xd, bd, initd, DT, DA, STP, PRP, FNP,
                       n_items, seqlen, nchunks,
                       H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                       R: gl.constexpr, NWC: gl.constexpr, HAS_INIT: gl.constexpr, STORE_PRE: gl.constexpr,
                       STORE_FIN: gl.constexpr, INIT_BYTES: gl.constexpr, PASSED: gl.constexpr):
    """Compute partition (NWC warps): B_tile[k, :] *= exp(min(dA_last - dA[k], 0)) dt[k], rounded to bf16, in place."""
    NKT: gl.constexpr = CS // BK
    L: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [NWC, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(1, L)
    offs_kl = gl.arange(0, BK, layout=LK)
    G = gl.num_programs(0)
    t = 0
    for it in range(gl.program_id(0), n_items, G):
        b = it // H
        h = it - b * H
        for c in range(nchunks):
            base = ((b * H + h) * nchunks + c) * CS
            da_last = gl.load(DA + base + (CS - 1))
            lim = seqlen - c * CS                                      # tokens of the chunk inside the sequence
            for kt in gl.static_range(NKT):
                slot = t % R
                # the K step's dA / dt loaded in the tile's row layout: no layout conversion in a worker partition
                # (its scratch would be placed over the warp_specialize capture buffer, which partitions that start
                # later still read)
                dk = gl.load(DA + base + kt * BK + offs_kl)
                tk = gl.load(DT + base + kt * BK + offs_kl)
                scale = gl.exp(gl.minimum(da_last - dk, 0.0)) * tk     # the scale of every token of the K step
                mbarrier.wait(ready.index(slot), (t // R) & 1)
                bt = b_s.index(slot).load(L).to(gl.float32)
                bs = bt * scale[:, None]
                if kt * BK + BK > lim:                                 # a partial chunk: rows past the sequence weigh 0
                    bs = gl.where((kt * BK + offs_kl < lim)[:, None], bs, 0.0)
                b_s.index(slot).store(bs.to(gl.bfloat16))
                fence_async_shared()
                gl.thread_barrier()
                mbarrier.arrive(scaled.index(slot))
                t += 1


@gluon.jit
def _mma_partition(x_s, b_s, ready, scaled, free, acc_done, acc_free, acc, init_s, init_bar, init_free, xd, bd, initd, DT, DA, STP, PRP, FNP,
                   n_items, seqlen, nchunks,
                   H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                   R: gl.constexpr, NWC: gl.constexpr, HAS_INIT: gl.constexpr, STORE_PRE: gl.constexpr,
                   STORE_FIN: gl.constexpr, INIT_BYTES: gl.constexpr, PASSED: gl.constexpr):
    """MMA warp: A_c (P, N) = sum over the chunk's K steps of x_tile^T (P, BK) . B_tile (BK, N)."""
    NKT: gl.constexpr = CS // BK
    G = gl.num_programs(0)
    t = 0
    cc = 0
    for it in range(gl.program_id(0), n_items, G):
        for c in range(nchunks):
            a = cc % 2
            mbarrier.wait(acc_free.index(a), ((cc // 2) & 1) ^ 1)
            _tc_fence(1, 1)
            for kt in gl.static_range(NKT):
                slot = t % R
                mbarrier.wait(ready.index(slot), (t // R) & 1)
                mbarrier.wait(scaled.index(slot), (t // R) & 1)
                tcgen05_mma(x_s.index(slot).permute((1, 0)), b_s.index(slot), acc.index(a), use_acc=kt > 0,
                            mbarriers=[free.index(slot)])
                t += 1
            tcgen05_commit(acc_done.index(a))
            cc += 1


# ------------------------------------------------------------------------------------------------ kernel
@gluon.jit
def _chunk_states_kernel(xd, bd, initd, DT, DA, STP, PRP, FNP, n_items, seqlen, nchunks,
                         H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                         R: gl.constexpr, NWC: gl.constexpr, REGS_W: gl.constexpr, REGS_C: gl.constexpr,
                         HAS_INIT: gl.constexpr, STORE_PRE: gl.constexpr, STORE_FIN: gl.constexpr, INIT_BYTES: gl.constexpr, PASSED: gl.constexpr):
    TL: gl.constexpr = TensorMemoryLayout(block=(P, N), unpacked=True)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    XL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BK, P], gl.bfloat16)
    BLY: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BK, N], gl.bfloat16)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [R, BK, P], XL)
    b_s = gl.allocate_shared_memory(gl.bfloat16, [R, BK, N], BLY)
    if HAS_INIT:
        init_s = gl.allocate_shared_memory(initd.dtype, [P, N], initd.layout)
    else:
        init_s = b_s.index(0)                                          # placeholder (no initial state)
    ready = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    scaled = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    acc_done = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    acc_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    if HAS_INIT:
        init_bar = gl.allocate_shared_memory(gl.int64, [1], BL)
        init_free = gl.allocate_shared_memory(gl.int64, [1], BL)
    else:                                                              # placeholders (no initial state)
        init_bar = acc_done.index(0)
        init_free = acc_free.index(0)
    for i in gl.static_range(R):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(scaled.index(i), count=1)
        mbarrier.init(free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(acc_done.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=1)
    if HAS_INIT:
        mbarrier.init(init_bar, count=1)
        mbarrier.init(init_free, count=1)
    fence_async_shared()
    acc = allocate_tensor_memory(gl.float32, [2, P, N], TL)
    gl.warp_specialize(
        (x_s, b_s, ready, scaled, free, acc_done, acc_free, acc, init_s, init_bar, init_free, xd, bd, initd, DT, DA, STP, PRP, FNP,
         n_items, seqlen, nchunks, H, P, N, CS, BK, R, NWC, HAS_INIT, STORE_PRE, STORE_FIN, INIT_BYTES, PASSED),
        _epilogue_partition,
        (x_s, b_s, ready, scaled, free, acc_done, acc_free, acc, init_s, init_bar, init_free, xd, bd, initd, DT, DA, STP, PRP, FNP,
         n_items, seqlen, nchunks, H, P, N, CS, BK, R, NWC, HAS_INIT, STORE_PRE, STORE_FIN, INIT_BYTES, PASSED),
        [_load_partition, _compute_partition, _mma_partition],
        [1, NWC, 1],
        [REGS_W, REGS_C, REGS_W])


# ------------------------------------------------------------------------------------------------ launchers
def _check(B, x, dt, dA_cumsum, name):
    require_cuda(x, name)
    require_sm100(x.device, name)
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk = dt.shape
    _, _, ngroups, dstate = B.shape
    if (ngroups != 1 or headdim != 64 or dstate != 128 or chunk % BK
            or not (nchunks - 1) * chunk < seqlen <= nchunks * chunk
            or B.shape != (batch, seqlen, 1, dstate) or dt.shape != (batch, nheads, nchunks, chunk)
            or dA_cumsum.shape != dt.shape):
        raise RuntimeError(f"neuronspark.{name}: shapes x {tuple(x.shape)} B {tuple(B.shape)} dt {tuple(dt.shape)} "
                           f"(ngroups 1, P 64, N 128, chunks of a multiple of {BK} tokens covering the sequence)")
    if x.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise RuntimeError(f"neuronspark.{name}: bf16 x and B expected")
    if x.stride(3) != 1 or x.stride(2) != headdim or x.stride(1) % 8 or x.stride(0) % 8:
        raise RuntimeError(f"neuronspark.{name}: x strides {tuple(x.stride())}")
    if B.stride(3) != 1 or B.stride(1) % 8 or B.stride(0) % 8:
        raise RuntimeError(f"neuronspark.{name}: B strides {tuple(B.stride())}")
    if (not dt.is_contiguous() or not dA_cumsum.is_contiguous() or dt.dtype != torch.float32
            or dA_cumsum.dtype != torch.float32):
        raise RuntimeError(f"neuronspark.{name}: contiguous fp32 dt / dA_cumsum (b, h, c, l) expected")
    return batch, seqlen, nheads, headdim, nchunks, chunk, dstate


def _launch(B, x, dt, dA_cumsum, init, pre, states, fin, dims, passed=False):
    batch, seqlen, nheads, headdim, nchunks, chunk, dstate = dims
    # 3-D views (batch, seqlen, .): any batch stride (time slices of a longer buffer); rows past seqlen read as zeros
    xd = TensorDescriptor(x, [batch, seqlen, nheads * headdim], [x.stride(0), x.stride(1), 1], [1, BK, headdim],
                          gl.NVMMASharedLayout.get_default_for([1, BK, headdim], gl.bfloat16))
    bd = TensorDescriptor(B, [batch, seqlen, dstate], [B.stride(0), B.stride(1), 1], [1, BK, dstate],
                          gl.NVMMASharedLayout.get_default_for([1, BK, dstate], gl.bfloat16))
    has_init = init is not None
    if has_init:
        s0, s1, s2, s3 = (int(v) for v in init.stride())
        if s3 != 1 or s2 != dstate or s1 != headdim * s2 or s0 != nheads * s1:
            raise RuntimeError(f"neuronspark.chunk_states_prefill: initial_state strides {tuple(init.stride())} (rows of "
                               "(h, P, N) blocks packed per batch row expected)")
        initd = TensorDescriptor(init, [batch * nheads * headdim, dstate], [dstate, 1], [headdim, dstate],
                                 gl.NVMMASharedLayout.get_default_for([headdim, dstate], _GL_DT[init.dtype]))
    else:
        initd = xd                                                     # placeholder (no initial state)
    n_items = batch * nheads
    G = min(CTAS_PER_SM * num_sms(x.device), n_items)
    with torch.cuda.device(x.device.index):
        _chunk_states_kernel[(G,)](
            xd, bd, initd, dt, dA_cumsum, states, pre if pre is not None else states, fin if fin is not None else states,
            n_items, seqlen, nchunks,
            H=nheads, P=headdim, N=dstate, CS=chunk, BK=BK, R=R, NWC=NWC, REGS_W=REGS_W, REGS_C=REGS_C,
            HAS_INIT=has_init, STORE_PRE=pre is not None, STORE_FIN=fin is not None,
            INIT_BYTES=headdim * dstate * (init.element_size() if has_init else 0), PASSED=passed, num_warps=4)


def chunk_states_train(B, x, dt, dA_cumsum, out_dtype, passed_fp32=False):
    """(states_pre (b, nc, h, P, N) fp32, states (b, nc, h, P, N) in out_dtype): see the module docstring.
    ``passed_fp32``: the fp32 output holds the passed states s_c (the state entering chunk c, s_0 = 0) instead of the
    chunks' own states -- the input of :func:`neuronspark.ops.gluon.dendritic_dstates.dstates_train`."""
    dims = _check(B, x, dt, dA_cumsum, "chunk_states_train")
    batch, seqlen, nheads, headdim, nchunks, chunk, dstate = dims
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"neuronspark.chunk_states_train: out_dtype {out_dtype}")
    pre = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=torch.float32)
    states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=out_dtype)
    _launch(B, x, dt, dA_cumsum, None, pre, states, None, dims, passed=passed_fp32)
    return pre, states


def chunk_states_prefill(B, x, dt, dA_cumsum, initial_state, out_dtype):
    """(states (b, nc, h, P, N) in out_dtype, final_state (b, h, P, N) fp32) from the carried ``initial_state``
    (b, h, P, N) fp32 or bf16 (unit N stride, (h, P, N) packed per batch row; a batch-row slice of a larger state buffer
    qualifies) or from zero (``None``, a row's first call): see the module docstring."""
    dims = _check(B, x, dt, dA_cumsum, "chunk_states_prefill")
    batch, seqlen, nheads, headdim, nchunks, chunk, dstate = dims
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"neuronspark.chunk_states_prefill: out_dtype {out_dtype}")
    if initial_state is not None and (tuple(initial_state.shape) != (batch, nheads, headdim, dstate)
                                      or initial_state.dtype not in (torch.float32, torch.bfloat16)):
        raise RuntimeError(f"neuronspark.chunk_states_prefill: initial_state {initial_state.dtype} "
                           f"{tuple(initial_state.shape)}")
    states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=out_dtype)
    fin = torch.empty((batch, nheads, headdim, dstate), device=x.device, dtype=torch.float32)
    _launch(B, x, dt, dA_cumsum, initial_state, None, states, fin, dims)
    return states, fin


def chunk_states_train_ref(B, x, dt, dA_cumsum, out_dtype):
    """Pure-torch reference: :func:`neuronspark.ops.triton.dendritic_states.csp_fused_ref`."""
    from neuronspark.ops.triton.dendritic_states import csp_fused_ref
    return csp_fused_ref(B, x, dt, dA_cumsum, out_dtype)


def chunk_states_prefill_ref(B, x, dt, dA_cumsum, initial_state, out_dtype):
    """Pure-torch reference: :func:`neuronspark.ops.triton.dendritic_states.csp_prefill_ref`."""
    from neuronspark.ops.triton.dendritic_states import csp_prefill_ref
    return csp_prefill_ref(B, x, dt, dA_cumsum, initial_state, out_dtype)
