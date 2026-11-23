"""dendritic-layer chunk states of the SSD scan, written in Gluon for Blackwell GPUs: the chunk-state product and the
state passing of one (b, head group) item walking its chunks in order, with the state carried in tensor memory.

For chunk c of head h (chunk length CS, dA_last = dA_cumsum[b, h, c, CS - 1]):

    A_c     = sum_k xs[k]^T B[k]      xs[k] = x[k] scale[k],  scale[k] = exp(min(dA_last - dA_cumsum[k], 0)) dt[k]
    s_{c+1} = exp(dA_last) s_c + A_c                                                          (P, N) fp32

the chunk-state body and the state-passing step.  The scale is a per-token, per-head factor, so it rides on x (one
64-column tile per head) and B is shared by the heads of an item: one MMA of M = NH x 64 rows serves NH heads.  The fp32
scaled x enters the tensor cores as two bf16 parts, hi = bf16(xs) and lo = bf16(xs - hi): 16 significant bits of the
fp32 product instead of the 8 of one bf16 operand.
Forms:

* :func:`chunk_states_train` (no initial state): ``states[c] = s_c`` in the output dtype (``states[0] = 0``; the final
  state is not stored, so the last chunk's product is not formed) -- the states the chunk scan reads and the training
  backward's state gradients take;
* :func:`chunk_states_prefill` (the carried state ``s_0``, or zero): ``states[c] = s_c`` in the output dtype and the final
  state ``s_nc`` fp32 -- the same quantities from the initial state.

Per CTA (persistent, items (b, group of NH heads) in round-robin): a load warp streams the (BK, NH P) x tiles of the
group and the (BK, N) B tiles of each K step through an R-deep mbarrier ring; a compute partition of NWC warps scales
each x tile (one factor per token and head) and writes its hi part in place and its lo part to a second ring tile; an
MMA warp issues A_c += hi^T . B_tile + lo^T . B_tile (the x tiles as permuted MN-major views, M = NH P, N = N, K = BK)
into one of two TMEM accumulators; the default partition (4 warps) holds the carried state in a third TMEM buffer and,
per chunk, column slice by column slice, stores the entering state and updates the carry (plain stores from registers).

Numerics: the scale and the passing step are the reference's expressions (``exp``), the scaled x split into its hi /
lo bf16 parts; the fp32 accumulation over the chunk's tokens is the tensor cores' order (the per-operator rule class of
:mod:`neuronspark.utils.testing` against :func:`chunk_states_train_ref` / :func:`chunk_states_prefill_ref`).  A partial
last chunk is supported (prefill): its rows past the sequence get a zero scaled x (dA_last is the chunk's last element,
as the passing reads it).  x and B are read through 3-D (batch, seqlen, .) descriptors: any batch stride (a time slice of
a longer buffer qualifies).  The initial state is read by plain loads (no barrier that the training form leaves unused).

Launch constants: BK 64 tokens per K step, R 4 ring slots, NWC 8 compute warps, registers 40 (load / MMA warps) and 128
(compute), ES 32 columns per epilogue slice; NH = 2 heads per item when single heads would take more than one wave of
the SMs, else 1; grid = min(#SM, items) persistent CTAs.  No partition uses an operation that needs compiler scratch
memory (a layout conversion, a reduction across warps): the allocator places such scratch over the warp_specialize
capture buffer, which partitions that start later still read.
"""
from __future__ import annotations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout, allocate_tensor_memory, get_tmem_32x32b_reg_layout, tcgen05_mma, tcgen05_commit, mbarrier, tma,
    fence_async_shared)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from ....neuronspark.ops.gluon._ptx import _tc_fence
from ....neuronspark.ops.triton import num_sms, require_cuda, require_sm100

__all__ = ["BK", "R", "NWC", "REGS_W", "REGS_C", "ES", "heads_per_item", "chunk_states_train", "chunk_states_prefill",
           "chunk_states_train_ref", "chunk_states_prefill_ref"]

# ------------------------------------------------------------------------------------------------ launch constants
BK = 64          # tokens per K step
R = 4            # ring slots (K steps in flight)
NWC = 8          # compute warps (the x scaling)
REGS_W = 40      # registers of the load / MMA warps
REGS_C = 128     # registers of the compute partition
ES = 32          # columns of N per epilogue slice


def heads_per_item(batch: int, nheads: int, sms: int) -> int:
    """2 (head pairs, M = 128 MMAs, B read once per pair) when one head per item would take more than one wave of the
    SMs, else 1 (one wave of single heads walks faster than a wave of pairs)."""
    return 2 if nheads % 2 == 0 and batch * nheads > sms else 1


# ------------------------------------------------------------------------------------------------ partitions
@gluon.jit
def _epilogue_partition(x_s, xl_s, b_s, ready, scaled, free, acc_done, acc_free, acc, xd, bd, DT, DA, STP, FNP, INI,
                        n_items, seqlen, nchunks,
                        H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                        R: gl.constexpr, NWC: gl.constexpr, NH: gl.constexpr, ES: gl.constexpr,
                        HAS_INIT: gl.constexpr, STORE_FIN: gl.constexpr):
    """Default partition (4 warps): per item the carried state in TMEM (zero or the initial state); per walked chunk,
    slice by slice, the entering state stored in the output dtype and the carry updated with the chunk's product;
    then the last state (training: the entering state of the last chunk; prefill: the final state in fp32)."""
    M: gl.constexpr = NH * P
    SL: gl.constexpr = get_tmem_32x32b_reg_layout(M, ES, [M, ES], 4)
    rows = gl.arange(0, M, layout=gl.SliceLayout(1, SL))
    cols = gl.arange(0, ES, layout=gl.SliceLayout(0, SL))
    hr = rows // P                                                     # the head of each row within the item
    pr = rows - hr * P
    NG: gl.constexpr = H // NH
    NS: gl.constexpr = N // ES
    cr = acc.index(2)                                                  # the carried state
    G = gl.num_programs(0)
    nwalk = nchunks if STORE_FIN else nchunks - 1                      # chunks whose product is formed
    cc = 0                                                             # chunks consumed by this CTA (accumulator index)
    for it in range(gl.program_id(0), n_items, G):
        b = it // NG
        hrow = (it - b * NG) * NH + hr                                 # the head of each row
        for j in gl.static_range(NS):
            if HAS_INIT:
                off = ((b * H + hrow) * P + pr).to(gl.int64)[:, None] * N + (j * ES + cols)[None, :]
                cr.slice(j * ES, ES).store(gl.load(INI + off).to(gl.float32))
            else:
                cr.slice(j * ES, ES).store(gl.zeros([M, ES], gl.float32, layout=SL))
        for c in range(nwalk):
            da_last = gl.load(DA + ((b * H + hrow) * nchunks + c) * CS + (CS - 1))
            decay = gl.exp(da_last)
            u = cc % 2
            base = (((b * nchunks + c) * H + hrow) * P + pr).to(gl.int64)[:, None] * N
            mbarrier.wait(acc_done.index(u), (cc // 2) & 1)
            _tc_fence(1, 4)
            for j in gl.static_range(NS):
                s = cr.slice(j * ES, ES).load(SL)
                gl.store(STP + base + (j * ES + cols)[None, :], s.to(STP.dtype.element_ty))   # the state entering chunk c
                ac = acc.index(u).slice(j * ES, ES).load(SL)
                cr.slice(j * ES, ES).store(decay[:, None] * s + ac)   # the state-passing step
            _tc_fence(0, 4)
            gl.thread_barrier()
            mbarrier.arrive(acc_free.index(u))
            cc += 1
        if STORE_FIN:
            base = ((b * H + hrow) * P + pr).to(gl.int64)[:, None] * N
            for j in gl.static_range(NS):
                gl.store(FNP + base + (j * ES + cols)[None, :], cr.slice(j * ES, ES).load(SL))
        else:
            base = (((b * nchunks + nchunks - 1) * H + hrow) * P + pr).to(gl.int64)[:, None] * N
            for j in gl.static_range(NS):
                s = cr.slice(j * ES, ES).load(SL)
                gl.store(STP + base + (j * ES + cols)[None, :], s.to(STP.dtype.element_ty))
        gl.thread_barrier()                                            # every thread's carry reads before the next item


@gluon.jit
def _load_partition(x_s, xl_s, b_s, ready, scaled, free, acc_done, acc_free, acc, xd, bd, DT, DA, STP, FNP, INI,
                    n_items, seqlen, nchunks,
                    H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                    R: gl.constexpr, NWC: gl.constexpr, NH: gl.constexpr, ES: gl.constexpr,
                    HAS_INIT: gl.constexpr, STORE_FIN: gl.constexpr):
    """Load warp: per K step the (BK, NH P) x tile of the item's heads and the (BK, N) B tile into the ring."""
    M: gl.constexpr = NH * P
    NKT: gl.constexpr = CS // BK
    NG: gl.constexpr = H // NH
    G = gl.num_programs(0)
    nwalk = nchunks if STORE_FIN else nchunks - 1
    t = 0
    for it in range(gl.program_id(0), n_items, G):
        b = it // NG
        h0 = (it - b * NG) * NH
        for c in range(nwalk):
            for kt in gl.static_range(NKT):
                slot = t % R
                mbarrier.wait(free.index(slot), ((t // R) & 1) ^ 1)
                mbarrier.expect(ready.index(slot), BK * M * 2 + BK * N * 2)
                t0 = c * CS + kt * BK
                # the 2-D ring tiles seen as the descriptors' (1, BK, .) boxes (the same bytes and swizzle)
                tma.async_copy_global_to_shared(xd, [b, t0, h0 * P], ready.index(slot),
                                                x_s.index(slot)._reinterpret(gl.bfloat16, [1, BK, M], xd.layout))
                tma.async_copy_global_to_shared(bd, [b, t0, 0], ready.index(slot),
                                                b_s.index(slot)._reinterpret(gl.bfloat16, [1, BK, N], bd.layout))
                t += 1


@gluon.jit
def _compute_partition(x_s, xl_s, b_s, ready, scaled, free, acc_done, acc_free, acc, xd, bd, DT, DA, STP, FNP, INI,
                       n_items, seqlen, nchunks,
                       H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                       R: gl.constexpr, NWC: gl.constexpr, NH: gl.constexpr, ES: gl.constexpr,
                       HAS_INIT: gl.constexpr, STORE_FIN: gl.constexpr):
    """Compute partition (NWC warps): xs[k, (h, p)] = x_tile[k, (h, p)] exp(min(dA_last_h - dA_h[k], 0)) dt_h[k] in
    fp32; hi = bf16(xs) in place of the x tile, lo = bf16(xs - hi) in the lo tile."""
    M: gl.constexpr = NH * P
    NKT: gl.constexpr = CS // BK
    NG: gl.constexpr = H // NH
    TC: gl.constexpr = M // 8                                          # threads across the tile's columns
    L: gl.constexpr = gl.BlockedLayout([1, 8], [32 // TC, TC], [NWC, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(1, L)
    offs_kl = gl.arange(0, BK, layout=LK)
    first = (gl.arange(0, M, layout=gl.SliceLayout(0, L)) < P)[None, :]   # the columns of the item's first head
    G = gl.num_programs(0)
    nwalk = nchunks if STORE_FIN else nchunks - 1
    t = 0
    for it in range(gl.program_id(0), n_items, G):
        b = it // NG
        h0 = (it - b * NG) * NH
        for c in range(nwalk):
            base0 = ((b * H + h0) * nchunks + c) * CS
            dl0 = gl.load(DA + base0 + (CS - 1))
            if NH == 2:
                base1 = base0 + nchunks * CS                           # head h0 + 1
                dl1 = gl.load(DA + base1 + (CS - 1))
            lim = seqlen - c * CS                                      # tokens of the chunk inside the sequence
            for kt in gl.static_range(NKT):
                slot = t % R
                # the K step's dA / dt loaded in the tile's row layout: no layout conversion in a worker partition
                w0 = gl.exp(gl.minimum(dl0 - gl.load(DA + base0 + kt * BK + offs_kl), 0.0)) \
                    * gl.load(DT + base0 + kt * BK + offs_kl)
                if NH == 2:
                    w1 = gl.exp(gl.minimum(dl1 - gl.load(DA + base1 + kt * BK + offs_kl), 0.0)) \
                        * gl.load(DT + base1 + kt * BK + offs_kl)
                mbarrier.wait(ready.index(slot), (t // R) & 1)
                xt = x_s.index(slot).load(L).to(gl.float32)
                if NH == 2:
                    xs = gl.where(first, xt * w0[:, None], xt * w1[:, None])
                else:
                    xs = xt * w0[:, None]
                if kt * BK + BK > lim:                                 # a partial chunk: rows past the sequence weigh 0
                    xs = gl.where((kt * BK + offs_kl < lim)[:, None], xs, 0.0)
                hi = xs.to(gl.bfloat16)
                x_s.index(slot).store(hi)
                xl_s.index(slot).store((xs - hi.to(gl.float32)).to(gl.bfloat16))
                fence_async_shared()
                gl.thread_barrier()
                mbarrier.arrive(scaled.index(slot))
                t += 1


@gluon.jit
def _mma_partition(x_s, xl_s, b_s, ready, scaled, free, acc_done, acc_free, acc, xd, bd, DT, DA, STP, FNP, INI,
                   n_items, seqlen, nchunks,
                   H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr, BK: gl.constexpr,
                   R: gl.constexpr, NWC: gl.constexpr, NH: gl.constexpr, ES: gl.constexpr,
                   HAS_INIT: gl.constexpr, STORE_FIN: gl.constexpr):
    """MMA warp: A_c (NH P, N) = sum over the chunk's K steps of hi^T (NH P, BK) . B_tile (BK, N) + lo^T . B_tile."""
    NKT: gl.constexpr = CS // BK
    G = gl.num_programs(0)
    nwalk = nchunks if STORE_FIN else nchunks - 1
    t = 0
    cc = 0
    for it in range(gl.program_id(0), n_items, G):
        for c in range(nwalk):
            a = cc % 2
            mbarrier.wait(acc_free.index(a), ((cc // 2) & 1) ^ 1)
            _tc_fence(1, 1)
            for kt in gl.static_range(NKT):
                slot = t % R
                mbarrier.wait(ready.index(slot), (t // R) & 1)
                mbarrier.wait(scaled.index(slot), (t // R) & 1)
                tcgen05_mma(x_s.index(slot).permute((1, 0)), b_s.index(slot), acc.index(a), use_acc=kt > 0)
                tcgen05_mma(xl_s.index(slot).permute((1, 0)), b_s.index(slot), acc.index(a), use_acc=True,
                            mbarriers=[free.index(slot)])
                t += 1
            tcgen05_commit(acc_done.index(a))
            cc += 1


# ------------------------------------------------------------------------------------------------ kernel
@gluon.jit
def _chunk_states_group_kernel(xd, bd, DT, DA, STP, FNP, INI, n_items, seqlen, nchunks,
                               H: gl.constexpr, P: gl.constexpr, N: gl.constexpr, CS: gl.constexpr,
                               BK: gl.constexpr, R: gl.constexpr, NWC: gl.constexpr, NH: gl.constexpr,
                               ES: gl.constexpr, REGS_W: gl.constexpr, REGS_C: gl.constexpr,
                               HAS_INIT: gl.constexpr, STORE_FIN: gl.constexpr):
    M: gl.constexpr = NH * P
    TL: gl.constexpr = TensorMemoryLayout(block=(M, N), unpacked=True)
    BL: gl.constexpr = mbarrier.MBarrierLayout()
    XL: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BK, M], gl.bfloat16)
    BLY: gl.constexpr = gl.NVMMASharedLayout.get_default_for([BK, N], gl.bfloat16)
    x_s = gl.allocate_shared_memory(gl.bfloat16, [R, BK, M], XL)
    xl_s = gl.allocate_shared_memory(gl.bfloat16, [R, BK, M], XL)
    b_s = gl.allocate_shared_memory(gl.bfloat16, [R, BK, N], BLY)
    ready = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    scaled = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    free = gl.allocate_shared_memory(gl.int64, [R, 1], BL)
    acc_done = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    acc_free = gl.allocate_shared_memory(gl.int64, [2, 1], BL)
    for i in gl.static_range(R):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(scaled.index(i), count=1)
        mbarrier.init(free.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(acc_done.index(i), count=1)
        mbarrier.init(acc_free.index(i), count=1)
    fence_async_shared()
    acc = allocate_tensor_memory(gl.float32, [3, M, N], TL)          # two chunk products and the carried state
    gl.warp_specialize(
        (x_s, xl_s, b_s, ready, scaled, free, acc_done, acc_free, acc, xd, bd, DT, DA, STP, FNP, INI,
         n_items, seqlen, nchunks, H, P, N, CS, BK, R, NWC, NH, ES, HAS_INIT, STORE_FIN),
        _epilogue_partition,
        (x_s, xl_s, b_s, ready, scaled, free, acc_done, acc_free, acc, xd, bd, DT, DA, STP, FNP, INI,
         n_items, seqlen, nchunks, H, P, N, CS, BK, R, NWC, NH, ES, HAS_INIT, STORE_FIN),
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


def _launch(B, x, dt, dA_cumsum, init, states, fin, dims):
    batch, seqlen, nheads, headdim, nchunks, chunk, dstate = dims
    sms = num_sms(x.device)
    nh = heads_per_item(batch, nheads, sms)
    m = nh * headdim
    # 3-D views (batch, seqlen, .): any batch stride (time slices of a longer buffer); rows past seqlen read as zeros
    xd = TensorDescriptor(x, [batch, seqlen, nheads * headdim], [x.stride(0), x.stride(1), 1], [1, BK, m],
                          gl.NVMMASharedLayout.get_default_for([1, BK, m], gl.bfloat16))
    bd = TensorDescriptor(B, [batch, seqlen, dstate], [B.stride(0), B.stride(1), 1], [1, BK, dstate],
                          gl.NVMMASharedLayout.get_default_for([1, BK, dstate], gl.bfloat16))
    has_init = init is not None
    if has_init:
        s0, s1, s2, s3 = (int(v) for v in init.stride())
        if s3 != 1 or s2 != dstate or s1 != headdim * s2 or s0 != nheads * s1:
            raise RuntimeError(f"neuronspark.chunk_states_prefill: initial_state strides {tuple(init.stride())} (rows of "
                               "(h, P, N) blocks packed per batch row expected)")
    n_items = batch * (nheads // nh)
    G = min(sms, n_items)
    with torch.cuda.device(x.device.index):
        _chunk_states_group_kernel[(G,)](
            xd, bd, dt, dA_cumsum, states, fin if fin is not None else states, init if has_init else states,
            n_items, seqlen, nchunks,
            H=nheads, P=headdim, N=dstate, CS=chunk, BK=BK, R=R, NWC=NWC, NH=nh, ES=ES, REGS_W=REGS_W, REGS_C=REGS_C,
            HAS_INIT=has_init, STORE_FIN=fin is not None, num_warps=4)


def chunk_states_train(B, x, dt, dA_cumsum, out_dtype):
    """states (b, nc, h, P, N) in out_dtype: the state entering each chunk (``states[0] = 0``), see the module
    docstring.  In bf16 it is both the chunk scan's operand and the input of
    :func:`neuronspark.ops.gluon.dendritic_dstates.dstates_train`."""
    dims = _check(B, x, dt, dA_cumsum, "chunk_states_train")
    batch, seqlen, nheads, headdim, nchunks, chunk, dstate = dims
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"neuronspark.chunk_states_train: out_dtype {out_dtype}")
    states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=out_dtype)
    _launch(B, x, dt, dA_cumsum, None, states, None, dims)
    return states


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
    _launch(B, x, dt, dA_cumsum, initial_state, states, fin, dims)
    return states, fin


def chunk_states_train_ref(B, x, dt, dA_cumsum, out_dtype):
    """Pure-torch reference: the states of :func:`neuronspark.ops.triton.dendritic_states.csp_fused_ref`."""
    from ....neuronspark.ops.triton.dendritic_states import csp_fused_ref
    return csp_fused_ref(B, x, dt, dA_cumsum, out_dtype)[1]


def chunk_states_prefill_ref(B, x, dt, dA_cumsum, initial_state, out_dtype):
    """Pure-torch reference: :func:`neuronspark.ops.triton.dendritic_states.csp_prefill_ref`."""
    from ....neuronspark.ops.triton.dendritic_states import csp_prefill_ref
    return csp_prefill_ref(B, x, dt, dA_cumsum, initial_state, out_dtype)
