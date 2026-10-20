"""Inference ring kernels of the memory layers' pool read.

Inside an inference cache every memory head keeps a ring of the last ``W`` synaptic events (``utils/cache.py``
``MemoryState``): ``pool_k`` (B, H, W, dk) bf16 unit key tags, ``pool_gv`` (B, H, W, dv) bf16 tagged values
``bf16(sigmoid(g + wbias - sparsity)) * v`` and ``pool_cum`` (B, H, W) fp32 absolute eligibility clocks (+inf = empty
slot).  Token ``t`` of a call writes slot ``(ring_ptr + t) % W``.  The pool read of a query row is

    s_w    = bf16(bf16(bf16(q) . k_w) * gain)                      (gain = sqrt(dk) = 16.0)
    bias_w = bf16((cum_t - cum_w) [-inf for later slots of this call] + sum_n m_n[t, w] beta[n, h])
    p      = softmax_w(s + bias)                                   (fp32)
    out    = bf16(bf16(p) . gv)

which is the bf16-autocast expression chain of the memory layer's inference forward (two bf16 bmm calls, the clock bias
cast to the pattern dtype, an fp32 softmax).  The last head additionally reads the ring with ``M = 4`` extra queries.
The induction term is formed in the kernels from the bool match matrices of the model-level induction state (``m_n``
for the orders 2, 3, 4; products with m in {0, 1} are exact, so the term's value does not depend on contraction).

Operators (each: private launcher, ``torch.library`` custom op with a fake, public function, pure-torch ``*_ref``):

``ring_read_t1``  decode step (T = 1), split-W form.  Kernel 1, program (row b, head h, chunk): the logits of the
    chunk's live slots into an fp32 scratch and the chunk's (running max, sum of exp) per query row (libdevice expf);
    kernel 2, same grid: the row's max / denominator combined from the chunk pairs, ``p = bf16(div.rn(exp(l - m), Z))``
    times the tagged values, an fp32 partial read per (row, query, chunk); the partials are summed over the chunks in
    fp32 and rounded to bf16.  Only the row's live slots are read: the ``L = min(pos + 1, W)`` slots ending at this
    step's slot ``ring_ptr % W`` (every other slot has an infinite clock: weight 0 in the softmax).  Deterministic (fixed
    chunk order, no atomics).  Compiled without FMA contraction.  The memory layer reads the ring with this kernel at
    decode batch sizes 1 and 32 (``RING_READ_KERNEL_BATCHES``); at the other sizes it reads with the torch expression
    chain (the module's own torch form), which ``ring_read_t1_ref`` evaluates.
``xq_form_t1``  decode step: the last head's extra-query formation in one kernel per (row, query): the rank-r
    projection (bf16 operands, fp32 sum, bf16 result), SiLU (bf16 result), interleaved RoPE in bf16 arithmetic (each
    product and sum rounded to bf16), L2 normalisation, division and the gain ``exp(clamp(psi, -3, 3)) * head_gain`` in
    fp32, one rounding to bf16 at the store.  The memory layer uses it at the decode batch sizes whose ring read is the
    torch chain.
``ring_write_prefill``  prefill (T >= 2): the call's ring write as one kernel over the (t, b, h) rows: the k row's fp32
    norm (Triton's row sum of the squares, sqrt), ``max(norm, 1e-12)``, the fp32 division, one rounding to bf16 into
    ``pool_k``; the q row the same way times the head's fp32 gain, stored fp32 (the recall's query); the fp32 gate
    through IEEE add / sub and ATen's sigmoid ``1 / (1 + expf(-x))`` (inline PTX without flush-to-zero: the sigmoid is
    rounded to bf16 before the product, so its exact form is kept), rounded to bf16, times v, rounded, into ``pool_gv``.  Operands whose channel dimension is not unit-stride (the T = 2 form of the qkv
    conv) take the reference expression chain ``ring_write_prefill_ref``.
``ring_recall_prefill``  prefill: the pool read of every query row of the call over the whole ring (earlier calls'
    events at their clocks, this call's events causal), for the six heads (``xq=False``: group g reads ring head g) or
    the extra queries (``xq=True``: every group reads ring head ``hr``, the query clocks are those of this call's
    slots).  T >= 64: two launches -- a statistics pass (the scores of every valid (64 x 128) tile into an fp32
    scratch, the online row max / sum, a per-tile validity flag) and a value pass (``p = bf16(exp(x - m) * (1 / l))``
    times the values on bf16 tensor cores, fp32 accumulation); T < 64: one kernel with 16-row tiles recomputing the
    scores in the value pass.  Both use ``tl.exp`` (ex2.approx after the log2 e multiply) and one reciprocal per row.

Numerics: every bf16 rounding point of the reference chain is kept; the fp32 summation orders of the tensor-core dots,
the chunked / online softmax denominators and the approximate exponential of the prefill recall differ from it.
Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the references.

Interface (shared counters of ``InferenceParams``, read during a layer call, advanced by the model after the call):
``pos`` (B,) int64 = tokens each row consumed before this call; ``ring_ptr`` / ``p0`` (1,) int64 = the ring pointer at
the start of this call (token t of the call is at slot ``(ptr + t) % W``).  ``ring_read_t1`` takes the tokens consumed
*before* the step (the live length is ``pos + 1``); the extra-query and induction operands are explicit arguments.
The launch configurations are the fixed constants below.
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (Triton resolves the helpers through this module's globals)
    _add_rn, _div_rn, _mul_rn, _rn_add_f32, _rn_mul_f32, _sub_rn,
)

__all__ = [
    "RING_READ_KERNEL_BATCHES", "RING2_QUERY_ROWS", "RING2_CONFIGS", "XQ_FORM_WARPS", "RING_WRITE_HEAD_DIM",
    "RING_WRITE_ROWS", "RING_WRITE_WARPS", "RECALL_BT", "RECALL_BT_SHORT", "RECALL_BW", "RECALL_WARPS",
    "RECALL_STAGES", "RECALL_STATS_WARPS", "RECALL_STATS_STAGES", "RECALL_VALUE_WARPS", "RECALL_VALUE_STAGES",
    "RECALL_VALUE_BDV",
    "ring_read_t1_takes", "ring_read_t1", "ring_read_t1_ref",
    "xq_form_t1", "xq_form_t1_ref", "xq_weight_stack", "xq_head_gain",
    "ring_write_prefill", "ring_write_prefill_ref",
    "ring_recall_prefill", "ring_recall_prefill_ref",
]

# ================================================================================================ fixed launch constants
# Decode ring read (split-W): the decode batch sizes at which the memory layer reads the ring with this kernel; at every
# other size it reads with the torch expression chain of the memory layer.
RING_READ_KERNEL_BATCHES = (1, 32)
# query rows of a program's dot tile: row 0 the head's query, rows 1..M the extra queries of the last head (padded)
RING2_QUERY_ROWS = 16
# (largest decode batch, live slots per program, slots per tile, warps, stages); the first row with b <= its bound
RING2_CONFIGS = (
    (1, 64, 64, 8, 2),
    (8, 128, 64, 4, 2),
    (None, 512, 64, 4, 3),
)
# extra-query formation: one program per (row, query)
XQ_FORM_WARPS = 4
# prefill ring write: head width, (t, b, h) rows per program, warps; the norm's reduction order is fixed (see above)
RING_WRITE_HEAD_DIM = 256
RING_WRITE_ROWS = 4
RING_WRITE_WARPS = 4
# prefill recall: query rows per program (T >= 64 / T < 64), ring slots per tile, warps / stages of the one-kernel form
RECALL_BT = 64
RECALL_BT_SHORT = 16
RECALL_BW = 128
RECALL_WARPS = 8
RECALL_STAGES = 1
# prefill recall, two-launch form (T >= 64): statistics pass, value pass, value columns per value program
RECALL_STATS_WARPS = 4
RECALL_STATS_STAGES = 1
RECALL_VALUE_WARPS = 8
RECALL_VALUE_STAGES = 1
RECALL_VALUE_BDV = 256

_OUT_TL = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ================================================================================================ decode ring read (T = 1)

@triton.jit
def _ring2_logits_kernel(qb_ptr, qx_ptr, k_ptr, cum_ptr, cumt_ptr, pos_ptr, ptr_ptr,
                         m0_ptr, m1_ptr, m2_ptr, m3_ptr, beta_ptr, lg_ptr, st_ptr,
                         H, W, XH, gain, s_cumt_b, s_cumt_h, s_m_b, s_m_w,
                         DK: tl.constexpr, MX: tl.constexpr, N_ORD: tl.constexpr, QP: tl.constexpr,
                         BLOCK_W: tl.constexpr, CHUNK: tl.constexpr, NS: tl.constexpr):
    """Kernel 1, program (b, h, chunk): the logits of the live positions [chunk * CHUNK, min((chunk + 1) * CHUNK, L))
    to the scratch and the chunk's (running max, sum of exp) per query row to st (b, row, chunk, 2); a chunk past the
    live length stores (-inf, 0)."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    b64 = b.to(tl.int64)
    bh = b64 * H + h
    NR = H + MX
    offs_r = tl.arange(0, QP)
    offs_dk = tl.arange(0, DK)
    is_x = h == XH
    qbase = tl.load(qb_ptr + bh * DK + offs_dk)
    if MX > 0:
        xrow = (offs_r >= 1) & (offs_r <= MX) & is_x
        qx = tl.load(qx_ptr + (b64 * MX + (offs_r[:, None] - 1)) * DK + offs_dk[None, :],
                     mask=xrow[:, None], other=0.0)
        q = tl.where(offs_r[:, None] == 0, qbase[None, :], qx)
        rmask = (offs_r == 0) | xrow
    else:
        q = tl.where(offs_r[:, None] == 0, qbase[None, :], tl.zeros([QP, DK], qbase.dtype))
        rmask = offs_r == 0
    lrow = tl.where(offs_r == 0, h, H + offs_r - 1)
    lbase = (b64 * NR + lrow) * W
    L = tl.load(pos_ptr + b).to(tl.int32) + 1                    # live length: the tokens before this step + this one
    L = tl.minimum(tl.maximum(L, 1), W)
    p = (tl.load(ptr_ptr) % W).to(tl.int32)
    start = p - L + 1 + W
    j_lo = sp * CHUNK
    j_hi = tl.minimum(j_lo + CHUNK, L)
    cum_t = tl.load(cumt_ptr + b64 * s_cumt_b + h * s_cumt_h)
    if N_ORD > 0:
        be0 = tl.load(beta_ptr + 0 * H + h)
    if N_ORD > 1:
        be1 = tl.load(beta_ptr + 1 * H + h)
    if N_ORD > 2:
        be2 = tl.load(beta_ptr + 2 * H + h)
    if N_ORD > 3:
        be3 = tl.load(beta_ptr + 3 * H + h)
    kbase = k_ptr + bh * W * DK
    cbase = cum_ptr + bh * W
    m_i = tl.full([QP], float("-inf"), tl.float32)
    z_i = tl.zeros([QP], tl.float32)
    for j0 in range(j_lo, j_hi, BLOCK_W):
        offs_j = j0 + tl.arange(0, BLOCK_W)
        valid = offs_j < j_hi
        slot = (start + offs_j) % W
        kt = tl.load(kbase + slot.to(tl.int64)[:, None] * DK + offs_dk[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kt))                                  # (QP, BLOCK_W) fp32 accumulation
        s = s.to(tl.bfloat16).to(tl.float32)
        s = (s * gain).to(tl.bfloat16).to(tl.float32)
        cj = tl.load(cbase + slot, mask=valid, other=float("inf"))
        bias = cum_t - cj
        if N_ORD > 0:
            mo = m0_ptr + b64 * s_m_b + slot.to(tl.int64) * s_m_w
            t = tl.load(mo, mask=valid, other=0).to(tl.float32) * be0
            if N_ORD > 1:
                t = t + tl.load(m1_ptr + b64 * s_m_b + slot.to(tl.int64) * s_m_w, mask=valid,
                                other=0).to(tl.float32) * be1
            if N_ORD > 2:
                t = t + tl.load(m2_ptr + b64 * s_m_b + slot.to(tl.int64) * s_m_w, mask=valid,
                                other=0).to(tl.float32) * be2
            if N_ORD > 3:
                t = t + tl.load(m3_ptr + b64 * s_m_b + slot.to(tl.int64) * s_m_w, mask=valid,
                                other=0).to(tl.float32) * be3
            bias = bias + t
        bias = bias.to(tl.bfloat16).to(tl.float32)
        lg = s + bias[None, :]
        lg = tl.where(valid[None, :], lg, float("-inf"))
        tl.store(lg_ptr + lbase[:, None] + offs_j[None, :], lg, mask=rmask[:, None] & valid[None, :])
        m_new = tl.maximum(m_i, tl.max(lg, axis=1))
        z_i = z_i * libdevice.exp(m_i - m_new) + tl.sum(libdevice.exp(lg - m_new[:, None]), axis=1)
        m_i = m_new
    sb = ((b64 * NR + lrow) * NS + sp) * 2
    tl.store(st_ptr + sb, m_i, mask=rmask)
    tl.store(st_ptr + sb + 1, z_i, mask=rmask)


@triton.jit
def _ring2_pv_kernel(lg_ptr, st_ptr, gv_ptr, pos_ptr, ptr_ptr, part_ptr, H, W, XH,
                     DV: tl.constexpr, MX: tl.constexpr, QP: tl.constexpr, BLOCK_W: tl.constexpr,
                     CHUNK: tl.constexpr, NS: tl.constexpr, NSP: tl.constexpr):
    """Kernel 2, program (b, h, chunk): the row's maximum m and denominator Z from the NS chunk pairs, then
    p = bf16(exp(l - m) / Z) over the chunk times the tagged values -> the fp32 partial read part (b, row, chunk, DV)
    (zeros past L)."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    b64 = b.to(tl.int64)
    bh = b64 * H + h
    NR = H + MX
    offs_r = tl.arange(0, QP)
    offs_dv = tl.arange(0, DV)
    is_x = h == XH
    if MX > 0:
        rmask = (offs_r == 0) | ((offs_r >= 1) & (offs_r <= MX) & is_x)
    else:
        rmask = offs_r == 0
    lrow = tl.where(offs_r == 0, h, H + offs_r - 1)
    lbase = (b64 * NR + lrow) * W
    L = tl.load(pos_ptr + b).to(tl.int32) + 1                    # live length: the tokens before this step + this one
    L = tl.minimum(tl.maximum(L, 1), W)
    p = (tl.load(ptr_ptr) % W).to(tl.int32)
    start = p - L + 1 + W
    j_lo = sp * CHUNK
    j_hi = tl.minimum(j_lo + CHUNK, L)
    offs_s = tl.arange(0, NSP)
    smask = rmask[:, None] & (offs_s[None, :] < NS)
    sb = ((b64 * NR + lrow) * NS)[:, None] + offs_s[None, :]
    ms = tl.load(st_ptr + sb * 2, mask=smask, other=float("-inf"))
    zs = tl.load(st_ptr + sb * 2 + 1, mask=smask, other=0.0)
    m_i = tl.max(ms, axis=1)
    m_i = tl.where(rmask, m_i, 0.0)
    z_i = tl.sum(zs * libdevice.exp(ms - m_i[:, None]), axis=1)
    z_i = tl.where(rmask, z_i, 1.0)
    gbase = gv_ptr + bh * W * DV
    acc = tl.zeros([QP, DV], tl.float32)
    for j0 in range(j_lo, j_hi, BLOCK_W):
        offs_j = j0 + tl.arange(0, BLOCK_W)
        valid = offs_j < j_hi
        slot = (start + offs_j) % W
        lg = tl.load(lg_ptr + lbase[:, None] + offs_j[None, :], mask=rmask[:, None] & valid[None, :],
                     other=float("-inf"))
        pr = tl.div_rn(libdevice.exp(lg - m_i[:, None]), z_i[:, None]).to(tl.bfloat16)
        gt = tl.load(gbase + slot.to(tl.int64)[:, None] * DV + offs_dv[None, :], mask=valid[:, None], other=0.0)
        acc = tl.dot(pr, gt, acc)
    pb = ((b64 * NR + lrow) * NS + sp) * DV
    tl.store(part_ptr + pb[:, None] + offs_dv[None, :], acc, mask=rmask[:, None])


def _ring2_config(B: int) -> Tuple[int, int, int, int]:
    """(chunk, block_w, warps, stages) of the split-W read for decode batch B."""
    for bmax, chunk, block_w, warps, stages in RING2_CONFIGS:
        if bmax is None or int(B) <= bmax:
            return chunk, block_w, warps, stages
    raise AssertionError("RING2_CONFIGS has no catch-all row")  # pragma: no cover


def _ring2_shapes(B: int, H: int, W: int, M: int) -> Tuple[int, int, int, int]:
    """(chunk, NS chunks, NS padded to a power of two, query rows per batch row) of the split-W read."""
    ch = _ring2_config(B)[0]
    ns = -(-int(W) // ch)
    nsp = 1
    while nsp < ns:
        nsp *= 2
    return ch, ns, nsp, int(H) + int(M)


def _ring_read_t1_fwd(qb: torch.Tensor, qx: torch.Tensor, pool_k: torch.Tensor, pool_gv: torch.Tensor,
                      pool_cum: torch.Tensor, cum_t: torch.Tensor, pos: torch.Tensor, ring_ptr: torch.Tensor,
                      matches: List[torch.Tensor], beta: Optional[torch.Tensor], gain: float,
                      mx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The two split-W kernels on prepared operands: qb (B, H, dk) bf16 contiguous, qx (B, M, dk) bf16 contiguous (qb
    when M = 0), the contiguous rings, cum_t (B, H) fp32, pos (B,) int64, ring_ptr (1,) int64, matches (n_ord <= 4
    bool (B, 1, W) tensors, [] without a term), beta (n_ord, H) fp32 or None.  Returns (part (B, H + M, NS, dv) fp32,
    st (B, H + M, NS, 2) fp32): row h = head h's query, row H + m = extra query m of the last head."""
    B, H, W, dk = (int(s) for s in pool_k.shape)
    dv = int(pool_gv.shape[-1])
    M = int(mx)
    dev = pool_k.device
    n_ord = len(matches)
    if n_ord:
        ms = [m.view(torch.uint8) for m in matches]
        s_mb, s_mw = int(ms[0].stride(0)), int(ms[0].stride(2))
        if any((int(m.stride(0)), int(m.stride(2))) != (s_mb, s_mw) for m in ms):
            ms = [m.contiguous() for m in ms]
            s_mb, s_mw = int(ms[0].stride(0)), int(ms[0].stride(2))
        ms = ms + [ms[0]] * (4 - n_ord)
        beta_c = beta.contiguous()
    else:
        ms = [pool_cum] * 4                                                    # never read (N_ORD = 0)
        s_mb, s_mw = 0, 0
        beta_c = pool_cum
    ch, ns, nsp, nr = _ring2_shapes(B, H, W, M)
    _, bw, nwarp, nst = _ring2_config(B)
    lg = torch.empty((B, nr, W), dtype=torch.float32, device=dev)
    st = torch.empty((B, nr, ns, 2), dtype=torch.float32, device=dev)
    part = torch.empty((B, nr, ns, dv), dtype=torch.float32, device=dev)
    grid = (B, H, ns)
    _ring2_logits_kernel[grid](qb, qx, pool_k, pool_cum, cum_t, pos, ring_ptr, ms[0], ms[1], ms[2], ms[3], beta_c,
                               lg, st, H, W, H - 1, float(gain), int(cum_t.stride(0)), int(cum_t.stride(1)),
                               s_mb, s_mw, DK=dk, MX=M, N_ORD=n_ord, QP=RING2_QUERY_ROWS, BLOCK_W=bw, CHUNK=ch,
                               NS=ns, num_warps=nwarp, num_stages=nst, enable_fp_fusion=False)
    _ring2_pv_kernel[grid](lg, st, pool_gv, pos, ring_ptr, part, H, W, H - 1,
                           DV=dv, MX=M, QP=RING2_QUERY_ROWS, BLOCK_W=bw, CHUNK=ch, NS=ns, NSP=nsp,
                           num_warps=nwarp, num_stages=nst, enable_fp_fusion=False)
    return part, st


@torch.library.custom_op("neuronspark::ring_t1_partials2", mutates_args=())
def _ring_t1_partials2_op(qb: torch.Tensor, qx: torch.Tensor, pool_k: torch.Tensor, pool_gv: torch.Tensor,
                          pool_cum: torch.Tensor, cum_t: torch.Tensor, pos: torch.Tensor, ring_ptr: torch.Tensor,
                          matches: List[torch.Tensor], beta: Optional[torch.Tensor], gain: float,
                          mx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return _ring_read_t1_fwd(qb, qx, pool_k, pool_gv, pool_cum, cum_t, pos, ring_ptr, list(matches), beta, gain, mx)


@_ring_t1_partials2_op.register_fake
def _ring_t1_partials2_fake(qb, qx, pool_k, pool_gv, pool_cum, cum_t, pos, ring_ptr, matches, beta, gain, mx):
    B, H, W = int(pool_k.shape[0]), int(pool_k.shape[1]), int(pool_k.shape[2])
    _, ns, _, nr = _ring2_shapes(B, H, W, int(mx))
    return (pool_k.new_empty((B, nr, ns, int(pool_gv.shape[-1])), dtype=torch.float32),
            pool_k.new_empty((B, nr, ns, 2), dtype=torch.float32))


def ring_read_t1_takes(batch_size: int) -> bool:
    """True at the decode batch sizes whose ring read is :func:`ring_read_t1` (``RING_READ_KERNEL_BATCHES``)."""
    return int(batch_size) in RING_READ_KERNEL_BATCHES


def _t1_matches(matches: Optional[Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    if matches is None:
        return []
    return [m.unsqueeze(1) if m.dim() == 2 else m for m in matches]


def _t1_xq(xq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if xq is not None and xq.dim() == 4:
        if int(xq.shape[2]) != 1:
            raise RuntimeError(f"ring_read_t1: extra queries {tuple(xq.shape)} are not a T = 1 step")
        xq = xq[:, :, 0]
    return xq


def _check_t1(q, xq, cum_t, pool_k, pool_gv, pool_cum, pos, ring_ptr, ms, beta) -> Tuple[int, int, int, int, int, int]:
    for name, t in (("q", q), ("cum_t", cum_t), ("pool_k", pool_k), ("pool_gv", pool_gv), ("pool_cum", pool_cum),
                    ("pos", pos), ("ring_ptr", ring_ptr)):
        require_cuda(t, f"ring_read_t1 ({name})")
    if pool_k.dim() != 4 or pool_gv.dim() != 4 or pool_cum.dim() != 3:
        raise RuntimeError("ring_read_t1: rings must be pool_k / pool_gv (B, H, W, d) and pool_cum (B, H, W)")
    B, H, W, dk = (int(s) for s in pool_k.shape)
    dv = int(pool_gv.shape[-1])
    if tuple(pool_gv.shape[:3]) != (B, H, W) or tuple(pool_cum.shape) != (B, H, W):
        raise RuntimeError(f"ring_read_t1: ring shapes {tuple(pool_k.shape)} {tuple(pool_gv.shape)} "
                           f"{tuple(pool_cum.shape)} differ")
    if pool_k.dtype != torch.bfloat16 or pool_gv.dtype != torch.bfloat16 or pool_cum.dtype != torch.float32:
        raise RuntimeError(f"ring_read_t1: bf16 rings and fp32 clocks are required, got {pool_k.dtype} / "
                           f"{pool_gv.dtype} / {pool_cum.dtype}")
    for t_, nm in ((pool_k, "pool_k"), (pool_gv, "pool_gv"), (pool_cum, "pool_cum")):
        if not t_.is_contiguous():
            raise RuntimeError(f"ring_read_t1: {nm} must be contiguous")
    for d in (dk, dv):
        if d < 16 or d > 256 or not _pow2(d):
            raise RuntimeError(f"ring_read_t1: head dims must be powers of two in [16, 256], got dk {dk} dv {dv}")
    if tuple(q.shape) != (B, H, dk) or not q.is_floating_point():
        raise RuntimeError(f"ring_read_t1: q must be a float ({B}, {H}, {dk}) tensor, got {q.dtype} {tuple(q.shape)}")
    M = 0
    if xq is not None:
        if xq.dim() != 3 or int(xq.shape[0]) != B or int(xq.shape[2]) != dk or not xq.is_floating_point():
            raise RuntimeError(f"ring_read_t1: extra queries must be ({B}, M, {dk}), got {tuple(xq.shape)}")
        M = int(xq.shape[1])
        if M < 1:
            raise RuntimeError("ring_read_t1: pass xq=None for a read without extra queries")
    if 1 + M > RING2_QUERY_ROWS:
        raise RuntimeError(f"ring_read_t1: 1 + {M} queries exceed the kernel's {RING2_QUERY_ROWS} query rows")
    if tuple(cum_t.shape) != (B, H) or cum_t.dtype != torch.float32:
        raise RuntimeError(f"ring_read_t1: cum_t must be ({B}, {H}) fp32, got {cum_t.dtype} {tuple(cum_t.shape)}")
    if tuple(pos.shape) != (B,) or pos.dtype != torch.int64:
        raise RuntimeError(f"ring_read_t1: pos must be a ({B},) int64 tensor, got {pos.dtype} {tuple(pos.shape)}")
    if ring_ptr.numel() != 1 or ring_ptr.dtype != torch.int64:
        raise RuntimeError("ring_read_t1: ring_ptr must be a (1,) int64 tensor")
    n_ord = len(ms)
    if n_ord > 4:
        raise RuntimeError(f"ring_read_t1: at most 4 induction orders, got {n_ord}")
    if n_ord:
        if any(tuple(m.shape) != (B, 1, W) or m.dtype != torch.bool or m.device != pool_k.device for m in ms):
            raise RuntimeError(f"ring_read_t1: induction matches {[tuple(m.shape) for m in ms]} for the ring "
                               f"({B}, 1, {W}) (bool)")
        if beta is None or tuple(beta.shape) != (n_ord, H) or beta.dtype != torch.float32:
            raise RuntimeError(f"ring_read_t1: beta must be ({n_ord}, {H}) fp32 with {n_ord} match orders")
    elif beta is not None:
        raise RuntimeError("ring_read_t1: beta given without induction matches")
    return B, H, W, dk, dv, M


def ring_read_t1(q: torch.Tensor, xq: Optional[torch.Tensor], cum_t: torch.Tensor, pool_k: torch.Tensor,
                 pool_gv: torch.Tensor, pool_cum: torch.Tensor, pos: torch.Tensor, ring_ptr: torch.Tensor,
                 matches: Optional[Sequence[torch.Tensor]], beta: Optional[torch.Tensor],
                 gain: float) -> torch.Tensor:
    """Pool read of a decode step (T = 1) for every head and the last head's extra queries, after the step's ring
    write.

    q (B, H, dk) float: the step's gained unit queries (rounded to bf16 here, as the reference's ``.to(bf16)``);
    xq (B, M, dk) float (or (B, M, 1, dk)) or None: the last head's extra queries; cum_t (B, H) fp32: this step's
    clocks; pool_k / pool_gv (B, H, W, dk / dv) bf16 contiguous, pool_cum (B, H, W) fp32 contiguous: the rings with
    this step's event written at slot ``ring_ptr % W``; pos (B,) int64: tokens consumed before this step; ring_ptr (1,)
    int64; matches: n_ord (<= 4) bool (B, 1, W) (or (B, W)) induction match rows of this step or None; beta (n_ord, H)
    fp32 or None; gain: the match scale (python float, 16.0).

    Returns (B, 1, (H + M) * dv) bf16: columns [h dv, (h + 1) dv) the read of head h, then [(H + m) dv, ...) the read
    of extra query m (to be projected by the module)."""
    xq = _t1_xq(xq)
    ms = _t1_matches(matches)
    B, H, W, dk, dv, M = _check_t1(q, xq, cum_t, pool_k, pool_gv, pool_cum, pos, ring_ptr, ms, beta)
    qb = q.to(torch.bfloat16).contiguous()
    qx = xq.to(torch.bfloat16).contiguous() if M else qb
    part, st = torch.ops.neuronspark.ring_t1_partials2(qb, qx, pool_k, pool_gv, pool_cum, cum_t, pos,
                                                       ring_ptr.reshape(1), ms, beta if ms else None, float(gain), M)
    red = part.sum(dim=2)                                                     # (B, H + M, dv) fp32 over the chunks
    return red.to(torch.bfloat16).reshape(B, 1, (H + M) * dv)


def _ring_live_mask(pos: torch.Tensor, ring_ptr: torch.Tensor, W: int) -> torch.Tensor:
    """(B, W) bool: slot w is one of row b's live slots -- the L_b = clamp(pos_b + 1, 1, W) slots ending at
    p = ring_ptr % W."""
    L = (pos + 1).clamp(1, W).view(-1, 1)
    w = torch.arange(W, device=pos.device, dtype=torch.int64).view(1, W)
    back = torch.remainder(torch.remainder(ring_ptr.reshape(1), W).view(1, 1) - w, W)    # 0 at p, 1 at p - 1, ...
    return back < L


def _ring_term(matches: Sequence[torch.Tensor], beta: torch.Tensor, H: int) -> torch.Tensor:
    """(B, H, T, W) fp32 induction term m_0 b_0 (+ m_n b_n) of every head from (B, T, W) matches."""
    out = matches[0].unsqueeze(1).to(torch.float32) * beta[0].view(1, H, 1, 1)
    for n in range(1, len(matches)):
        out = out + matches[n].unsqueeze(1).to(torch.float32) * beta[n].view(1, H, 1, 1)
    return out


def ring_read_t1_ref(q: torch.Tensor, xq: Optional[torch.Tensor], cum_t: torch.Tensor, pool_k: torch.Tensor,
                     pool_gv: torch.Tensor, pool_cum: torch.Tensor, pos: torch.Tensor, ring_ptr: torch.Tensor,
                     matches: Optional[Sequence[torch.Tensor]], beta: Optional[torch.Tensor],
                     gain: float) -> torch.Tensor:
    """The torch expression chain of the T = 1 ring read (the memory layer's T = 1 decode branch and the extra-query
    read at T = 1) over all W slots under bf16 autocast (its einsums are bf16 bmm calls with bf16 outputs), the slots
    outside the live range set to -inf before the bias cast.  Same arguments and output as :func:`ring_read_t1`.  Run it
    under ``regime("infer")`` for the inference GEMM settings."""
    xq = _t1_xq(xq)
    ms = _t1_matches(matches)
    B, H, W, dk = (int(s) for s in pool_k.shape)
    dv = int(pool_gv.shape[-1])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        live = _ring_live_mask(pos, ring_ptr, W)                                  # (B, W)
        allt = _ring_term(ms, beta, H) if ms else None                            # (B, H, 1, W)
        qh = q.to(pool_k.dtype).float()
        scores = torch.einsum("bhd,bhwd->bhw", qh, pool_k.float()) * gain
        bias = cum_t.unsqueeze(-1) - pool_cum
        if allt is not None:
            bias = bias + allt[:, :, 0]
        bias = bias.masked_fill(~live.unsqueeze(1), float("-inf"))
        bias = bias.to(pool_k.dtype).float()
        probs = torch.softmax(scores + bias, dim=-1)
        out = torch.einsum("bhw,bhwd->bhd", probs.to(pool_gv.dtype).float(), pool_gv.float())
        parts = [out.to(torch.bfloat16).reshape(B, 1, H * dv)]
        if xq is not None:
            i = H - 1
            qxh = xq.to(pool_k.dtype).float().unsqueeze(2)                        # (B, M, 1, dk)
            k_ring, gv_ring, cum_ring = pool_k[:, i], pool_gv[:, i], pool_cum[:, i]
            sc = torch.einsum("bmtd,bwd->bmtw", qxh, k_ring.float()) * gain
            bx = cum_t[:, i].view(B, 1).unsqueeze(-1) - cum_ring.unsqueeze(1)      # (B, 1, W)
            if allt is not None:
                bx = bx + allt[:, i]
            bx = bx.masked_fill(~live.unsqueeze(1), float("-inf"))
            bx = bx.to(pool_k.dtype).float()
            px = torch.softmax(sc + bx.unsqueeze(1), dim=-1)                       # (B, M, 1, W)
            reads = torch.einsum("bmtw,bwd->bmtd", px.to(pool_gv.dtype).float(), gv_ring.float())[:, :, 0]
            parts.append(reads.to(torch.bfloat16).reshape(B, 1, -1))
    return torch.cat(parts, dim=-1)


# ================================================================================================ extra-query formation

@triton.jit
def _bf16(x):
    """Round an fp32 value to the nearest bf16 (ties to even) and widen it back through the bit pattern (the plain
    ``x.to(bf16).to(f32)`` round trip can be folded away by the compiler inside larger expressions); finite values, an
    infinity stays itself."""
    u = x.to(tl.int32, bitcast=True)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    u = u & -65536                     # 0xFFFF0000 as a signed int32
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _xq_form_kernel(U, WB, COS, SIN, PSI, HG, OUT,
                    s_u_b, s_wb_m, s_cs_b, s_psi_b, s_out_b,
                    M: tl.constexpr, R: tl.constexpr, DK: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M
    offs_k = tl.arange(0, R)
    offs_i = tl.arange(0, HALF)
    # ---- the projection of the query's rank-R slice: q[n] = sum_k WB[m, n, k] u[b, m R + k]  (bf16 operands, fp32 sum)
    u = tl.load(U + b * s_u_b + m * R + offs_k).to(tl.float32)                                   # (R,)
    w_e = tl.load(WB + m * s_wb_m + (2 * offs_i)[:, None] * R + offs_k[None, :]).to(tl.float32)  # even rows (HALF, R)
    w_o = tl.load(WB + m * s_wb_m + (2 * offs_i + 1)[:, None] * R + offs_k[None, :]).to(tl.float32)
    q_e = _bf16(tl.sum(w_e * u[None, :], axis=1))                                               # the linear's bf16 output
    q_o = _bf16(tl.sum(w_o * u[None, :], axis=1))
    out_row = OUT + b * s_out_b + m * DK
    # ---- SiLU (fp32, rounded to bf16 as the bf16 op's output)
    q_e = _bf16(q_e * tl.sigmoid(q_e))
    q_o = _bf16(q_o * tl.sigmoid(q_o))
    # ---- interleaved RoPE in bf16 arithmetic: y_even = x_even cos - x_odd sin, y_odd = x_odd cos + x_even sin
    c = tl.load(COS + b * s_cs_b + offs_i).to(tl.float32)                                  # the row's position (stride 0 = shared)
    s = tl.load(SIN + b * s_cs_b + offs_i).to(tl.float32)
    y_e = _bf16(_bf16(q_e * c) - _bf16(q_o * s))
    y_o = _bf16(_bf16(q_o * c) + _bf16(q_e * s))
    # ---- L2 normalisation (fp32 sum of squares, sqrt, floor 1e-12, division) and the gain
    ss = tl.sum(y_e * y_e, axis=0) + tl.sum(y_o * y_o, axis=0)
    nrm = tl.maximum(libdevice.sqrt(ss), 1e-12)
    psi = tl.load(PSI + b * s_psi_b + m).to(tl.float32)
    psi = tl.minimum(tl.maximum(psi, -3.0), 3.0)
    g = libdevice.exp(psi) * tl.load(HG).to(tl.float32)
    o_e = (y_e / nrm) * g
    o_o = (y_o / nrm) * g
    tl.store(out_row + 2 * offs_i, o_e.to(tl.bfloat16))
    tl.store(out_row + 2 * offs_i + 1, o_o.to(tl.bfloat16))


def _check_xq(u, wb, cos, sin, psi, head_gain) -> Tuple[int, int, int, int]:
    for name, t in (("u", u), ("wb", wb), ("cos", cos), ("sin", sin), ("psi", psi), ("head_gain", head_gain)):
        require_cuda(t, f"xq_form_t1 ({name})")
    if wb.dim() != 3 or u.dim() != 2 or psi.dim() != 2 or cos.dim() != 2 or sin.dim() != 2:
        raise RuntimeError("xq_form_t1: u (B, M r), wb (M, dk, r), cos / sin (B, dk / 2), psi (B, M) are required")
    B = int(u.shape[0])
    M, dk, r = (int(x) for x in wb.shape)
    if u.dtype != torch.bfloat16 or wb.dtype != torch.bfloat16 or psi.dtype != torch.bfloat16:
        raise RuntimeError("xq_form_t1: u / wb / psi must be bf16")
    if cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
        raise RuntimeError("xq_form_t1: cos / sin must be bf16 (the step's tables cast to the query dtype)")
    if (tuple(u.shape) != (B, M * r) or tuple(psi.shape) != (B, M) or tuple(cos.shape) != (B, dk // 2)
            or tuple(sin.shape) != (B, dk // 2)):
        raise RuntimeError(f"xq_form_t1: shapes u {tuple(u.shape)} wb {tuple(wb.shape)} psi {tuple(psi.shape)} "
                           f"cos {tuple(cos.shape)} sin {tuple(sin.shape)}")
    if cos.stride(0) != sin.stride(0) or cos.stride(1) != 1 or sin.stride(1) != 1:
        raise RuntimeError(f"xq_form_t1: cos / sin strides {tuple(cos.stride())} {tuple(sin.stride())} (unit inner "
                           "stride, equal row strides)")
    if tuple(head_gain.shape) != (1,) or head_gain.dtype != torch.float32:
        raise RuntimeError("xq_form_t1: head_gain must be a (1,) fp32 tensor")
    if not (wb.is_contiguous() and u.stride(1) == 1 and psi.stride(1) == 1):
        raise RuntimeError("xq_form_t1: operand layouts (wb contiguous, u / psi unit inner stride)")
    if not _pow2(dk) or not _pow2(r) or dk < 2:
        raise RuntimeError("xq_form_t1: dk and r must be powers of two")
    return B, M, dk, r


def _xq_form_t1_fwd(u: torch.Tensor, wb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, psi: torch.Tensor,
                    head_gain: torch.Tensor) -> torch.Tensor:
    B, M, dk, r = _check_xq(u, wb, cos, sin, psi, head_gain)
    out = torch.empty((B, M, dk), dtype=torch.bfloat16, device=u.device)
    _xq_form_kernel[(B * M,)](u, wb, cos, sin, psi, head_gain, out,
                              u.stride(0), wb.stride(0), cos.stride(0), psi.stride(0), out.stride(0),
                              M=M, R=r, DK=dk, HALF=dk // 2, num_warps=XQ_FORM_WARPS)
    return out


@torch.library.custom_op("neuronspark::xq_form_t1", mutates_args=())
def _xq_form_t1_op(u: torch.Tensor, wb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, psi: torch.Tensor,
                   head_gain: torch.Tensor) -> torch.Tensor:
    return _xq_form_t1_fwd(u, wb, cos, sin, psi, head_gain)


@_xq_form_t1_op.register_fake
def _xq_form_t1_fake(u, wb, cos, sin, psi, head_gain):
    return u.new_empty((u.shape[0], wb.shape[0], wb.shape[1]), dtype=torch.bfloat16)


def xq_form_t1(u: torch.Tensor, wb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, psi: torch.Tensor,
               head_gain: torch.Tensor) -> torch.Tensor:
    """The last head's M extra queries of a decode step.

    u (B, M r) bf16 (unit inner stride): the low-rank map of the normed input; wb (M, dk, r) bf16 contiguous: the
    stacked per-query projections (:func:`xq_weight_stack`); cos / sin (B, dk / 2) bf16 with unit inner stride: the
    last head's band of the step's RoPE tables per row (an expanded view with row stride 0 for a shared position);
    psi (B, M) bf16: the gain projection (unclamped); head_gain (1,) fp32: the last head's dendritic query gain
    (:func:`xq_head_gain`).  Returns (B, M, dk) bf16 contiguous: the gained unit queries (the read rounds them to bf16
    in any case)."""
    return torch.ops.neuronspark.xq_form_t1(u, wb, cos, sin, psi, head_gain)


def xq_form_t1_ref(u: torch.Tensor, wb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, psi: torch.Tensor,
                   head_gain: torch.Tensor) -> torch.Tensor:
    """The inference forward's arithmetic of the extra-query loop with explicit dtypes: the projection (``torch.mm``,
    bf16 operands and output), SiLU and the interleaved RoPE as bf16 ops; the L2 normalisation, the division and the
    gain in fp32 with one rounding to bf16 at the stack.  Same arguments and output as :func:`xq_form_t1`."""
    M, dk, r = (int(x) for x in wb.shape)
    with torch.autocast("cuda", enabled=False):
        psi32 = psi.to(torch.float32).clamp(-3.0, 3.0)
        c, s = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        qs = []
        for m in range(M):
            q = torch.mm(u[:, m * r:(m + 1) * r], wb[m].t())                       # (B, dk) bf16
            q = F.silu(q)
            x_even, x_odd = q[..., ::2], q[..., 1::2]
            y = torch.empty_like(q)
            y[..., ::2] = x_even * c - x_odd * s
            y[..., 1::2] = x_odd * c + x_even * s
            y = y.to(torch.float32)
            nrm = y.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-12)
            g = torch.exp(psi32[:, m:m + 1]) * head_gain.to(torch.float32)
            qs.append(((y / nrm) * g).to(torch.bfloat16))
    return torch.stack(qs, dim=1)


@torch.no_grad()
def xq_weight_stack(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """(M, dk, r) bf16 contiguous stack of the extra-query projections ``pool_xq_b.{0..M-1}`` (dk, r) -- the values
    the bf16 projection uses (RNE of the fp32 masters).  Build it once per weight version (e.g. at
    ``prepare_inference``), not per step."""
    return torch.stack([w.detach().to(torch.bfloat16) for w in weights], dim=0).contiguous()


@torch.no_grad()
def xq_head_gain(gain_ratio: torch.Tensor, q_dtype: torch.dtype) -> torch.Tensor:
    """(1,) fp32 last-head dendritic query gain as the extra-query read forms it: the (H,) fp32 ratio
    ``exp(log_gain)`` cast to the query dtype, the last head's entry widened back to fp32."""
    return gain_ratio.to(q_dtype)[-1:].to(torch.float32).clone()


# ================================================================================================ prefill ring write

@triton.jit
def _normalize_rows(x):
    # x: (R, 256) fp32 (the bf16 operands upcast, exact) -> fp32 (R, 256): F.normalize(x, dim=-1) -- the fp32 norm
    # (Triton's row sum of the squares, sqrt), clamp_min(1e-12), the fp32 division
    nrm = tl.sqrt(tl.sum(x * x, axis=1))
    denom = tl.maximum(nrm, 1e-12)
    return x / denom[:, None]


@triton.jit
def _ring_write_kernel(K, Q, G, V, GAIN, WBIAS, PK, PGV, QOUT, PTR,
                       NROWS, B, H, W, sparsity,
                       sKt, sKb, sKh, sQt, sQb, sQh, sGt, sGb, sGh, sVt, sVb, sVh,
                       HAS_GAIN: tl.constexpr, HAS_WBIAS: tl.constexpr, HAS_SPARSITY: tl.constexpr,
                       R: tl.constexpr, DKC: tl.constexpr):
    pid = tl.program_id(axis=0)
    rows = pid * R + tl.arange(0, R)                                      # flattened (t, b, h) rows, t-major
    rmask = rows < NROWS
    t = rows // (B * H)
    b = (rows // H) % B
    h = rows % H
    cols = tl.arange(0, DKC)
    m2 = rmask[:, None] & (cols[None, :] < DKC)
    ptr = tl.load(PTR)
    slot = (ptr + t.to(tl.int64)) % W                                     # this call's ring slots, (ptr + arange(T)) % W
    ring_row = ((b.to(tl.int64) * H + h) * W + slot)[:, None] * DKC + cols[None, :]
    # ---- k: unit tag into pool_k
    koff = t.to(tl.int64)[:, None] * sKt + b.to(tl.int64)[:, None] * sKb + h.to(tl.int64)[:, None] * sKh + cols[None, :]
    k = tl.load(K + koff, mask=m2, other=0.0).to(tl.float32)
    tl.store(PK + ring_row, _normalize_rows(k).to(tl.bfloat16), mask=m2)          # k_hat fp32 -> bf16 at the ring write
    # ---- q: the unit query (fp32) times the head's fp32 gain, into QOUT (T, B, H, DK) fp32 contiguous
    qoff = t.to(tl.int64)[:, None] * sQt + b.to(tl.int64)[:, None] * sQb + h.to(tl.int64)[:, None] * sQh + cols[None, :]
    q = _normalize_rows(tl.load(Q + qoff, mask=m2, other=0.0).to(tl.float32))
    if HAS_GAIN:
        gain = tl.load(GAIN + h, mask=rmask, other=0.0)                  # (H,) fp32 gain, applied after the normalisation
        q = q * gain[:, None]
    tl.store(QOUT + rows.to(tl.int64)[:, None] * DKC + cols[None, :], q, mask=m2)
    # ---- gate / value: sigmoid(g.float() + wbias - sparsity).to(bf16) * v into pool_gv
    goff = t.to(tl.int64)[:, None] * sGt + b.to(tl.int64)[:, None] * sGb + h.to(tl.int64)[:, None] * sGh + cols[None, :]
    gf = tl.load(G + goff, mask=m2, other=0.0).to(tl.float32)             # the gate fp32 (or bf16 upcast): g_h.float()
    if HAS_WBIAS:
        wb = tl.load(WBIAS + h, mask=rmask, other=0.0)                    # (H,) fp32
        gf = _add_rn(gf, wb[:, None])
    if HAS_SPARSITY:
        gf = _sub_rn(gf, sparsity)
    one = tl.full(gf.shape, 1.0, tl.float32)
    sig = _div_rn(one, _add_rn(one, libdevice.exp(-gf))).to(tl.bfloat16)  # ATen sigmoid fp32: one / (one + exp(-x))
    voff = t.to(tl.int64)[:, None] * sVt + b.to(tl.int64)[:, None] * sVb + h.to(tl.int64)[:, None] * sVh + cols[None, :]
    v = tl.load(V + voff, mask=m2, other=0.0)
    gv = _mul_rn(sig.to(tl.float32), v.to(tl.float32)).to(tl.bfloat16)   # bf16 * bf16: the exact fp32 product, rounded
    tl.store(PGV + ring_row, gv, mask=m2)


def _ring_write_layout_ok(k_h, q_h, g_h, v_h) -> bool:
    """False for operands whose channel dimension is not unit-stride (the T = 2 form of the qkv conv)."""
    return all(x.dim() != 4 or x.stride(-1) == 1 for x in (q_h, k_h, v_h, g_h))


def _check_ring_write(k_h, q_h, g_h, v_h, qgain, wbias, pool_k, pool_gv, ring_ptr, T, B, H, W) -> None:
    DK = RING_WRITE_HEAD_DIM
    for name, t in (("k_h", k_h), ("q_h", q_h), ("g_h", g_h), ("v_h", v_h), ("pool_k", pool_k),
                    ("pool_gv", pool_gv), ("ring_ptr", ring_ptr)):
        require_cuda(t, f"ring_write_prefill ({name})")
    shp = (int(T), int(B), int(H), DK)
    for name, x in (("q", q_h), ("k", k_h), ("v", v_h)):
        if x.dtype != torch.bfloat16 or tuple(x.shape) != shp:
            raise RuntimeError(f"ring_write_prefill: {name} must be bf16 {shp}, got {x.dtype} {tuple(x.shape)}")
    if g_h.dtype not in (torch.float32, torch.bfloat16) or tuple(g_h.shape) != shp:
        raise RuntimeError(f"ring_write_prefill: the gate must be fp32 or bf16 {shp}, got {g_h.dtype} "
                           f"{tuple(g_h.shape)}")
    for name, p in (("pool_k", pool_k), ("pool_gv", pool_gv)):
        if p.dtype != torch.bfloat16 or not p.is_contiguous() or tuple(p.shape) != (int(B), int(H), int(W), DK):
            raise RuntimeError(f"ring_write_prefill: {name} must be a contiguous bf16 ({B}, {H}, {W}, {DK}) ring, got "
                               f"{p.dtype} {tuple(p.shape)} contiguous={p.is_contiguous()}")
    if ring_ptr.dtype != torch.int64 or ring_ptr.numel() != 1:
        raise RuntimeError(f"ring_write_prefill: ring_ptr must be a (1,) int64 tensor, got {ring_ptr.dtype} "
                           f"numel {ring_ptr.numel()}")
    if not 1 <= int(T) <= int(W):
        raise RuntimeError(f"ring_write_prefill: T {T} must lie in [1, W = {W}]")
    for name, x in (("qgain", qgain), ("wbias", wbias)):
        if x is not None and (x.dim() != 1 or x.numel() != int(H) or x.dtype != torch.float32):
            raise RuntimeError(f"ring_write_prefill: {name} must be None or ({H},) fp32, got {x.dtype} "
                               f"{tuple(x.shape)}")


def _ring_write_prefill_fwd(k_h: torch.Tensor, q_h: torch.Tensor, g_h: torch.Tensor, v_h: torch.Tensor,
                            qgain: Optional[torch.Tensor], wbias: Optional[torch.Tensor], sparsity: float,
                            pool_k: torch.Tensor, pool_gv: torch.Tensor, ring_ptr: torch.Tensor,
                            T: int, B: int, H: int, W: int) -> torch.Tensor:
    DK = RING_WRITE_HEAD_DIM
    qout = torch.empty((T, B, H, DK), dtype=torch.float32, device=q_h.device)
    nrows = T * B * H
    gain = qgain.contiguous() if qgain is not None else qout            # dummy pointers when absent (constexpr off)
    wb = wbias.contiguous() if wbias is not None else qout
    has_sp = float(sparsity) > 0.0
    grid = (triton.cdiv(nrows, RING_WRITE_ROWS),)
    _ring_write_kernel[grid](k_h, q_h, g_h, v_h, gain, wb, pool_k, pool_gv, qout, ring_ptr,
                             nrows, B, H, W, float(sparsity),
                             k_h.stride(0), k_h.stride(1), k_h.stride(2), q_h.stride(0), q_h.stride(1), q_h.stride(2),
                             g_h.stride(0), g_h.stride(1), g_h.stride(2), v_h.stride(0), v_h.stride(1), v_h.stride(2),
                             HAS_GAIN=qgain is not None, HAS_WBIAS=wbias is not None, HAS_SPARSITY=has_sp,
                             R=RING_WRITE_ROWS, DKC=DK, num_warps=RING_WRITE_WARPS)
    return qout


@torch.library.custom_op("neuronspark::ring_write_prefill", mutates_args=("pool_k", "pool_gv"))
def _ring_write_prefill_op(k_h: torch.Tensor, q_h: torch.Tensor, g_h: torch.Tensor, v_h: torch.Tensor,
                           qgain: Optional[torch.Tensor], wbias: Optional[torch.Tensor], sparsity: float,
                           pool_k: torch.Tensor, pool_gv: torch.Tensor, ring_ptr: torch.Tensor,
                           T: int, B: int, H: int, W: int) -> torch.Tensor:
    return _ring_write_prefill_fwd(k_h, q_h, g_h, v_h, qgain, wbias, sparsity, pool_k, pool_gv, ring_ptr, T, B, H, W)


@_ring_write_prefill_op.register_fake
def _ring_write_prefill_fake(k_h, q_h, g_h, v_h, qgain, wbias, sparsity, pool_k, pool_gv, ring_ptr, T, B, H, W):
    return q_h.new_empty((T, B, H, RING_WRITE_HEAD_DIM), dtype=torch.float32)


def ring_write_prefill(k_h: torch.Tensor, q_h: torch.Tensor, g_h: torch.Tensor, v_h: torch.Tensor,
                       qgain: Optional[torch.Tensor], wbias: Optional[torch.Tensor], sparsity: float,
                       pool_k: torch.Tensor, pool_gv: torch.Tensor, ring_ptr: torch.Tensor,
                       T: int, B: int, H: int, W: int) -> torch.Tensor:
    """Write a prefill call's unit key tags and tagged values into the rings and form its gained unit queries.

    k_h / q_h / v_h (T, B, H, 256) bf16, g_h (T, B, H, 256) fp32 (or bf16): the per-head key / query / value / gate
    of the call (after RoPE, before any normalisation); qgain (H,) fp32 or None: the dendritic query gain (formed in
    fp32, the dtype of the normalised query); wbias (H,) fp32 or None: the write bias; sparsity: the write threshold
    (2.0); pool_k / pool_gv (B, H, W, 256) bf16 contiguous, written in place at slots ``(ring_ptr + t) % W``;
    ring_ptr (1,) int64 at the start of the call.  Returns the gained unit queries (T, B, H, 256) fp32.

    Operands with a unit channel stride run the kernel; operands whose channel dimension is strided (the T = 2 conv
    form) run :func:`ring_write_prefill_ref`, the expression chain of the memory layer's prefill on that layout.
    The ring clocks are written by the caller."""
    _check_ring_write(k_h, q_h, g_h, v_h, qgain, wbias, pool_k, pool_gv, ring_ptr, T, B, H, W)
    if not _ring_write_layout_ok(k_h, q_h, g_h, v_h):
        return ring_write_prefill_ref(k_h, q_h, g_h, v_h, qgain, wbias, sparsity, pool_k, pool_gv, ring_ptr,
                                      T, B, H, W)
    return torch.ops.neuronspark.ring_write_prefill(k_h, q_h, g_h, v_h, qgain, wbias, float(sparsity), pool_k,
                                                    pool_gv, ring_ptr.reshape(1), int(T), int(B), int(H), int(W))


def ring_write_prefill_ref(k_h: torch.Tensor, q_h: torch.Tensor, g_h: torch.Tensor, v_h: torch.Tensor,
                           qgain: Optional[torch.Tensor], wbias: Optional[torch.Tensor], sparsity: float,
                           pool_k: torch.Tensor, pool_gv: torch.Tensor, ring_ptr: torch.Tensor,
                           T: int, B: int, H: int, W: int) -> torch.Tensor:
    """The prefill ring-write expression chain under the inference forward's bf16 autocast (``F.normalize`` of the
    bf16 k / q in fp32, the fp32 gain, the fp32 gate chain, the bf16 sigmoid and product), writing pool_k / pool_gv in
    place; returns the gained queries (fp32).  Same arguments and output as :func:`ring_write_prefill`."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        k_hat = F.normalize(k_h, dim=-1)
        q_n = F.normalize(q_h, dim=-1)
        if qgain is not None:
            q_n = q_n * (qgain.view(1, 1, H, 1) if qgain.dim() == 1 else qgain.unsqueeze(-1))
        g = g_h
        if wbias is not None:
            g = g.float() + wbias.view(1, 1, H, 1)
        g_raw = g.float()
        if sparsity > 0.0:
            g_raw = g_raw - sparsity
        gv = torch.sigmoid(g_raw).to(v_h.dtype) * v_h
        slots = (ring_ptr.reshape(1) + torch.arange(T, device=q_h.device, dtype=torch.int64)) % W
        pool_k.index_copy_(2, slots, k_hat.permute(1, 2, 0, 3).to(pool_k.dtype))
        pool_gv.index_copy_(2, slots, gv.permute(1, 2, 0, 3).to(pool_gv.dtype))
    return q_n


# ================================================================================================ prefill ring recall

@triton.jit
def _ind_from_matches(M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h, hb, b, t, w, tm, N_IND: tl.constexpr):
    """The induction term of head hb at (b, t, w): m_0 b_0, then + m_n b_n (fp32 RN; m in {0, 1})."""
    off = b * sm_b + t[:, None] * sm_t + w[None, :] * sm_w
    iv = _rn_mul_f32(tl.load(M0 + off, mask=tm[:, None], other=0).to(tl.float32),
                     tl.broadcast_to(tl.load(BETA + hb * sbe_h), off.shape))
    if N_IND > 1:
        iv = _rn_add_f32(iv, _rn_mul_f32(tl.load(M1 + off, mask=tm[:, None], other=0).to(tl.float32),
                                         tl.broadcast_to(tl.load(BETA + sbe_n + hb * sbe_h), off.shape)))
    if N_IND > 2:
        iv = _rn_add_f32(iv, _rn_mul_f32(tl.load(M2 + off, mask=tm[:, None], other=0).to(tl.float32),
                                         tl.broadcast_to(tl.load(BETA + 2 * sbe_n + hb * sbe_h), off.shape)))
    if N_IND > 3:
        iv = _rn_add_f32(iv, _rn_mul_f32(tl.load(M3 + off, mask=tm[:, None], other=0).to(tl.float32),
                                         tl.broadcast_to(tl.load(BETA + 3 * sbe_n + hb * sbe_h), off.shape)))
    return iv


@triton.jit
def _rf_valid(CW, scw_b, scw_h, scw_w, b, hr, w, t, tm, p0, T, W):
    """The slot clocks of a tile and its causal / validity masks: a slot is valid for query t when its clock is finite
    and it is not a later slot of this call (call position j(w) > t)."""
    cw = tl.load(CW + b * scw_b + hr * scw_h + w * scw_w)
    jw = (w.to(tl.int64) - p0) % W
    jw = tl.where(jw < 0, jw + W, jw)
    causal = (jw[None, :] < T) & (jw[None, :] > t[:, None].to(tl.int64))
    valid = (~causal) & (cw[None, :] != float("inf")) & tm[:, None]
    return cw, causal, valid


@triton.jit
def _rf_x(q, ct, cw, causal, valid, KR, sk_b, sk_h, sk_w, sk_d,
          M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h, scale, b, hr, t, tm, w,
          DK: tl.constexpr, N_IND: tl.constexpr):
    """The logits x of a (BT, BW) tile: s = bf16(bf16(q . k) * scale) + bf16(clock bias [+ induction term]),
    -inf where invalid."""
    dd = tl.arange(0, DK)
    k = tl.load(KR + b * sk_b + hr * sk_h + w[:, None] * sk_w + dd[None, :] * sk_d)
    s = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)            # autocast bmm: bf16 output
    s = (s * scale).to(tl.bfloat16).to(tl.float32)                      # bf16 tensor x gain: bf16 result
    bias = ct[:, None] - cw[None, :]
    bias = tl.where(causal, float("-inf"), bias)
    if N_IND > 0:
        iv = _ind_from_matches(M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h, hr, b, t, w, tm, N_IND)
        bias = bias + iv
    bias = bias.to(tl.bfloat16).to(tl.float32)
    return tl.where(valid, s + bias, float("-inf"))


@triton.jit
def _ring_fast_kernel(Q, sq_b, sq_g, sq_t, sq_d, KR, sk_b, sk_h, sk_w, sk_d, GV, sg_b, sg_h, sg_w, sg_d,
                      CA, sca_b, sca_g, sca_t, CW, scw_b, scw_h, scw_w, P0,
                      OUT, so_t, so_b, so_c, scale, T, W, G, HR,
                      M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h,
                      XQ: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                      BT: tl.constexpr, BW: tl.constexpr, N_IND: tl.constexpr, OUT_DTYPE: tl.constexpr,
                      BDV: tl.constexpr):
    """One-kernel recall (short calls): pass 1 the online row max / sum over the valid tiles, pass 2 the scores again,
    p = bf16(exp(x - m) * (1 / l)) times the values."""
    pt = tl.program_id(0)
    pbg = tl.program_id(1)
    pd = tl.program_id(2)                 # the program's block of BDV value columns
    b = pbg // G
    g = pbg % G
    if XQ:
        hr = HR
    else:
        hr = g
    t = pt * BT + tl.arange(0, BT)
    tm = t < T
    p0 = tl.load(P0)
    dd = tl.arange(0, DK)
    q = tl.load(Q + b * sq_b + g * sq_g + t[:, None] * sq_t + dd[None, :] * sq_d, mask=tm[:, None],
                other=0.0).to(tl.bfloat16)
    if XQ:
        ct = tl.load(CW + b * scw_b + hr * scw_h + ((p0 + t) % W) * scw_w, mask=tm, other=0.0)
    else:
        ct = tl.load(CA + b * sca_b + g * sca_g + t * sca_t, mask=tm, other=0.0)
    m_i = tl.full([BT], float("-inf"), tl.float32)
    l_i = tl.zeros([BT], dtype=tl.float32)
    for w0 in range(0, W, BW):
        w = w0 + tl.arange(0, BW)
        cw, causal, valid = _rf_valid(CW, scw_b, scw_h, scw_w, b, hr, w, t, tm, p0, T, W)
        if tl.sum(valid.to(tl.int32)) > 0:
            x = _rf_x(q, ct, cw, causal, valid, KR, sk_b, sk_h, sk_w, sk_d,
                      M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h, scale, b, hr, t, tm, w, DK, N_IND)
            m_new = tl.maximum(m_i, tl.max(x, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
            pe = tl.where(valid, tl.exp(x - m_safe[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(pe, axis=1)
            m_i = m_new
    m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    inv_l = 1.0 / l_safe                                              # one reciprocal per row
    dvv = pd * BDV + tl.arange(0, BDV)
    acc = tl.zeros([BT, BDV], dtype=tl.float32)
    for w0 in range(0, W, BW):
        w = w0 + tl.arange(0, BW)
        cw, causal, valid = _rf_valid(CW, scw_b, scw_h, scw_w, b, hr, w, t, tm, p0, T, W)
        if tl.sum(valid.to(tl.int32)) > 0:
            x = _rf_x(q, ct, cw, causal, valid, KR, sk_b, sk_h, sk_w, sk_d,
                      M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h, scale, b, hr, t, tm, w, DK, N_IND)
            pr = tl.where(valid, tl.exp(x - m_safe[:, None]) * inv_l[:, None], 0.0).to(tl.bfloat16)
            gv = tl.load(GV + b * sg_b + hr * sg_h + w[:, None] * sg_w + dvv[None, :] * sg_d)
            acc = tl.dot(pr, gv, acc)
    tl.store(OUT + t[:, None] * so_t + b * so_b + (g * DV + dvv)[None, :] * so_c,
             acc.to(tl.bfloat16).to(OUT_DTYPE), mask=tm[:, None])                    # autocast bmm: bf16 output


@triton.jit
def _ring_stats_kernel(Q, sq_b, sq_g, sq_t, sq_d, KR, sk_b, sk_h, sk_w, sk_d,
                       CA, sca_b, sca_g, sca_t, CW, scw_b, scw_h, scw_w, P0,
                       scale, T, W, G, HR,
                       M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h, SCR, MST, LST, VFL,
                       XQ: tl.constexpr, DK: tl.constexpr,
                       BT: tl.constexpr, BW: tl.constexpr, N_IND: tl.constexpr):
    """Two-launch recall, launch 1: the scores x of every valid (BT, BW) tile into SCR, the row max / sum into
    MST / LST, a per-(program, tile) validity flag into VFL.  A tile is valid for the block when one of its slots has
    a finite clock and is not causal for the block's last valid row."""
    pt = tl.program_id(0)
    pbg = tl.program_id(1)
    b = pbg // G
    g = pbg % G
    if XQ:
        hr = HR
    else:
        hr = g
    t = pt * BT + tl.arange(0, BT)
    tm = t < T
    p0 = tl.load(P0)
    dd = tl.arange(0, DK)
    q = tl.load(Q + b * sq_b + g * sq_g + t[:, None] * sq_t + dd[None, :] * sq_d, mask=tm[:, None],
                other=0.0).to(tl.bfloat16)
    if XQ:
        ct = tl.load(CW + b * scw_b + hr * scw_h + ((p0 + t) % W) * scw_w, mask=tm, other=0.0)
    else:
        ct = tl.load(CA + b * sca_b + g * sca_g + t * sca_t, mask=tm, other=0.0)
    nT = tl.cdiv(T, BT)
    prog = pbg * nT + pt
    scr = SCR + (prog * BT + tl.arange(0, BT))[:, None].to(tl.int64) * W
    t_max = tl.minimum(pt * BT + BT, T) - 1
    m_i = tl.full([BT], float("-inf"), tl.float32)
    l_i = tl.zeros([BT], dtype=tl.float32)
    for w0 in range(0, W, BW):
        w = w0 + tl.arange(0, BW)
        cw, causal, valid = _rf_valid(CW, scw_b, scw_h, scw_w, b, hr, w, t, tm, p0, T, W)
        jw = (w.to(tl.int64) - p0) % W
        jw = tl.where(jw < 0, jw + W, jw)
        anyv = tl.sum(((cw != float("inf")) & ((jw >= T) | (jw <= t_max))).to(tl.int32)) > 0
        if anyv:
            x = _rf_x(q, ct, cw, causal, valid, KR, sk_b, sk_h, sk_w, sk_d,
                      M0, M1, M2, M3, sm_b, sm_t, sm_w, BETA, sbe_n, sbe_h, scale, b, hr, t, tm, w, DK, N_IND)
            tl.store(scr + w[None, :], x)
            m_new = tl.maximum(m_i, tl.max(x, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
            pe = tl.where(valid, tl.exp(x - m_safe[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(pe, axis=1)
            m_i = m_new
            tl.store(VFL + prog * (W // BW) + w0 // BW, 1)
        else:
            tl.store(VFL + prog * (W // BW) + w0 // BW, 0)
    m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(MST + prog * BT + tl.arange(0, BT), m_safe)
    tl.store(LST + prog * BT + tl.arange(0, BT), l_safe)


@triton.jit
def _ring_value_kernel(GV, sg_b, sg_h, sg_w, sg_d, OUT, so_t, so_b, so_c, T, W, G, HR, SCR, MST, LST, VFL,
                       XQ: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr,
                       OUT_DTYPE: tl.constexpr, BDV: tl.constexpr):
    """Two-launch recall, launch 2: p = exp(x - m) * (1 / l) from the stored scores, rounded to bf16, times the ring
    values on the valid tiles (VFL) into a (BT, BDV) fp32 accumulator."""
    pt = tl.program_id(0)
    pbg = tl.program_id(1)
    pd = tl.program_id(2)
    b = pbg // G
    g = pbg % G
    if XQ:
        hr = HR
    else:
        hr = g
    t = pt * BT + tl.arange(0, BT)
    tm = t < T
    nT = tl.cdiv(T, BT)
    prog = pbg * nT + pt
    scr = SCR + (prog * BT + tl.arange(0, BT))[:, None].to(tl.int64) * W
    m_safe = tl.load(MST + prog * BT + tl.arange(0, BT))
    l_safe = tl.load(LST + prog * BT + tl.arange(0, BT))
    inv_l = 1.0 / l_safe
    dvv = pd * BDV + tl.arange(0, BDV)
    acc = tl.zeros([BT, BDV], dtype=tl.float32)
    for w0 in range(0, W, BW):
        w = w0 + tl.arange(0, BW)
        vf = tl.load(VFL + prog * (W // BW) + w0 // BW)
        if vf > 0:
            x = tl.load(scr + w[None, :])
            valid = x != float("-inf")
            pr = tl.where(valid, tl.exp(x - m_safe[:, None]) * inv_l[:, None], 0.0).to(tl.bfloat16)
            gv = tl.load(GV + b * sg_b + hr * sg_h + w[:, None] * sg_w + dvv[None, :] * sg_d)
            acc = tl.dot(pr, gv, acc)
    tl.store(OUT + t[:, None] * so_t + b * so_b + (g * DV + dvv)[None, :] * so_c,
             acc.to(tl.bfloat16).to(OUT_DTYPE), mask=tm[:, None])                    # autocast bmm: bf16 output


def _recall_ind(ind4) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
    """``ind4`` -> (matches, beta): None, (matches, beta) or ("matches", matches, beta)."""
    if ind4 is None:
        return [], None
    if isinstance(ind4, torch.Tensor) or not isinstance(ind4, (tuple, list)):
        raise RuntimeError("ring_recall_prefill: ind4 must be None or (matches, beta): the induction term is formed "
                           "in the kernel from the bool match matrices")
    if len(ind4) == 3 and isinstance(ind4[0], str):
        ind4 = ind4[1:]
    if len(ind4) != 2:
        raise RuntimeError("ring_recall_prefill: ind4 must be (matches, beta)")
    matches, beta = ind4
    return list(matches), beta


def _check_recall(q4, k_ring, gv_ring, cum_ring, ct3, p0, matches, beta, T, B, G, W, dk, dv, out_dtype, xq, hr):
    for name, t in (("q4", q4), ("k_ring", k_ring), ("gv_ring", gv_ring), ("cum_ring", cum_ring), ("p0", p0)):
        require_cuda(t, f"ring_recall_prefill ({name})")
    if W % RECALL_BW or dk not in (16, 32, 64, 128, 256) or dv not in (16, 32, 64, 128, 256) or T < 1:
        raise RuntimeError(f"ring_recall_prefill: unsupported W {W} dk {dk} dv {dv} T {T} (W % {RECALL_BW} == 0, "
                           "dk / dv powers of two in [16, 256])")
    if T > W:
        raise RuntimeError(f"ring_recall_prefill: a call of {T} tokens exceeds the ring ({W} slots)")
    if T >= RECALL_BT and dv % RECALL_VALUE_BDV:
        raise RuntimeError(f"ring_recall_prefill: dv {dv} is not a multiple of the value block {RECALL_VALUE_BDV}")
    if k_ring.dtype != torch.bfloat16 or gv_ring.dtype != torch.bfloat16 or cum_ring.dtype != torch.float32:
        raise RuntimeError("ring_recall_prefill: bf16 rings and fp32 clocks are required")
    H = int(k_ring.shape[1])
    if (tuple(k_ring.shape) != (B, H, W, dk) or tuple(gv_ring.shape) != (B, H, W, dv)
            or tuple(cum_ring.shape) != (B, H, W)):
        raise RuntimeError(f"ring_recall_prefill: ring shapes {tuple(k_ring.shape)} {tuple(gv_ring.shape)} "
                           f"{tuple(cum_ring.shape)} for B {B} W {W} dk {dk} dv {dv}")
    if tuple(q4.shape) != (B, G, T, dk) or not q4.is_floating_point():
        raise RuntimeError(f"ring_recall_prefill: q4 must be a float ({B}, {G}, {T}, {dk}) view, got "
                           f"{q4.dtype} {tuple(q4.shape)}")
    if xq:
        if not 0 <= int(hr) < H:
            raise RuntimeError(f"ring_recall_prefill: ring head {hr} outside [0, {H})")
    else:
        if G != H:
            raise RuntimeError(f"ring_recall_prefill: the head read needs G = H = {H} groups, got {G}")
        if ct3 is None or tuple(ct3.shape) != (B, G, T) or ct3.dtype != torch.float32:
            raise RuntimeError(f"ring_recall_prefill: ct3 must be the ({B}, {G}, {T}) fp32 query clocks")
    if p0.dtype != torch.int64 or p0.numel() != 1:
        raise RuntimeError("ring_recall_prefill: p0 must be a (1,) int64 tensor")
    if out_dtype not in _OUT_TL:
        raise RuntimeError(f"ring_recall_prefill: out_dtype {out_dtype} (fp32 / bf16 / fp16)")
    n_ind = len(matches)
    if n_ind:
        if not 1 <= n_ind <= 4 or any(m.dtype != torch.bool or tuple(m.shape) != (B, T, W) for m in matches):
            raise RuntimeError(f"ring_recall_prefill: matches {[tuple(m.shape) for m in matches]} for "
                               f"({B}, {T}, {W}) (bool, at most 4 orders)")
        if any(m.stride() != matches[0].stride() for m in matches):
            raise RuntimeError("ring_recall_prefill: the matches must share one layout")
        if beta is None or beta.dtype != torch.float32 or beta.dim() != 2 or tuple(beta.shape) != (n_ind, H):
            raise RuntimeError(f"ring_recall_prefill: beta must be ({n_ind}, {H}) fp32")


def _ring_recall_prefill_fwd(q4: torch.Tensor, k_ring: torch.Tensor, gv_ring: torch.Tensor, cum_ring: torch.Tensor,
                             ct3: Optional[torch.Tensor], p0: torch.Tensor, matches: List[torch.Tensor],
                             beta: Optional[torch.Tensor], scale: float, T: int, B: int, G: int, W: int, dk: int,
                             dv: int, out_dtype: torch.dtype, xq: bool, hr: int) -> torch.Tensor:
    dev = q4.device
    ct = ct3 if ct3 is not None else cum_ring
    sct = (ct3.stride(0), ct3.stride(1), ct3.stride(2)) if ct3 is not None else (0, 0, 0)
    n_ind = len(matches)
    mlist = list(matches) + [matches[0]] * (4 - n_ind) if n_ind else [cum_ring] * 4
    sm = tuple(matches[0].stride()) if n_ind else (0, 0, 0)
    be = beta if n_ind else cum_ring
    sbe = (beta.stride(0), beta.stride(1)) if n_ind else (0, 0)
    out = torch.empty((T, B, G * dv), dtype=out_dtype, device=dev)
    BW = RECALL_BW
    qs = (q4, q4.stride(0), q4.stride(1), q4.stride(2), q4.stride(3))
    ks = (k_ring, k_ring.stride(0), k_ring.stride(1), k_ring.stride(2), k_ring.stride(3))
    gs = (gv_ring, gv_ring.stride(0), gv_ring.stride(1), gv_ring.stride(2), gv_ring.stride(3))
    cs = (ct, sct[0], sct[1], sct[2], cum_ring, cum_ring.stride(0), cum_ring.stride(1), cum_ring.stride(2))
    inds = (mlist[0], mlist[1], mlist[2], mlist[3], sm[0], sm[1], sm[2], be, sbe[0], sbe[1])
    if T >= RECALL_BT:
        # two launches: statistics (no accumulator) and values
        BT = RECALL_BT
        nT = triton.cdiv(T, BT)
        scr = torch.empty((B * G * nT * BT, W), dtype=torch.float32, device=dev)
        mst = torch.empty((B * G * nT * BT,), dtype=torch.float32, device=dev)
        lst = torch.empty((B * G * nT * BT,), dtype=torch.float32, device=dev)
        vfl = torch.empty((B * G * nT, W // BW), dtype=torch.int32, device=dev)
        _ring_stats_kernel[(nT, B * G)](
            *qs, *ks, *cs, p0, float(scale), T, W, G, int(hr), *inds, scr, mst, lst, vfl,
            XQ=bool(xq), DK=dk, BT=BT, BW=BW, N_IND=n_ind,
            num_warps=RECALL_STATS_WARPS, num_stages=RECALL_STATS_STAGES)
        _ring_value_kernel[(nT, B * G, dv // RECALL_VALUE_BDV)](
            *gs, out, out.stride(0), out.stride(1), out.stride(2), T, W, G, int(hr), scr, mst, lst, vfl,
            XQ=bool(xq), DV=dv, BT=BT, BW=BW, OUT_DTYPE=_OUT_TL[out_dtype], BDV=RECALL_VALUE_BDV,
            num_warps=RECALL_VALUE_WARPS, num_stages=RECALL_VALUE_STAGES)
    else:
        BT = RECALL_BT_SHORT
        _ring_fast_kernel[(triton.cdiv(T, BT), B * G, 1)](
            *qs, *ks, *gs, *cs, p0, out, out.stride(0), out.stride(1), out.stride(2), float(scale), T, W, G, int(hr),
            *inds, XQ=bool(xq), DK=dk, DV=dv, BT=BT, BW=BW, N_IND=n_ind, OUT_DTYPE=_OUT_TL[out_dtype], BDV=dv,
            num_warps=RECALL_WARPS, num_stages=RECALL_STAGES)
    return out


@torch.library.custom_op("neuronspark::ring_recall_prefill", mutates_args=())
def _ring_recall_prefill_op(q4: torch.Tensor, k_ring: torch.Tensor, gv_ring: torch.Tensor, cum_ring: torch.Tensor,
                            ct3: Optional[torch.Tensor], p0: torch.Tensor, matches: List[torch.Tensor],
                            beta: Optional[torch.Tensor], scale: float, T: int, B: int, G: int, W: int, dk: int,
                            dv: int, out_dtype: torch.dtype, xq: bool, hr: int) -> torch.Tensor:
    return _ring_recall_prefill_fwd(q4, k_ring, gv_ring, cum_ring, ct3, p0, list(matches), beta, scale, T, B, G, W,
                                    dk, dv, out_dtype, xq, hr)


@_ring_recall_prefill_op.register_fake
def _ring_recall_prefill_fake(q4, k_ring, gv_ring, cum_ring, ct3, p0, matches, beta, scale, T, B, G, W, dk, dv,
                              out_dtype, xq, hr):
    return q4.new_empty((T, B, G * dv), dtype=out_dtype)


def ring_recall_prefill(q4: torch.Tensor, k_ring: torch.Tensor, gv_ring: torch.Tensor, cum_ring: torch.Tensor,
                        ct3: Optional[torch.Tensor], p0: torch.Tensor, ind4, scale: float, T: int, B: int, G: int,
                        W: int, dk: int, dv: int, out_dtype: torch.dtype, xq: bool = False,
                        hr: int = 0) -> torch.Tensor:
    """Pool read of every query row of a prefill call over the whole ring, after the call's ring write.

    q4 (B, G, T, dk) float view (any strides; rounded to bf16 in the kernel): the queries -- the six heads' gained unit
    queries (``xq=False``, group g reads ring head g) or the last head's extra queries (``xq=True``, every group reads
    ring head ``hr``); k_ring / gv_ring (B, H, W, dk / dv) bf16 and cum_ring (B, H, W) fp32: the rings with this call's
    events written; ct3 (B, G, T) fp32 view of the query clocks (``xq=False``) or None (``xq=True``: the clocks of this
    call's slots of head ``hr``); p0 (1,) int64: the ring pointer at the start of the call; ind4: None or
    ``(matches, beta)`` with n (<= 4) bool (B, T, W) match matrices sharing one layout and beta (n, H) fp32 (the head's
    column is read); scale: the match scale (16.0).  Returns (T, B, G * dv) in ``out_dtype`` (the bf16 read widened)."""
    matches, beta = _recall_ind(ind4)
    T, B, G, W, dk, dv = int(T), int(B), int(G), int(W), int(dk), int(dv)
    _check_recall(q4, k_ring, gv_ring, cum_ring, ct3, p0, matches, beta, T, B, G, W, dk, dv, out_dtype, xq, hr)
    return torch.ops.neuronspark.ring_recall_prefill(q4, k_ring, gv_ring, cum_ring, ct3, p0.reshape(1), matches,
                                                     beta if matches else None, float(scale), T, B, G, W, dk, dv,
                                                     out_dtype, bool(xq), int(hr))


def ring_recall_prefill_ref(q4: torch.Tensor, k_ring: torch.Tensor, gv_ring: torch.Tensor, cum_ring: torch.Tensor,
                            ct3: Optional[torch.Tensor], p0: torch.Tensor, ind4, scale: float, T: int, B: int, G: int,
                            W: int, dk: int, dv: int, out_dtype: torch.dtype, xq: bool = False,
                            hr: int = 0) -> torch.Tensor:
    """The torch expressions of the prefill ring read (the memory layer's ring-chunk branch; the extra-query read)
    under the inference forward's bf16 autocast: bf16 bmm scores, the clock bias with this call's later slots at -inf
    plus the induction term, cast to bf16, an fp32 softmax over all W slots, a bf16 bmm with the values.  Same
    arguments and output as :func:`ring_recall_prefill`."""
    matches, beta = _recall_ind(ind4)
    T, W = int(T), int(W)
    dev = q4.device
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ind = None
        if matches:
            Hh = int(beta.shape[1])
            allt = _ring_term(matches, beta, Hh)                                  # (B, H, T, W)
            ind = allt if not xq else allt[:, hr].unsqueeze(1).expand(B, G, T, W)
        qh = q4.to(k_ring.dtype).float()
        ar = torch.arange(T, device=dev, dtype=torch.int64)
        slots = (p0.reshape(1) + ar) % W
        tj = torch.full((W,), -1, dtype=torch.int64, device=dev)
        tj[slots] = ar
        if not xq:
            scores = torch.einsum("bhtd,bhwd->bhtw", qh, k_ring.float()) * scale
            bias = (ct3.unsqueeze(-1) - cum_ring.unsqueeze(2))
            bias = bias.masked_fill(tj.view(1, 1, 1, W) > ar.view(1, 1, T, 1), float("-inf"))
            if ind is not None:
                bias = bias + ind
            bias = bias.to(k_ring.dtype).float()
            probs = torch.softmax(scores + bias, dim=-1)
            out = torch.einsum("bhtw,bhwd->bhtd", probs.to(gv_ring.dtype).float(), gv_ring.float())
            return out.permute(2, 0, 1, 3).reshape(T, B, G * dv).to(out_dtype)
        k_r, gv_r, cum_r = k_ring[:, hr], gv_ring[:, hr], cum_ring[:, hr]
        cum_t = cum_r.index_select(1, slots)
        scores = torch.einsum("bmtd,bwd->bmtw", qh, k_r.float()) * scale
        bias = cum_t.unsqueeze(-1) - cum_r.unsqueeze(1)
        if T > 1:
            bias = bias.masked_fill(tj.view(1, 1, W) > ar.view(1, T, 1), float("-inf"))
        if ind is not None:
            bias = bias + ind[:, 0]
        bias = bias.to(k_ring.dtype).float()
        probs = torch.softmax(scores + bias.unsqueeze(1), dim=-1)
        out = torch.einsum("bmtw,bwd->bmtd", probs.to(gv_ring.dtype).float(), gv_r.float())
        return out.permute(2, 0, 1, 3).reshape(T, B, -1).to(out_dtype)
