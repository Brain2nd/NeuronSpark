"""Extra-query formation of a prefill call in one kernel.

The last memory head reads its ring with M extra queries (1.27B: M = 4, dk = 256).  Query m of row (t, b) is formed
from its bf16 projection ``q_m`` (T, B, dk) (``bf16(u[..., r m : r m + r] @ pool_xq_b[m]^T)``, a cuBLAS GEMM outside
this op) by the expressions of the hippocampal layer under the bf16 autocast of the inference regime::

    x   = silu(q_m)                                   fp32 x / (1 + expf(-x)), rounded to bf16
    y   = rope(x, bf16(cos), bf16(sin))               interleaved pairs: y_2p = bf16(bf16(x_2p c) - bf16(x_2p+1 s)),
                                                      y_2p+1 = bf16(bf16(x_2p+1 c) + bf16(x_2p s))
    n   = max(sqrt(sum_i y_i^2), 1e-12)               fp32 (F.normalize of a bf16 tensor under autocast: fp32 norm)
    q   = (y / n) * (exp(psi_m) * f32(head_gain))     fp32: the IEEE division, the gain and the product
    out = bf16(q)

``psi`` (T, B, >= M) fp32 is the clamped gain projection, ``head_gain`` the last head's match gain in the query dtype
(one element).  The chain returns the fp32 queries; the ring read (``ring_recall_prefill``, extra-query form) rounds
its queries to bf16 before the score dot, so the bf16 output of this op is exactly the operand the read forms.  One
program forms ``XQ_QUERY_ROWS`` (t, b) rows of one query and writes them into the stacked (M, T, B, dk) bf16 buffer;
the public function returns its (B, M, T, dk) permuted view (the layout of the chain's stack + permute).

Numerics: every rounding point of the chain is kept (fp32 arithmetic through IEEE division and square root, no FMA
contraction); the fp32 sum of squares runs in another order than ATen's norm reduction.  Tests: the per-operator
tolerance rule of :mod:`neuronspark.utils.testing` against :func:`xq_queries_prefill_ref`.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.modules.rope import apply_rope_interleaved_ref
from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _rn_div_f32  # noqa: F401  (resolved by the kernel through this module's globals)
from neuronspark.ops.triton.pool_ring import _bf16  # noqa: F401  (the bf16 rounding helper of the ring kernels)

__all__ = [
    "XQ_QUERY_MAX",
    "XQ_QUERY_ROWS",
    "XQ_QUERY_WARPS",
    "XQ_NORM_EPS",
    "xq_queries_prefill",
    "xq_queries_prefill_ref",
]

# ================================================================================================ fixed constants
#: largest number of extra queries one launch forms
XQ_QUERY_MAX = 4
#: (t, b) rows per program, warps
XQ_QUERY_ROWS = 8
XQ_QUERY_WARPS = 4
#: floor of the query norm (F.normalize's eps)
XQ_NORM_EPS = 1e-12


# ================================================================================================ kernel
@triton.jit
def _xq_query_kernel(Q0, Q1, Q2, Q3, PSI, s_psi_t, s_psi_b, GAIN, COS, SIN, s_c_t, s_c_b, s_s_t, s_s_b, OUT, R, B,
                     D2: tl.constexpr, BLOCK_R: tl.constexpr):
    pid_r = tl.program_id(0)
    m = tl.program_id(1)
    if m == 0:
        Q = Q0
    elif m == 1:
        Q = Q1
    elif m == 2:
        Q = Q2
    else:
        Q = Q3
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mr = r < R
    p = tl.arange(0, D2)
    t = r // B
    b = r - t * B
    r64 = r.to(tl.int64)
    t64 = t.to(tl.int64)
    b64 = b.to(tl.int64)
    base = Q + r64[:, None] * (2 * D2) + 2 * p[None, :]
    xe = tl.load(base, mask=mr[:, None], other=0.0).to(tl.float32)
    xo = tl.load(base + 1, mask=mr[:, None], other=0.0).to(tl.float32)
    # SiLU: fp32 x / (1 + expf(-x)), rounded to bf16
    xe = _bf16(_rn_div_f32(xe, 1.0 + libdevice.exp(-xe)))
    xo = _bf16(_rn_div_f32(xo, 1.0 + libdevice.exp(-xo)))
    # interleaved RoPE in bf16 arithmetic (the tables cast to the query dtype)
    c = tl.load(COS + t64[:, None] * s_c_t + b64[:, None] * s_c_b + p[None, :], mask=mr[:, None], other=0.0)
    s = tl.load(SIN + t64[:, None] * s_s_t + b64[:, None] * s_s_b + p[None, :], mask=mr[:, None], other=0.0)
    c = _bf16(c.to(tl.float32))
    s = _bf16(s.to(tl.float32))
    ye = _bf16(_bf16(xe * c) - _bf16(xo * s))
    yo = _bf16(_bf16(xo * c) + _bf16(xe * s))
    # unit rows: fp32 norm, floor, IEEE division (no rounding to bf16)
    ss = tl.sum(ye * ye, axis=1) + tl.sum(yo * yo, axis=1)
    nrm = tl.maximum(tl.sqrt_rn(ss), 1e-12)
    ue = _rn_div_f32(ye, nrm[:, None])
    uo = _rn_div_f32(yo, nrm[:, None])
    # the gain exp(psi_m) * head_gain in fp32, the fp32 product, one rounding to bf16 at the store
    psi = tl.load(PSI + t64 * s_psi_t + b64 * s_psi_b + m, mask=mr, other=0.0)
    g = libdevice.exp(psi) * tl.load(GAIN).to(tl.float32)
    oe = (ue * g[:, None]).to(tl.bfloat16)
    oo = (uo * g[:, None]).to(tl.bfloat16)
    ob = OUT + (m.to(tl.int64) * R + r64)[:, None] * (2 * D2) + 2 * p[None, :]
    tl.store(ob, oe, mask=mr[:, None])
    tl.store(ob + 1, oo, mask=mr[:, None])


# ================================================================================================ launcher
def _table_stride(tab: torch.Tensor, i: int) -> int:
    return 0 if int(tab.shape[i]) == 1 else int(tab.stride(i))


def _check(qs: Sequence[torch.Tensor], psi: torch.Tensor, head_gain: torch.Tensor, cos: torch.Tensor,
           sin: torch.Tensor):
    M = len(qs)
    if not 1 <= M <= XQ_QUERY_MAX:
        raise RuntimeError(f"neuronspark.xq_queries_prefill: 1 .. {XQ_QUERY_MAX} queries per launch, got {M}")
    q0 = qs[0]
    for name, t in (("q", q0), ("psi", psi), ("head_gain", head_gain), ("cos", cos), ("sin", sin)):
        require_cuda(t, f"xq_queries_prefill ({name})")
    if q0.dim() != 3 or q0.dtype != torch.bfloat16:
        raise RuntimeError(f"neuronspark.xq_queries_prefill: the projections must be bf16 (T, B, dk), got "
                           f"{q0.dtype} {tuple(q0.shape)}")
    T, B, dk = (int(v) for v in q0.shape)
    half = dk // 2
    if dk % 2 or half < 16 or half & (half - 1):
        raise RuntimeError(f"neuronspark.xq_queries_prefill: dk = {dk} (dk / 2 a power of two >= 16)")
    if any(tuple(q.shape) != (T, B, dk) or q.dtype != torch.bfloat16 or not q.is_contiguous() for q in qs):
        raise RuntimeError("neuronspark.xq_queries_prefill: every projection must be a contiguous bf16 "
                           f"({T}, {B}, {dk}) tensor")
    if (psi.dtype != torch.float32 or psi.dim() != 3 or tuple(psi.shape[:2]) != (T, B) or int(psi.shape[-1]) < M
            or psi.stride(-1) != 1):
        raise RuntimeError(f"neuronspark.xq_queries_prefill: psi must be fp32 ({T}, {B}, >= {M}) with a unit inner "
                           f"stride, got {psi.dtype} {tuple(psi.shape)} strides {tuple(psi.stride())}")
    if head_gain.numel() != 1 or not head_gain.is_floating_point():
        raise RuntimeError("neuronspark.xq_queries_prefill: head_gain must be a one-element float tensor")
    for name, tab in (("cos", cos), ("sin", sin)):
        if (tab.dim() != 3 or tab.stride(-1) != 1 or int(tab.shape[-1]) != half
                or int(tab.shape[0]) not in (1, T) or int(tab.shape[1]) not in (1, B)
                or tab.dtype not in (torch.float32, torch.bfloat16)):
            raise RuntimeError(f"neuronspark.xq_queries_prefill: {name} must be fp32 / bf16 (T or 1, B or 1, {half}) "
                               f"with a unit inner stride, got {tab.dtype} {tuple(tab.shape)} "
                               f"strides {tuple(tab.stride())}")
    if M * T * B * dk >= 2 ** 31:
        raise RuntimeError("neuronspark.xq_queries_prefill: too many elements for the row addressing")
    return M, T, B, dk


def _xq_queries_prefill_fwd(qs: List[torch.Tensor], psi: torch.Tensor, head_gain: torch.Tensor, cos: torch.Tensor,
                            sin: torch.Tensor) -> torch.Tensor:
    M, T, B, dk = _check(qs, psi, head_gain, cos, sin)
    R = T * B
    out = torch.empty((M, T, B, dk), dtype=torch.bfloat16, device=qs[0].device)
    ptrs = list(qs) + [qs[0]] * (XQ_QUERY_MAX - M)
    _xq_query_kernel[(triton.cdiv(R, XQ_QUERY_ROWS), M)](
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], psi, int(psi.stride(0)), int(psi.stride(1)), head_gain.reshape(1),
        cos, sin, _table_stride(cos, 0), _table_stride(cos, 1), _table_stride(sin, 0), _table_stride(sin, 1),
        out, R, B, D2=dk // 2, BLOCK_R=XQ_QUERY_ROWS, num_warps=XQ_QUERY_WARPS, enable_fp_fusion=False)
    return out


@torch.library.custom_op("neuronspark::xq_queries_prefill", mutates_args=())
def _xq_queries_prefill_op(qs: List[torch.Tensor], psi: torch.Tensor, head_gain: torch.Tensor, cos: torch.Tensor,
                           sin: torch.Tensor) -> torch.Tensor:
    return _xq_queries_prefill_fwd(list(qs), psi, head_gain, cos, sin)


@_xq_queries_prefill_op.register_fake
def _xq_queries_prefill_fake(qs, psi, head_gain, cos, sin):
    T, B, dk = (int(v) for v in qs[0].shape)
    return qs[0].new_empty((len(qs), T, B, dk), dtype=torch.bfloat16)


def xq_queries_prefill(qs: Sequence[torch.Tensor], psi: torch.Tensor, head_gain: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor) -> torch.Tensor:
    """(B, M, T, dk) bf16: the M extra queries of a prefill call (the permuted view of a contiguous (M, T, B, dk)
    buffer).

    ``qs``: M (<= 4) bf16 contiguous (T, B, dk) projections; ``psi`` (T, B, >= M) fp32 with a unit inner stride: the
    clamped gain projection (column m is query m's log gain); ``head_gain``: the last head's match gain, one element
    (bf16, the query dtype); ``cos`` / ``sin`` (T or 1, B or 1, dk / 2) fp32 / bf16 with a unit inner stride: the
    last head's band of the RoPE tables at each row's positions."""
    buf = torch.ops.neuronspark.xq_queries_prefill(list(qs), psi, head_gain, cos, sin)
    return buf.permute(2, 0, 1, 3)


# ================================================================================================ reference
def xq_queries_prefill_ref(qs: Sequence[torch.Tensor], psi: torch.Tensor, head_gain: torch.Tensor, cos: torch.Tensor,
                           sin: torch.Tensor) -> torch.Tensor:
    """The hippocampal layer's torch formation of the extra queries after their projections, under the bf16 autocast of
    the inference regime (same arguments as :func:`xq_queries_prefill`): SiLU and the interleaved RoPE as bf16 ops,
    ``F.normalize`` (fp32 under autocast), the gain ``exp(psi_m) * f32(head_gain)`` in fp32 and the product with the
    unit query; stacked and permuted to (B, M, T, dk).  Returns the fp32 queries."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        queries = []
        for m, q in enumerate(qs):
            q_m = F.silu(q)
            q_m = F.normalize(apply_rope_interleaved_ref(q_m, cos.to(q_m.dtype), sin.to(q_m.dtype)), dim=-1)
            g = torch.exp(psi[..., m:m + 1]) * head_gain.to(torch.float32)
            queries.append(q_m * g.to(q_m.dtype))
        return torch.stack(queries, dim=0).permute(2, 0, 1, 3)
