"""Weight gradient of the linear operator for the products that leave cuBLAS's GEMM short of the GPU: ``dW (n, K) fp32 =
dy[:, :n]^T x`` over all rows as S partial products of row chunks.

cuBLAS runs the weight-gradient GEMM of a projection with few output tiles (the gates, low-rank factors and head
projections of every layer) as a few long programs over all rows: 12 us for a (512, 512) weight over 8,192 rows on a
B200, where its operands take 2 us to read.  Here the rows are cut into S chunks of equal length and cuBLAS's batched
GEMM computes the S partial products at once (``torch.bmm`` with an fp32 output: S times the programs, each over a
chunk); one kernel then adds the S partials in the order s = 0, 1, ..., S - 1 and writes the sum (``out = sum``) or
adds it into the buffer (``out = out + sum``), and the buffer's version counter is advanced as an ``out=`` operation
advances it.  The partial sum's order is fixed and the batched GEMM's for a shape on one stream: the same bits at every
call on one stream (one GPU, one cuBLAS version).  The sums are fp32 as in the single GEMM: values that cancel across
rows lose what an fp32 addition loses in any order.

:func:`chunks` gives S from (n, K, rows): the chunk length 1,024 rows when the product is narrow (min(n, K) at most
``WG_NARROW`` or at most 4 tiles of 128 x 128) and 2,048 rows otherwise, S the power of two of rows / chunk, at most
``WG_MAX_CHUNKS``, halved while S x tiles exceeds ``WG_MAX_PROGRAMS``; products of more than ``WG_TILES_PER_KROW`` tiles
per 1,024 rows, and those S < 2 leaves, stay one cuBLAS GEMM (the rule measured on the weight-gradient shapes of the
four README scales at 1,024 to 8,192 rows).

:func:`neuronspark.ops.linear.linear`'s backward computes every weight gradient through :func:`weight_grad` on CUDA.
Reference :func:`weight_grad_ref`: the bf16 operands multiplied in float64, the fp32 result (added into ``out`` in
fp32).  The sums run in another order than cuBLAS's single GEMM.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["WG_NARROW", "WG_MAX_CHUNKS", "WG_MAX_PROGRAMS", "WG_TILES_PER_KROW", "SUM_BLOCK", "SUM_WARPS", "chunks",
           "weight_grad", "weight_grad_ref"]

# ================================================================================================ fixed constants
#: a product with min(n, K) at most this many features (or at most 4 tiles) takes chunks of 1,024 rows, else 2,048
WG_NARROW = 32
#: chunks at most; S x tiles (of 128 x 128) at most
WG_MAX_CHUNKS = 8
WG_MAX_PROGRAMS = 128
#: a product of more tiles than this per 1,024 rows stays one GEMM
WG_TILES_PER_KROW = 6
#: the partial sum: elements per program, warps
SUM_BLOCK = 1024
SUM_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _sum_partials_kernel(PS, OUT, NK, S: tl.constexpr, ACC: tl.constexpr, BLOCK: tl.constexpr):
    """out[i] = (out[i] +) ps[0, i] + ps[1, i] + ... + ps[S - 1, i], in this order."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < NK
    acc = tl.load(PS + i, mask=m, other=0.0)
    for s in tl.static_range(1, S):
        acc += tl.load(PS + s * NK + i, mask=m, other=0.0)
    if ACC:
        acc = tl.load(OUT + i, mask=m, other=0.0) + acc
    tl.store(OUT + i, acc, mask=m)


# ================================================================================================ launchers
def chunks(n: int, k: int, rows: int) -> int:
    """S, the row chunks of the weight gradient of an (n, K) weight over ``rows`` rows (1: one cuBLAS GEMM)."""
    tiles = triton.cdiv(n, 128) * triton.cdiv(k, 128)
    if tiles * 1024 > WG_TILES_PER_KROW * rows:
        return 1
    length = 1024 if (min(n, k) <= WG_NARROW or tiles <= 4) else 2048
    if rows < 2 * length:
        return 1
    s = min(WG_MAX_CHUNKS, 1 << ((rows // length).bit_length() - 1))
    while s > 1 and s * tiles > WG_MAX_PROGRAMS:
        s //= 2
    return s if s >= 2 and rows % s == 0 else 1


def weight_grad(dy: torch.Tensor, x: torch.Tensor, n: int, out: Optional[torch.Tensor] = None,
                accumulate: bool = False) -> torch.Tensor:
    """``dy[:, :n]^T x`` in fp32: ``dy`` (rows, P) bf16 with unit last stride (P >= n), ``x`` (rows, K) bf16 with unit
    last stride, both CUDA tensors on one device; written into ``out`` (an fp32 (n, K) tensor on their device that is
    not a negative view, a new one when None), or added into it with ``accumulate``.  Returns ``out``.  Products where
    :func:`chunks` gives S >= 2 (and ``out`` is contiguous) run as S chunk products and one partial sum, the others as
    one cuBLAS GEMM (``torch.mm`` / ``torch.addmm`` with ``out_dtype=float32``); the arguments are checked on both."""
    require_cuda(dy, "weight_grad")
    if dy.dim() != 2 or x.dim() != 2 or dy.dtype != torch.bfloat16 or x.dtype != torch.bfloat16 \
            or int(dy.shape[0]) != int(x.shape[0]) or int(dy.shape[1]) < n or dy.stride(1) != 1 or x.stride(1) != 1 \
            or x.device != dy.device:
        raise RuntimeError(f"neuronspark.weight_grad: dy (rows, >= {n}) and x (rows, K) bf16 with unit last strides on "
                           f"one device, got {dy.dtype} {tuple(dy.shape)} {tuple(dy.stride())}, {x.dtype} "
                           f"{tuple(x.shape)} {tuple(x.stride())}")
    rows, k = int(x.shape[0]), int(x.shape[1])
    if out is None:
        if accumulate:
            raise RuntimeError("neuronspark.weight_grad: accumulate needs out")
        out = torch.empty((n, k), dtype=torch.float32, device=x.device)
    elif out.dtype != torch.float32 or tuple(out.shape) != (n, k) or out.device != dy.device or out.is_neg():
        raise RuntimeError(f"neuronspark.weight_grad: out must be an fp32 ({n}, {k}) tensor on {dy.device} that is not "
                           f"a negative view, got {out.dtype} {tuple(out.shape)} on {out.device}")
    s = chunks(n, k, rows) if out.is_contiguous() else 1
    if s == 1:
        if accumulate:
            torch.addmm(out, dy[:, :n].t(), x, out_dtype=torch.float32, out=out)
        else:
            torch.mm(dy[:, :n].t(), x, out_dtype=torch.float32, out=out)
        return out
    c = rows // s
    ps = torch.bmm(dy[:, :n].reshape(s, c, n).transpose(1, 2), x.reshape(s, c, k), out_dtype=torch.float32)
    nk = n * k
    with torch.cuda.device(dy.device):
        _sum_partials_kernel[(triton.cdiv(nk, SUM_BLOCK),)](ps, out, nk, S=s, ACC=bool(accumulate), BLOCK=SUM_BLOCK,
                                                             num_warps=SUM_WARPS)
    torch.autograd.graph.increment_version(out)                    # as the out= GEMM: saved copies of out see it
    return out


# ================================================================================================ reference
def weight_grad_ref(dy: torch.Tensor, x: torch.Tensor, n: int, out: Optional[torch.Tensor] = None,
                    accumulate: bool = False) -> torch.Tensor:
    """:func:`weight_grad` in float64: the bf16 operands' products summed in float64, rounded to fp32, then written into
    ``out`` or added into it in fp32 (a new tensor; ``out`` is not modified)."""
    g = (dy[:, :n].double().t() @ x.double()).float()
    if accumulate:
        return out + g
    return g
