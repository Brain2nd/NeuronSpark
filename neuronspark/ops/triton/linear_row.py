"""Inference product of one row: ``y = x @ w^T`` with ``x`` (1, K) and up to 4,096 output features (the decode step
of one sequence: every projection of a layer but the dendritic in-projection, the stacked projections of z and the
readout).

cuBLAS serves a one-row product of this size with a GEMM kernel built for many rows (4 - 5 us in a replayed graph
whatever the weight's size; a split-K product and a separate reduction kernel for a long K, a dot and a reduction for
a few output features).  Here a program computes ``BN`` output features: the bf16 operands widened to fp32, the
products accumulated in fp32 over K in tiles of ``BK`` columns spread over the block's threads (each thread issues one
or two loads per tile), one tree sum per output feature, the result rounded to bf16 once.  The sums run in another
order than cuBLAS's; the reference is :func:`linear_row_ref` (= :func:`neuronspark.ops.linear.linear_ref`: the bf16
operands multiplied in float64, one bf16 rounding).

:func:`neuronspark.ops.linear.linear` routes its inference calls here when :func:`takes_row` holds.  The bound on the
output features is where cuBLAS is the faster of the two on the B200 (measured in a replayed graph on cold weights:
N = 5,504 and 18,832 at K = 1,536).
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["ROW_MAX_N", "ROW_MAX_K", "ROW_LONG_K", "ROW_TILE", "ROW_LONG_TILE", "row_tile", "takes_row", "linear_row",
           "linear_row_ref"]

# ================================================================================================ fixed constants
#: output features and input features (inclusive bounds; the kernel is compiled per K with one unrolled tile per
#: K tile: 8 at the bound)
ROW_MAX_N = 4096
ROW_MAX_K = 16384
#: (K tile, elements of a program's tile, warps) for K <= ROW_LONG_K and for a longer K.  A program takes
#: ``tile elements // BK`` output features, with BK = min(next_power_of_2(K), K tile).  One tile: 16 elements a
#: thread; several tiles one after the other (the dendritic out-projection, K = 9,216): 8 elements a thread
ROW_LONG_K = 2048
ROW_TILE = (2048, 2048, 4)
ROW_LONG_TILE = (2048, 2048, 8)


def row_tile(k: int) -> Tuple[int, int, int]:
    """(BN, BK, warps) of the kernel for an input width ``k``."""
    bk_max, elems, warps = ROW_TILE if int(k) <= ROW_LONG_K else ROW_LONG_TILE
    bk = min(triton.next_power_of_2(int(k)), bk_max)
    return max(1, elems // bk), bk, warps


# ================================================================================================ kernel
@triton.jit
def _linear_row_kernel(X, W, Y, N, s_w, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    mn = n < N
    kk = tl.arange(0, BK)
    rows = W + n[:, None].to(tl.int64) * s_w
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BK):
        if k0 + BK <= K:
            x = tl.load(X + k0 + kk).to(tl.float32)
            w = tl.load(rows + (k0 + kk)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        else:
            mk = k0 + kk < K
            x = tl.load(X + k0 + kk, mask=mk, other=0.0).to(tl.float32)
            w = tl.load(rows + (k0 + kk)[None, :], mask=mn[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += w * x[None, :]
    tl.store(Y + n, tl.sum(acc, axis=1).to(tl.bfloat16), mask=mn)


# ================================================================================================ launcher
def takes_row(x2: torch.Tensor, w: torch.Tensor, n: int) -> bool:
    """x2 (1, K) bf16 with a unit last stride and w (P >= n, K) bf16 contiguous on one CUDA device (plain tensors:
    no lazy negation), 1 <= K <= ROW_MAX_K, 1 <= n <= ROW_MAX_N: the product of x2 with the first ``n`` rows of w"""
    if x2.dim() != 2 or w.dim() != 2 or x2.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return False
    if not x2.is_cuda or w.device != x2.device or x2.stride(1) != 1 or not w.is_contiguous():
        return False
    if x2.is_neg() or w.is_neg():
        return False
    return (int(x2.shape[0]) == 1 and 1 <= int(x2.shape[1]) == int(w.shape[1]) <= ROW_MAX_K
            and 1 <= int(n) <= min(ROW_MAX_N, int(w.shape[0])))


def linear_row(x2: torch.Tensor, w: torch.Tensor, n: int) -> torch.Tensor:
    """(1, n) bf16 = x2 @ w[:n]^T for operands that satisfy :func:`takes_row` (raises otherwise)."""
    require_cuda(x2, "linear_row")
    if not takes_row(x2, w, n):
        raise RuntimeError(f"neuronspark.linear_row: operands x {tuple(x2.shape)} {x2.dtype} stride "
                           f"{tuple(x2.stride())}, w {tuple(w.shape)} {w.dtype}, n {n} outside its form (one row, "
                           f"K <= {ROW_MAX_K}, 1 <= n <= {ROW_MAX_N} rows of a contiguous bf16 weight)")
    K, N = int(x2.shape[1]), int(n)
    bn, bk, warps = row_tile(K)
    y = torch.empty((1, N), dtype=torch.bfloat16, device=x2.device)
    _linear_row_kernel[(triton.cdiv(N, bn),)](x2, w, y, N, int(w.stride(0)), K=K, BN=bn, BK=bk, num_warps=warps)
    return y


def linear_row_ref(x2: torch.Tensor, w: torch.Tensor, n: int) -> torch.Tensor:
    """Pure-torch reference: the bf16 operands multiplied in float64, the output rounded to bf16 once."""
    return torch.matmul(x2.double(), w[:int(n)].double().t()).to(torch.bfloat16)
