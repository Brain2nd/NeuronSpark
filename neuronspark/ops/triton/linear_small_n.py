"""Inference product of a few rows and a few output features: ``y = x @ w^T`` with N < 16 output features and 2 to 32
rows (the decode step's per-head projections, e.g. the value-residual gate (6, 1536)).

cuBLAS serves such a product as a split-K GEMM followed by a separate reduction kernel (two launches, 12-23 us at 8-32
rows); here one program per row computes all N outputs: the bf16 operands widened to fp32, products accumulated in fp32
over K (tiles of 256, 128 or 64 columns, then one tree sum per output), the result rounded to bf16 once.  The sums run
in another order than cuBLAS's; the reference is :func:`linear_small_n_ref` (= :func:`neuronspark.ops.linear.linear_ref`:
the bf16 operands multiplied in float64, one bf16 rounding).

:func:`neuronspark.ops.linear.linear` routes its inference calls here when :func:`takes` holds (one row stays on cuBLAS:
the single-row product is as fast there).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["SMALL_N_MAX", "SMALL_N_MIN_ROWS", "SMALL_N_MAX_ROWS", "SMALL_N_WARPS", "takes", "linear_small_n",
           "linear_small_n_ref"]

# ================================================================================================ fixed constants
#: output features (exclusive bound), rows (inclusive bounds), warps; K tile = 256 / 128 / 64 (the largest dividing K)
SMALL_N_MAX = 16
SMALL_N_MIN_ROWS = 2
SMALL_N_MAX_ROWS = 32
SMALL_N_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _linear_small_n_kernel(X, s_x, W, Y, K, N, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, BN)
    mn = n < N
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), dtype=tl.float32)
    for k0 in tl.range(0, K, BK):
        x = tl.load(X + row * s_x + k0 + kk).to(tl.float32)
        w = tl.load(W + n[:, None].to(tl.int64) * K + (k0 + kk)[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
        acc += w * x[None, :]
    tl.store(Y + row * N + n, tl.sum(acc, axis=1).to(tl.bfloat16), mask=mn)


# ================================================================================================ launcher
def takes(x2: torch.Tensor, w: torch.Tensor) -> bool:
    """x2 (M, K) bf16 with a unit last stride and w (N, K) bf16 contiguous on one CUDA device, 2 <= M <= 32, N < 16,
    K % 64 == 0"""
    if x2.dim() != 2 or w.dim() != 2 or x2.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        return False
    if not x2.is_cuda or w.device != x2.device or x2.stride(1) != 1 or not w.is_contiguous():
        return False
    M, K = (int(s) for s in x2.shape)
    N = int(w.shape[0])
    return (SMALL_N_MIN_ROWS <= M <= SMALL_N_MAX_ROWS and 1 <= N < SMALL_N_MAX and int(w.shape[1]) == K
            and K % 64 == 0)


def linear_small_n(x2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """(M, N) bf16 = x2 @ w^T for operands that satisfy :func:`takes` (raises otherwise)."""
    require_cuda(x2, "linear_small_n")
    if not takes(x2, w):
        raise RuntimeError(f"neuronspark.linear_small_n: operands x {tuple(x2.shape)} {x2.dtype} stride "
                           f"{tuple(x2.stride())}, w {tuple(w.shape)} {w.dtype} outside its form (2..32 rows, N < 16, "
                           f"K % 64 == 0, bf16)")
    M, K = (int(s) for s in x2.shape)
    N = int(w.shape[0])
    bk = 256 if K % 256 == 0 else (128 if K % 128 == 0 else 64)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x2.device)
    _linear_small_n_kernel[(M,)](x2, int(x2.stride(0)), w, y, K, N, BN=max(2, triton.next_power_of_2(N)), BK=bk,
                                 num_warps=SMALL_N_WARPS)
    return y


def linear_small_n_ref(x2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Pure-torch reference: the bf16 operands multiplied in float64, the output rounded to bf16 once."""
    return torch.matmul(x2.double(), w.double().t()).to(torch.bfloat16)
