"""The bf16 operand of the projections that read one fp32 tensor (training).

``shared_operand(x, uses)`` -> ``uses`` tensors, each the row-major bf16 rounding of the fp32 (T, B, C) ``x`` (they
share one buffer); the gradient of ``x`` is the fp32 sum of the gradients of the ``uses`` tensors.

The linear operator (:mod:`neuronspark.ops.linear`) rounds an fp32 operand to bf16 itself and returns that operand's
gradient, a bf16 GEMM product, widened to fp32; ``n`` projections of one fp32 tensor then cost ``n`` casts in the
forward, ``n`` widenings in the backward and ``n - 1`` fp32 additions of full-size tensors, and a layer input that is a
view of batch-major memory costs a row-major copy before them.  Here the rounding is made once (one copy kernel, which
writes the row-major layout the GEMMs take), every projection takes a bf16 tensor (its gradient comes back as the
bf16 product), and the backward widens and sums the ``uses`` gradients in one kernel: the same bf16 operand for every
projection and the same fp32 sum of the same bf16 gradients, added from the last use to the first (the order in which
autograd accumulates the gradients of uses created one after the other).

The ``uses`` outputs are views of one tensor: they must not be modified in place.  A use without a gradient is left
out of the sum.  The rule of :mod:`neuronspark.utils.testing` against :func:`shared_operand_ref`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _sr_bf16_to_f32 as _w  # noqa: F401  (jit helper)

__all__ = ["SharedOperandFn", "shared_operand", "shared_operand_ref", "MAX_USES", "SUM_BLOCK", "SUM_WARPS"]

# ------------------------------------------------------------------------------------------------ fixed constants
MAX_USES = 8                          # gradient pointers of the backward kernel
SUM_BLOCK, SUM_WARPS = 1024, 4        # elements per program, warps


@triton.jit
def _sum_kernel(G0, G1, G2, G3, G4, G5, G6, G7, OUT, N, K: tl.constexpr, BLOCK: tl.constexpr):
    """OUT[i] = the fp32 sum of the first K bf16 gradients at i, added in the order G0, G1, ...; contiguous tensors of
    N elements."""
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    acc = _w(tl.load(G0 + i, mask=m, other=0.0))
    if K > 1:
        acc += _w(tl.load(G1 + i, mask=m, other=0.0))
    if K > 2:
        acc += _w(tl.load(G2 + i, mask=m, other=0.0))
    if K > 3:
        acc += _w(tl.load(G3 + i, mask=m, other=0.0))
    if K > 4:
        acc += _w(tl.load(G4 + i, mask=m, other=0.0))
    if K > 5:
        acc += _w(tl.load(G5 + i, mask=m, other=0.0))
    if K > 6:
        acc += _w(tl.load(G6 + i, mask=m, other=0.0))
    if K > 7:
        acc += _w(tl.load(G7 + i, mask=m, other=0.0))
    tl.store(OUT + i, acc, mask=m)


class SharedOperandFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, uses):
        xb = torch.empty(tuple(x.shape), dtype=torch.bfloat16, device=x.device)
        xb.copy_(x)                                    # one kernel: the rounding and the row-major layout
        ctx.set_materialize_grads(False)
        return tuple(xb.view_as(xb) for _ in range(uses))

    @staticmethod
    def backward(ctx, *gs):
        gs = [g.contiguous() for g in gs if g is not None]
        if not gs:
            return None, None
        if any(g.dtype != torch.bfloat16 for g in gs):
            raise RuntimeError("neuronspark.shared_operand backward: the gradients of the uses must be bf16")
        out = torch.empty(tuple(gs[0].shape), dtype=torch.float32, device=gs[0].device)
        n = out.numel()
        ptrs = gs[::-1] + [gs[0]] * (MAX_USES - len(gs))          # from the last use to the first
        _sum_kernel[(triton.cdiv(n, SUM_BLOCK),)](*ptrs, out, n, K=len(gs), BLOCK=SUM_BLOCK, num_warps=SUM_WARPS)
        return out, None


def shared_operand(x: torch.Tensor, uses: int) -> Tuple[torch.Tensor, ...]:
    """``uses`` views of the contiguous bf16 rounding of the fp32 CUDA tensor ``x`` (1 <= uses <= 8); the gradient of
    ``x`` is the fp32 sum of their gradients (see the module docstring).  Training regime."""
    require_cuda(x, "shared_operand")
    uses = int(uses)
    if x.dtype != torch.float32 or x.numel() == 0 or not 1 <= uses <= MAX_USES:
        raise RuntimeError(f"neuronspark.shared_operand: an fp32 tensor and 1 .. {MAX_USES} uses, got {x.dtype} "
                           f"{tuple(x.shape)}, {uses}")
    return SharedOperandFn.apply(x, uses)


def shared_operand_ref(x: torch.Tensor, uses: int) -> Tuple[torch.Tensor, ...]:
    """The torch form: every use rounds the fp32 tensor itself and returns its gradient widened to fp32; autograd sums
    the fp32 gradients (from the last use to the first when the uses' gradients arrive together)."""
    return tuple(x.to(torch.bfloat16) for _ in range(int(uses)))
