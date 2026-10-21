"""Bias-free linear op with an fp32 weight gradient (every projection of the model, the tied readout included).

Numerics contract (training, whole-row regime):

* forward: ``y = F.linear(x.to(bf16), weight.to(bf16))``: bf16 RNE operands, fp32 accumulation in cuBLAS, bf16
  output.  These are the operands bf16 autocast gives ``F.linear`` (an fp32 operand is cast with ``.to(bf16)``, a bf16
  operand passes through), i.e. the GEMM of the model's autocast forward; the two casts are made once and the same
  tensors are saved for backward.  The op does not depend on the autocast state.
* weight gradient: ``dW = torch.mm(dy2.t(), x2, out_dtype=torch.float32)`` with ``dy2`` / ``x2`` the row-major
  ``(rows, N)`` / ``(rows, K)`` bf16 views of ``dy`` / the saved input: one GEMM with bf16 operands, fp32 accumulation
  and an fp32 result.  It is returned as the weight's gradient (fp32 ``.grad`` through AccumulateGrad); nothing is
  rounded to bf16.  A weight used twice in one backward (the tied embedding: readout + lookup) receives the fp32
  sum of its contributions from autograd.
* input gradient: ``dx = (dy2 @ weight_bf16)`` (bf16 GEMM output) widened to ``x.dtype``.
* an output-feature count N that is not a multiple of 8 (the odd vocabulary of the tied readout, 151,669) runs the three
  GEMMs on a padded layout: P = N rounded up to a multiple of 128 (151,680), the bf16 weight
  ``wp = [weight.to(bf16); 0]`` (P, K) with zero pad rows; forward ``F.linear(xb, wp)`` (..., P) whose pad columns are
  exactly 0, returned as the (..., N) view (row stride P); backward on the (rows, P) gradient whose pad columns are 0
  (the slice's backward writes them): ``dW = torch.mm(dyp[:, :N].t(), x2, out_dtype=float32)`` (the stride-P view, no
  copy) and ``dx = dyp @ wp`` (K = P).  The real columns are the same dot products of the same bf16 operands; the
  aligned extent keeps cuBLAS off the unaligned fallback kernel, which on an odd leading dimension runs split-K with
  bf16 partial sums (several bf16 roundings per output instead of one).  The weight gradient is one beta-0 GEMM over
  all rows.

Inference (no autograd needed): ``F.linear(x.to(bf16), w)`` where ``w`` is the bf16 copy made by :func:`cache_bf16_`
when one is present and current, else ``weight.to(bf16)``; the two hold the same values while the cache is current.  A
cache whose weight was modified after caching (a version bump), moved or re-typed raises: call
``prepare_inference()`` again.  The input's view and strides are kept (no copy), as in the autocast call.  A product of
2 to 32 rows and fewer than 16 output features (the decode step's per-head projections) runs
:func:`neuronspark.ops.triton.linear_small_n.linear_small_n` instead of cuBLAS's split-K GEMM and reduction: one kernel,
fp32 accumulation, one bf16 rounding (the same reference).

Pure-torch reference :func:`linear_ref`: the same bf16-rounded operands multiplied in float64, the output rounded
once to bf16; its gradients are float64 products delivered in the inputs' dtypes.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "Fp32WgradLinearFn",
    "linear",
    "linear_ref",
    "cache_bf16_",
    "drop_bf16_",
    "has_bf16_cache",
    "padded_extent",
]

_CACHE_ATTR = "_ns_bf16"
_VERSION_ATTR = "_ns_bf16_version"
# training GEMMs of an output-feature count that is not a multiple of ALIGN run on the extent rounded up to PAD_MULTIPLE
# (the tied readout's vocabulary: 151,669 -> 151,680)
ALIGN = 8
PAD_MULTIPLE = 128


def padded_extent(n: int) -> int:
    """Output-feature extent of the training GEMMs: ``n`` when it is a multiple of 8, else ``n`` rounded up to a multiple
    of 128."""
    n = int(n)
    if n % ALIGN == 0:
        return n
    return (n + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE


# ------------------------------------------------------------------------------------------------ checks
def _check(x: torch.Tensor, weight: torch.Tensor) -> None:
    if weight.dim() != 2:
        raise RuntimeError(f"neuronspark.ops.linear: weight must be 2-D (N, K), got shape {tuple(weight.shape)}")
    if x.dim() < 1 or x.shape[-1] != weight.shape[1]:
        raise RuntimeError(f"neuronspark.ops.linear: x {tuple(x.shape)} does not match weight {tuple(weight.shape)}")
    if not x.is_floating_point() or x.dtype == torch.float64:
        raise RuntimeError(f"neuronspark.ops.linear: x must be fp32, bf16 or fp16, got {x.dtype}")
    if weight.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"neuronspark.ops.linear: weight must be fp32 (or bf16 in inference), got {weight.dtype}")
    if x.device != weight.device:
        raise RuntimeError(f"neuronspark.ops.linear: x on {x.device}, weight on {weight.device}")


def _cached_bf16(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """The current bf16 cache of ``weight`` or None; raises on a stale cache."""
    w = getattr(weight, _CACHE_ATTR, None)
    if w is None:
        return None
    if getattr(weight, _VERSION_ATTR, None) != weight._version:
        raise RuntimeError("neuronspark.ops.linear: the bf16 inference copy of a weight is stale (the weight was "
                           "modified after prepare_inference()); call prepare_inference() again")
    if w.device != weight.device or w.shape != weight.shape or w.dtype != torch.bfloat16:
        raise RuntimeError("neuronspark.ops.linear: the bf16 inference copy of a weight no longer matches it "
                           f"(copy {tuple(w.shape)} {w.dtype} on {w.device}, weight {tuple(weight.shape)} on "
                           f"{weight.device}); call prepare_inference() again")
    return w


# ------------------------------------------------------------------------------------------------ bf16 cache
@torch.no_grad()
def cache_bf16_(param: torch.Tensor) -> torch.Tensor:
    """Attach (or refresh) the bf16 inference copy of ``param``: ``param.detach().to(bf16)``, contiguous, with the
    parameter's version recorded.  Returns the copy."""
    if param.dim() != 2 or param.dtype not in (torch.float32, torch.bfloat16):
        raise RuntimeError(f"cache_bf16_: expected a 2-D fp32/bf16 weight, got {tuple(param.shape)} {param.dtype}")
    w = param.detach().to(torch.bfloat16).contiguous()
    setattr(param, _CACHE_ATTR, w)
    setattr(param, _VERSION_ATTR, param._version)
    return w


def drop_bf16_(param: torch.Tensor) -> None:
    """Remove the bf16 inference copy of ``param`` (no-op without one)."""
    for name in (_CACHE_ATTR, _VERSION_ATTR):
        if hasattr(param, name):
            delattr(param, name)


def has_bf16_cache(param: torch.Tensor) -> bool:
    return getattr(param, _CACHE_ATTR, None) is not None


# ------------------------------------------------------------------------------------------------ autograd Function
class Fp32WgradLinearFn(torch.autograd.Function):
    """``y = x @ W^T`` with bf16 operands, bf16 output, fp32 weight gradient, input gradient in ``x.dtype``.

    The output has ``padded_extent(N)`` features: for an N that is not a multiple of 8 the pad columns are exactly 0 and
    :func:`linear` returns the (..., N) view; the gradient then reaches backward as the (..., P) tensor of the slice's
    backward (pad columns 0)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xb = x.to(torch.bfloat16)
        n, k = int(weight.shape[0]), int(weight.shape[1])
        p = padded_extent(n)
        if p == n:
            wb = weight.to(torch.bfloat16)
        else:
            wb = torch.empty((p, k), dtype=torch.bfloat16, device=weight.device)
            wb[:n].copy_(weight)                     # RNE fp32 -> bf16: the bits of weight.to(bf16)
            wb[n:].zero_()
        out = F.linear(xb, wb)
        if out.dtype != torch.bfloat16:
            raise RuntimeError(f"neuronspark.ops.linear: the GEMM produced {out.dtype}, expected bf16 (an fp16 "
                               "autocast region is not supported)")
        ctx.input_dtype = x.dtype
        ctx.input_shape = x.shape
        ctx.out_features = n
        ctx.save_for_backward(xb, wb)
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        xb, wb = ctx.saved_tensors
        n = ctx.out_features
        dx = dw = None
        with torch.autocast(xb.device.type, enabled=False):
            dy2 = dy.reshape(-1, dy.shape[-1]).contiguous()
            if dy2.dtype != torch.bfloat16:
                raise RuntimeError(f"neuronspark.ops.linear: incoming gradient is {dy2.dtype}, expected bf16")
            if int(dy2.shape[1]) != int(wb.shape[0]):
                raise RuntimeError(f"neuronspark.ops.linear: incoming gradient has {int(dy2.shape[1])} features, "
                                   f"expected {int(wb.shape[0])}")
            if ctx.needs_input_grad[1]:
                x2 = xb.reshape(-1, xb.shape[-1]).contiguous()
                dw = torch.mm(dy2[:, :n].t(), x2, out_dtype=torch.float32)
            if ctx.needs_input_grad[0]:
                dx = dy2.matmul(wb).reshape(ctx.input_shape).to(ctx.input_dtype)
        return dx, dw


# ------------------------------------------------------------------------------------------------ public function
def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x (..., K)`` fp32 / bf16, ``weight (N, K)`` fp32 -> ``(..., N)`` bf16.

    With autograd (grad mode on and ``x`` or ``weight`` requiring grad) the :class:`Fp32WgradLinearFn` path runs and the
    weight must be fp32; for an N that is not a multiple of 8 the result is the (..., N) view of the padded (..., P)
    output (row stride P, see the module docstring).  Otherwise the inference path runs: the current bf16 cache of the
    weight (see :func:`cache_bf16_`) or ``weight.to(bf16)``.
    """
    _check(x, weight)
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        if weight.dtype != torch.float32:
            raise RuntimeError(f"neuronspark.ops.linear: a trained weight must be fp32, got {weight.dtype}")
        n = int(weight.shape[0])
        out = Fp32WgradLinearFn.apply(x, weight)
        # the slice is taken outside the Function: an in-place update of the (..., N) view (the readout balance) is
        # then an ordinary view update for autograd, and the slice's backward hands the Function the (..., P)
        # gradient with zero pad columns
        return out if int(out.shape[-1]) == n else out[..., :n]
    w = _cached_bf16(weight)
    if w is None:
        w = weight.to(torch.bfloat16)
    xb = x.to(torch.bfloat16)
    if int(w.shape[0]) < 16 and xb.is_cuda and xb.dim() >= 2:
        from neuronspark.ops.triton.linear_small_n import linear_small_n, takes
        x2 = xb.reshape(-1, int(xb.shape[-1]))
        if takes(x2, w):
            return linear_small_n(x2, w).view(*xb.shape[:-1], int(w.shape[0]))
    return F.linear(xb, w)


def linear_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Pure-torch reference of :func:`linear`: the bf16-rounded operands multiplied in float64, the output rounded to
    bf16 once.  Differentiable: the gradients are float64 products of the bf16-rounded operands (straight-through
    rounding), delivered as ``x.dtype`` / ``weight.dtype`` by autograd."""
    _check(x, weight)
    x64 = x.double()
    w64 = weight.double()
    xr = x64 + (x.detach().to(torch.bfloat16).double() - x64.detach())
    wr = w64 + (weight.detach().to(torch.bfloat16).double() - w64.detach())
    return torch.matmul(xr, wr.t()).to(torch.bfloat16)
