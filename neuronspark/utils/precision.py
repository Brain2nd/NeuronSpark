"""Numerics regimes: the one place where the floating-point settings of a forward are decided.

The model has two numerics regimes with different process settings; each regime's code path is written for its own:

=============================================  ===================  ====================================
setting                                        ``regime("train")``  ``regime("infer")``
=============================================  ===================  ====================================
autocast                                       CUDA, bf16           CUDA, bf16
``torch.backends.cuda.matmul.allow_tf32``      True                 False
``torch.backends.cudnn.allow_tf32``            True                 unchanged
``matmul.allow_bf16_reduced_precision_red..``  unchanged            False
=============================================  ===================  ====================================

``regime("train")`` is the whole-row regime (training, and teacher-forced scoring without an inference cache): the fp32
matmuls of the hippocampal layers (PKM half-query scores, copy-detector cosines) run in TF32.  ``regime("infer")`` is
prefill and decode with an inference cache: those matmuls run in IEEE fp32 and bf16 GEMMs accumulate without
reduced-precision reductions.

The contexts save the previous values on entry and restore them on exit (nesting restores the outer regime).  Every
setting is host-side state, so entering a regime inside CUDA-graph capture or around a replayed graph is safe.

Backward.  The regime is a forward context: a training step calls ``loss.backward()`` after the context has exited,
and the autograd engine reads the process-wide TF32 setting when each backward GEMM runs.  The whole-row regime's
contract is forward and backward under one setting (TF32 on), so an fp32 product of the whole-row regime whose
gradient GEMMs are TF32-eligible (fp32 operands: the PKM half-query scores, the novelty predictor's matvecs) is formed
by :func:`regime_product`: its backward runs under the ``matmul.allow_tf32`` value captured at its forward, whatever
the setting is when ``backward()`` is called.  The bf16-operand GEMMs (every projection, the pool attention) do not
depend on the TF32 setting.
"""
from __future__ import annotations

import contextlib
from typing import Callable, Iterator

import torch

__all__ = ["REGIMES", "regime", "regime_product", "check_fp32_parameters", "current_regime_flags"]

REGIMES = ("train", "infer")


def current_regime_flags() -> dict:
    """The process settings a regime touches (for reports)."""
    return {
        "autocast_cuda": bool(torch.is_autocast_enabled("cuda")),
        "autocast_cuda_dtype": str(torch.get_autocast_dtype("cuda")),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_bf16_reduced_precision_reduction":
            bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
    }


@contextlib.contextmanager
def regime(name: str) -> Iterator[None]:
    """Enter the ``"train"`` or ``"infer"`` regime (see the module docstring)."""
    if name not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {name!r}")
    mm = torch.backends.cuda.matmul
    saved_tf32 = mm.allow_tf32
    saved_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    saved_rr = mm.allow_bf16_reduced_precision_reduction
    try:
        if name == "train":
            mm.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            mm.allow_tf32 = False
            mm.allow_bf16_reduced_precision_reduction = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    finally:
        mm.allow_tf32 = saved_tf32
        torch.backends.cudnn.allow_tf32 = saved_cudnn_tf32
        mm.allow_bf16_reduced_precision_reduction = saved_rr


class _RegimeProductFn(torch.autograd.Function):
    """``fn(a, b)`` whose backward runs under the ``matmul.allow_tf32`` value of its forward.

    Forward: ``fn(a, b)`` itself (the same kernels; grad mode is off inside, which does not change them).  Backward:
    ``fn`` is evaluated again on detached copies of the saved operands with the captured TF32 setting and the forward's
    autocast state, and ``torch.autograd.grad`` of that product gives the operand gradients: the backward kernels of
    ``fn``'s own autograd graph, under the forward's settings.  First-order gradients only."""

    @staticmethod
    def forward(ctx, fn, a, b):
        dev = a.device.type
        ctx.fn = fn
        ctx.allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        ctx.autocast = (dev, bool(torch.is_autocast_enabled(dev)), torch.get_autocast_dtype(dev))
        ctx.save_for_backward(a, b)
        return fn(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        need_a, need_b = bool(ctx.needs_input_grad[1]), bool(ctx.needs_input_grad[2])
        if not (need_a or need_b):
            return None, None, None
        mm = torch.backends.cuda.matmul
        saved = mm.allow_tf32
        dev, ac_on, ac_dtype = ctx.autocast
        try:
            mm.allow_tf32 = ctx.allow_tf32
            with torch.enable_grad(), torch.autocast(dev, dtype=ac_dtype, enabled=ac_on):
                a_ = a.detach().requires_grad_(need_a)
                b_ = b.detach().requires_grad_(need_b)
                out = ctx.fn(a_, b_)
                wrt = [t for t, need in ((a_, need_a), (b_, need_b)) if need]
                grads = iter(torch.autograd.grad(out, wrt, g))
        finally:
            mm.allow_tf32 = saved
        ga = next(grads) if need_a else None
        gb = next(grads) if need_b else None
        return None, ga, gb


def regime_product(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], a: torch.Tensor,
                   b: torch.Tensor) -> torch.Tensor:
    """``fn(a, b)`` for an fp32 product (an einsum / matmul of two fp32 operands) of the whole-row regime.  With
    autograd its backward runs under the TF32 setting current at this call (the regime's), not the one current when
    ``backward()`` runs (see the module docstring); without autograd it is ``fn(a, b)``."""
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return _RegimeProductFn.apply(fn, a, b)
    return fn(a, b)


def check_fp32_parameters(module: torch.nn.Module) -> None:
    """Raise unless every floating parameter of ``module`` is fp32 (the model keeps fp32 master weights; bf16 copies
    exist only inside the ops and in the explicit inference weight cache)."""
    bad = [(n, p.dtype) for n, p in module.named_parameters() if p.is_floating_point() and p.dtype != torch.float32]
    if bad:
        shown = ", ".join(f"{n}: {d}" for n, d in bad[:8])
        more = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
        raise RuntimeError(f"neuronspark: parameters must be fp32, found {shown}{more}")
