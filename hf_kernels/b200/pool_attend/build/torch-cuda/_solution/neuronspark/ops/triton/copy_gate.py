"""Copy arbitration factors of a prefill call in one kernel.

A hippocampal layer scales its pool read by ``1 + f`` and its PKM read by ``rho``
(:mod:`neuronspark.modules.copy_gate`).  Per position (t, b), all in fp32::

    pi  = exp(32 maximum - logaddexp(32 tau, lse))          the copy confidence (maximum, lse: the copy detector's
                                                            statistics; tau: the layer's sink)
    c   = sigmoid(zw + bias + rv)                           the novelty factor (zw = z . w, rv = rms(r) . v)
    f   = (kappa pi) c                                      the read gain's factor
    rho = 2 sigmoid(-strength pi)                           the PKM copy gate

The torch chain (:func:`copy_gate_prefill_ref`) runs sixteen elementwise kernels on (T, B) tensors.  Here one program
forms COPY_GATE_BLOCK positions with the same fp32 operations in the same order: ``logaddexp`` as ATen's functor
(``max + log1p(exp(-|a - b|))``, the common infinite value returned as it is), ``sigmoid`` as ``1 / (1 + exp(-x))``,
libdevice ``expf`` / ``log1pf``, no FMA contraction.  A position without a copy source (maximum = lse = -inf) gives
pi = 0, f = 0, rho = 1.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....neuronspark.ops.triton import require_cuda

__all__ = ["COPY_GATE_BLOCK", "COPY_GATE_WARPS", "COPY_LOGIT_SCALE", "copy_gate_prefill", "copy_gate_prefill_ref"]

# ================================================================================================ fixed constants
#: positions per program, warps; grid (position blocks,)
COPY_GATE_BLOCK = 1024
COPY_GATE_WARPS = 4
#: logit scale of the copy detector's scores and of the sink (``neuronspark.modules.copy_detector``)
COPY_LOGIT_SCALE = 32.0


# ================================================================================================ kernel
@triton.jit
def _copy_gate_kernel(MX, LSE, ZW, RV, TAU, BIAS, STRENGTH, F_OUT, RHO_OUT, n, kappa, scale, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    mx = tl.load(MX + i, mask=m, other=0.0)
    lse = tl.load(LSE + i, mask=m, other=0.0)
    sink = scale * tl.load(TAU) + tl.zeros([BLOCK], dtype=tl.float32)
    both_inf = (tl.abs(sink) == float("inf")) & (sink == lse)
    la = tl.maximum(sink, lse) + libdevice.log1p(libdevice.exp(-tl.abs(sink - lse)))
    la = tl.where(both_inf, sink, la)
    pi = libdevice.exp(scale * mx - la)
    logit = tl.load(ZW + i, mask=m, other=0.0) + tl.load(BIAS)
    logit = logit + tl.load(RV + i, mask=m, other=0.0)
    c = 1.0 / (1.0 + libdevice.exp(-logit))
    tl.store(F_OUT + i, (kappa * pi) * c, mask=m)
    g = (-tl.load(STRENGTH)) * pi
    tl.store(RHO_OUT + i, 2.0 * (1.0 / (1.0 + libdevice.exp(-g))), mask=m)


# ================================================================================================ launcher
def _check(maximum, lse, zw, rv, tau, bias, strength) -> Tuple[int, int]:
    for name, x in (("maximum", maximum), ("lse", lse), ("zw", zw), ("rv", rv), ("tau", tau), ("bias", bias),
                    ("strength", strength)):
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"neuronspark.copy_gate_prefill: {name} must be a tensor, got {type(x).__name__}")
        require_cuda(x, f"copy_gate_prefill ({name})")
        if x.dtype != torch.float32 or x.device != maximum.device:
            raise RuntimeError(f"neuronspark.copy_gate_prefill: {name} must be fp32 on {maximum.device}, got "
                               f"{x.dtype} on {x.device}")
    if maximum.dim() != 2:
        raise RuntimeError(f"neuronspark.copy_gate_prefill: maximum must be (T, B), got {tuple(maximum.shape)}")
    T, B = (int(s) for s in maximum.shape)
    for name, x in (("maximum", maximum), ("lse", lse), ("zw", zw), ("rv", rv)):
        if tuple(x.shape) != (T, B) or not x.is_contiguous():
            raise RuntimeError(f"neuronspark.copy_gate_prefill: {name} must be a contiguous ({T}, {B}) tensor, got "
                               f"{tuple(x.shape)} with strides {tuple(x.stride())}")
    for name, x in (("tau", tau), ("bias", bias), ("strength", strength)):
        if x.numel() != 1:
            raise RuntimeError(f"neuronspark.copy_gate_prefill: {name} must hold one value, got {tuple(x.shape)}")
    return T, B


def _copy_gate_prefill_fwd(maximum, lse, zw, rv, tau, bias, strength, kappa):
    T, B = _check(maximum, lse, zw, rv, tau, bias, strength)
    f = torch.empty((T, B), dtype=torch.float32, device=maximum.device)
    rho = torch.empty((T, B), dtype=torch.float32, device=maximum.device)
    n = T * B
    if n:
        _copy_gate_kernel[(triton.cdiv(n, COPY_GATE_BLOCK),)](
            maximum, lse, zw, rv, tau, bias, strength, f, rho, n, float(kappa), COPY_LOGIT_SCALE,
            BLOCK=COPY_GATE_BLOCK, num_warps=COPY_GATE_WARPS, enable_fp_fusion=False)
    return f, rho


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::copy_gate_prefill", mutates_args=())
def _copy_gate_prefill_op(maximum: torch.Tensor, lse: torch.Tensor, zw: torch.Tensor, rv: torch.Tensor,
                          tau: torch.Tensor, bias: torch.Tensor, strength: torch.Tensor,
                          kappa: float) -> Tuple[torch.Tensor, torch.Tensor]:
    return _copy_gate_prefill_fwd(maximum, lse, zw, rv, tau, bias, strength, kappa)


@_copy_gate_prefill_op.register_fake
def _copy_gate_prefill_fake(maximum, lse, zw, rv, tau, bias, strength, kappa):
    shape = tuple(maximum.shape)
    return (torch.empty(shape, dtype=torch.float32, device=maximum.device),
            torch.empty(shape, dtype=torch.float32, device=maximum.device))


def copy_gate_prefill(maximum: torch.Tensor, lse: torch.Tensor, zw: torch.Tensor, rv: torch.Tensor,
                      tau: torch.Tensor, bias: torch.Tensor, strength: torch.Tensor,
                      kappa: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(f, rho)``, each (T, B) fp32 (see the module docstring): ``maximum`` / ``lse`` the copy detector's
    statistics, ``zw`` / ``rv`` the novelty predictor's two products, each (T, B) fp32 contiguous; ``tau``, ``bias``,
    ``strength`` one fp32 value each (read on the device); ``kappa`` the read gain.  No autograd."""
    return _KERNEL_OPS.copy_gate_prefill(maximum, lse, zw, rv, tau, bias, strength, float(kappa))


# ================================================================================================ reference
def copy_gate_prefill_ref(maximum: torch.Tensor, lse: torch.Tensor, zw: torch.Tensor, rv: torch.Tensor,
                          tau: torch.Tensor, bias: torch.Tensor, strength: torch.Tensor,
                          kappa: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """The hippocampal layer's torch chain (same arguments and outputs as :func:`copy_gate_prefill`)."""
    sink = torch.ones_like(lse) * (COPY_LOGIT_SCALE * tau.float())
    pi = torch.exp(COPY_LOGIT_SCALE * maximum - torch.logaddexp(sink, lse))
    logit = zw + bias.float().reshape(1)
    logit = logit + rv
    c = torch.sigmoid(logit)
    f = (float(kappa) * pi) * c
    rho = 2.0 * torch.sigmoid(-strength.float() * pi)
    return f, rho
