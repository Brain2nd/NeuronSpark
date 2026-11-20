"""Decode-step operators of the PLIF populations of the hippocampal layers (T = 1, carried state updated in place).

``plif_step``  one token of a PLIF population in one kernel: the projected gate, one leaky step from the carried
    membrane, the bipolar release and the soft reset.  Per row b and channel c, with ``xw = fp32(x)`` and every fp32
    operation rounded once (round to nearest even, no FMA contraction)::

        z    = ((w_eta * xw) + b_eta) + fp32(zp)        zp = the eta projection (B, C) bf16, formed by the caller
        eta  = 1 / (1 + expf(-z))                         libdevice expf, add.rn, div.rn (the ATen fp32 sigmoid)
        pre  = ((1 - eta) * v) + (eta * xw)               v = the carried membrane
        S    = max(pre - th, 0) - max(-pre - th, 0)       th = v_th (C,)
        v'   = pre - S                                    written over v

    S is stored in ``out_dtype`` (one rounding when it is bf16).
``centred_step``  the strictly-past centring of one token over a carried running sum, in one kernel::

        out  = (xw - S / max(n, 1)) [n > 0]               S = run_sum (B, C) fp32, n = count (B,) tokens before this one
        S'   = S + xw                                     written over run_sum

    stored in ``out_dtype`` (bf16 for the consumers of the model: the centred value is the operand of a bf16
    projection, whose input cast is the same rounding).

These are the operations of the torch expression chains they replace (``plif_step_ref`` / ``centred_step_ref``), in
the same order; the references are those chains.  Tests: the per-operator tolerance rule of
:mod:`neuronspark.utils.testing` against the references.
"""
from __future__ import annotations

from typing import Dict

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (resolved by the kernels below through this module's globals)
    _add_rn,
    _div_rn,
    _mul_rn,
    _sub_rn,
)

__all__ = ["STEP_BLOCK", "STEP_WARPS", "plif_step", "plif_step_ref", "centred_step", "centred_step_ref"]

# ================================================================================================ fixed constants
#: elements per program and warps of both kernels (a population of 1536 channels at batch 1 is two programs)
STEP_BLOCK = 1024
STEP_WARPS = 4

_DTYPES: Dict[str, torch.dtype] = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.plif_step: " + msg)


def _dtype_name(dt: torch.dtype) -> str:
    for name, d in _DTYPES.items():
        if d == dt:
            return name
    raise RuntimeError(f"neuronspark.ops.plif_step: the output is stored in float32 or bfloat16, got {dt}")


# ================================================================================================ kernels
@triton.jit
def _plif_step_kernel(X, Z, W, BE, TH, V, S, N, C, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    c = offs % C
    xf = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    z = tl.load(Z + offs, mask=m, other=0.0).to(tl.float32)
    w = tl.load(W + c, mask=m, other=0.0)
    b = tl.load(BE + c, mask=m, other=0.0)
    th = tl.load(TH + c, mask=m, other=0.0)
    v = tl.load(V + offs, mask=m, other=0.0)
    one = tl.full(xf.shape, 1.0, tl.float32)
    zero = tl.zeros(xf.shape, tl.float32)
    z2 = _add_rn(_add_rn(_mul_rn(w, xf), b), z)
    eta = _div_rn(one, _add_rn(one, libdevice.exp(-z2)))
    a = _sub_rn(one, eta)
    d = _mul_rn(eta, xf)
    pre = _add_rn(_mul_rn(a, v), d)
    s = _sub_rn(tl.maximum(_sub_rn(pre, th), zero), tl.maximum(_sub_rn(-pre, th), zero))
    tl.store(V + offs, _sub_rn(pre, s), mask=m)
    tl.store(S + offs, s.to(S.dtype.element_ty), mask=m)


@triton.jit
def _centred_step_kernel(X, SUM, CNT, OUT, N, C, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    xf = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
    s = tl.load(SUM + offs, mask=m, other=0.0)
    n = tl.load(CNT + offs // C, mask=m, other=0).to(tl.float32)
    one = tl.full(xf.shape, 1.0, tl.float32)
    zero = tl.zeros(xf.shape, tl.float32)
    out = tl.where(n > zero, _sub_rn(xf, _div_rn(s, tl.maximum(n, one))), zero)
    tl.store(SUM + offs, _add_rn(s, xf), mask=m)
    tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=m)


# ================================================================================================ launchers
def _rows(name: str, t: torch.Tensor) -> torch.Tensor:
    _need(t.dim() == 2 and t.dtype in (torch.float32, torch.bfloat16),
          f"{name} must be fp32 or bf16 (B, C), got {t.dtype} {tuple(t.shape)}")
    return t if t.is_contiguous() else t.contiguous()


def _plif_step_fwd(x: torch.Tensor, zp: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor, v_th: torch.Tensor,
                   v: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    require_cuda(x, "plif_step")
    xc = _rows("x", x)
    B, C = (int(s) for s in xc.shape)
    _need(tuple(zp.shape) == (B, C), f"the projection zp must be ({B}, {C}), got {tuple(zp.shape)}")
    zc = _rows("zp", zp)
    for name, t in (("w_eta", w_eta), ("b_eta", b_eta), ("v_th", v_th)):
        _need(t.dtype == torch.float32 and tuple(t.shape) == (C,) and t.device == x.device and t.is_contiguous(),
              f"{name} must be a contiguous fp32 ({C},) tensor, got {t.dtype} {tuple(t.shape)}")
    _need(v.dtype == torch.float32 and tuple(v.shape) == (B, C) and v.is_contiguous() and v.device == x.device,
          f"the membrane v must be a contiguous fp32 ({B}, {C}) tensor, got {v.dtype} {tuple(v.shape)}")
    N = B * C
    _need(0 < N < 2 ** 31, "the element count must be positive and int32-addressable")
    S = torch.empty((B, C), dtype=out_dtype, device=x.device)
    _plif_step_kernel[(triton.cdiv(N, STEP_BLOCK),)](xc, zc, w_eta, b_eta, v_th, v, S, N, C, BLOCK=STEP_BLOCK,
                                                     num_warps=STEP_WARPS, enable_fp_fusion=False)
    return S


def _centred_step_fwd(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor,
                      out_dtype: torch.dtype) -> torch.Tensor:
    require_cuda(x, "centred_step")
    xc = _rows("x", x)
    B, C = (int(s) for s in xc.shape)
    _need(run_sum.dtype == torch.float32 and tuple(run_sum.shape) == (B, C) and run_sum.is_contiguous()
          and run_sum.device == x.device,
          f"the carried sum must be a contiguous fp32 ({B}, {C}) tensor, got {run_sum.dtype} {tuple(run_sum.shape)}")
    _need(count.dtype == torch.int64 and tuple(count.shape) == (B,) and count.is_contiguous()
          and count.device == x.device, f"the count must be a contiguous ({B},) int64 tensor")
    N = B * C
    _need(0 < N < 2 ** 31, "the element count must be positive and int32-addressable")
    out = torch.empty((B, C), dtype=out_dtype, device=x.device)
    _centred_step_kernel[(triton.cdiv(N, STEP_BLOCK),)](xc, run_sum, count, out, N, C, BLOCK=STEP_BLOCK,
                                                        num_warps=STEP_WARPS, enable_fp_fusion=False)
    return out


@torch.library.custom_op("neuronspark::plif_step", mutates_args=("v",))
def _plif_step_op(x: torch.Tensor, zp: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor, v_th: torch.Tensor,
                  v: torch.Tensor, out_dtype: str) -> torch.Tensor:
    return _plif_step_fwd(x, zp, w_eta, b_eta, v_th, v, _DTYPES[out_dtype])


@_plif_step_op.register_fake
def _plif_step_fake(x, zp, w_eta, b_eta, v_th, v, out_dtype):
    return x.new_empty(tuple(x.shape), dtype=_DTYPES[out_dtype])


@torch.library.custom_op("neuronspark::centred_step", mutates_args=("run_sum",))
def _centred_step_op(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor, out_dtype: str) -> torch.Tensor:
    return _centred_step_fwd(x, run_sum, count, _DTYPES[out_dtype])


@_centred_step_op.register_fake
def _centred_step_fake(x, run_sum, count, out_dtype):
    return x.new_empty(tuple(x.shape), dtype=_DTYPES[out_dtype])


# ================================================================================================ public functions
def plif_step(x: torch.Tensor, zp: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor, v_th: torch.Tensor,
              v: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Spikes S (B, C) in ``out_dtype`` of one token: ``x`` (B, C) fp32 / bf16, ``zp`` (B, C) the eta projection (bf16,
    or its fp32 widening), ``w_eta`` / ``b_eta`` / ``v_th`` fp32 (C,), ``v`` (B, C) fp32 contiguous: the carried
    membrane, overwritten with the membrane after this token."""
    return torch.ops.neuronspark.plif_step(x, zp, w_eta, b_eta, v_th, v, _dtype_name(out_dtype))


def centred_step(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """(B, C) in ``out_dtype``: ``x`` (B, C) minus the mean of the row's earlier inputs (zero at the row start);
    ``run_sum`` (B, C) fp32 contiguous, the sum of the earlier inputs, advanced in place by ``x``; ``count`` (B,) int64,
    the tokens before this one."""
    return torch.ops.neuronspark.centred_step(x, run_sum, count, _dtype_name(out_dtype))


# ================================================================================================ pure-torch references
@torch.no_grad()
def plif_step_ref(x: torch.Tensor, zp: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor, v_th: torch.Tensor,
                  v: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Pure-torch reference of :func:`plif_step`: the eager gate chain, one leaky step, the bipolar release and the
    soft reset; ``v`` is overwritten."""
    xw = x.to(torch.float32)
    z_eta = w_eta.to(torch.float32) * xw + b_eta.to(torch.float32)
    z_eta = z_eta + zp.to(torch.float32)
    eta = torch.sigmoid(z_eta)
    a_eta = 1.0 - eta
    drive = eta * xw
    th = v_th.to(torch.float32)
    pre = a_eta * v.to(torch.float32) + drive
    s = torch.clamp_min(pre - th, 0.0)
    s = s - torch.clamp_min(-pre - th, 0.0)
    v.copy_(pre - s)
    return s.to(out_dtype)


@torch.no_grad()
def centred_step_ref(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor,
                     out_dtype: torch.dtype) -> torch.Tensor:
    """Pure-torch reference of :func:`centred_step`: c = (x - S / max(n, 1)) [n > 0] in fp32, cast to ``out_dtype``;
    ``run_sum`` advanced in place by x (the whole row's cumulative sum's next addition, in its order)."""
    xf = x.to(torch.float32)
    n = count.view(-1, 1)
    xn = n.to(torch.float32).clamp_min(1.0)
    has_past = (n > 0).to(torch.float32)
    out = (xf - run_sum / xn) * has_past
    run_sum.add_(xf)
    return out.to(out_dtype)
