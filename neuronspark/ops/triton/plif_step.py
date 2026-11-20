"""Decode-step operators of the PLIF populations of the hippocampal layers (T = 1, carried state updated in place).

``plif_step``  one token of a PLIF population in one kernel, one program per row: the token term of the gate (the
    rank-r eta projection of the input, or of its strictly-past centring for the centred neuron), the projected gate,
    one leaky step from the carried membrane, the bipolar release and the soft reset.  Per row b and channel c, with
    ``xw = fp32(x)``, f(.) the exact widening to fp32 and every fp32 operation of the gate chain rounded once (round to
    nearest even, no FMA contraction)::

        xc   = xw                                         the centred neuron: (xw - S / max(n, 1)) [n > 0], S' = S + xw
        t    = bf16(bf16(eta_V) @ bf16(xc))               (r,)   fp32 accumulation, one rounding
        zp   = bf16(bf16(eta_U) @ t)                      (C,)   fp32 accumulation, one rounding
        z    = ((w_eta * xw) + b_eta) + f(zp)
        eta  = 1 / (1 + expf(-z))                         libdevice expf, add.rn, div.rn (the ATen fp32 sigmoid)
        pre  = ((1 - eta) * v) + (eta * xw)               v = the carried membrane
        S    = max(pre - th, 0) - max(-pre - th, 0)       th = v_th (C,)
        v'   = pre - S                                    written over v

    S is stored in ``out_dtype`` (one rounding when it is bf16).  The two projections are the bf16 products of
    :func:`neuronspark.ops.linear.linear` at its rounding points, summed in another order.
``centred_step``  the strictly-past centring of one token over a carried running sum, in one kernel::

        out  = (xw - S / max(n, 1)) [n > 0]               S = run_sum (B, C) fp32, n = count (B,) tokens before this one
        S'   = S + xw                                     written over run_sum

    stored in ``out_dtype`` (bf16 for the consumers of the model: the centred value is the operand of a bf16
    projection, whose input cast is the same rounding).

These are the operations of the torch expression chains they replace (``plif_step_ref`` / ``centred_step_ref``), in
the same order (the sums of the two projections apart); the references are those chains.  Tests: the per-operator
tolerance rule of :mod:`neuronspark.utils.testing` against the references.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.linear import inference_weight, linear_ref
from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (resolved by the kernels below through this module's globals)
    _add_rn,
    _div_rn,
    _mul_rn,
    _sub_rn,
)

__all__ = ["STEP_BLOCK", "STEP_WARPS", "PLIF_STEP_WARPS", "plif_step", "plif_step_ref",
           "plif_gate_step_ref", "centred_step", "centred_step_ref"]

# ================================================================================================ fixed constants
#: centring: elements per program and warps
STEP_BLOCK = 1024
STEP_WARPS = 4
#: PLIF step: one program per row over the whole population (a power-of-two block with a mask), warps.  The eta
#: weights are not in the cache when the step reaches them: the kernel's time is the number of loads a thread issues
#: one after the other, so the two weight tiles are spread over as many threads as a block has
PLIF_STEP_WARPS = 32

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
def _bf16(x):
    """Round an fp32 value to the nearest bf16 (ties to even) and widen it back through the bit pattern (finite
    values)."""
    u = x.to(tl.int32, bitcast=True)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    u = u & -65536                     # 0xFFFF0000 as a signed int32
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _plif_step_kernel(X, EV, EU, W, BE, TH, V, S, SUM, CNT, s_x, s_ev, s_eu, C, R,
                      RB: tl.constexpr, BC: tl.constexpr, CENTRED: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    rr = tl.arange(0, RB)
    rm = rr < R
    c = tl.arange(0, BC)
    m = c < C
    one = tl.full([BC], 1.0, tl.float32)
    zero = tl.zeros([BC], tl.float32)
    # ---- every read of the row, of its state and of the weights (the program holds the whole row: BC >= C)
    xf = tl.load(X + b * s_x + c, mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + b * C + c, mask=m, other=0.0)
    w = tl.load(W + c, mask=m, other=0.0)
    be = tl.load(BE + c, mask=m, other=0.0)
    th = tl.load(TH + c, mask=m, other=0.0)
    if CENTRED:
        n = zero + tl.load(CNT + b).to(tl.float32)
        past = tl.load(SUM + b * C + c, mask=m, other=0.0)
        xc = tl.where(n > zero, _sub_rn(xf, _div_rn(past, tl.maximum(n, one))), zero)
    else:
        xc = xf
    # ---- the token term: t = bf16(eta_V @ bf16(xc)) (RB,), zp = bf16(eta_U @ t) (BC,)
    ev = tl.load(EV + rr[:, None].to(tl.int64) * s_ev + c[None, :], mask=rm[:, None] & m[None, :], other=0.0)
    t = _bf16(tl.sum(_bf16(ev.to(tl.float32)) * _bf16(xc)[None, :], axis=1))
    eu = tl.load(EU + c[:, None].to(tl.int64) * s_eu + rr[None, :], mask=m[:, None] & rm[None, :], other=0.0)
    zp = _bf16(tl.sum(_bf16(eu.to(tl.float32)) * t[None, :], axis=1))
    # ---- the gate chain, the leaky step, the release and the reset
    z2 = _add_rn(_add_rn(_mul_rn(w, xf), be), zp)
    eta = _div_rn(one, _add_rn(one, libdevice.exp(-z2)))
    a = _sub_rn(one, eta)
    d = _mul_rn(eta, xf)
    pre = _add_rn(_mul_rn(a, v), d)
    sp = _sub_rn(tl.maximum(_sub_rn(pre, th), zero), tl.maximum(_sub_rn(-pre, th), zero))
    tl.debug_barrier()                 # every thread has read the carried state before a thread writes it
    tl.store(V + b * C + c, _sub_rn(pre, sp), mask=m)
    tl.store(S + b * C + c, sp.to(S.dtype.element_ty), mask=m)
    if CENTRED:
        tl.store(SUM + b * C + c, _add_rn(past, xf), mask=m)


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


def _plif_step_fwd(x: torch.Tensor, eta_V: torch.Tensor, eta_U: torch.Tensor, w_eta: torch.Tensor,
                   b_eta: torch.Tensor, v_th: torch.Tensor, v: torch.Tensor, run_sum: Optional[torch.Tensor],
                   count: Optional[torch.Tensor], out_dtype: torch.dtype) -> torch.Tensor:
    require_cuda(x, "plif_step")
    _need(x.dim() == 2 and x.dtype in (torch.float32, torch.bfloat16),
          f"x must be fp32 or bf16 (B, C), got {x.dtype} {tuple(x.shape)}")
    if x.stride(1) != 1:
        x = x.contiguous()
    B, C = (int(s) for s in x.shape)
    dev = x.device
    _need(eta_V.dim() == 2 and int(eta_V.shape[1]) == C and eta_V.dtype in (torch.float32, torch.bfloat16)
          and eta_V.stride(1) == 1 and eta_V.device == dev,
          f"eta_V must be an fp32 / bf16 (rank, {C}) tensor with unit inner stride, got {eta_V.dtype} "
          f"{tuple(eta_V.shape)}")
    R = int(eta_V.shape[0])
    _need(tuple(eta_U.shape) == (C, R) and eta_U.dtype in (torch.float32, torch.bfloat16) and eta_U.stride(1) == 1
          and eta_U.device == dev,
          f"eta_U must be an fp32 / bf16 ({C}, {R}) tensor with unit inner stride, got {eta_U.dtype} "
          f"{tuple(eta_U.shape)}")
    for name, t in (("w_eta", w_eta), ("b_eta", b_eta), ("v_th", v_th)):
        _need(t.dtype == torch.float32 and tuple(t.shape) == (C,) and t.device == dev and t.is_contiguous(),
              f"{name} must be a contiguous fp32 ({C},) tensor, got {t.dtype} {tuple(t.shape)}")
    _need(v.dtype == torch.float32 and tuple(v.shape) == (B, C) and v.is_contiguous() and v.device == dev,
          f"the membrane v must be a contiguous fp32 ({B}, {C}) tensor, got {v.dtype} {tuple(v.shape)}")
    centred = run_sum is not None
    if centred:
        _need(run_sum.dtype == torch.float32 and tuple(run_sum.shape) == (B, C) and run_sum.is_contiguous()
              and run_sum.device == dev,
              f"the carried sum must be a contiguous fp32 ({B}, {C}) tensor, got {run_sum.dtype} "
              f"{tuple(run_sum.shape)}")
        _need(count is not None and count.dtype == torch.int64 and tuple(count.shape) == (B,)
              and count.is_contiguous() and count.device == dev,
              f"the centred neuron needs the count as a contiguous ({B},) int64 tensor")
    else:
        _need(count is None, "a count without a carried sum")
    _need(0 < B * C < 2 ** 31, "the element count must be positive and int32-addressable")
    S = torch.empty((B, C), dtype=out_dtype, device=dev)
    _plif_step_kernel[(B,)](x, eta_V, eta_U, w_eta, b_eta, v_th, v, S, run_sum if centred else v,
                            count if centred else v, int(x.stride(0)), int(eta_V.stride(0)), int(eta_U.stride(0)),
                            C, R, RB=triton.next_power_of_2(R), BC=triton.next_power_of_2(C), CENTRED=int(centred),
                            num_warps=PLIF_STEP_WARPS, enable_fp_fusion=False)
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


@torch.library.custom_op("neuronspark::plif_step", mutates_args=("v", "run_sum"))
def _plif_step_op(x: torch.Tensor, eta_V: torch.Tensor, eta_U: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor,
                  v_th: torch.Tensor, v: torch.Tensor, out_dtype: str, run_sum: Optional[torch.Tensor],
                  count: Optional[torch.Tensor]) -> torch.Tensor:
    return _plif_step_fwd(x, eta_V, eta_U, w_eta, b_eta, v_th, v, run_sum, count, _DTYPES[out_dtype])


@_plif_step_op.register_fake
def _plif_step_fake(x, eta_V, eta_U, w_eta, b_eta, v_th, v, out_dtype, run_sum, count):
    return x.new_empty(tuple(x.shape), dtype=_DTYPES[out_dtype])


@torch.library.custom_op("neuronspark::centred_step", mutates_args=("run_sum",))
def _centred_step_op(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor, out_dtype: str) -> torch.Tensor:
    return _centred_step_fwd(x, run_sum, count, _DTYPES[out_dtype])


@_centred_step_op.register_fake
def _centred_step_fake(x, run_sum, count, out_dtype):
    return x.new_empty(tuple(x.shape), dtype=_DTYPES[out_dtype])


# ================================================================================================ public functions
def plif_step(x: torch.Tensor, eta_V: torch.Tensor, eta_U: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor,
              v_th: torch.Tensor, v: torch.Tensor, out_dtype: torch.dtype, run_sum: Optional[torch.Tensor] = None,
              count: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Spikes S (B, C) in ``out_dtype`` of one token: ``x`` (B, C) fp32 / bf16, ``eta_V`` (rank, C) / ``eta_U``
    (C, rank): the eta projection's weights, fp32 (the kernel rounds them to bf16) or their bf16 roundings (the
    inference copies of :func:`neuronspark.ops.linear.cache_bf16_`), ``w_eta`` / ``b_eta`` / ``v_th`` fp32 (C,),
    ``v`` (B, C) fp32 contiguous: the carried membrane, overwritten with the membrane after this token.  The centred
    neuron: ``run_sum`` (B, C) fp32 contiguous, the sum of the row's earlier inputs (advanced in place by ``x``), and
    ``count`` (B,) int64, the tokens before this one; the projection then reads the input minus the mean of the
    earlier ones.  ``v`` and ``run_sum`` share no memory with another argument."""
    return torch.ops.neuronspark.plif_step(x, inference_weight(eta_V), inference_weight(eta_U), w_eta, b_eta, v_th, v,
                                           _dtype_name(out_dtype), run_sum, count)


def centred_step(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """(B, C) in ``out_dtype``: ``x`` (B, C) minus the mean of the row's earlier inputs (zero at the row start);
    ``run_sum`` (B, C) fp32 contiguous, the sum of the earlier inputs, advanced in place by ``x``; ``count`` (B,) int64,
    the tokens before this one."""
    return torch.ops.neuronspark.centred_step(x, run_sum, count, _dtype_name(out_dtype))


# ================================================================================================ pure-torch references
@torch.no_grad()
def plif_gate_step_ref(x: torch.Tensor, zp: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor, v_th: torch.Tensor,
                       v: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """The step from a given token term ``zp`` (B, C): the eager gate chain, one leaky step, the bipolar release and
    the soft reset; ``v`` is overwritten."""
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
def plif_step_ref(x: torch.Tensor, eta_V: torch.Tensor, eta_U: torch.Tensor, w_eta: torch.Tensor, b_eta: torch.Tensor,
                  v_th: torch.Tensor, v: torch.Tensor, out_dtype: torch.dtype,
                  run_sum: Optional[torch.Tensor] = None, count: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pure-torch reference of :func:`plif_step`: the centring (centred neuron), the two eta projections (the linear
    operator's reference) and :func:`plif_gate_step_ref`; ``v`` (and ``run_sum``) overwritten."""
    xc = x if run_sum is None else centred_step_ref(x, run_sum, count, torch.bfloat16)
    zp = linear_ref(linear_ref(xc, eta_V), eta_U)
    return plif_gate_step_ref(x, zp, w_eta, b_eta, v_th, v, out_dtype)


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
