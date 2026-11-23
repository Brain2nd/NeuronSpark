"""Decode-step output norm and gate of the hippocampal layer's attention sub-block (T = 1), in one kernel, one program
per row.

From the ring read ``red`` (B, (H + M) dv) bf16 (the H heads' reads, then the extra queries' reads), the projected
extra-query addend ``xq_add`` (B, dv) bf16 of the last head and the gate logits ``gate`` (B, D) bf16 (D = H dv)::

    a   = f(red[:, :D])                                   the heads' reads widened to fp32
    a   = a + f(xq_add)        on the last head's dv channels (fp32)
    y   = (a rsqrt(mean_c a^2 + eps)) (1 + w)             the zero-centred RMS norm, fp32
    out = bf16(bf16(sigmoid(f(gate))) y)                  the gate (a bf16 sigmoid) times the norm, rounded to bf16

f(.) is the exact widening to fp32; the output is the operand of the output projection.  These are the operations of
the torch expression chain of the step (``attn_out_step_ref``: the cast, the addition, the concatenation and
:func:`neuronspark.ops.triton.hippo_norms.rmsnorm_gate_rows`) at its rounding points; the row sum is formed in another
order.  Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the reference.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....neuronspark.ops.triton import require_cuda
from ....neuronspark.ops.triton._ptx import _add_rn, _div_rn, _mul_rn





__all__ = ["ATTN_OUT_STEP_WARPS", "attn_out_step", "attn_out_step_ref"]

# ================================================================================================ fixed constants
#: one program per row over the whole width (a power-of-two block with a mask), warps
ATTN_OUT_STEP_WARPS = 8


# ================================================================================================ kernel
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
def _attn_out_step_kernel(RED, XQA, G, WN, OUT, s_red, s_xqa, s_g, D, LAST, eps, BD: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    dd = tl.arange(0, BD)
    m = dd < D
    zero = tl.zeros([BD], tl.float32)
    one = tl.full([BD], 1.0, tl.float32)
    a = tl.load(RED + b * s_red + dd, mask=m, other=0.0).to(tl.float32)
    mx = m & (dd >= LAST)                                          # the last head's channels
    a = _add_rn(a, tl.load(XQA + b * s_xqa + (dd - LAST), mask=mx, other=0.0).to(tl.float32))
    r = libdevice.rsqrt(tl.sum(a * a, axis=0) / D + eps)
    y = _mul_rn(_mul_rn(a, zero + r), _add_rn(one, tl.load(WN + dd, mask=m, other=0.0)))
    g = tl.load(G + b * s_g + dd, mask=m, other=0.0).to(tl.float32)
    sg = _bf16(_div_rn(one, _add_rn(one, libdevice.exp(-g))))
    tl.store(OUT + b * D + dd, _mul_rn(sg, y).to(tl.bfloat16), mask=m)


# ================================================================================================ launcher
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.attn_out_step: " + msg)


def _attn_out_step_fwd(red: torch.Tensor, xq_add: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                       eps: float) -> torch.Tensor:
    require_cuda(red, "attn_out_step")
    dev = red.device
    _need(gate.dim() == 2 and gate.dtype == torch.bfloat16 and gate.stride(1) == 1 and gate.device == dev,
          f"the gate logits must be bf16 (B, D) with unit inner stride, got {gate.dtype} {tuple(gate.shape)}")
    B, D = (int(s) for s in gate.shape)
    _need(xq_add.dim() == 2 and int(xq_add.shape[0]) == B and xq_add.dtype == torch.bfloat16
          and xq_add.stride(1) == 1 and xq_add.device == dev,
          f"the extra-query addend must be bf16 ({B}, dv) with unit inner stride, got {xq_add.dtype} "
          f"{tuple(xq_add.shape)}")
    dv = int(xq_add.shape[1])
    _need(0 < dv <= D, f"the addend's width {dv} exceeds the output width {D}")
    _need(red.dim() == 2 and int(red.shape[0]) == B and int(red.shape[1]) >= D and red.dtype == torch.bfloat16
          and red.stride(1) == 1,
          f"the ring read must be bf16 ({B}, >= {D}) with unit inner stride, got {red.dtype} {tuple(red.shape)}")
    _need(tuple(weight.shape) == (D,) and weight.dtype == torch.float32 and weight.is_contiguous()
          and weight.device == dev, f"the norm weight must be a contiguous fp32 ({D},) tensor")
    out = torch.empty((B, D), dtype=torch.bfloat16, device=dev)
    _attn_out_step_kernel[(B,)](red, xq_add, gate, weight, out, int(red.stride(0)), int(xq_add.stride(0)),
                                int(gate.stride(0)), D, D - dv, float(eps), BD=triton.next_power_of_2(D),
                                num_warps=ATTN_OUT_STEP_WARPS, enable_fp_fusion=False)
    return out


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::attn_out_step", mutates_args=())
def _attn_out_step_op(red: torch.Tensor, xq_add: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> torch.Tensor:
    return _attn_out_step_fwd(red, xq_add, gate, weight, eps)


@_attn_out_step_op.register_fake
def _attn_out_step_fake(red, xq_add, gate, weight, eps):
    return gate.new_empty(tuple(gate.shape), dtype=torch.bfloat16)


# ================================================================================================ public function
def attn_out_step(red: torch.Tensor, xq_add: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                  eps: float) -> torch.Tensor:
    """(B, D) bf16: the gated output norm of the attention read of a decode step (module docstring).

    red (B, >= D) bf16: the ring read, its first D = H dv columns the heads' reads; xq_add (B, dv) bf16: the projected
    extra-query reads, added on the last head's channels; gate (B, D) bf16: the gate logits; weight (D,) fp32: the
    zero-centred norm weight; eps: the norm's eps.  Unit inner strides."""
    return _KERNEL_OPS.attn_out_step(red, xq_add, gate, weight, float(eps))


# ================================================================================================ pure-torch reference
@torch.no_grad()
def attn_out_step_ref(red: torch.Tensor, xq_add: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> torch.Tensor:
    """The torch expression chain of the step: the heads' reads cast to fp32, the extra-query addend on the last
    head's channels, the zero-centred RMS norm, the bf16 sigmoid gate, one rounding to bf16.  Same arguments and
    output as :func:`attn_out_step`."""
    D = int(gate.shape[1])
    last = D - int(xq_add.shape[1])
    with torch.autocast("cuda", enabled=False):
        base = red[:, :D].to(torch.float32)
        attn = torch.cat([base[:, :last], base[:, last:] + xq_add], dim=-1)
        y = attn * torch.rsqrt(attn.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + weight.to(torch.float32))
        return (torch.sigmoid(gate).to(torch.float32) * y).to(torch.bfloat16)
