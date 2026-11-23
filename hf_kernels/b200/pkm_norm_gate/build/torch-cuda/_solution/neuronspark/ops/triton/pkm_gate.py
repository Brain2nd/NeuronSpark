"""Norm and gate of the PKM read (training; the prefill calls it without autograd): one forward kernel and one
backward kernel.

The value read ``val`` (T, B, dv) fp32 of the product-key memory (:mod:`neuronspark.modules.pkm`) is normalised with a
zero-centred gain and gated per channel before the output projection::

    r = bf16( val * rsqrt(mean(val^2) + eps) * (1 + gain) * sigmoid(f32(glog)) )

``glog`` (T, B, dv) bf16 are the gate logits (a projection of the FFN input); the product is fp32 and its one rounding
to bf16 is the output projection's operand rounding.

Operator ``pkm_norm_gate(val, gain, glog, eps)`` (:class:`PKMNormGateFn`):

* forward: one program per (t, b) row; the row factor ``rsqrt(mean(val^2) + eps)`` is kept (T, B);
* backward from the bf16 gradient ``g`` of r, with ``gf = f32(g)``, ``n = val rr``, ``s = sigmoid(f32(glog))``: one
  kernel writes ``dglog = bf16(gf n (1 + gain) s (1 - s))``, ``dval = a rr - (sum_c a val) rr^3 val / dv`` with
  ``a = gf s (1 + gain)``, and per program the column partial sums of ``dgain = sum_rows gf s n``, which one torch
  ``sum`` reduces.

Numerics: fp32 register sums and products (another summation order than the torch chain's reductions).  Tests: the
per-operator tolerance rule of :mod:`neuronspark.utils.testing` against :func:`pkm_norm_gate_ref`, the torch chain.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE

from typing import Tuple

import torch
import triton
import triton.language as tl

from ....neuronspark.ops.triton import require_cuda

__all__ = ["PKMNormGateFn", "pkm_norm_gate", "pkm_norm_gate_ref"]

# ================================================================================================ fixed constants
#: forward: one (t, b) row per program, warps
FWD_WARPS = 4
#: backward: rows per program (the gain's column partial sums are per program), warps
BWD_ROWS = 4
BWD_WARPS = 8


# ================================================================================================ kernels
@triton.jit
def _pg_fwd_kernel(VAL, GAIN, GL, OUT, RR, B, D, vs0, vs1, ls0, ls1, eps, BD: tl.constexpr):
    """out[r] = bf16(val rr (1 + gain) sigmoid(glog)) (row-major), rr = rsqrt(mean(val^2) + eps) stored at RR[r]."""
    r = tl.program_id(0).to(tl.int64)
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD)
    cm = c < D
    v = tl.load(VAL + (t * vs0 + b * vs1) + c, mask=cm, other=0.0)
    rr = tl.rsqrt(tl.sum(v * v, axis=0) / D + eps)
    tl.store(RR + r, rr)
    g1 = 1.0 + tl.load(GAIN + c, mask=cm, other=0.0)
    s = tl.sigmoid(tl.load(GL + (t * ls0 + b * ls1) + c, mask=cm, other=0.0).to(tl.float32))
    tl.store(OUT + r * D + c, (((v * rr) * g1) * s).to(tl.bfloat16), mask=cm)


@triton.jit
def _pg_bwd_kernel(G, VAL, GAIN, GL, RR, DVAL, DGL, PART, R, B, D, gs0, gs1, gs2, vs0, vs1, ls0, ls1,
                   dv0, dv1, dl0, dl1, BR: tl.constexpr, BD: tl.constexpr):
    """BR rows per program: dval (fp32) and dglog (bf16), each written through its own strides, PART[program, c] = the
    program's partial sum of the gain's gradient."""
    pid = tl.program_id(0).to(tl.int64)
    r = pid * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    c = tl.arange(0, BD)
    cm = c < D
    m = rm[:, None] & cm[None, :]
    gf = tl.load(G + (t * gs0 + b * gs1)[:, None] + c.to(tl.int64)[None, :] * gs2, mask=m,
                 other=0.0).to(tl.float32)
    v = tl.load(VAL + (t * vs0 + b * vs1)[:, None] + c[None, :], mask=m, other=0.0)
    s = tl.sigmoid(tl.load(GL + (t * ls0 + b * ls1)[:, None] + c[None, :], mask=m, other=0.0).to(tl.float32))
    g1 = 1.0 + tl.load(GAIN + c, mask=cm, other=0.0)
    rr = tl.load(RR + r, mask=rm, other=0.0)
    n = v * rr[:, None]
    tl.store(DGL + (t * dl0 + b * dl1)[:, None] + c[None, :], (gf * (n * g1[None, :]) * s * (1.0 - s)).to(tl.bfloat16),
             mask=m)
    dy = gf * s
    tl.store(PART + pid * D + c, tl.sum(dy * n, axis=0), mask=cm)
    a = dy * g1[None, :]
    sa = tl.sum(a * v, axis=1) * (rr * rr * rr) / D
    tl.store(DVAL + (t * dv0 + b * dv1)[:, None] + c[None, :], a * rr[:, None] - sa[:, None] * v, mask=m)


# ================================================================================================ launchers
def _dest_like(t: torch.Tensor) -> torch.Tensor:
    """An uninitialised tensor like ``t`` that the kernels write row by row: the layout torch gives a tensor like t
    when its inner stride is 1 (a dense t: t's own layout), row-major otherwise (a t whose strides are not dense)."""
    d = torch.empty_like(t)
    return d if d.stride(-1) == 1 else torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)


def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_norm_gate_fwd", mutates_args=())
def _pkm_norm_gate_fwd(val: torch.Tensor, gain: torch.Tensor, glog: torch.Tensor,
                       eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(r bf16 (T, B, dv) contiguous, the row factors fp32 (T, B))."""
    T, B, D = (int(v) for v in val.shape)
    out = torch.empty((T, B, D), dtype=torch.bfloat16, device=val.device)
    rr = torch.empty((T, B), dtype=torch.float32, device=val.device)
    if T * B:
        _pg_fwd_kernel[(T * B,)](val, gain, glog, out, rr, B, D, int(val.stride(0)), int(val.stride(1)),
                                 int(glog.stride(0)), int(glog.stride(1)), float(eps), BD=_pow2(D),
                                 num_warps=FWD_WARPS)
    return out, rr


@_pkm_norm_gate_fwd.register_fake
def _pkm_norm_gate_fwd_fake(val, gain, glog, eps):
    return (val.new_empty(tuple(val.shape), dtype=torch.bfloat16),
            val.new_empty((val.shape[0], val.shape[1]), dtype=torch.float32))


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_norm_gate_bwd", mutates_args=())
def _pkm_norm_gate_bwd(g: torch.Tensor, val: torch.Tensor, gain: torch.Tensor, glog: torch.Tensor,
                       rr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dval fp32 (T, B, dv), dgain fp32 (dv,), dglog bf16 (T, B, dv); dval / dglog in the layout torch gives a
    tensor like val / glog)."""
    T, B, D = (int(v) for v in val.shape)
    R = T * B
    dev = val.device
    dval = _dest_like(val)
    dgl = _dest_like(glog)
    n_prog = triton.cdiv(R, BWD_ROWS)
    part = torch.empty((n_prog, D), dtype=torch.float32, device=dev)
    if R:
        _pg_bwd_kernel[(n_prog,)](g, val, gain, glog, rr, dval, dgl, part, R, B, D,
                                  int(g.stride(0)), int(g.stride(1)), int(g.stride(2)),
                                  int(val.stride(0)), int(val.stride(1)), int(glog.stride(0)), int(glog.stride(1)),
                                  int(dval.stride(0)), int(dval.stride(1)), int(dgl.stride(0)), int(dgl.stride(1)),
                                  BR=BWD_ROWS, BD=_pow2(D), num_warps=BWD_WARPS)
    return dval, part.sum(0), dgl


@_pkm_norm_gate_bwd.register_fake
def _pkm_norm_gate_bwd_fake(g, val, gain, glog, rr):
    return (val.new_empty(tuple(val.shape)), gain.new_empty(tuple(gain.shape)),
            glog.new_empty(tuple(glog.shape)))


class PKMNormGateFn(torch.autograd.Function):
    """r = bf16(val rsqrt(mean(val^2) + eps) (1 + gain) sigmoid(f32(glog))); gradients for val (fp32), gain (fp32) and
    glog (bf16) from the gradient of r."""

    @staticmethod
    def forward(ctx, val, gain, glog, eps):
        out, rr = _pkm_norm_gate_fwd(val, gain, glog, float(eps))
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(val, gain, glog, rr)
        return out

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None, None
        val, gain, glog, rr = ctx.saved_tensors
        if g.dtype not in (torch.bfloat16, torch.float32) or tuple(g.shape) != tuple(val.shape):
            raise RuntimeError(f"neuronspark.pkm_norm_gate backward: gradient {g.dtype} {tuple(g.shape)} for "
                               f"{tuple(val.shape)}")
        dval, dgain, dgl = _pkm_norm_gate_bwd(g, val, gain, glog, rr)
        need = ctx.needs_input_grad
        return dval if need[0] else None, dgain if need[1] else None, dgl if need[2] else None, None


# ================================================================================================ public function
def pkm_norm_gate(val: torch.Tensor, gain: torch.Tensor, glog: torch.Tensor, eps: float) -> torch.Tensor:
    """``bf16(val * rsqrt(mean(val^2, -1) + eps) * (1 + gain) * sigmoid(glog.float()))`` (T, B, dv) contiguous.

    ``val`` (T, B, dv) fp32 and ``glog`` (T, B, dv) bf16 with unit inner strides (any (t, b) strides); ``gain`` (dv,)
    fp32 contiguous, the zero-centred norm gain; ``eps`` the norm's eps.  Differentiable in val, gain and glog.  The
    training forward and, without autograd, the prefill."""
    for name, t in (("val", val), ("gain", gain), ("glog", glog)):
        require_cuda(t, f"pkm_norm_gate ({name})")
    if val.dim() != 3 or val.dtype != torch.float32 or val.stride(-1) != 1 or val.numel() >= 2 ** 31:
        raise RuntimeError(f"neuronspark.pkm_norm_gate: val must be fp32 (T, B, dv) with a unit inner stride, got "
                           f"{val.dtype} {tuple(val.shape)} {tuple(val.stride())}")
    if glog.dtype != torch.bfloat16 or tuple(glog.shape) != tuple(val.shape) or glog.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.pkm_norm_gate: glog must be bf16 {tuple(val.shape)} with a unit inner stride, "
                           f"got {glog.dtype} {tuple(glog.shape)}")
    if gain.dtype != torch.float32 or tuple(gain.shape) != (int(val.shape[-1]),) or not gain.is_contiguous():
        raise RuntimeError(f"neuronspark.pkm_norm_gate: gain must be a contiguous fp32 ({int(val.shape[-1])},) tensor")
    return PKMNormGateFn.apply(val, gain, glog, float(eps))


# ================================================================================================ reference
def pkm_norm_gate_ref(val: torch.Tensor, gain: torch.Tensor, glog: torch.Tensor, eps: float) -> torch.Tensor:
    """The torch chain of the read's norm and gate (same arguments and output as :func:`pkm_norm_gate`)."""
    r = val * torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + gain.to(torch.float32))
    return (r * torch.sigmoid(glog.to(torch.float32))).to(torch.bfloat16)
