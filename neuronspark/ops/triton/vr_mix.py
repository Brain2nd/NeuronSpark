"""Value-residual mix and GLU tail of a prefill call in one kernel.

The value residual of a hippocampal layer mixes its values with the per-head gate of the centred normed input, and the
GLU
tail adds its addend to the last head's channels (:mod:`neuronspark.modules.pool_attention`)::

    lam[t, b, c] = lam0[c] + tanh(f32(gl[t, b, c // (D / H)]))
    m            = bf16(f32(v) * (1 + lam))            hippocampal layer 0 (carry None)
                 = bf16(f32(v) + lam * f32(carry))     later layers (carry = layer 0's value stream)
    out          = m;  out[..., D - d:] = bf16(f32(m[..., D - d:]) + glu_add)          (d = glu_add.shape[-1])

with ``gl`` (T, B, H) the bf16 gate projection, ``lam0`` (D,) fp32 the static part (``pool_value_residual`` plus the
layer's anchor) and ``glu_add`` (T, B, d) fp32.  The torch chain (:func:`vr_mix_prefill_ref`) materialises ``lam``
(T, B, D) fp32 and runs about ten elementwise kernels and the concatenation.  Here one program forms a (VR_MIX_BT
tokens, VR_MIX_BC channels) tile of one row b with the same fp32 operations in the same order (libdevice ``tanhf``,
ATen's device tanh for fp32; no FMA contraction) and the same bf16 roundings, so the output equals the chain's.  The
inputs may have any strides; the output is (T, B, D) contiguous, the layout of the chain's concatenation.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda

__all__ = ["VR_MIX_BT", "VR_MIX_BC", "VR_MIX_WARPS", "vr_mix_prefill", "vr_mix_prefill_ref"]

# ================================================================================================ fixed constants
#: tokens and channels per program, warps; grid (token blocks, channel blocks, B)
VR_MIX_BT = 32
VR_MIX_BC = 128
VR_MIX_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _vr_mix_kernel(V, XC, GL, LAM0, GV, OUT, T, B, D, DH, DT, sv_t, sv_b, sv_c, sx_t, sx_b, sx_c, sg_t, sg_b, sg_h,
                   sa_t, sa_b, FIRST: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(2).to(tl.int64)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mc = c < D
    m = (t < T)[:, None] & mc[None, :]
    v = tl.load(V + b * sv_b + t[:, None] * sv_t + c[None, :] * sv_c, mask=m, other=0.0).to(tl.float32)
    g = tl.load(GL + b * sg_b + t[:, None] * sg_t + (c // DH)[None, :] * sg_h, mask=m, other=0.0).to(tl.float32)
    lam = tl.load(LAM0 + c, mask=mc, other=0.0)[None, :] + libdevice.tanh(g)
    if FIRST:
        mix = v * (1.0 + lam)
    else:
        xc = tl.load(XC + b * sx_b + t[:, None] * sx_t + c[None, :] * sx_c, mask=m, other=0.0).to(tl.float32)
        mix = v + lam * xc
    mix = mix.to(tl.bfloat16)
    ct = c - (D - DT)
    tail = ct >= 0
    ga = tl.load(GV + b * sa_b + t[:, None] * sa_t + ct[None, :], mask=m & tail[None, :], other=0.0)
    out = tl.where(tail[None, :], (mix.to(tl.float32) + ga).to(tl.bfloat16), mix)
    tl.store(OUT + (t[:, None] * B + b) * D + c[None, :], out, mask=m)


# ================================================================================================ launcher
def _check(v, carry, gl, lam0, glu_add):
    for name, x in (("v", v), ("gl", gl), ("lam0", lam0), ("glu_add", glu_add)):
        require_cuda(x, f"vr_mix_prefill ({name})")
    if v.dim() != 3 or v.dtype != torch.bfloat16:
        raise RuntimeError(f"neuronspark.vr_mix_prefill: v must be bf16 (T, B, D), got {v.dtype} {tuple(v.shape)}")
    T, B, D = (int(x) for x in v.shape)
    if carry is not None and (carry.dtype != torch.bfloat16 or tuple(carry.shape) != (T, B, D)
                              or carry.device != v.device):
        raise RuntimeError("neuronspark.vr_mix_prefill: the carry must be bf16 of v's shape")
    if gl.dim() != 3 or tuple(gl.shape[:2]) != (T, B) or gl.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"neuronspark.vr_mix_prefill: gl must be bf16 / fp32 (T, B, H), got {gl.dtype} "
                           f"{tuple(gl.shape)}")
    H = int(gl.shape[2])
    if H < 1 or D % H:
        raise RuntimeError(f"neuronspark.vr_mix_prefill: D = {D} channels for {H} heads")
    if lam0.dtype != torch.float32 or tuple(lam0.shape) != (D,) or not lam0.is_contiguous():
        raise RuntimeError(f"neuronspark.vr_mix_prefill: lam0 must be a contiguous fp32 ({D},) tensor, got "
                           f"{lam0.dtype} {tuple(lam0.shape)}")
    d = int(glu_add.shape[-1]) if glu_add.dim() == 3 else -1
    if (glu_add.dtype != torch.float32 or tuple(glu_add.shape) != (T, B, d) or not 0 < d <= D
            or glu_add.stride(2) != 1):
        raise RuntimeError(f"neuronspark.vr_mix_prefill: glu_add must be fp32 (T, B, d) with 0 < d <= {D} and unit "
                           f"inner stride, got {glu_add.dtype} {tuple(glu_add.shape)}")
    for name, x in (("v", v), ("carry", carry), ("glu_add", glu_add)):
        if x is not None and max(abs(s) * n for s, n in zip(x.stride(), x.shape)) >= 2 ** 31:
            raise RuntimeError(f"neuronspark.vr_mix_prefill: {name} spans more than 2^31 elements")
    if T * B * D >= 2 ** 31:
        raise RuntimeError("neuronspark.vr_mix_prefill: tensor too large")
    return T, B, D, H, d


def _vr_mix_prefill_fwd(v, carry, gl, lam0, glu_add):
    T, B, D, H, d = _check(v, carry, gl, lam0, glu_add)
    out = torch.empty((T, B, D), dtype=torch.bfloat16, device=v.device)
    if T * B == 0:
        return out
    xc = carry if carry is not None else v
    _vr_mix_kernel[(triton.cdiv(T, VR_MIX_BT), triton.cdiv(D, VR_MIX_BC), B)](
        v, xc, gl, lam0, glu_add, out, T, B, D, D // H, d, *v.stride(), *xc.stride(), *gl.stride(),
        int(glu_add.stride(0)), int(glu_add.stride(1)), FIRST=carry is None, BT=VR_MIX_BT, BC=VR_MIX_BC,
        num_warps=VR_MIX_WARPS, enable_fp_fusion=False)
    return out


@torch.library.custom_op("neuronspark::vr_mix_prefill", mutates_args=())
def _vr_mix_prefill_op(v: torch.Tensor, carry: Optional[torch.Tensor], gl: torch.Tensor, lam0: torch.Tensor,
                       glu_add: torch.Tensor) -> torch.Tensor:
    return _vr_mix_prefill_fwd(v, carry, gl, lam0, glu_add)


@_vr_mix_prefill_op.register_fake
def _vr_mix_prefill_fake(v, carry, gl, lam0, glu_add):
    return torch.empty(tuple(v.shape), dtype=torch.bfloat16, device=v.device)


def vr_mix_prefill(v: torch.Tensor, carry: Optional[torch.Tensor], gl: torch.Tensor, lam0: torch.Tensor,
                   glu_add: torch.Tensor) -> torch.Tensor:
    """(T, B, D) bf16 contiguous mixed value stream (see the module docstring): ``v`` (T, B, D) bf16 and ``carry``
    (T, B, D) bf16 or None (hippocampal layer 0) at any strides, ``gl`` (T, B, H) the bf16 gate projection, ``lam0``
    (D,)
    fp32 contiguous the static mix, ``glu_add`` (T, B, d) fp32 the GLU addend of the last d channels.  No autograd."""
    return torch.ops.neuronspark.vr_mix_prefill(v, carry, gl, lam0, glu_add)


# ================================================================================================ reference
def vr_mix_prefill_ref(v: torch.Tensor, carry: Optional[torch.Tensor], gl: torch.Tensor, lam0: torch.Tensor,
                       glu_add: torch.Tensor) -> torch.Tensor:
    """The hippocampal layer's torch chain (same arguments and output as :func:`vr_mix_prefill`): the fp32 tanh of the
    gate
    projection repeated over each head's channels plus the static mix, the value residual, the GLU tail."""
    H, D, d = int(gl.shape[-1]), int(v.shape[-1]), int(glu_add.shape[-1])
    lam = lam0.view(1, 1, -1) + torch.tanh(gl.to(torch.float32)).repeat_interleave(D // H, dim=-1)
    if carry is None:
        mix = (v.float() * (1.0 + lam)).to(v.dtype)
    else:
        mix = (v.float() + lam * carry.float()).to(v.dtype)
    return torch.cat([mix[..., :D - d], (mix[..., D - d:].float() + glu_add).to(v.dtype)], dim=-1)
