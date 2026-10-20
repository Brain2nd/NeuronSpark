"""Value-residual mix of a prefill call in one kernel.

The value residual of a memory layer mixes its values with the per-head gate of the centred normed input
(:mod:`neuronspark.modules.pool_attention`)::

    lam[t, b, c] = lam0[c] + tanh(f32(gl[t, b, c // (D / H)]))        (T, B, D) fp32

with ``gl`` (T, B, H) the bf16 gate projection and ``lam0`` (D,) fp32 the static part (``pool_value_residual`` plus
the layer's anchor).  The torch chain (:func:`vr_lambda_prefill_ref`) is a cast, a tanh, the materialised
``repeat_interleave`` over each head's D / H channels and the broadcast add.  Here one program writes
``VR_LAMBDA_BLOCK`` channels of one (t, b) row: the same tanh (libdevice ``tanhf``, ATen's device tanh for fp32) and
the same single fp32 addition, so the output equals the chain's.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda

__all__ = ["VR_LAMBDA_BLOCK", "VR_LAMBDA_WARPS", "vr_lambda_prefill", "vr_lambda_prefill_ref"]

# ================================================================================================ fixed constants
#: channels per program, warps; grid (T * B rows, channel blocks)
VR_LAMBDA_BLOCK = 512
VR_LAMBDA_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _vr_lambda_kernel(GL, LAM, OUT, B, D, DH, s_gl0, s_gl1, s_gl2, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    pc = tl.program_id(1)
    t = (r // B).to(tl.int64)
    b = (r % B).to(tl.int64)
    c = pc * BLOCK + tl.arange(0, BLOCK)
    m = c < D
    g = tl.load(GL + t * s_gl0 + b * s_gl1 + (c // DH) * s_gl2, mask=m, other=0.0).to(tl.float32)
    lam = tl.load(LAM + c, mask=m, other=0.0)
    tl.store(OUT + r.to(tl.int64) * D + c, lam + libdevice.tanh(g), mask=m)


# ================================================================================================ launcher
def _check(gl: torch.Tensor, lam: torch.Tensor):
    require_cuda(gl, "vr_lambda_prefill (gl)")
    require_cuda(lam, "vr_lambda_prefill (lam)")
    if gl.dim() != 3 or gl.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"neuronspark.vr_lambda_prefill: gl must be bf16 / fp32 (T, B, H), got {gl.dtype} "
                           f"{tuple(gl.shape)}")
    T, B, H = (int(v) for v in gl.shape)
    if lam.dtype != torch.float32 or lam.dim() != 1 or not lam.is_contiguous():
        raise RuntimeError(f"neuronspark.vr_lambda_prefill: lam must be a contiguous fp32 (D,) tensor, got {lam.dtype} "
                           f"{tuple(lam.shape)}")
    D = int(lam.shape[0])
    if H < 1 or D % H:
        raise RuntimeError(f"neuronspark.vr_lambda_prefill: D = {D} channels for {H} heads")
    return T, B, H, D


def _vr_lambda_prefill_fwd(gl: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    T, B, H, D = _check(gl, lam)
    out = torch.empty((T, B, D), dtype=torch.float32, device=gl.device)
    if T * B == 0:
        return out
    _vr_lambda_kernel[(T * B, triton.cdiv(D, VR_LAMBDA_BLOCK))](
        gl, lam, out, B, D, D // H, int(gl.stride(0)), int(gl.stride(1)), int(gl.stride(2)), BLOCK=VR_LAMBDA_BLOCK,
        num_warps=VR_LAMBDA_WARPS, enable_fp_fusion=False)
    return out


@torch.library.custom_op("neuronspark::vr_lambda_prefill", mutates_args=())
def _vr_lambda_prefill_op(gl: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return _vr_lambda_prefill_fwd(gl, lam)


@_vr_lambda_prefill_op.register_fake
def _vr_lambda_prefill_fake(gl, lam):
    return gl.new_empty((int(gl.shape[0]), int(gl.shape[1]), int(lam.shape[0])), dtype=torch.float32)


def vr_lambda_prefill(gl: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """(T, B, D) fp32 contiguous ``lam + tanh(f32(gl))`` repeated over each head's D / H channels: ``gl`` (T, B, H)
    the bf16 gate projection (any strides), ``lam`` (D,) fp32 contiguous the static mix."""
    return torch.ops.neuronspark.vr_lambda_prefill(gl, lam)


# ================================================================================================ reference
def vr_lambda_prefill_ref(gl: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """The memory layer's torch chain (same arguments and output as :func:`vr_lambda_prefill`): the fp32 tanh of the
    gate projection repeated over each head's channels, added to the broadcast static mix."""
    H, D = int(gl.shape[-1]), int(lam.shape[-1])
    return lam.view(1, 1, -1) + torch.tanh(gl.to(torch.float32)).repeat_interleave(D // H, dim=-1)
