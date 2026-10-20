"""spiking-layer residual addition of the inference prefill: the mixer's bf16 contribution added to the fp32 residual
stream.

Op (registry name ``spiking_res_add``; no autograd)::

    h_new = spiking_res_add(h, upd)         # h (T, B, D) fp32, upd (T, B, D) bf16 -> (T, B, D) fp32 contiguous

    h_new[t, b, :] = h[t, b, :] + fp32(upd[t, b, :])

In the spiking layer's prefill ``h`` is the T-major residual stream and ``upd`` the (B, T, D) output of the W_out GEMM
viewed as (T, B, D) by a transpose.  With one operand transposed, ``h + upd`` runs through ATen's generic (not
vectorised) elementwise kernel; both operands keep D contiguous, so this kernel adds one (t, b) row of D per program
with vector loads from both operands and a coalesced store.  The same fp32 addition of the same operands (the bf16
operand converted exactly): the values equal ``h + upd``.

Launch constants: one program per (t, b) row and BLOCK columns, BLOCK 1024 (512 for D <= 512), 4 warps.

Contract (anything else raises): CUDA tensors on one device; ``h`` fp32 and ``upd`` bf16 of one (T, B, D) shape, each
with a unit last stride (any T / B strides); batch offsets below 2^31 elements.  Reference: :func:`spiking_res_add_ref`
(``h + upd``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["NUM_WARPS", "block_cols", "spiking_res_add", "spiking_res_add_ref"]

# ------------------------------------------------------------------------------------------------ launch constants
NUM_WARPS = 4


def block_cols(D: int) -> int:
    """Columns per program: 512 for D <= 512, else 1024."""
    return 512 if int(D) <= 512 else 1024


# ------------------------------------------------------------------------------------------------ kernel
@triton.jit
def _res_add_kernel(H, U, OUT, B, D, s_ht, s_hb, s_ut, s_ub, BLOCK: tl.constexpr):
    """program (r = t * B + b, column block): OUT[r, cols] = H[t, b, cols] + fp32(U[t, b, cols]); OUT contiguous."""
    r = tl.program_id(0)
    pc = tl.program_id(1)
    t = r // B
    b = r % B
    c = pc * BLOCK + tl.arange(0, BLOCK)
    m = c < D
    h = tl.load(H + t.to(tl.int64) * s_ht + b * s_hb + c, mask=m)
    u = tl.load(U + t.to(tl.int64) * s_ut + b * s_ub + c, mask=m).to(tl.float32)
    tl.store(OUT + r.to(tl.int64) * D + c, h + u, mask=m)


# ------------------------------------------------------------------------------------------------ launcher
def _spiking_res_add_fwd(h: torch.Tensor, upd: torch.Tensor) -> torch.Tensor:
    """(T, B, D) fp32 contiguous ``h + upd`` (operands checked by the caller)."""
    T, B, D = (int(s) for s in h.shape)
    out = torch.empty((T, B, D), dtype=torch.float32, device=h.device)
    BLOCK = block_cols(D)
    with torch.cuda.device(h.device.index):
        _res_add_kernel[(T * B, triton.cdiv(D, BLOCK))](h, upd, out, B, D, h.stride(0), h.stride(1), upd.stride(0),
                                                         upd.stride(1), BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out


@torch.library.custom_op("neuronspark::spiking_res_add", mutates_args=(), schema="(Tensor h, Tensor upd) -> Tensor")
def _spiking_res_add_op(h, upd):
    """(T, B, D) fp32 contiguous"""
    return _spiking_res_add_fwd(h, upd)


@_spiking_res_add_op.register_fake
def _spiking_res_add_fake(h, upd):
    return torch.empty(tuple(h.shape), dtype=torch.float32, device=h.device)


# ------------------------------------------------------------------------------------------------ public function
def _fail(msg: str) -> None:
    raise RuntimeError(f"neuronspark.spiking_res_add: {msg}")


def spiking_res_add(h: torch.Tensor, upd: torch.Tensor) -> torch.Tensor:
    """``h + upd`` for the fp32 residual stream ``h`` (T, B, D) and the bf16 mixer output ``upd`` (T, B, D) (e.g. the
    transposed view of a (B, T, D) GEMM output), unit last strides, any T / B strides -> a new (T, B, D) fp32
    contiguous tensor (see the module docstring).  No autograd; the inputs are not modified."""
    for name, t in (("h", h), ("upd", upd)):
        if not isinstance(t, torch.Tensor):
            _fail(f"{name} must be a tensor, got {type(t).__name__}")
    require_cuda(h, "spiking_res_add")
    if upd.device != h.device:
        _fail(f"upd is on {upd.device}, h on {h.device}")
    if h.dim() != 3 or tuple(upd.shape) != tuple(h.shape):
        _fail(f"h and upd must be (T, B, D) of one shape, got {tuple(h.shape)} {tuple(upd.shape)}")
    if h.dtype != torch.float32 or upd.dtype != torch.bfloat16:
        _fail(f"h fp32 and upd bf16 expected, got {h.dtype} {upd.dtype}")
    if h.stride(-1) != 1 or upd.stride(-1) != 1:
        _fail(f"unit last strides expected, got h {tuple(h.stride())} upd {tuple(upd.stride())}")
    T, B, D = (int(s) for s in h.shape)
    if (B - 1) * max(int(h.stride(1)), int(upd.stride(1)), 0) + D >= 2 ** 31:
        _fail(f"batch offsets of {tuple(h.shape)} with strides {tuple(h.stride())} / {tuple(upd.stride())} exceed 2^31 "
              "elements")
    if torch.is_grad_enabled() and (h.requires_grad or upd.requires_grad):
        _fail("an inference operator (no autograd): call it under torch.no_grad()")
    if T * B == 0 or D == 0:
        return torch.empty((T, B, D), dtype=torch.float32, device=h.device)
    return torch.ops.neuronspark.spiking_res_add(h, upd)


# ------------------------------------------------------------------------------------------------ reference
def spiking_res_add_ref(h: torch.Tensor, upd: torch.Tensor) -> torch.Tensor:
    """Pure torch: ``h + upd`` (the bf16 operand converted exactly to fp32; ATen's result layout follows ``h``).  Any
    device."""
    return h + upd
