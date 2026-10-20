"""Interleaved RoPE of the hippocampal layer's queries and keys.

Operators (registry names):

* ``silu_rope(x, cos, sin)`` (training): ``rope(bf16(silu(x)))`` for the bf16 (T, B, C) q or k slice of the short
  conv's output, forward and backward.
* ``rope_paired(x, cos, sin)`` (training): ``rope(x)`` for a bf16 (T, B, C) x (the extra queries of the last head),
  forward and backward.
* ``rope_interleaved(x, cos, sin)`` (prefill, no autograd): ``rope(x)`` for a bf16 (T, B, C) x, one kernel per call.

``rope`` is the interleaved layout: pair ``p`` = channels ``(2p, 2p + 1)``, ``c = cos[t, p]``,
``s = sin[t, p]``::

    y[2p]     = bf16( bf16(x[2p] c)     - bf16(x[2p + 1] s) )
    y[2p + 1] = bf16( bf16(x[2p + 1] c) + bf16(x[2p] s) )

The tables are bf16, either (T, C / 2) or broadcastable (T | 1, B | 1, C / 2); they receive no gradient.

Numerics contract
-----------------
Every product and every sum is one fp32 operation on the exact bf16 operands, rounded to bf16 (inline PTX
``mul / add / sub .rn.f32`` and ``cvt.rn.bf16.f32``, ``enable_fp_fusion=False``), i.e. the rounding points of the
eager bf16 ops.  The SiLU is ATen's functor ``x / (1 + expf(-x))`` (libdevice ``expf``, ``add.rn``, ``div.rn``) on the widened
bf16 value, rounded to bf16.  Backward: the slice-buffer sums of autograd's graph (each use rounded to bf16, then the
"+ 0" of the other slice's zero fill), then ATen's ``silu_backward`` functor ``dy s (1 + x (1 - s))`` with the
``1 + x (.)`` contracted to one ``fma.rn`` as nvcc compiles it.  The training kernels read / write channel-contiguous
(BT, 2 BP) tiles (the even / odd elements split / joined in registers, int32 in-tile offsets); x, y and the gradients
need channel stride 1 (other layouts raise).  ``rope_interleaved`` forms the same four products and two sums.

Launchers are ``torch.library`` custom ops (``neuronspark::hippo_*rope*``) with fake implementations of the same
layouts.  The references apply :func:`neuronspark.modules.rope.apply_rope_interleaved_ref` (the eager bf16 ops).
"""
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from neuronspark.modules.rope import apply_rope_interleaved_ref
from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton.hippo_norms import (  # noqa: F401  (jit helpers resolved through this module's globals)
    _g_add,
    _g_mul,
    _g_n,
    _g_neg,
    _g_r16,
    _g_silu,
    _g_silu_bwd,
    _g_sub,
    _g_w,
    _meta,
)

__all__ = ["SiluRopeFn", "silu_rope", "silu_rope_ref", "RopePairedFn", "rope_paired", "rope_paired_ref",
           "rope_interleaved", "rope_interleaved_ref"]

# ------------------------------------------------------------------------------------------------ fixed constants
SROPE_FWD_CFG = (16, 64, 4)   # silu_rope forward (BT positions, BP pairs, warps): channel-contiguous tiles
SROPE_BWD_CFG = (16, 64, 4)   # silu_rope backward (x, gy, gx all channel-contiguous tiles)
ROPE_CFG = (16, 64, 4)        # rope_paired forward / backward
# rope_interleaved (prefill): BLOCK_P = min(64, pow2(C / 2)) pairs, BLOCK_R = 16 rows (32 when BLOCK_P < 64), 4 warps


# ================================================================================================ training kernels
@triton.jit
def _silu_rope_fwdp_kernel(X, C, S, Y, T, P, xs0, xs1, cs0, cs1, cs2, ss0, ss1, ss2, ys0, ys1,
                     BT: tl.constexpr, BP: tl.constexpr):
    """y = rope(bf16 silu(x)) on a (BT, 2 BP) channel tile of one b (x read and y written as one contiguous tile each,
    the even / odd elements split / joined in registers; channel stride 1; int32 in-tile offsets)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    b = tl.program_id(1)
    p = tl.program_id(2) * BP + tl.arange(0, BP)
    c2 = tl.program_id(2) * (2 * BP) + tl.arange(0, 2 * BP)
    m = (t < T)[:, None] & (p < P)[None, :]
    m2 = (t < T)[:, None] & (c2 < 2 * P)[None, :]
    x2 = tl.load(X + (t[:, None] * xs0 + b * xs1 + c2[None, :]), mask=m2, other=0.0)
    xe16, xd16 = tl.split(tl.reshape(x2, [BT, BP, 2]))
    xe = _g_r16(_g_silu(_g_w(xe16)))
    xd = _g_r16(_g_silu(_g_w(xd16)))
    c = _g_w(tl.load(C + (t[:, None] * cs0 + b * cs1 + p[None, :] * cs2), mask=m, other=0.0))
    s = _g_w(tl.load(S + (t[:, None] * ss0 + b * ss1 + p[None, :] * ss2), mask=m, other=0.0))
    pp = _g_r16(_g_mul(xe, c))
    q = _g_r16(_g_mul(xd, s))
    r = _g_r16(_g_mul(xd, c))
    u = _g_r16(_g_mul(xe, s))
    y2 = tl.reshape(tl.join(_g_n(_g_sub(pp, q)), _g_n(_g_add(r, u))), [BT, 2 * BP])
    tl.store(Y + (t[:, None] * ys0 + b * ys1 + c2[None, :]), y2, mask=m2)


@triton.jit
def _silu_rope_bwdx_kernel(GY, X, C, S, GX, T, P, gs0, gs1, xs0, xs1, cs0, cs1, cs2, ss0, ss1, ss2, os0, os1,
                     BT: tl.constexpr, BP: tl.constexpr):
    """silu_rope backward on a (BT, 2 BP) channel tile: gy, x and gx read / written as contiguous channel tiles."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    b = tl.program_id(1)
    p = tl.program_id(2) * BP + tl.arange(0, BP)
    c2 = tl.program_id(2) * (2 * BP) + tl.arange(0, 2 * BP)
    m = (t < T)[:, None] & (p < P)[None, :]
    m2 = (t < T)[:, None] & (c2 < 2 * P)[None, :]
    g2 = tl.load(GY + (t[:, None] * gs0 + b * gs1 + c2[None, :]), mask=m2, other=0.0)
    ga16, gb16 = tl.split(tl.reshape(g2, [BT, BP, 2]))
    ga = _g_w(ga16)
    gb = _g_w(gb16)
    c = _g_w(tl.load(C + (t[:, None] * cs0 + b * cs1 + p[None, :] * cs2), mask=m, other=0.0))
    s = _g_w(tl.load(S + (t[:, None] * ss0 + b * ss1 + p[None, :] * ss2), mask=m, other=0.0))
    pp = _g_r16(_g_mul(ga, c))
    q = _g_r16(_g_mul(_g_neg(ga), s))
    r = _g_r16(_g_mul(gb, c))
    u = _g_r16(_g_mul(gb, s))
    ge = _g_r16(_g_add(_g_r16(_g_add(pp, u)), 0.0))
    gd = _g_r16(_g_add(_g_r16(_g_add(q, r)), 0.0))
    x2 = tl.load(X + (t[:, None] * xs0 + b * xs1 + c2[None, :]), mask=m2, other=0.0)
    xe16, xd16 = tl.split(tl.reshape(x2, [BT, BP, 2]))
    xe = _g_w(xe16)
    xd = _g_w(xd16)
    o2 = tl.reshape(tl.join(_g_n(_g_silu_bwd(ge, xe)), _g_n(_g_silu_bwd(gd, xd))), [BT, 2 * BP])
    tl.store(GX + (t[:, None] * os0 + b * os1 + c2[None, :]), o2, mask=m2)


@triton.jit
def _rope_fwdp_kernel(X, C, S, Y, T, P, xs0, xs1, cs0, cs1, cs2, ss0, ss1, ss2, ys0, ys1,
                      BT: tl.constexpr, BP: tl.constexpr):
    """rope(x) with x read and y written as ONE contiguous (BT, 2 BP) channel tile each (channel stride 1, int32
    in-tile offsets), program (t tile, b, pair tile)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    b = tl.program_id(1)
    p = tl.program_id(2) * BP + tl.arange(0, BP)
    c2 = tl.program_id(2) * (2 * BP) + tl.arange(0, 2 * BP)
    m = (t < T)[:, None] & (p < P)[None, :]
    m2 = (t < T)[:, None] & (c2 < 2 * P)[None, :]
    x2 = tl.load(X + (t[:, None] * xs0 + b * xs1 + c2[None, :]), mask=m2, other=0.0)
    xe16, xd16 = tl.split(tl.reshape(x2, [BT, BP, 2]))
    xe = _g_w(xe16)
    xd = _g_w(xd16)
    c = _g_w(tl.load(C + (t[:, None] * cs0 + b * cs1 + p[None, :] * cs2), mask=m, other=0.0))
    s = _g_w(tl.load(S + (t[:, None] * ss0 + b * ss1 + p[None, :] * ss2), mask=m, other=0.0))
    pp = _g_r16(_g_mul(xe, c))
    q = _g_r16(_g_mul(xd, s))
    r = _g_r16(_g_mul(xd, c))
    u = _g_r16(_g_mul(xe, s))
    y2 = tl.reshape(tl.join(_g_n(_g_sub(pp, q)), _g_n(_g_add(r, u))), [BT, 2 * BP])
    tl.store(Y + (t[:, None] * ys0 + b * ys1 + c2[None, :]), y2, mask=m2)


@triton.jit
def _rope_bwdp_kernel(GY, C, S, GX, T, P, gs0, gs1, cs0, cs1, cs2, ss0, ss1, ss2, os0, os1,
                      BT: tl.constexpr, BP: tl.constexpr):
    """rope backward with gy read and gx written as contiguous (BT, 2 BP) channel tiles: g_e = bf16(bf16(gA c) +
    bf16(gB s)) + 0, g_o = bf16(bf16(-gA s) + bf16(gB c)) + 0."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    b = tl.program_id(1)
    p = tl.program_id(2) * BP + tl.arange(0, BP)
    c2 = tl.program_id(2) * (2 * BP) + tl.arange(0, 2 * BP)
    m = (t < T)[:, None] & (p < P)[None, :]
    m2 = (t < T)[:, None] & (c2 < 2 * P)[None, :]
    g2 = tl.load(GY + (t[:, None] * gs0 + b * gs1 + c2[None, :]), mask=m2, other=0.0)
    ga16, gb16 = tl.split(tl.reshape(g2, [BT, BP, 2]))
    ga = _g_w(ga16)
    gb = _g_w(gb16)
    c = _g_w(tl.load(C + (t[:, None] * cs0 + b * cs1 + p[None, :] * cs2), mask=m, other=0.0))
    s = _g_w(tl.load(S + (t[:, None] * ss0 + b * ss1 + p[None, :] * ss2), mask=m, other=0.0))
    pp = _g_r16(_g_mul(ga, c))
    q = _g_r16(_g_mul(_g_neg(ga), s))
    r = _g_r16(_g_mul(gb, c))
    u = _g_r16(_g_mul(gb, s))
    ge = _g_r16(_g_add(pp, u))
    gd = _g_r16(_g_add(q, r))
    o2 = tl.reshape(tl.join(_g_n(_g_add(ge, 0.0)), _g_n(_g_add(gd, 0.0))), [BT, 2 * BP])
    tl.store(GX + (t[:, None] * os0 + b * os1 + c2[None, :]), o2, mask=m2)


# ================================================================================================ prefill kernel
@triton.jit
def _rope_interleaved_kernel(X, COS, SIN, Y, R, D2, s_x_r, s_c_t, s_c_b, s_s_t, s_s_b, B,
                             BLOCK_R: tl.constexpr, BLOCK_P: tl.constexpr):
    """rows r = t * B + b of a (T, B, D) bf16 tensor (row stride s_x_r, unit last stride); pairs p of D / 2.  The (even,
    odd) pair is loaded and stored as one contiguous (BLOCK_R, 2 BLOCK_P) tile, split / joined on the last axis; every
    product rounded to bf16 before the difference / sum."""
    pid_r = tl.program_id(axis=0)
    pid_p = tl.program_id(axis=1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_e = pid_p * (2 * BLOCK_P) + tl.arange(0, 2 * BLOCK_P)         # the 2 BLOCK_P contiguous elements of the pair block
    mr = offs_r < R
    m2 = mr[:, None] & (offs_e[None, :] < 2 * D2)
    mp = mr[:, None] & (offs_p[None, :] < D2)
    t = offs_r // B
    b = offs_r - t * B
    x2 = tl.load(X + offs_r[:, None] * s_x_r + offs_e[None, :], mask=m2, other=0.0).to(tl.float32)
    x_e, x_o = tl.split(tl.reshape(x2, (BLOCK_R, BLOCK_P, 2)))         # even = element 2p, odd = element 2p + 1
    c = tl.load(COS + t[:, None] * s_c_t + b[:, None] * s_c_b + offs_p[None, :], mask=mp, other=0.0).to(tl.float32)
    s = tl.load(SIN + t[:, None] * s_s_t + b[:, None] * s_s_b + offs_p[None, :], mask=mp, other=0.0).to(tl.float32)
    p1 = (x_e * c).to(tl.bfloat16).to(tl.float32)
    p2 = (x_o * s).to(tl.bfloat16).to(tl.float32)
    p3 = (x_o * c).to(tl.bfloat16).to(tl.float32)
    p4 = (x_e * s).to(tl.bfloat16).to(tl.float32)
    y_e = (p1 - p2).to(tl.bfloat16)
    y_o = (p3 + p4).to(tl.bfloat16)
    y2 = tl.reshape(tl.join(y_e, y_o), (BLOCK_R, 2 * BLOCK_P))
    tl.store(Y + offs_r[:, None] * s_x_r + offs_e[None, :], y2, mask=m2)


# ================================================================================================ host helpers
def _tables(cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, what: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos / sin as bf16 (T | 1, B | 1, C / 2) tensors (a 2-D (T, C / 2) table gains a broadcast batch axis)."""
    T, B, D = (int(v) for v in x.shape)
    out = []
    for n, t in (("cos", cos), ("sin", sin)):
        if not isinstance(t, torch.Tensor):
            raise RuntimeError(f"neuronspark.{what}: {n} must be a tensor")
        if t.dim() == 2:
            t = t.unsqueeze(1)
        if (t.dtype != torch.bfloat16 or t.device != x.device or t.dim() != 3 or t.shape[0] not in (1, T)
                or t.shape[1] not in (1, B) or int(t.shape[2]) != D // 2):
            raise RuntimeError(f"neuronspark.{what}: {n} must be a bf16 (T|1, B|1, C/2) or (T, C/2) table on x's "
                               f"device, got {t.dtype} {tuple(t.shape)} for x {tuple(x.shape)}")
        if t.requires_grad:
            raise RuntimeError(f"neuronspark.{what}: {n} requires grad (the fused backward gives no table gradient)")
        out.append(t)
    return out[0], out[1]


def _tab_st(t: torch.Tensor):
    """a (T|1, B|1, P) table's strides on (t, b, p) (0 on a size-1 axis)."""
    return (int(t.stride(0)) if t.shape[0] != 1 else 0, int(t.stride(1)) if t.shape[1] != 1 else 0, int(t.stride(2)))


def _paired_ok(what: str, **ts) -> None:
    for n, t in ts.items():
        if int(t.stride(2)) != 1:
            raise RuntimeError(f"neuronspark.{what}: {n} needs channel stride 1 for the paired tile, got strides "
                               f"{tuple(t.stride())}")
        span = sum((int(a) - 1) * abs(int(b)) for a, b in zip(t.shape, t.stride())) + 1
        if span >= 2 ** 31:
            raise RuntimeError(f"neuronspark.{what}: {n} offsets beyond the int32 range of the paired kernel")


def _check_x(x: torch.Tensor, what: str) -> None:
    require_cuda(x, what)
    if not (x.dtype == torch.bfloat16 and x.dim() == 3 and int(x.shape[-1]) % 2 == 0 and x.numel() > 0):
        raise RuntimeError(f"neuronspark.{what}: x must be a non-empty 3-D bf16 CUDA tensor with an even last axis, "
                           f"got {x.dtype} {tuple(x.shape)}")


# ================================================================================================ silu_rope
def _srope_fwd_alloc(xp):
    T, B, D = (int(v) for v in xp.shape)
    sm = F.silu(_meta(xp))
    return torch.empty_strided((T, B, D), torch.empty_like(sm).stride(), dtype=torch.bfloat16, device=xp.device)


@torch.library.custom_op("neuronspark::hippo_silu_rope_fwd", mutates_args=(),
                         schema="(Tensor x, Tensor cos, Tensor sin) -> Tensor")
def _srope_fwd(x, cos, sin):
    T, B, D = (int(v) for v in x.shape)
    P = D // 2
    y = _srope_fwd_alloc(x)
    _paired_ok("silu_rope", x=x, y=y)
    bt, bp, nw = SROPE_FWD_CFG
    _silu_rope_fwdp_kernel[(triton.cdiv(T, bt), B, triton.cdiv(P, bp))](
        x, cos, sin, y, T, P, int(x.stride(0)), int(x.stride(1)), *_tab_st(cos), *_tab_st(sin),
        int(y.stride(0)), int(y.stride(1)), BT=bt, BP=bp, num_warps=nw, enable_fp_fusion=False)
    return y


@_srope_fwd.register_fake
def _srope_fwd_fake(x, cos, sin):
    return _srope_fwd_alloc(x)


def _srope_bwd_alloc(x):
    T, B, D = (int(v) for v in x.shape)
    gsm = torch.empty((T, B, D), dtype=torch.bfloat16, device="meta")
    gx_st = torch.ops.aten.silu_backward(gsm, _meta(x)).stride()
    return torch.empty_strided((T, B, D), gx_st, dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("neuronspark::hippo_silu_rope_bwd", mutates_args=(),
                         schema="(Tensor gy, Tensor x, Tensor cos, Tensor sin) -> Tensor")
def _srope_bwd(gy, x, cos, sin):
    T, B, D = (int(v) for v in x.shape)
    P = D // 2
    gx = _srope_bwd_alloc(x)
    _paired_ok("silu_rope backward", gy=gy, x=x, gx=gx)
    bt, bp, nw = SROPE_BWD_CFG
    _silu_rope_bwdx_kernel[(triton.cdiv(T, bt), B, triton.cdiv(P, bp))](
        gy, x, cos, sin, gx, T, P, int(gy.stride(0)), int(gy.stride(1)), int(x.stride(0)), int(x.stride(1)),
        *_tab_st(cos), *_tab_st(sin), int(gx.stride(0)), int(gx.stride(1)),
        BT=bt, BP=bp, num_warps=nw, enable_fp_fusion=False)
    return gx


@_srope_bwd.register_fake
def _srope_bwd_fake(gy, x, cos, sin):
    return _srope_bwd_alloc(x)


class SiluRopeFn(torch.autograd.Function):
    """rope(bf16 silu(x)) for a bf16 (T, B, C) x with channel stride 1; backward x's bf16 gradient."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        y = _srope_fwd(x, cos, sin)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, cos, sin)
        return y

    @staticmethod
    def backward(ctx, gy):
        if gy is None:
            return None, None, None
        x, cos, sin = ctx.saved_tensors
        if gy.dtype != torch.bfloat16 or tuple(gy.shape) != tuple(x.shape):
            raise RuntimeError(f"neuronspark.silu_rope backward: gradient {gy.dtype} {tuple(gy.shape)} for "
                               f"{tuple(x.shape)}")
        return _srope_bwd(gy, x, cos, sin), None, None


def silu_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``rope(F.silu(x))`` (interleaved layout) for a bf16 (T, B, C) CUDA ``x`` with channel stride 1 (the q or k slice
    of the short conv's T-major output); ``cos`` / ``sin`` bf16 tables (T, C / 2) or (T|1, B|1, C / 2).  Returns a bf16
    tensor in the layout ``F.silu`` gives (contiguous for those slices).  Training regime."""
    _check_x(x, "silu_rope")
    c3, s3 = _tables(cos, sin, x, "silu_rope")
    return SiluRopeFn.apply(x, c3, s3)


def silu_rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``apply_rope_interleaved_ref(F.silu(x), cos, sin)`` in bf16."""
    c3, s3 = _tables(cos, sin, x, "silu_rope_ref")
    return apply_rope_interleaved_ref(F.silu(x), c3, s3)


# ================================================================================================ rope_paired
@torch.library.custom_op("neuronspark::hippo_rope_paired_fwd", mutates_args=(),
                         schema="(Tensor x, Tensor cos, Tensor sin) -> Tensor")
def _rope_fwd(x, cos, sin):
    T, B, D = (int(v) for v in x.shape)
    y = torch.empty_like(x)
    _paired_ok("rope_paired", x=x, y=y)
    bt, bp, nw = ROPE_CFG
    _rope_fwdp_kernel[(triton.cdiv(T, bt), B, triton.cdiv(D // 2, bp))](
        x, cos, sin, y, T, D // 2, int(x.stride(0)), int(x.stride(1)), *_tab_st(cos), *_tab_st(sin),
        int(y.stride(0)), int(y.stride(1)), BT=bt, BP=bp, num_warps=nw, enable_fp_fusion=False)
    return y


@_rope_fwd.register_fake
def _rope_fwd_fake(x, cos, sin):
    return torch.empty_like(x)


@torch.library.custom_op("neuronspark::hippo_rope_paired_bwd", mutates_args=(),
                         schema="(Tensor gy, Tensor cos, Tensor sin) -> Tensor")
def _rope_bwd(gy, cos, sin):
    T, B, D = (int(v) for v in gy.shape)
    gx = torch.empty((T, B, D), dtype=torch.bfloat16, device=gy.device)
    _paired_ok("rope_paired backward", gy=gy, gx=gx)
    bt, bp, nw = ROPE_CFG
    _rope_bwdp_kernel[(triton.cdiv(T, bt), B, triton.cdiv(D // 2, bp))](
        gy, cos, sin, gx, T, D // 2, int(gy.stride(0)), int(gy.stride(1)), *_tab_st(cos), *_tab_st(sin),
        int(gx.stride(0)), int(gx.stride(1)), BT=bt, BP=bp, num_warps=nw, enable_fp_fusion=False)
    return gx


@_rope_bwd.register_fake
def _rope_bwd_fake(gy, cos, sin):
    return torch.empty(tuple(gy.shape), dtype=torch.bfloat16, device=gy.device)


class RopePairedFn(torch.autograd.Function):
    """rope(x) for a dense bf16 (T, B, C) x with channel stride 1; backward the bf16 gradient (contiguous)."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(cos, sin)
        ctx.xshape = tuple(int(v) for v in x.shape)
        return _rope_fwd(x, cos, sin)

    @staticmethod
    def backward(ctx, gy):
        if gy is None:
            return None, None, None
        cos, sin = ctx.saved_tensors
        if gy.dtype != torch.bfloat16 or tuple(gy.shape) != ctx.xshape:
            raise RuntimeError(f"neuronspark.rope_paired backward: gradient {gy.dtype} {tuple(gy.shape)} for "
                               f"{ctx.xshape}")
        return _rope_bwd(gy, cos, sin), None, None


def rope_paired(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``rope(x)`` (interleaved layout) for a dense bf16 (T, B, C) CUDA ``x`` with channel stride 1 (the extra queries'
    projections); ``cos`` / ``sin`` bf16 (T, C / 2) or (T|1, B|1, C / 2) tables (the last head's column slice).
    Training regime."""
    _check_x(x, "rope_paired")
    if torch.empty_like(x).stride() != x.stride():
        raise RuntimeError(f"neuronspark.rope_paired: x strides {tuple(x.stride())} are not preserved by empty_like")
    c3, s3 = _tables(cos, sin, x, "rope_paired")
    return RopePairedFn.apply(x, c3, s3)


def rope_paired_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    c3, s3 = _tables(cos, sin, x, "rope_paired_ref")
    return apply_rope_interleaved_ref(x, c3, s3)


# ================================================================================================ rope_interleaved
@torch.library.custom_op("neuronspark::hippo_rope_interleaved", mutates_args=(),
                         schema="(Tensor x, Tensor cos, Tensor sin) -> Tensor")
def _rope_interleaved_op(x, cos, sin):
    T, B, D = (int(v) for v in x.shape)
    D2 = D // 2
    if x.stride(1) != D or x.stride(0) != B * D:
        x = x.contiguous()                   # a (T, B, D) tensor whose rows are not D apart: as the reference reads it
    sct, scb = _bcast_strides(cos, T, B, D2)
    sst, ssb = _bcast_strides(sin, T, B, D2)
    y = torch.empty((T, B, D), dtype=x.dtype, device=x.device)
    R = T * B
    BLOCK_P = min(64, triton.next_power_of_2(D2))
    BLOCK_R = 16 if BLOCK_P >= 64 else 32
    grid = (triton.cdiv(R, BLOCK_R), triton.cdiv(D2, BLOCK_P))
    _rope_interleaved_kernel[grid](x, cos, sin, y, R, D2, D, sct, scb, sst, ssb, B, BLOCK_R=BLOCK_R, BLOCK_P=BLOCK_P,
                                   num_warps=4, enable_fp_fusion=False)
    return y


@_rope_interleaved_op.register_fake
def _rope_interleaved_fake(x, cos, sin):
    return torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device)


def _bcast_strides(c: torch.Tensor, T: int, B: int, D2: int):
    """(stride_t, stride_b) of a cos / sin table broadcastable to (T, B, D2) with unit last stride."""
    if c.dim() != 3 or c.shape[-1] != D2 or c.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.rope_interleaved: cos / sin must be (T or 1, B or 1, D / 2) with unit last "
                           f"stride, got {tuple(c.shape)} {tuple(c.stride())}")
    if c.shape[0] not in (1, T) or c.shape[1] not in (1, B):
        raise RuntimeError(f"neuronspark.rope_interleaved: cos / sin {tuple(c.shape)} not broadcastable to "
                           f"({T}, {B}, {D2})")
    return (0 if c.shape[0] == 1 else c.stride(0)), (0 if c.shape[1] == 1 else c.stride(1))


def rope_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``rope(x)`` (interleaved layout) for a bf16 (T, B, C) CUDA ``x`` with unit last stride, one kernel; ``cos`` /
    ``sin`` bf16 (T, C / 2) or (T|1, B|1, C / 2) with unit last stride.  Returns a contiguous bf16 (T, B, C) tensor.
    Prefill regime (no autograd)."""
    _check_x(x, "rope_interleaved")
    if x.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.rope_interleaved: x needs unit last stride, got {tuple(x.stride())}")
    c3, s3 = _tables(cos, sin, x, "rope_interleaved")
    return _rope_interleaved_op(x, c3, s3)


def rope_interleaved_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    c3, s3 = _tables(cos, sin, x, "rope_interleaved_ref")
    return apply_rope_interleaved_ref(x, c3, s3)
