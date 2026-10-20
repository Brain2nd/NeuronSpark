"""The memory layer's qkv short convolution: depthwise, kernel 4, causal, no bias, on the (T, B, C) projection.

Operators (registry names):

* ``short_conv_train(qkv, weight)`` (training): ``out[t, b, c] = bf16(fma(w3, x[t], fma(w2, x[t-1], fma(w1, x[t-2],
  fma(w0, x[t-3], +0)))))`` with ``w = bf16(weight[c, 0, :])`` widened and ``x[t < 0] = 0``; ``qkv`` bf16 (T, B, C)
  with channel stride 1, ``weight`` fp32 (C, 1, 4); returns (T, B, C) bf16 contiguous (T-major).  Backward: the input
  gradient by the same fma chain in the reference tap order, and the fp32 weight gradient ``dW[c, 0, k] =
  sum_{b, t} x[t - 3 + k] dy[t]`` returned as ``weight.grad``.
* ``short_conv_silu_prefill(x, hist, weight)`` (prefill, T >= 3, no autograd): the same chain with the carried three taps
  ``hist`` (B, C, 3) (oldest first) standing for ``x[t < 0]``; ``hist`` is rolled in place to the call's last three
  inputs.  Returns (T, B, C) bf16 contiguous.
* ``short_conv_step(x, hist, weight)`` (decode, T == 1, no autograd): ``y = w0 h0 + w1 h1 + w2 h2 + w3 x`` as the fma
  chain, ``hist`` rolled in place to ``(h1, h2, x)``; ``x`` (B, C) bf16 with unit row stride -> (B, C) bf16.

Numerics contract
-----------------
Forward (all regimes): the fp32 fma chain from +0 in tap order (oldest tap first) on bf16 operands, one rounding to
bf16 -- the arithmetic of PyTorch's depthwise conv under bf16 autocast.  Training backward: the input gradient by
the same chain on the upstream gradient (the reference tap order); the weight gradient as per-tile fp32
partial sums (the products of two bf16 values are exact) into a ``(programs, 4 C)`` workspace, summed per (c, k) by a
finish kernel in a fixed order (deterministic; fp32 reduction-order class against other orders).  The inference
history is fp32 holding bf16-exact values; the taps are rounded to bf16 explicitly (the staging copy's rounding).

Weight gradient: the fp32 ``(C, 1, 4)`` gradient is returned as the weight's gradient (written into a zero-filled
buffer by the finish kernel; several uses of the weight are summed in fp32 by autograd).

Launchers are ``torch.library`` custom ops (``neuronspark::short_conv_*``; the inference ops declare their in-place
update of ``hist``) with fake implementations.  Reference of the training form: the module's
``qkv_conv(F.pad(qkv.permute(1, 2, 0), (3, 0))).permute(2, 0, 1)``.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (jit helpers resolved through this module's globals)
    _rn_fma_f32 as _scfma,
    _sr_bf16_to_f32 as _scw,
    _sr_f32_to_bf16 as _scn,
)

__all__ = ["ShortConvFn", "short_conv_train", "short_conv_train_ref", "short_conv_silu_prefill", "short_conv_silu_prefill_ref",
           "short_conv_step", "short_conv_step_ref"]

K = 4
# ------------------------------------------------------------------------------------------------ fixed constants
TRAIN_FWD_CFG = (64, 128, 4)     # training forward (BT positions, BC channels, warps), output layout T-major
TRAIN_BWD_CFG = (64, 64, 2)      # training backward (input gradient + weight-gradient partials)
WFIN_CFG = (16, 128, 4)          # weight-gradient finish (PB partial rows per step, BJ (c, k) entries, warps)
PREFILL_BT, PREFILL_BC, PREFILL_WARPS = 32, 256, 4
STEP_BLOCK, STEP_WARPS = 1024, 4


# ================================================================================================ training kernels
@triton.jit
def _sc_fwd_kernel(X, W, OUT, T, C, sxt, sxb, sot, sob, soc, BT: tl.constexpr, BC: tl.constexpr):
    pt = tl.program_id(0)
    b = tl.program_id(1)
    pc = tl.program_id(2)
    t = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    cm = c < C
    tm = t < T
    w0 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 0, mask=cm, other=0.0))[None, :], (BT, BC))
    w1 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 1, mask=cm, other=0.0))[None, :], (BT, BC))
    w2 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 2, mask=cm, other=0.0))[None, :], (BT, BC))
    w3 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 3, mask=cm, other=0.0))[None, :], (BT, BC))
    xb = X + b * sxb + c[None, :]
    r0 = t - 3
    r1 = t - 2
    r2 = t - 1
    x0 = _scw(tl.load(xb + r0[:, None] * sxt, mask=((r0 >= 0) & tm)[:, None] & cm[None, :], other=0.0))
    x1 = _scw(tl.load(xb + r1[:, None] * sxt, mask=((r1 >= 0) & tm)[:, None] & cm[None, :], other=0.0))
    x2 = _scw(tl.load(xb + r2[:, None] * sxt, mask=((r2 >= 0) & tm)[:, None] & cm[None, :], other=0.0))
    x3 = _scw(tl.load(xb + t[:, None] * sxt, mask=tm[:, None] & cm[None, :], other=0.0))
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    acc = _scfma(w0, x0, acc)
    acc = _scfma(w1, x1, acc)
    acc = _scfma(w2, x2, acc)
    acc = _scfma(w3, x3, acc)
    tl.store(OUT + b * sob + t[:, None] * sot + c[None, :] * soc, _scn(acc), mask=tm[:, None] & cm[None, :])


@triton.jit
def _sc_bwd_kernel(X, W, DY, DX, PART, T, C, NTB, sxt, sxb, sdt, sdb, sdc, sgt, sgb,
                   BT: tl.constexpr, BC: tl.constexpr):
    pt = tl.program_id(0)
    b = tl.program_id(1)
    pc = tl.program_id(2)
    t = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    cm = c < C
    tm = t < T
    w0 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 0, mask=cm, other=0.0))[None, :], (BT, BC))
    w1 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 1, mask=cm, other=0.0))[None, :], (BT, BC))
    w2 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 2, mask=cm, other=0.0))[None, :], (BT, BC))
    w3 = tl.broadcast_to(_scw(tl.load(W + c * 4 + 3, mask=cm, other=0.0))[None, :], (BT, BC))
    # upstream gradient rows t .. t + 3 (index >= T masked to 0)
    db = DY + b * sdb + c[None, :] * sdc
    s1 = t + 1
    s2 = t + 2
    s3 = t + 3
    d0 = _scw(tl.load(db + t[:, None] * sdt, mask=tm[:, None] & cm[None, :], other=0.0))
    d1 = _scw(tl.load(db + s1[:, None] * sdt, mask=(s1 < T)[:, None] & cm[None, :], other=0.0))
    d2 = _scw(tl.load(db + s2[:, None] * sdt, mask=(s2 < T)[:, None] & cm[None, :], other=0.0))
    d3 = _scw(tl.load(db + s3[:, None] * sdt, mask=(s3 < T)[:, None] & cm[None, :], other=0.0))
    # input gradient: the reference tap order (kw = 0 first: w0 * dy[t + 3])
    acc = tl.zeros((BT, BC), dtype=tl.float32)
    acc = _scfma(w0, d3, acc)
    acc = _scfma(w1, d2, acc)
    acc = _scfma(w2, d1, acc)
    acc = _scfma(w3, d0, acc)
    tl.store(DX + b * sgb + t[:, None] * sgt + c[None, :], _scn(acc), mask=tm[:, None] & cm[None, :])
    # weight-gradient partials of this tile: sum over its rows t < T of x[t - 3 + k] * dy[t] (d0 is 0 for t >= T)
    xb = X + b * sxb + c[None, :]
    r0 = t - 3
    r1 = t - 2
    r2 = t - 1
    x0 = _scw(tl.load(xb + r0[:, None] * sxt, mask=((r0 >= 0) & tm)[:, None] & cm[None, :], other=0.0))
    x1 = _scw(tl.load(xb + r1[:, None] * sxt, mask=((r1 >= 0) & tm)[:, None] & cm[None, :], other=0.0))
    x2 = _scw(tl.load(xb + r2[:, None] * sxt, mask=((r2 >= 0) & tm)[:, None] & cm[None, :], other=0.0))
    x3 = _scw(tl.load(xb + t[:, None] * sxt, mask=tm[:, None] & cm[None, :], other=0.0))
    a0 = tl.sum(x0 * d0, axis=0)
    a1 = tl.sum(x1 * d0, axis=0)
    a2 = tl.sum(x2 * d0, axis=0)
    a3 = tl.sum(x3 * d0, axis=0)
    pb = PART + (b * NTB + pt) * (4 * C) + c * 4
    tl.store(pb + 0, a0, mask=cm)
    tl.store(pb + 1, a1, mask=cm)
    tl.store(pb + 2, a2, mask=cm)
    tl.store(pb + 3, a3, mask=cm)


@triton.jit
def _sc_wfin_kernel(PART, MAIN, P, N4, PB: tl.constexpr, BJ: tl.constexpr):
    """MAIN[j] += sum over the P partial rows of PART[:, j] (row chunks of PB, each summed by tl.sum, in order)."""
    j = tl.program_id(0) * BJ + tl.arange(0, BJ)
    jm = j < N4
    acc = tl.zeros((BJ,), dtype=tl.float32)
    for p0 in range(0, P, PB):
        p = p0 + tl.arange(0, PB)
        v = tl.load(PART + p[:, None] * N4 + j[None, :], mask=(p < P)[:, None] & jm[None, :], other=0.0)
        acc += tl.sum(v, axis=0)
    m = tl.load(MAIN + j, mask=jm, other=0.0)
    tl.store(MAIN + j, m + acc, mask=jm)


# ================================================================================================ inference kernels
@triton.jit
def _bf16(x):
    """round an fp32 value to the nearest bf16 (ties to even) and widen it back, through the bit pattern (the natural
    round trip can be folded away by the compiler in some expression contexts); finite values, an infinity stays
    itself"""
    u = x.to(tl.int32, bitcast=True)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    u = u & -65536                     # 0xFFFF0000 as a signed int32
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _conv_step_kernel(X, HIST, WT, OUT, s_x_b, s_h_b, s_h_c, s_out_b, C, BLOCK: tl.constexpr):
    """one row b, BLOCK channels: y = w0 h0 + w1 h1 + w2 h2 + w3 x as the fp32 FMA chain of PyTorch's depthwise conv on
    bf16 operands (the staged history and the weight in bf16), y rounded to bf16; the history rolled: (h1, h2, x)"""
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    x16 = tl.load(X + b * s_x_b + c, mask=m, other=0.0)                       # bf16
    hp = HIST + b * s_h_b + c * s_h_c
    h0 = _bf16(tl.load(hp, mask=m, other=0.0).to(tl.float32))                # the staging copy's bf16 rounding
    h1 = _bf16(tl.load(hp + 1, mask=m, other=0.0).to(tl.float32))
    h2 = _bf16(tl.load(hp + 2, mask=m, other=0.0).to(tl.float32))
    x = x16.to(tl.float32)
    w0 = _bf16(tl.load(WT + c * 4 + 0, mask=m, other=0.0).to(tl.float32))     # autocast's bf16 weight
    w1 = _bf16(tl.load(WT + c * 4 + 1, mask=m, other=0.0).to(tl.float32))
    w2 = _bf16(tl.load(WT + c * 4 + 2, mask=m, other=0.0).to(tl.float32))
    w3 = _bf16(tl.load(WT + c * 4 + 3, mask=m, other=0.0).to(tl.float32))
    acc = tl.zeros_like(x)
    acc = tl.math.fma(w0, h0, acc)
    acc = tl.math.fma(w1, h1, acc)
    acc = tl.math.fma(w2, h2, acc)
    acc = tl.math.fma(w3, x, acc)
    tl.store(OUT + b * s_out_b + c, acc.to(tl.bfloat16), mask=m)
    hdt = HIST.dtype.element_ty
    tl.store(hp, h1.to(hdt), mask=m)
    tl.store(hp + 1, h2.to(hdt), mask=m)
    tl.store(hp + 2, x.to(hdt), mask=m)


@triton.jit
def _conv_prefill_kernel(X, HIST, WT, OUT, s_x_t, s_x_b, s_h_b, s_h_c, s_o_t, s_o_b, T, C,
                         BT: tl.constexpr, BC: tl.constexpr):
    """one (b, t-tile, c-tile): y[t, c] = w3 x[t] + w2 x[t-1] + w1 x[t-2] + w0 x[t-3] as the fp32 FMA chain of
    PyTorch's depthwise conv (tap 0 oldest) on bf16 operands, x[t < 0] = the carried history (bf16-valued), rounded to
    bf16, then its SiLU (fp32 with Triton's exponential and division, one rounding to bf16); the last three rows of the
    call roll into the history (written by the tile that holds t = T - 1)"""
    b = tl.program_id(0)
    pt = tl.program_id(1)
    pc = tl.program_id(2)
    t = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    tm = t < T
    cm = c < C
    m = tm[:, None] & cm[None, :]
    hp = HIST + b * s_h_b + c * s_h_c
    w0 = _bf16(tl.load(WT + c * 4 + 0, mask=cm, other=0.0).to(tl.float32))
    w1 = _bf16(tl.load(WT + c * 4 + 1, mask=cm, other=0.0).to(tl.float32))
    w2 = _bf16(tl.load(WT + c * 4 + 2, mask=cm, other=0.0).to(tl.float32))
    w3 = _bf16(tl.load(WT + c * 4 + 3, mask=cm, other=0.0).to(tl.float32))
    h0 = _bf16(tl.load(hp + 0, mask=cm, other=0.0).to(tl.float32))
    h1 = _bf16(tl.load(hp + 1, mask=cm, other=0.0).to(tl.float32))
    h2 = _bf16(tl.load(hp + 2, mask=cm, other=0.0).to(tl.float32))
    xb = X + b * s_x_b + c[None, :]
    x0 = tl.load(xb + t[:, None] * s_x_t, mask=m, other=0.0).to(tl.float32)                             # x[t]
    t1 = t - 1
    x1 = tl.where((t1 >= 0)[:, None], tl.load(xb + t1[:, None] * s_x_t, mask=m & (t1 >= 0)[:, None], other=0.0).to(tl.float32), h2[None, :])
    t2 = t - 2
    x2 = tl.where((t2 >= 0)[:, None], tl.load(xb + t2[:, None] * s_x_t, mask=m & (t2 >= 0)[:, None], other=0.0).to(tl.float32), h1[None, :])
    t3 = t - 3
    x3 = tl.where((t3 >= 0)[:, None], tl.load(xb + t3[:, None] * s_x_t, mask=m & (t3 >= 0)[:, None], other=0.0).to(tl.float32), h0[None, :])
    # x[t-2] for t == 1 is history tap 2 (h2), for t == 0 it is h1; x[t-3] for t == 2 / 1 / 0 is h2 / h1 / h0
    x2 = tl.where((t == 1)[:, None], h2[None, :], x2)
    x3 = tl.where((t == 2)[:, None], h2[None, :], tl.where((t == 1)[:, None], h1[None, :], x3))
    acc = tl.zeros_like(x0)
    acc = tl.math.fma(w0[None, :], x3, acc)
    acc = tl.math.fma(w1[None, :], x2, acc)
    acc = tl.math.fma(w2[None, :], x1, acc)
    acc = tl.math.fma(w3[None, :], x0, acc)
    of = acc.to(tl.bfloat16).to(tl.float32)
    o16 = (of / (1.0 + tl.exp(-of))).to(tl.bfloat16)                    # SiLU of the bf16 conv output (fp32, one rounding)
    tl.store(OUT + t[:, None] * s_o_t + b * s_o_b + c[None, :], o16, mask=m)
    # the history roll: x[T-3], x[T-2], x[T-1] (the tile holding t = T - 1)
    last = (T - 1) // BT
    if pt == last:
        hdt = HIST.dtype.element_ty
        for j in tl.static_range(3):
            tt = T - 3 + j
            v = tl.load(X + b * s_x_b + tt * s_x_t + c, mask=cm, other=0.0).to(tl.float32)          # T >= 3: tt >= 0
            tl.store(hp + j, v.to(hdt), mask=cm)


# ================================================================================================ training
def _check_train_x(x: torch.Tensor, wb: torch.Tensor):
    if x.device.type != "cuda" or wb.device != x.device:
        raise RuntimeError(f"neuronspark.short_conv_train: CUDA tensors on one device required, got x {x.device}, "
                           f"weight {wb.device}")
    if x.dtype != torch.bfloat16 or x.dim() != 3 or x.stride(2) != 1:
        raise RuntimeError(f"neuronspark.short_conv_train: x must be a 3-D bf16 (T, B, C) tensor with channel stride 1, "
                           f"got {tuple(x.shape)} {x.dtype} strides {tuple(x.stride())}")
    T, B, C = (int(v) for v in x.shape)
    if T < 1 or B < 1 or C < 1:
        raise RuntimeError(f"neuronspark.short_conv_train: empty x {tuple(x.shape)}")
    if tuple(wb.shape) != (C, 1, K) or wb.dtype != torch.bfloat16 or not wb.is_contiguous():
        raise RuntimeError(f"neuronspark.short_conv_train: the bf16 weight must be a contiguous ({C}, 1, {K}) tensor, "
                           f"got {tuple(wb.shape)} {wb.dtype}")
    span = (T - 1) * abs(x.stride(0)) + (B - 1) * abs(x.stride(1)) + C
    if T * B * C >= 2 ** 31 or span >= 2 ** 31:
        raise RuntimeError("neuronspark.short_conv_train: offsets beyond the kernels' int32 range")
    return T, B, C


@torch.library.custom_op("neuronspark::short_conv_fwd", mutates_args=(), schema="(Tensor x, Tensor wb) -> Tensor")
def _sc_fwd(x, wb):
    """(T, B, C) bf16 contiguous causal depthwise conv of x (T, B, C) with the bf16 (C, 1, 4) weight."""
    T, B, C = _check_train_x(x, wb)
    BT, BC, nw = TRAIN_FWD_CFG
    out = torch.empty((T, B, C), dtype=torch.bfloat16, device=x.device)
    sot, sob, soc = B * C, C, 1
    grid = (triton.cdiv(T, BT), B, triton.cdiv(C, BC))
    with torch.cuda.device(x.device):
        _sc_fwd_kernel[grid](x, wb, out, T, C, x.stride(0), x.stride(1), sot, sob, soc, BT=BT, BC=BC, num_warps=nw,
                             enable_fp_fusion=False)
    return out


@_sc_fwd.register_fake
def _sc_fwd_fake(x, wb):
    T, B, C = x.shape
    return torch.empty((T, B, C), dtype=torch.bfloat16, device=x.device)


@torch.library.custom_op("neuronspark::short_conv_bwd", mutates_args=(),
                         schema="(Tensor x, Tensor wb, Tensor dy) -> (Tensor, Tensor)")
def _sc_bwd(x, wb, dy):
    """(dx (T, B, C) bf16 contiguous, dW (C, 1, 4) fp32) for the upstream gradient dy ((T, B, C), any strides)."""
    T, B, C = _check_train_x(x, wb)
    if dy.dtype != torch.bfloat16 or tuple(dy.shape) != (T, B, C):
        raise RuntimeError(f"neuronspark.short_conv_train backward: the upstream gradient must be bf16 {(T, B, C)}, got "
                           f"{dy.dtype} {tuple(dy.shape)}")
    dspan = sum((n - 1) * abs(s) for n, s in zip(dy.shape, dy.stride())) + 1
    if dspan >= 2 ** 31:
        raise RuntimeError("neuronspark.short_conv_train backward: gradient offsets beyond the kernels' int32 range")
    BT, BC, nw = TRAIN_BWD_CFG
    PB, BJ, nwf = WFIN_CFG
    ntb = triton.cdiv(T, BT)
    P = B * ntb
    dx = torch.empty((T, B, C), dtype=torch.bfloat16, device=x.device)
    part = torch.empty((P, C * K), dtype=torch.float32, device=x.device)
    dw = torch.zeros((C, 1, K), dtype=torch.float32, device=x.device)
    grid = (ntb, B, triton.cdiv(C, BC))
    with torch.cuda.device(x.device):
        _sc_bwd_kernel[grid](x, wb, dy, dx, part, T, C, ntb, x.stride(0), x.stride(1), dy.stride(0), dy.stride(1),
                             dy.stride(2), dx.stride(0), dx.stride(1), BT=BT, BC=BC, num_warps=nw, enable_fp_fusion=False)
        _sc_wfin_kernel[(triton.cdiv(C * K, BJ),)](part, dw, P, C * K, PB=PB, BJ=BJ, num_warps=nwf)
    return dx, dw


@_sc_bwd.register_fake
def _sc_bwd_fake(x, wb, dy):
    T, B, C = x.shape
    return (torch.empty((T, B, C), dtype=torch.bfloat16, device=x.device),
            torch.empty((C, 1, K), dtype=torch.float32, device=x.device))


class ShortConvFn(torch.autograd.Function):
    """The causal depthwise k = 4 conv on the (T, B, C) qkv -> (T, B, C) contiguous; backward the input gradient and the
    fp32 (C, 1, 4) weight gradient."""

    @staticmethod
    def forward(ctx, x, weight):
        wb = weight.to(torch.bfloat16)                 # (C, 1, 4): one RN cast (autocast's cast of the conv weight)
        out = _sc_fwd(x, wb)
        ctx.save_for_backward(x, wb)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, wb = ctx.saved_tensors
        with torch.autocast(x.device.type, enabled=False):
            dx, dw = _sc_bwd(x, wb, dy)
        return (dx if ctx.needs_input_grad[0] else None), (dw if ctx.needs_input_grad[1] else None)


def _check_weight(weight: torch.Tensor, C: int, what: str) -> None:
    if not (isinstance(weight, torch.Tensor) and tuple(weight.shape) == (C, 1, K) and weight.is_contiguous()
            and weight.dtype == torch.float32):
        raise RuntimeError(f"neuronspark.{what}: weight must be a contiguous fp32 ({C}, 1, {K}) tensor (a 4-tap depthwise "
                           f"conv), got {getattr(weight, 'dtype', None)} {tuple(getattr(weight, 'shape', ()))}")


def short_conv_train(qkv: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The causal depthwise k = 4 conv of the bf16 (T, B, C) projection (channel stride 1) with the fp32 (C, 1, 4)
    weight -> (T, B, C) bf16 contiguous; the weight's gradient is fp32.  Training regime."""
    require_cuda(qkv, "short_conv_train")
    if qkv.dim() != 3:
        raise RuntimeError(f"neuronspark.short_conv_train: qkv must be (T, B, C), got {tuple(qkv.shape)}")
    _check_weight(weight, int(qkv.shape[2]), "short_conv_train")
    return ShortConvFn.apply(qkv, weight)


def short_conv_train_ref(qkv: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The fp32 tap chain on the bf16-rounded operands (zero left pad), rounded to bf16 once; the weight enters as
    its bf16 value with a straight-through fp32 gradient (the reference of the fp32 weight gradient)."""
    T = int(qkv.shape[0])
    w32 = weight.float()
    wr = w32 + (weight.detach().to(torch.bfloat16).float() - w32.detach())           # (C, 1, 4) bf16-valued
    x = qkv.float()
    xp = F.pad(x, (0, 0, 0, 0, K - 1, 0))                                              # (T + 3, B, C), zeros first
    acc = torch.zeros_like(x)
    for k in range(K):
        acc = acc + wr[:, 0, k] * xp[k:k + T]
    return acc.to(torch.bfloat16)


# ================================================================================================ inference
def _check_hist(hist: torch.Tensor, B: int, C: int, what: str) -> None:
    if (not isinstance(hist, torch.Tensor) or tuple(hist.shape) != (B, C, 3) or hist.stride(2) != 1
            or hist.stride(1) != 3):
        raise RuntimeError(f"neuronspark.{what}: hist {tuple(getattr(hist, 'shape', ()))} strides "
                           f"{tuple(getattr(hist, 'stride', lambda: ())())} (need (B, C, 3) with contiguous taps)")


@torch.library.custom_op("neuronspark::short_conv_silu_prefill", mutates_args=("hist",),
                         schema="(Tensor x, Tensor(a!) hist, Tensor weight) -> Tensor")
def _sc_prefill(x, hist, weight):
    T, B, C = (int(v) for v in x.shape)
    out = torch.empty((T, B, C), dtype=torch.bfloat16, device=x.device)
    _conv_prefill_kernel[(B, triton.cdiv(T, PREFILL_BT), triton.cdiv(C, PREFILL_BC))](
        x, hist, weight, out, x.stride(0), x.stride(1), hist.stride(0), hist.stride(1), out.stride(0), out.stride(1),
        T, C, BT=PREFILL_BT, BC=PREFILL_BC, num_warps=PREFILL_WARPS)
    return out


@_sc_prefill.register_fake
def _sc_prefill_fake(x, hist, weight):
    return torch.empty(tuple(x.shape), dtype=torch.bfloat16, device=x.device)


def short_conv_silu_prefill(x: torch.Tensor, hist: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The causal conv of a prefill call followed by the SiLU of q / k / v: ``x`` (T, B, C) bf16 with unit inner stride
    and T >= 3, ``hist`` (B, C, 3) the carried taps (rolled in place to the call's last three inputs), ``weight`` fp32
    (C, 1, 4) -> (T, B, C) bf16 contiguous ``silu(conv(x))``.  No autograd."""
    require_cuda(x, "short_conv_silu_prefill")
    if x.dim() != 3 or x.dtype != torch.bfloat16 or x.stride(2) != 1 or int(x.shape[0]) < 3:
        raise RuntimeError(f"neuronspark.short_conv_silu_prefill: x {tuple(x.shape)} {x.dtype} strides "
                           f"{tuple(x.stride())} (bf16, unit inner stride, T >= 3)")
    T, B, C = (int(v) for v in x.shape)
    _check_hist(hist, B, C, "short_conv_silu_prefill")
    _check_weight(weight, C, "short_conv_silu_prefill")
    return _sc_prefill(x, hist, weight)


def short_conv_silu_prefill_ref(x: torch.Tensor, hist: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The torch path at T > 1: (T, B, C) -> (B, C, T), cat with the bf16 history, F.conv1d on bf16 operands, the
    roll (hist <- the last 3 taps of the staged input, in place), F.silu of the bf16 output."""
    T, B, C = x.shape
    xb = x.permute(1, 2, 0)
    full = torch.cat([hist.to(torch.bfloat16), xb], dim=2)
    out = F.conv1d(full, weight.to(torch.bfloat16), groups=C)                                   # (B, C, T)
    hist.copy_(full[:, :, full.shape[2] - 3:].to(hist.dtype))
    return F.silu(out.permute(2, 0, 1))


@torch.library.custom_op("neuronspark::short_conv_step", mutates_args=("hist",),
                         schema="(Tensor x, Tensor(a!) hist, Tensor weight) -> Tensor")
def _sc_step(x, hist, weight):
    B, C = int(x.shape[0]), int(x.shape[1])
    out = torch.empty((B, C), dtype=torch.bfloat16, device=x.device)
    _conv_step_kernel[(B, triton.cdiv(C, STEP_BLOCK))](x, hist, weight, out, x.stride(0), hist.stride(0),
                                                       hist.stride(1), out.stride(0), C, BLOCK=STEP_BLOCK,
                                                       num_warps=STEP_WARPS)
    return out


@_sc_step.register_fake
def _sc_step_fake(x, hist, weight):
    return torch.empty((x.shape[0], x.shape[1]), dtype=torch.bfloat16, device=x.device)


def short_conv_step(x: torch.Tensor, hist: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The causal conv of a decode step: ``x`` (B, C) bf16 with unit row stride (the step's projection), ``hist``
    (B, C, 3) the carried taps (rolled in place to (h1, h2, x)), ``weight`` fp32 (C, 1, 4) -> (B, C) bf16
    contiguous.  No autograd."""
    require_cuda(x, "short_conv_step")
    if x.dim() != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise RuntimeError("neuronspark.short_conv_step: x must be (B, C) bf16 with unit row stride")
    B, C = int(x.shape[0]), int(x.shape[1])
    _check_hist(hist, B, C, "short_conv_step")
    _check_weight(weight, C, "short_conv_step")
    return _sc_step(x, hist, weight)


def short_conv_step_ref(x: torch.Tensor, hist: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The torch path's expressions (the staging copies, F.conv1d on bf16 operands, the roll in place)."""
    C = x.shape[1]
    full = torch.cat([hist.to(torch.bfloat16), x.unsqueeze(-1)], dim=2)          # (B, C, 4) bf16
    out = F.conv1d(full, weight.to(torch.bfloat16), groups=C)                    # (B, C, 1) bf16
    hist.copy_(full[:, :, 1:].to(hist.dtype))
    return out[:, :, 0]
