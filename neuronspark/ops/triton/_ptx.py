"""Inline-PTX helpers shared by the Triton kernels.

Each helper issues one explicit instruction (or a fixed short sequence) so that a kernel's rounding points are exactly
those of the reference arithmetic it reproduces: IEEE fp32 round-to-nearest-even ``add / sub / mul / div / sqrt / fma``
(no FMA contraction, no flush-to-zero unless named ``_ftz``), exact bf16 <-> fp32 conversions, the ATen sigmoid
(``1 / (1 + expf(-x))`` with libdevice ``expf``), and the fast-math sequences of the reference CUDA kernels.

Kernels import the helpers they call into their own module (Triton resolves a called ``@triton.jit`` function through
the caller's globals), e.g. ``from neuronspark.ops.triton._ptx import _rn_add_f32, _rn_mul_f32``.  The helpers are
grouped by the kernel families that use them; helpers of different groups that issue the same instruction are
interchangeable (the groups only differ in register-constraint spelling).
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice as _libdevice

__all__ = [
    # IEEE fp32 RN arithmetic
    "_rn_mul_f32", "_rn_add_f32", "_rn_div_f32", "_rn_sqrt_f32", "_rn_sub_f32", "_rn_fma_f32", "_floor_f32",
    "_recip_rn_f32", "_sigmoid_rn_f32", "_release_gate_rn_f32", "gate_guard", "GATE_GUARD_MIN_ALPHA",
    # soma read-out group
    "_sr_add_rn", "_sr_sub_rn", "_sr_mul_rn", "_sr_div_rn", "_sr_neg", "_sr_max", "_sr_bf16_to_f32",
    "_sr_f32_to_bf16", "_sr_clamp_min0", "_sr_sigmoid", "_sr_release",
    # PLIF glue group
    "_pg_add", "_pg_sub", "_pg_mul", "_pg_div", "_pg_neg", "_pg_w", "_pg_n", "_pg_sigmoid",
    # decode-step fast-math group
    "_fma_rn_ftz", "_silu_fast_ftz", "_softplus",
    # ring-write group
    "_mul_rn", "_add_rn", "_sub_rn", "_div_rn", "_fma_rn", "_sqrt_rn",
    # approximate exp2
    "_ex2ftz",
]


# ============================================================================================ IEEE fp32 RN arithmetic

@triton.jit
def _rn_mul_f32(left, right):
    """One explicit IEEE FP32 round-to-nearest-even multiply."""
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_add_f32(left, right):
    """One explicit IEEE FP32 round-to-nearest-even addition."""
    return tl.inline_asm_elementwise(
        asm="add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_div_f32(left, right):
    """One explicit IEEE FP32 round-to-nearest-even division (Triton's ``/`` lowers to a non-IEEE form)."""
    return tl.inline_asm_elementwise(
        asm="div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_sqrt_f32(value):
    """One explicit IEEE correctly-rounded FP32 square root (torch.sqrt's result; tl.sqrt lowers to a non-IEEE
    form)."""
    return tl.inline_asm_elementwise(
        asm="sqrt.rn.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_sub_f32(left, right):
    """One explicit IEEE FP32 round-to-nearest-even subtraction (no FMA contraction with a neighbouring multiply)."""
    return tl.inline_asm_elementwise(
        asm="sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rn_fma_f32(a, b, c):
    """One explicit IEEE FP32 fused multiply-add a * b + c, rounded once (round-to-nearest-even)."""
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _floor_f32(value):
    """Exact FP32 floor (cvt.rmi: the integral value toward -inf)."""
    return tl.inline_asm_elementwise(
        asm="cvt.rmi.f32.f32 $0, $1;",
        constraints="=f,f",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _recip_rn_f32(den):
    """The correctly rounded reciprocal div.rn.f32(1, den) for every fp32 ``den``, computed without the division's slow
    path for den >= 2^125.

    For den >= 2^125 (including +inf) the quotient is taken on scaled operands, which stay in the fast path:
      dd = min(den * 2^-126, 2^25)   exact; in [0.5, 4) for finite den
      q  = div.rn(2^23, dd)          normal operands, normal quotient
    1/den = u * 2^-149 with u = 2^23 / dd, and every fp32 value below 2^-125 is an integer multiple of 2^-149, so the
    correctly rounded result is RN_int(u) * 2^-149.  For q >= 2^23 the fp32 grid of q is the integer grid, so k = q.
    Below it |q - u| <= 1/4, so RN_int(u) is floor(q) or floor(q) + 1: the latter exactly when
    2^23 - (floor(q) + 1/2) * dd > 0, whose sign the single rounding of the fma preserves.  +inf clamps to dd = 2^25:
    q = 1/4, k = 0, result +0 = div.rn(1, +inf).  k * 2^-149 is formed as (k * 2^-23) * 2^-126, both products exact.
    den < 2^125 and NaN go through the unchanged division."""
    big = den >= 4.253529586511731e+37                                   # 2^125
    dd = tl.where(big, tl.minimum(_rn_mul_f32(den, 1.1754943508222875e-38), 33554432.0), den)
    q = _rn_div_f32(tl.where(big, 8388608.0, 1.0), dd)
    f = _floor_f32(q)
    t = _rn_fma_f32(-_rn_add_f32(f, 0.5), dd, 8388608.0)
    k = tl.where((t > 0.0) & (q < 8388608.0), _rn_add_f32(f, 1.0), f)
    r = _rn_mul_f32(_rn_mul_f32(k, 1.1920928955078125e-07), 1.1754943508222875e-38)
    return tl.where(big, r, q)


@triton.jit
def _sigmoid_rn_f32(x, GUARD: tl.constexpr = 0):
    """torch.sigmoid for fp32 exactly: 1 / (1 + expf(-x)) with libdevice expf and IEEE RN add/div (tl.sigmoid uses
    the approximate exp/div).

    GUARD=1 takes the reciprocal through _recip_rn_f32: the same correctly rounded result, without the division's slow
    path (1 + e >= 2^125, i.e. x <= -86.64); it pays only where such arguments are frequent (see gate_guard).  GUARD=0
    is the plain instruction sequence."""
    e = _libdevice.exp(-x)
    den = _rn_add_f32(1.0, e)
    if GUARD:
        r = _recip_rn_f32(den)
    else:
        r = _rn_div_f32(1.0, den)
    return r


@triton.jit
def _release_gate_rn_f32(margin, alpha, EXACT: tl.constexpr,
                         GATE_GUARD: tl.constexpr = 0):
    """Surrogate gate of the soft-reset release: EXACT (a non-positive slope) gives the exact step derivative,
    otherwise the RN sigmoid of ``margin * alpha``.  At the threshold the step chooses the zero subgradient."""
    if EXACT:
        return (margin > 0.0).to(tl.float32)
    else:
        return _sigmoid_rn_f32(_rn_mul_f32(margin, alpha), GATE_GUARD)


# Surrogate slope from which the backward kernels evaluate their gate sigmoids with GATE_GUARD=1, per kernel family.
GATE_GUARD_MIN_ALPHA = {"plain": 64.0, "ath": 32.0}


def gate_guard(alpha: float, kind: str = "plain") -> int:
    """GATE_GUARD constexpr of a soft-reset backward launch at surrogate slope ``alpha`` (0 at the model's slope 8.0).

    Both settings compute the same correctly rounded sigmoid; it only decides whether the gates pay extra
    instructions to avoid the IEEE division's slow path."""
    return int(float(alpha) >= GATE_GUARD_MIN_ALPHA[kind])


# ============================================================================================ soma read-out group

@triton.jit
def _sr_add_rn(a, b):
    """One IEEE fp32 round-to-nearest-even addition (no FTZ, no contraction)."""
    return tl.inline_asm_elementwise(asm="add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sr_sub_rn(a, b):
    """One IEEE fp32 round-to-nearest-even subtraction (= ATen's a + (-1) * b)."""
    return tl.inline_asm_elementwise(asm="sub.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sr_mul_rn(a, b):
    """One IEEE fp32 round-to-nearest-even multiplication."""
    return tl.inline_asm_elementwise(asm="mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sr_div_rn(a, b):
    """One IEEE fp32 correctly rounded division (nvcc's default -prec-div=true form of ATen's '/')."""
    return tl.inline_asm_elementwise(asm="div.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sr_neg(a):
    """neg.f32: the sign flip of ATen's ``-a``."""
    return tl.inline_asm_elementwise(asm="neg.f32 $0, $1;", constraints="=f,f", args=[a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sr_max(a, b):
    """max.f32: what ``::max(float, float)`` (fmaxf) compiles to."""
    return tl.inline_asm_elementwise(asm="max.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sr_bf16_to_f32(h):
    """cvt.f32.bf16 (c10::BFloat16 -> float on sm_90+: __bfloat162float); exact."""
    return tl.inline_asm_elementwise(asm="cvt.f32.bf16 $0, $1;", constraints="=f,h", args=[h],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sr_f32_to_bf16(x):
    """cvt.rn.bf16.f32 (float -> c10::BFloat16 on sm_80+: __float2bfloat16); round to nearest even."""
    return tl.inline_asm_elementwise(asm="cvt.rn.bf16.f32 $0, $1;", constraints="=h,f", args=[x],
                                     dtype=tl.bfloat16, is_pure=True, pack=1)


@triton.jit
def _sr_clamp_min0(a):
    """ATen launch_clamp_scalar (Min): isnan(a) ? a : ::max(a, 0.f)."""
    return tl.where(a != a, a, _sr_max(a, 0.0))


@triton.jit
def _sr_sigmoid(x):
    """ATen sigmoid_kernel_cuda for float: one / (one + std::exp(-x)) with libdevice expf, RN add and div."""
    e = _libdevice.exp(_sr_neg(x))
    return _sr_div_rn(1.0, _sr_add_rn(1.0, e))


@triton.jit
def _sr_release(v, th):
    """Bipolar release S = clamp_min(v - th, 0) - clamp_min(-v - th, 0); also returns the two margins a = v - th and
    b = -v - th (the backward's sigmoid arguments before the slope)."""
    a = _sr_sub_rn(v, th)
    b = _sr_sub_rn(_sr_neg(v), th)
    s = _sr_sub_rn(_sr_clamp_min0(a), _sr_clamp_min0(b))
    return s, a, b


# ============================================================================================ PLIF glue group

@triton.jit
def _pg_add(a, b):
    return tl.inline_asm_elementwise(asm="add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pg_sub(a, b):
    return tl.inline_asm_elementwise(asm="sub.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pg_mul(a, b):
    return tl.inline_asm_elementwise(asm="mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pg_div(a, b):
    return tl.inline_asm_elementwise(asm="div.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pg_neg(a):
    return tl.inline_asm_elementwise(asm="neg.f32 $0, $1;", constraints="=f,f", args=[a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pg_w(h):
    return tl.inline_asm_elementwise(asm="cvt.f32.bf16 $0, $1;", constraints="=f,h", args=[h],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pg_n(x):
    return tl.inline_asm_elementwise(asm="cvt.rn.bf16.f32 $0, $1;", constraints="=h,f", args=[x],
                                     dtype=tl.bfloat16, is_pure=True, pack=1)


@triton.jit
def _pg_sigmoid(x):
    """ATen sigmoid functor (float): one / (one + std::exp(-x)) -- libdevice expf, RN add, div.rn."""
    return _pg_div(1.0, _pg_add(1.0, _libdevice.exp(_pg_neg(x))))


# ============================================================================================ decode-step fast-math group

@triton.jit
def _fma_rn_ftz(a, b, c):
    """FFMA.FTZ (nvcc --use_fast_math: a * b + c, one rounding, denormals flushed)"""
    return tl.inline_asm_elementwise(asm="fma.rn.ftz.f32 $0, $1, $2, $3;", constraints="=f,f,f,f", args=[a, b, c],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _silu_fast_ftz(v):
    """The reference conv-update kernel's ``out / (1 + expf(-out))`` under --use_fast_math, as its SASS: FMUL.FTZ(out,
    -log2 e), MUFU.EX2, FADD.FTZ(., 1), MUFU.RCP, FMUL.FTZ(out, .)"""
    return tl.inline_asm_elementwise(
        asm="{\n.reg .f32 t;\nmul.rn.ftz.f32 t, $1, 0fBFB8AA3B;\nex2.approx.ftz.f32 t, t;\n"
            "add.rn.ftz.f32 t, t, 0f3F800000;\nrcp.approx.ftz.f32 t, t;\nmul.rn.ftz.f32 $0, $1, t;\n}",
        constraints="=f,f", args=[v], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _softplus(dt):
    # the write-strength softplus: log(exp(dt) + 1) (callers take dt itself above 20)
    return tl.math.log(tl.math.exp(dt) + 1)


# ============================================================================================ ring-write group

@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _div_rn(a, b):
    return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _fma_rn(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=r,r,r,r", [a, b, c], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _sqrt_rn(a):
    return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=r,r", [a], dtype=tl.float32, is_pure=True, pack=1)


# ============================================================================================ approximate exp2

@triton.jit
def _ex2ftz(t):
    """ex2.approx.ftz.f32: the MUFU.EX2 that tl.exp issues after its log2(e) multiply, without the non-ftz wrapper
    ptxas emits around it; equal to tl.exp(x) when called on x * 1.4426950408889634 except that results below 2^-126
    (x < -87.3) are flushed to zero."""
    return tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [t], dtype=tl.float32, is_pure=True, pack=1)
