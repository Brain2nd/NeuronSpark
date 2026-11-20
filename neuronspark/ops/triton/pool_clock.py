"""The clock increments of a hippocampal layer's pool (training): from the decay logits to the gained mean log decay.

``clock_increments(logit, rate, gains, base, floor)`` -> (T, B, H) fp32::

    a   = exp(-float(rate[c]) softplus(float(logit[t, b, c]))) base          the decay per channel, fp32
    la  = mean_c log(max(a, floor))                                           (T, B), shared by the heads
    inc = la gains[h]                                                         (T, B, H)

``logit`` bf16 (T, B, C) (the decay projection of the layer input), ``rate`` bf16 (C,) (the rounded exp of the rate
exponent), ``gains`` fp32 (H,) (the heads' eligibility slopes), ``base`` the base decay, ``floor`` the decay floor.  The
cumulative sum of ``inc`` over the tokens is the pool's clock (:func:`neuronspark.ops.triton.cumsum.cumsum0`).
Gradients: logit (bf16), rate (bf16), gains (fp32).

One forward and one backward kernel (plus the two reductions of the backward's partial sums) for the torch chain of the
module (the cast, softplus, product, exp, product, clamp, log, mean, the gain product and their backward nodes: about
30 launches per layer).  One program is CLOCK_BR rows (t, b) with all C channels, so the mean and its backward are row
sums in registers; the rate's gradient is per-program column sums and the gains' gradient per-program sums, each
reduced by one torch sum (fp32 reduction-order class).

Numerics: fp32 throughout, as the chain (libdevice ``log1pf`` / ``expf`` / ``logf`` for the softplus, the decay and the
log; the softplus is the identity above its threshold 20); the logit's and the rate's gradients are rounded to bf16
once, where the chain casts them.  The rule of :mod:`neuronspark.utils.testing` against
:func:`clock_increments_ref`.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _sr_bf16_to_f32 as _w, _sr_f32_to_bf16 as _n  # noqa: F401  (jit helpers)

__all__ = ["ClockIncrementsFn", "clock_increments", "clock_increments_ref", "CLOCK_BR", "CLOCK_WARPS", "MAX_HEADS"]

# ------------------------------------------------------------------------------------------------ fixed constants
CLOCK_BR, CLOCK_WARPS = 16, 4         # rows (t, b) per program, warps (forward and backward)
MAX_HEADS = 16                        # head lanes of a program
SOFTPLUS_THRESHOLD = 20.0             # F.softplus's default threshold (beta 1)


# ================================================================================================ kernels
@triton.jit
def _decay(LOGIT, RATE, row, rm, c, cm, C, BASE, THR):
    """(x, softplus(x), a = exp(-rate softplus(x)) base) of the (BR, BD) tile of rows ``row``; LOGIT contiguous (R, C)."""
    m = rm[:, None] & cm[None, :]
    x = _w(tl.load(LOGIT + (row[:, None] * C + c[None, :]), mask=m, other=0.0))
    sp = tl.where(x > THR, x, libdevice.log1p(tl.exp(x)))
    rate = _w(tl.load(RATE + c, mask=cm, other=0.0))
    a = tl.exp(-(rate[None, :] * sp)) * BASE
    return m, x, sp, rate, a


@triton.jit
def _clock_fwd_kernel(LOGIT, RATE, GAINS, INC, R, C, H, BASE, FLOOR, INVC, THR,
                      BR: tl.constexpr, BD: tl.constexpr, HP: tl.constexpr):
    """BR rows: inc[row, h] = gains[h] mean_c log(max(a, floor)); INC contiguous (R, H)."""
    row = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    rm = row < R
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < C
    m, x, sp, rate, a = _decay(LOGIT, RATE, row, rm, c, cm, C, BASE, THR)
    la = tl.sum(tl.where(m, tl.log(tl.maximum(a, FLOOR)), 0.0), axis=1) * INVC
    h = tl.arange(0, HP).to(tl.int64)
    hm = h < H
    g = tl.load(GAINS + h, mask=hm, other=0.0)
    tl.store(INC + (row[:, None] * H + h[None, :]), la[:, None] * g[None, :], mask=rm[:, None] & hm[None, :])


@triton.jit
def _clock_bwd_kernel(DINC, LOGIT, RATE, GAINS, DLOGIT, PR, PG, R, C, H, BASE, FLOOR, INVC, THR,
                      BR: tl.constexpr, BD: tl.constexpr, HP: tl.constexpr):
    """BR rows, from DINC (R, H) fp32: the logit's gradient (bf16, contiguous (R, C)), PR[prog, c] = the program's
    sum of the rate's gradient, PG[prog, h] = the program's sum of dinc[row, h] la[row]."""
    prog = tl.program_id(0).to(tl.int64)
    row = (prog * BR + tl.arange(0, BR)).to(tl.int64)
    rm = row < R
    c = tl.arange(0, BD).to(tl.int64)
    cm = c < C
    m, x, sp, rate, a = _decay(LOGIT, RATE, row, rm, c, cm, C, BASE, THR)
    ac = tl.maximum(a, FLOOR)
    la = tl.sum(tl.where(m, tl.log(ac), 0.0), axis=1) * INVC
    h = tl.arange(0, HP).to(tl.int64)
    hm = h < H
    mh = rm[:, None] & hm[None, :]
    g = tl.load(GAINS + h, mask=hm, other=0.0)
    dinc = tl.load(DINC + (row[:, None] * H + h[None, :]), mask=mh, other=0.0)
    tl.store(PG + (prog * HP + h), tl.sum(dinc * la[:, None], axis=0), mask=hm)
    dla = tl.sum(dinc * g[None, :], axis=1)                              # the gain product's backward
    da = tl.where(a >= FLOOR, (dla * INVC)[:, None] / ac, 0.0)           # mean, log and clamp_min backward
    du = da * a                                                          # the base product's and the exp's backward
    du = tl.where(m, du, 0.0)
    tl.store(PR + (prog * C + c), tl.sum(-(du * sp), axis=0), mask=cm)   # u = -rate sp
    dsp = -(rate[None, :] * du)
    z = tl.exp(x)                                                        # softplus backward as torch forms it:
    dx = tl.where(x > THR, dsp, dsp * z / (z + 1.0))                     # g z / (z + 1), z = exp(x)
    tl.store(DLOGIT + (row[:, None] * C + c[None, :]), _n(dx), mask=m)


# ================================================================================================ launchers
def _dims(logit: torch.Tensor, gains: torch.Tensor) -> Tuple[int, int, int, int, int]:
    T, B, C = (int(s) for s in logit.shape)
    return T, B, C, int(gains.shape[0]), int(triton.next_power_of_2(C))


class ClockIncrementsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logit, rate, gains, base, floor):
        T, B, C, H, bd = _dims(logit, gains)
        inc = torch.empty((T, B, H), dtype=torch.float32, device=logit.device)
        _clock_fwd_kernel[(triton.cdiv(T * B, CLOCK_BR),)](
            logit, rate, gains, inc, T * B, C, H, base, floor, 1.0 / C, SOFTPLUS_THRESHOLD,
            BR=CLOCK_BR, BD=bd, HP=MAX_HEADS, num_warps=CLOCK_WARPS, enable_fp_fusion=False)
        ctx.save_for_backward(logit, rate, gains)
        ctx.consts = (float(base), float(floor))
        return inc

    @staticmethod
    def backward(ctx, dinc):
        logit, rate, gains = ctx.saved_tensors
        base, floor = ctx.consts
        T, B, C, H, bd = _dims(logit, gains)
        dev = logit.device
        dinc = dinc.contiguous()
        dlogit = torch.empty((T, B, C), dtype=torch.bfloat16, device=dev)
        nprog = triton.cdiv(T * B, CLOCK_BR)
        pr = torch.empty((nprog, C), dtype=torch.float32, device=dev)
        pg = torch.empty((nprog, MAX_HEADS), dtype=torch.float32, device=dev)
        _clock_bwd_kernel[(nprog,)](
            dinc, logit, rate, gains, dlogit, pr, pg, T * B, C, H, base, floor, 1.0 / C, SOFTPLUS_THRESHOLD,
            BR=CLOCK_BR, BD=bd, HP=MAX_HEADS, num_warps=CLOCK_WARPS, enable_fp_fusion=False)
        return dlogit, pr.sum(0).to(torch.bfloat16), pg[:, :H].sum(0), None, None


def clock_increments(logit: torch.Tensor, rate: torch.Tensor, gains: torch.Tensor, base: float,
                     floor: float) -> torch.Tensor:
    """(T, B, H) fp32 gained clock increments (see the module docstring): ``logit`` bf16 (T, B, C) CUDA, ``rate`` bf16
    (C,), ``gains`` fp32 (H,) with H <= 16, ``base`` the base decay, ``floor`` the decay floor.  Training regime."""
    require_cuda(logit, "clock_increments")
    if logit.dtype != torch.bfloat16 or logit.dim() != 3 or logit.numel() == 0 or logit.numel() >= 2 ** 31:
        raise RuntimeError(f"neuronspark.clock_increments: logit must be a bf16 (T, B, C) CUDA tensor, got "
                           f"{logit.dtype} {tuple(logit.shape)}")
    C = int(logit.shape[2])
    if rate.dtype != torch.bfloat16 or tuple(rate.shape) != (C,) or rate.device != logit.device:
        raise RuntimeError(f"neuronspark.clock_increments: rate must be bf16 ({C},), got {rate.dtype} "
                           f"{tuple(rate.shape)}")
    if (gains.dtype != torch.float32 or gains.dim() != 1 or not 1 <= int(gains.shape[0]) <= MAX_HEADS
            or gains.device != logit.device):
        raise RuntimeError(f"neuronspark.clock_increments: gains must be fp32 (H,) with H <= {MAX_HEADS}, got "
                           f"{gains.dtype} {tuple(gains.shape)}")
    if C > 4096:
        raise RuntimeError(f"neuronspark.clock_increments: at most 4096 channels, got {C}")
    return ClockIncrementsFn.apply(logit.contiguous(), rate.contiguous(), gains.contiguous(), float(base),
                                   float(floor))


def clock_increments_ref(logit: torch.Tensor, rate: torch.Tensor, gains: torch.Tensor, base: float,
                         floor: float) -> torch.Tensor:
    """The torch chain of the module: the decay, its floor, the mean log decay and the gain product."""
    a = torch.exp(-rate * F.softplus(logit.float())) * base
    return a.to(torch.float32).clamp_min(floor).log().mean(dim=-1).unsqueeze(-1) * gains
