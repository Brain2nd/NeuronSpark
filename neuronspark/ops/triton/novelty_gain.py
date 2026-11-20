"""Novelty predictor and confidence-gated read gain of the hippocampal layer (training): one forward kernel and one
backward kernel.

After the pool attention the layer adds the centred attention residual ``r`` (bf16) to the residual stream ``h`` with a
gain per token (:mod:`neuronspark.modules.copy_gate`)::

    rf  = f32(r);   rn = rf * rsqrt(mean(rf^2) + eps)
    c   = sigmoid(z . w + b + rn . v)                    novelty factor (T, B), z the layer's frozen-norm input
    f   = (kappa pi) c                                   pi the copy confidence (T, B), no gradient
    h'  = h + bf16(rf (1 + f))

Operator ``novelty_read_gain(h, z, r, pi, w, b, v, kappa, eps) -> (h', c)`` (:class:`NoveltyReadGainFn`):

* forward: one program per (t, b) row forms the two dot products, the rms of r, the sigmoid, the gained residual and the
  add; ``c`` and the row factor ``rsqrt(mean(rf^2) + eps)`` are kept (T, B).  ``c`` is returned for probes and carries
  no gradient of its own.
* backward from ``g = dL/dh'``: ``dh = g`` (the same tensor); with ``gf = f32(bf16(g))`` (the bf16 gradient of the
  gained residual), ``df = sum_c gf rf`` and ``dl = df kappa pi c (1 - c)`` one kernel writes ``dz = dl w``,
  ``dr = bf16(gf (1 + f) + a rr - (sum_c a rf) rr^3 rf / D)`` with ``a = dl v``, and per program the partial sums of
  ``dw = sum_rows dl z``, ``dv = sum_rows dl rn`` and ``db = sum_rows dl``, which one torch ``sum`` reduces.

Numerics: fp32 products and register sums (another summation order than the torch chain's matvecs), so ``f`` agrees
with the chain's to fp32 rounding.  The gained residual is rounded to bf16 before the add, as in the chain: where
``rf (1 + f)`` lies at a bf16 rounding boundary the two can round to neighbouring bf16 values (one bf16 step: at most
7.8e-3 for a gained residual below 2 in magnitude).  Tests: the per-operator tolerance rule of
:mod:`neuronspark.utils.testing` against :func:`novelty_read_gain_ref`, the torch chain.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = ["NoveltyReadGainFn", "novelty_read_gain", "novelty_read_gain_ref"]

# ================================================================================================ fixed constants
#: forward: one (t, b) row per program, warps
FWD_WARPS = 4
#: backward: rows per program (the partial sums of the predictor's parameter gradients are per program), warps
BWD_ROWS = 4
BWD_WARPS = 8


# ================================================================================================ kernels
@triton.jit
def _ng_fwd_kernel(H, Z, RS, PI, W, BIAS, V, OUT, C, RR, B, D,
                   hs0, hs1, zs0, zs1, rs0, rs1, ps0, ps1, us0, us1, kappa, eps, BD: tl.constexpr):
    """h' = h + f32(bf16(rf (1 + kappa pi c))) for one row; c and rr stored at [t * B + b]."""
    r = tl.program_id(0).to(tl.int64)
    t = r // B
    b = r - t * B
    col = tl.arange(0, BD)
    cm = col < D
    rf = tl.load(RS + (t * rs0 + b * rs1) + col, mask=cm, other=0.0).to(tl.float32)
    z = tl.load(Z + (t * zs0 + b * zs1) + col, mask=cm, other=0.0)
    w = tl.load(W + col, mask=cm, other=0.0)
    v = tl.load(V + col, mask=cm, other=0.0)
    rr = tl.rsqrt(tl.sum(rf * rf, axis=0) / D + eps)
    logit = (tl.sum(z * w, axis=0) + tl.load(BIAS)) + tl.sum(rf * v, axis=0) * rr
    c = 1.0 / (1.0 + tl.exp(-logit))
    tl.store(C + r, c)
    tl.store(RR + r, rr)
    f1 = 1.0 + (kappa * tl.load(PI + (t * ps0 + b * ps1))) * c
    h = tl.load(H + (t * hs0 + b * hs1) + col, mask=cm, other=0.0)
    tl.store(OUT + (t * us0 + b * us1) + col, h + (rf * f1).to(tl.bfloat16).to(tl.float32), mask=cm)


@triton.jit
def _ng_bwd_kernel(G, Z, RS, PI, W, V, C, RR, DZ, DR, PART, R, B, D,
                   gs0, gs1, gs2, zs0, zs1, rs0, rs1, ps0, ps1, ds0, ds1, es0, es1, kappa,
                   BR: tl.constexpr, BD: tl.constexpr):
    """BR rows per program: dz (fp32), dr (bf16), PART[program] = (D partial sums of dw, D of dv, the partial sum of
    db)."""
    pid = tl.program_id(0).to(tl.int64)
    r = pid * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    col = tl.arange(0, BD)
    cm = col < D
    m = rm[:, None] & cm[None, :]
    gf = tl.load(G + (t * gs0 + b * gs1)[:, None] + col.to(tl.int64)[None, :] * gs2, mask=m,
                 other=0.0).to(tl.bfloat16).to(tl.float32)
    rf = tl.load(RS + (t * rs0 + b * rs1)[:, None] + col[None, :], mask=m, other=0.0).to(tl.float32)
    c = tl.load(C + r, mask=rm, other=0.0)
    rr = tl.load(RR + r, mask=rm, other=0.0)
    kp = kappa * tl.load(PI + (t * ps0 + b * ps1), mask=rm, other=0.0)
    dl = tl.sum(gf * rf, axis=1) * kp * c * (1.0 - c)                                # [BR] logit gradient
    w = tl.load(W + col, mask=cm, other=0.0)
    v = tl.load(V + col, mask=cm, other=0.0)
    tl.store(DZ + (t * ds0 + b * ds1)[:, None] + col[None, :], dl[:, None] * w[None, :], mask=m)
    a = dl[:, None] * v[None, :]
    sa = tl.sum(a * rf, axis=1) * (rr * rr * rr) / D
    dr = gf * (1.0 + kp * c)[:, None] + a * rr[:, None] - sa[:, None] * rf
    tl.store(DR + (t * es0 + b * es1)[:, None] + col[None, :], dr.to(tl.bfloat16), mask=m)
    z = tl.load(Z + (t * zs0 + b * zs1)[:, None] + col[None, :], mask=m, other=0.0)
    prow = PART + pid * (2 * D + 1)
    tl.store(prow + col, tl.sum(dl[:, None] * z, axis=0), mask=cm)
    tl.store(prow + D + col, tl.sum((dl * rr)[:, None] * rf, axis=0), mask=cm)
    tl.store(prow + 2 * D, tl.sum(dl, axis=0))


# ================================================================================================ launchers
def _dest_like(t: torch.Tensor) -> torch.Tensor:
    """An uninitialised tensor like ``t`` that the kernels write row by row: the layout torch gives a tensor like t
    when its inner stride is 1 (a dense t: t's own layout), row-major otherwise (a t whose strides are not dense)."""
    d = torch.empty_like(t)
    return d if d.stride(-1) == 1 else torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)


def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


def _s2(t: torch.Tensor) -> Tuple[int, int]:
    return int(t.stride(0)), int(t.stride(1))


@torch.library.custom_op("neuronspark::novelty_read_gain_fwd", mutates_args=())
def _novelty_read_gain_fwd(h: torch.Tensor, z: torch.Tensor, r: torch.Tensor, pi: torch.Tensor, w: torch.Tensor,
                           b: torch.Tensor, v: torch.Tensor, kappa: float,
                           eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(h' fp32 in h's layout, c fp32 (T, B), the rms row factors fp32 (T, B))."""
    T, B, D = (int(x) for x in h.shape)
    out = _dest_like(h)
    c = torch.empty((T, B), dtype=torch.float32, device=h.device)
    rr = torch.empty((T, B), dtype=torch.float32, device=h.device)
    if T * B:
        _ng_fwd_kernel[(T * B,)](h, z, r, pi, w, b, v, out, c, rr, B, D, *_s2(h), *_s2(z), *_s2(r), *_s2(pi),
                                 *_s2(out), float(kappa), float(eps), BD=_pow2(D), num_warps=FWD_WARPS)
    return out, c, rr


@_novelty_read_gain_fwd.register_fake
def _novelty_read_gain_fwd_fake(h, z, r, pi, w, b, v, kappa, eps):
    shape = (h.shape[0], h.shape[1])
    return torch.empty_like(h), h.new_empty(shape, dtype=torch.float32), h.new_empty(shape, dtype=torch.float32)


@torch.library.custom_op("neuronspark::novelty_read_gain_bwd", mutates_args=())
def _novelty_read_gain_bwd(g: torch.Tensor, z: torch.Tensor, r: torch.Tensor, pi: torch.Tensor, w: torch.Tensor,
                           v: torch.Tensor, c: torch.Tensor, rr: torch.Tensor,
                           kappa: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dz fp32 in z's layout, dr bf16 in r's layout, sums fp32 (2 D + 1,): the gradients of w, of v and of b)."""
    T, B, D = (int(x) for x in z.shape)
    R = T * B
    dz = torch.empty_like(z)
    dr = _dest_like(r)
    n_prog = triton.cdiv(R, BWD_ROWS)
    part = torch.empty((n_prog, 2 * D + 1), dtype=torch.float32, device=z.device)
    if R:
        _ng_bwd_kernel[(n_prog,)](g, z, r, pi, w, v, c, rr, dz, dr, part, R, B, D,
                                  int(g.stride(0)), int(g.stride(1)), int(g.stride(2)), *_s2(z), *_s2(r), *_s2(pi),
                                  *_s2(dz), *_s2(dr), float(kappa), BR=BWD_ROWS, BD=_pow2(D), num_warps=BWD_WARPS)
    return dz, dr, part.sum(0)


@_novelty_read_gain_bwd.register_fake
def _novelty_read_gain_bwd_fake(g, z, r, pi, w, v, c, rr, kappa):
    return torch.empty_like(z), torch.empty_like(r), z.new_empty((2 * z.shape[-1] + 1,), dtype=torch.float32)


class NoveltyReadGainFn(torch.autograd.Function):
    """(h', c): h' = h + bf16(f32(r) (1 + kappa pi c)), c = sigmoid(z . w + b + rms(r) . v).  Gradients from h': h (the
    incoming gradient itself), z (fp32), r (bf16), w, b, v (fp32); pi has none, c is not differentiable."""

    @staticmethod
    def forward(ctx, h, z, r, pi, w, b, v, kappa, eps):
        out, c, rr = _novelty_read_gain_fwd(h, z, r, pi, w, b, v, float(kappa), float(eps))
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(c)
        ctx.save_for_backward(z, r, pi, w, v, c, rr)
        ctx.kappa = float(kappa)
        ctx.b_shape = tuple(b.shape)
        return out, c

    @staticmethod
    def backward(ctx, g, _gc):
        if g is None:
            return (None,) * 9
        z, r, pi, w, v, c, rr = ctx.saved_tensors
        if g.dtype != torch.float32 or tuple(g.shape) != tuple(z.shape):
            raise RuntimeError(f"neuronspark.novelty_read_gain backward: gradient {g.dtype} {tuple(g.shape)} for "
                               f"{tuple(z.shape)}")
        dz, dr, sums = _novelty_read_gain_bwd(g, z, r, pi, w, v, c, rr, ctx.kappa)
        D = int(z.shape[-1])
        need = ctx.needs_input_grad
        return (g if need[0] else None, dz if need[1] else None, dr if need[2] else None, None,
                sums[:D] if need[4] else None, sums[2 * D:].reshape(ctx.b_shape) if need[5] else None,
                sums[D:2 * D] if need[6] else None, None, None)


# ================================================================================================ public function
def novelty_read_gain(h: torch.Tensor, z: torch.Tensor, r: torch.Tensor, pi: torch.Tensor, w: torch.Tensor,
                      b: torch.Tensor, v: torch.Tensor, kappa: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(h + bf16(r.float() * (1 + kappa * pi * c)), c)`` with ``c = sigmoid(z . w + b + rms(r) . v)`` (T, B) fp32.

    ``h`` / ``z`` (T, B, D) fp32 and ``r`` (T, B, D) bf16 with unit inner strides (any (t, b) strides); ``pi`` (T, B)
    fp32 without gradient; ``w`` / ``v`` (D,) fp32 contiguous, ``b`` one-element fp32; ``kappa`` the read gain;
    ``eps`` the eps of the rms.  h' is fp32 in h's layout; differentiable in h, z, r, w, b, v.  Training regime."""
    for name, t in (("h", h), ("z", z), ("r", r), ("pi", pi), ("w", w), ("b", b), ("v", v)):
        require_cuda(t, f"novelty_read_gain ({name})")
    if h.dim() != 3 or h.dtype != torch.float32 or h.stride(-1) != 1 or h.numel() >= 2 ** 31:
        raise RuntimeError(f"neuronspark.novelty_read_gain: h must be fp32 (T, B, D) with a unit inner stride, got "
                           f"{h.dtype} {tuple(h.shape)} {tuple(h.stride())}")
    T, B, D = (int(x) for x in h.shape)
    for name, t, dt in (("z", z, torch.float32), ("r", r, torch.bfloat16)):
        if t.dtype != dt or tuple(t.shape) != (T, B, D) or t.stride(-1) != 1:
            raise RuntimeError(f"neuronspark.novelty_read_gain: {name} must be {dt} ({T}, {B}, {D}) with a unit inner "
                               f"stride, got {t.dtype} {tuple(t.shape)} {tuple(t.stride())}")
    if pi.dtype != torch.float32 or tuple(pi.shape) != (T, B):
        raise RuntimeError(f"neuronspark.novelty_read_gain: pi must be fp32 ({T}, {B}), got {pi.dtype} "
                           f"{tuple(pi.shape)}")
    if pi.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("neuronspark.novelty_read_gain: pi requires grad; the operator gives it no gradient (the "
                           "copy confidence is a constant of the call)")
    for name, t in (("w", w), ("v", v)):
        if t.dtype != torch.float32 or tuple(t.shape) != (D,) or not t.is_contiguous():
            raise RuntimeError(f"neuronspark.novelty_read_gain: {name} must be a contiguous fp32 ({D},) tensor")
    if b.dtype != torch.float32 or b.numel() != 1:
        raise RuntimeError("neuronspark.novelty_read_gain: b must be a one-element fp32 tensor")
    return NoveltyReadGainFn.apply(h, z, r, pi, w, b, v, float(kappa), float(eps))


# ================================================================================================ reference
def novelty_read_gain_ref(h: torch.Tensor, z: torch.Tensor, r: torch.Tensor, pi: torch.Tensor, w: torch.Tensor,
                          b: torch.Tensor, v: torch.Tensor, kappa: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """The torch chain (same arguments and outputs as :func:`novelty_read_gain`): the two fp32 matvecs, the rms of the
    residual, the sigmoid, the gained residual in bf16 and the fp32 add; autocast disabled."""
    with torch.autocast(h.device.type, enabled=False):
        logit = torch.matmul(z.float(), w.float()) + b.float()
        rf = r.float()
        rn = rf * torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + eps)
        c = torch.sigmoid(logit + torch.matmul(rn, v.float()))
        f = (float(kappa) * pi) * c
        return h + (rf * (1.0 + f).unsqueeze(-1)).to(r.dtype), c.detach()
