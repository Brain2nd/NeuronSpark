"""Lexical read of a whole-row call after its projections: one forward kernel and one backward kernel.

The first step of every hippocampal layer reads the model-level lexical vector m through a scalar gate per token
(:mod:`neuronspark.modules.lexical_read`).  After the three bf16 projections ``q_raw = bf16(hn @ lex_q^T)`` (T, B, K),
``k_raw = bf16(m @ lex_k^T)`` (T, B, K) and ``o = bf16(m @ lex_o^T)`` (T, B, D) the read is, per row::

    qf    = (f32(q_raw) * rsqrt(mean(f32(q_raw)^2) + eps)) * (1 + g_q)      fp32, K = 64
    kf    = (f32(k_raw) * rsqrt(mean(f32(k_raw)^2) + eps)) * (1 + g_k)
    alpha = sigmoid(sum(qf * kf) / sqrt(K) + beta)                           fp32 scalar per row
    out   = h + alpha * f32(o)                                               fp32 (T, B, D)

Operator ``lex_read_train(h, q_raw, k_raw, o, q_gain, k_gain, beta, eps)`` (training, :class:`LexReadTrainFn`):

* forward: one program per (t, b) row forms the two K-wide norms, the gate and the residual; alpha is kept (T, B);
* backward from ``g = dL/dout``: ``dh = g`` (the same tensor); one kernel writes ``do = bf16(alpha g)``, the row sums
  ``ga = sum_c g f32(o)``, the gate's logit gradient ``dx = ga alpha (1 - alpha)``, the bf16 gradients of ``q_raw`` /
  ``k_raw`` through their norms, and per program the partial sums of the gradients of ``q_gain``, ``k_gain`` and
  ``beta``, which one torch ``sum`` reduces.

Numerics: fp32 register sums and products (another summation order than the torch chain's reductions).  Tests: the
per-operator tolerance rule of :mod:`neuronspark.utils.testing` against :func:`lex_read_train_ref`, the torch chain.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE

import math
from typing import Tuple

import torch
import triton
import triton.language as tl

from ....neuronspark.ops.triton import require_cuda

__all__ = ["LexReadTrainFn", "lex_read_train", "lex_read_train_ref"]

# ================================================================================================ fixed constants
#: forward: one (t, b) row per program, warps
FWD_WARPS = 4
#: backward: rows per program (the gain / bias partial sums are per program), warps
BWD_ROWS = 4
BWD_WARPS = 8


# ================================================================================================ kernels
@triton.jit
def _lexr_fwd_kernel(H, Q, KR, O, GQ, GK, BETA, OUT, ALPHA, B, D, K,
                     hs0, hs1, qs0, qs1, ks0, ks1, os0, os1, us0, us1, eps, inv_sqrt_k,
                     KB: tl.constexpr, BD: tl.constexpr):
    """out = h + alpha f32(o) for one row; alpha stored at ALPHA[t * B + b]."""
    r = tl.program_id(0).to(tl.int64)
    t = r // B
    b = r - t * B
    kk = tl.arange(0, KB)
    km = kk < K
    q = tl.load(Q + (t * qs0 + b * qs1) + kk, mask=km, other=0.0).to(tl.float32)
    k = tl.load(KR + (t * ks0 + b * ks1) + kk, mask=km, other=0.0).to(tl.float32)
    qf = (q * tl.rsqrt(tl.sum(q * q, axis=0) / K + eps)) * (1.0 + tl.load(GQ + kk, mask=km, other=0.0))
    kf = (k * tl.rsqrt(tl.sum(k * k, axis=0) / K + eps)) * (1.0 + tl.load(GK + kk, mask=km, other=0.0))
    x = tl.sum(qf * kf, axis=0) * inv_sqrt_k + tl.load(BETA)
    alpha = 1.0 / (1.0 + tl.exp(-x))
    tl.store(ALPHA + r, alpha)
    c = tl.arange(0, BD)
    cm = c < D
    h = tl.load(H + (t * hs0 + b * hs1) + c, mask=cm, other=0.0)
    o = tl.load(O + (t * os0 + b * os1) + c, mask=cm, other=0.0).to(tl.float32)
    tl.store(OUT + (t * us0 + b * us1) + c, h + alpha * o, mask=cm)


@triton.jit
def _lexr_bwd_kernel(G, Q, KR, O, GQ, GK, ALPHA, GO, DQ, DK, PART, R, B, D, K,
                     gs0, gs1, gs2, qs0, qs1, ks0, ks1, os0, os1, go0, go1, dq0, dq1, dk0, dk1, eps, inv_sqrt_k,
                     KB: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    """BR rows per program: do = bf16(alpha g), dq / dk = the bf16 gradients of the projections (each written through
    its own strides), PART[program] = (the K partial sums of q_gain's gradient, of k_gain's, the partial sum of
    beta's)."""
    pid = tl.program_id(0).to(tl.int64)
    r = pid * BR + tl.arange(0, BR)
    rm = r < R
    t = r // B
    b = r - t * B
    kk = tl.arange(0, KB)
    km = kk < K
    mk = rm[:, None] & km[None, :]
    c = tl.arange(0, BD)
    cm = c < D
    m = rm[:, None] & cm[None, :]
    alpha = tl.load(ALPHA + r, mask=rm, other=0.0)
    g = tl.load(G + (t * gs0 + b * gs1)[:, None] + c.to(tl.int64)[None, :] * gs2, mask=m, other=0.0)
    o = tl.load(O + (t * os0 + b * os1)[:, None] + c[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(GO + (t * go0 + b * go1)[:, None] + c[None, :], (alpha[:, None] * g).to(tl.bfloat16), mask=m)
    dx = tl.sum(g * o, axis=1) * alpha * (1.0 - alpha)                               # [BR] gate-logit gradient
    q = tl.load(Q + (t * qs0 + b * qs1)[:, None] + kk[None, :], mask=mk, other=0.0).to(tl.float32)
    k = tl.load(KR + (t * ks0 + b * ks1)[:, None] + kk[None, :], mask=mk, other=0.0).to(tl.float32)
    gq1 = 1.0 + tl.load(GQ + kk, mask=km, other=0.0)
    gk1 = 1.0 + tl.load(GK + kk, mask=km, other=0.0)
    rq = tl.rsqrt(tl.sum(q * q, axis=1) / K + eps)
    rk = tl.rsqrt(tl.sum(k * k, axis=1) / K + eps)
    qn = q * rq[:, None]
    kn = k * rk[:, None]
    ds = (dx * inv_sqrt_k)[:, None]
    dqf = ds * (kn * gk1[None, :])
    dkf = ds * (qn * gq1[None, :])
    aq = dqf * gq1[None, :]
    ak = dkf * gk1[None, :]
    sq = tl.sum(aq * q, axis=1) * (rq * rq * rq) / K
    sk = tl.sum(ak * k, axis=1) * (rk * rk * rk) / K
    dq = aq * rq[:, None] - sq[:, None] * q
    dk = ak * rk[:, None] - sk[:, None] * k
    tl.store(DQ + (t * dq0 + b * dq1)[:, None] + kk[None, :], dq.to(tl.bfloat16), mask=mk)
    tl.store(DK + (t * dk0 + b * dk1)[:, None] + kk[None, :], dk.to(tl.bfloat16), mask=mk)
    prow = PART + pid * (2 * K + 1)
    tl.store(prow + kk, tl.sum(dqf * qn, axis=0), mask=km)
    tl.store(prow + K + kk, tl.sum(dkf * kn, axis=0), mask=km)
    tl.store(prow + 2 * K, tl.sum(dx, axis=0))


# ================================================================================================ launchers
def _dest_like(t: torch.Tensor) -> torch.Tensor:
    """An uninitialised tensor like ``t`` that the kernels write row by row: the layout torch gives a tensor like t
    when its inner stride is 1 (a dense t: t's own layout), row-major otherwise (a t whose strides are not dense)."""
    d = torch.empty_like(t)
    return d if d.stride(-1) == 1 else torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)


def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


def _check(h, q_raw, k_raw, o, q_gain, k_gain, beta) -> Tuple[int, int, int, int]:
    for name, t in (("h", h), ("q_raw", q_raw), ("k_raw", k_raw), ("o", o), ("q_gain", q_gain), ("k_gain", k_gain),
                    ("beta", beta)):
        require_cuda(t, f"lex_read_train ({name})")
    if h.dim() != 3 or h.dtype != torch.float32 or h.numel() >= 2 ** 31:
        raise RuntimeError(f"neuronspark.lex_read_train: h must be fp32 (T, B, D), got {h.dtype} {tuple(h.shape)}")
    T, B, D = (int(v) for v in h.shape)
    K = int(q_raw.shape[-1]) if q_raw.dim() == 3 else 0
    if K < 1:
        raise RuntimeError(f"neuronspark.lex_read_train: q_raw must be bf16 ({T}, {B}, K), got {tuple(q_raw.shape)}")
    for name, t, n in (("q_raw", q_raw, K), ("k_raw", k_raw, K), ("o", o, D)):
        if tuple(t.shape) != (T, B, n) or t.dtype != torch.bfloat16:
            raise RuntimeError(f"neuronspark.lex_read_train: {name} must be bf16 ({T}, {B}, {n}), got {t.dtype} "
                               f"{tuple(t.shape)}")
    for name, t in (("h", h), ("q_raw", q_raw), ("k_raw", k_raw), ("o", o)):
        if t.stride(-1) != 1:
            raise RuntimeError(f"neuronspark.lex_read_train: {name} needs a unit inner stride, got {tuple(t.stride())}")
    for name, t in (("q_gain", q_gain), ("k_gain", k_gain)):
        if t.dtype != torch.float32 or tuple(t.shape) != (K,) or not t.is_contiguous():
            raise RuntimeError(f"neuronspark.lex_read_train: {name} must be a contiguous fp32 ({K},) tensor")
    if beta.dtype != torch.float32 or beta.numel() != 1:
        raise RuntimeError("neuronspark.lex_read_train: beta must be a one-element fp32 tensor")
    return T, B, D, K


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::lex_read_train_fwd", mutates_args=())
def _lex_read_train_fwd(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                        q_gain: torch.Tensor, k_gain: torch.Tensor, beta: torch.Tensor,
                        eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(out fp32 in h's layout, alpha fp32 (T, B) contiguous)."""
    T, B, D = (int(v) for v in h.shape)
    K = int(q_raw.shape[-1])
    out = _dest_like(h)
    alpha = torch.empty((T, B), dtype=torch.float32, device=h.device)
    if T * B:
        _lexr_fwd_kernel[(T * B,)](
            h, q_raw, k_raw, o, q_gain, k_gain, beta, out, alpha, B, D, K,
            int(h.stride(0)), int(h.stride(1)), int(q_raw.stride(0)), int(q_raw.stride(1)),
            int(k_raw.stride(0)), int(k_raw.stride(1)), int(o.stride(0)), int(o.stride(1)),
            int(out.stride(0)), int(out.stride(1)), float(eps), 1.0 / math.sqrt(K),
            KB=_pow2(K), BD=_pow2(D), num_warps=FWD_WARPS)
    return out, alpha


@_lex_read_train_fwd.register_fake
def _lex_read_train_fwd_fake(h, q_raw, k_raw, o, q_gain, k_gain, beta, eps):
    return torch.empty_like(h), h.new_empty((h.shape[0], h.shape[1]), dtype=torch.float32)


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::lex_read_train_bwd", mutates_args=())
def _lex_read_train_bwd(g: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                        q_gain: torch.Tensor, k_gain: torch.Tensor, alpha: torch.Tensor,
                        eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(do bf16 (T, B, D), dq bf16 (T, B, K), dk bf16 (T, B, K), each in the layout torch gives a tensor like its
    projection; sums fp32 (2 K + 1,): the gradients of q_gain, of k_gain and of beta)."""
    T, B, D = (int(v) for v in o.shape)
    K = int(q_raw.shape[-1])
    R = T * B
    dev = g.device
    go = _dest_like(o)
    dq = _dest_like(q_raw)
    dk = _dest_like(k_raw)
    n_prog = triton.cdiv(R, BWD_ROWS)
    part = torch.empty((n_prog, 2 * K + 1), dtype=torch.float32, device=dev)
    if R:
        _lexr_bwd_kernel[(n_prog,)](
            g, q_raw, k_raw, o, q_gain, k_gain, alpha, go, dq, dk, part, R, B, D, K,
            int(g.stride(0)), int(g.stride(1)), int(g.stride(2)), int(q_raw.stride(0)), int(q_raw.stride(1)),
            int(k_raw.stride(0)), int(k_raw.stride(1)), int(o.stride(0)), int(o.stride(1)),
            int(go.stride(0)), int(go.stride(1)), int(dq.stride(0)), int(dq.stride(1)),
            int(dk.stride(0)), int(dk.stride(1)), float(eps), 1.0 / math.sqrt(K),
            KB=_pow2(K), BR=BWD_ROWS, BD=_pow2(D), num_warps=BWD_WARPS)
    return go, dq, dk, part.sum(0)


@_lex_read_train_bwd.register_fake
def _lex_read_train_bwd_fake(g, q_raw, k_raw, o, q_gain, k_gain, alpha, eps):
    K = q_raw.shape[-1]
    return (o.new_empty(tuple(o.shape)), q_raw.new_empty(tuple(q_raw.shape)), k_raw.new_empty(tuple(k_raw.shape)),
            g.new_empty((2 * K + 1,), dtype=torch.float32))


class LexReadTrainFn(torch.autograd.Function):
    """out = h + alpha f32(o) with the lexical gate alpha of every row; gradients for h (the incoming gradient itself),
    q_raw, k_raw, o (bf16), q_gain, k_gain, beta (fp32)."""

    @staticmethod
    def forward(ctx, h, q_raw, k_raw, o, q_gain, k_gain, beta, eps):
        out, alpha = _lex_read_train_fwd(h, q_raw, k_raw, o, q_gain, k_gain, beta, float(eps))
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q_raw, k_raw, o, q_gain, k_gain, alpha)
        ctx.eps = float(eps)
        ctx.beta_shape = tuple(beta.shape)
        return out

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return (None,) * 8
        q_raw, k_raw, o, q_gain, k_gain, alpha = ctx.saved_tensors
        if g.dtype != torch.float32 or tuple(g.shape) != tuple(o.shape):
            raise RuntimeError(f"neuronspark.lex_read_train backward: gradient {g.dtype} {tuple(g.shape)} for "
                               f"{tuple(o.shape)}")
        go, dq, dk, sums = _lex_read_train_bwd(g, q_raw, k_raw, o, q_gain, k_gain, alpha, ctx.eps)
        K = int(q_raw.shape[-1])
        need = ctx.needs_input_grad
        return (g if need[0] else None, dq if need[1] else None, dk if need[2] else None, go if need[3] else None,
                sums[:K] if need[4] else None, sums[K:2 * K] if need[5] else None,
                sums[2 * K:].reshape(ctx.beta_shape) if need[6] else None, None)


# ================================================================================================ public function
def lex_read_train(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                   q_gain: torch.Tensor, k_gain: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """(T, B, D) fp32 ``h + alpha * f32(o)`` (h's layout) with the lexical gate alpha of every row; differentiable in
    every tensor argument.

    ``h`` (T, B, D) fp32 (any (t, b) strides); ``q_raw`` / ``k_raw`` (T, B, K) and ``o`` (T, B, D) the bf16 projections
    (unit inner strides); ``q_gain`` / ``k_gain`` (K,) fp32 zero-centred norm gains; ``beta`` one-element fp32 gate
    bias; ``eps`` the norms' eps (1e-6).  Training regime."""
    _check(h, q_raw, k_raw, o, q_gain, k_gain, beta)
    return LexReadTrainFn.apply(h, q_raw, k_raw, o, q_gain, k_gain, beta, float(eps))


# ================================================================================================ reference
def lex_read_train_ref(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                       q_gain: torch.Tensor, k_gain: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """The torch chain of the read after its projections (same arguments and output as :func:`lex_read_train`): the
    zero-centred-gain RMS norms of q / k in fp32, the scaled dot plus the bias through a sigmoid, and the fp32 residual
    ``h + (alpha * f32(o))``."""
    K = int(q_raw.shape[-1])
    qf = q_raw.float()
    qf = qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + q_gain.float())
    kf = k_raw.float()
    kf = kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + k_gain.float())
    alpha = torch.sigmoid((qf * kf).sum(dim=-1, keepdim=True) / math.sqrt(K) + beta.float())
    return h + (alpha * o.float()).to(h.dtype)
