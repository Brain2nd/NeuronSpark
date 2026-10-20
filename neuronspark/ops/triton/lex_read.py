"""Lexical read of a prefill call after its projections, in one kernel.

The first step of every hippocampal layer reads the model-level lexical vector m through a scalar gate per token
(:mod:`neuronspark.modules.lexical_read`).  After the three bf16 projections ``q_raw = bf16(hn @ lex_q^T)``
(T, B, K), ``k_raw = bf16(m @ lex_k^T)`` (T, B, K) and ``o = bf16(m @ lex_o^T)`` (T, B, D) the read is, per row::

    qf    = (f32(q_raw) * rsqrt(mean(f32(q_raw)^2) + eps)) * (1 + g_q)      fp32, K = 64
    kf    = (f32(k_raw) * rsqrt(mean(f32(k_raw)^2) + eps)) * (1 + g_k)
    alpha = sigmoid(sum(qf * kf) / sqrt(K) + beta)                           fp32 scalar per row
    out   = h + alpha * f32(o)                                               fp32 (T, B, D)

The torch chain (:func:`lex_read_prefill_ref`) runs each step as its own (T, B, K) or (T, B, D) kernel.  Here one
program handles one (t, b) row: the K-wide norms, the gate and the residual add over D in blocks of
``LEX_READ_BLOCK_D``.  The operations are the chain's: the mean as the fp32 sum divided by K (IEEE division),
``rsqrtf``, the products in the chain's association, the IEEE division by sqrt(K), the logistic as 1 / (1 + expf(-x)),
then one product and one sum per output element; no FMA contraction.  The K-wide sums run in another order than
ATen's reductions.  Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the chain.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _rn_div_f32  # noqa: F401  (resolved by the kernel through this module's globals)

__all__ = ["LEX_READ_BLOCK_D", "LEX_READ_WARPS", "lex_read_prefill", "lex_read_prefill_ref"]

# ================================================================================================ fixed constants
#: residual channels per inner step, warps; one program per (t, b) row
LEX_READ_BLOCK_D = 512
LEX_READ_WARPS = 4


# ================================================================================================ kernel
@triton.jit
def _lex_read_kernel(H, Q, KR, O, GQ, GK, BETA, OUT, B, D,
                     s_h0, s_h1, s_q0, s_q1, s_k0, s_k1, s_o0, s_o1, eps, sqrt_k,
                     K: tl.constexpr, BLOCK_D: tl.constexpr):
    r = tl.program_id(0)
    t = (r // B).to(tl.int64)
    b = (r % B).to(tl.int64)
    kk = tl.arange(0, K)
    q = tl.load(Q + t * s_q0 + b * s_q1 + kk).to(tl.float32)
    k = tl.load(KR + t * s_k0 + b * s_k1 + kk).to(tl.float32)
    gq = 1.0 + tl.load(GQ + kk).to(tl.float32)
    gk = 1.0 + tl.load(GK + kk).to(tl.float32)
    mq = _rn_div_f32(tl.sum(q * q, axis=0), K * 1.0)
    mk = _rn_div_f32(tl.sum(k * k, axis=0), K * 1.0)
    qf = (q * libdevice.rsqrt(mq + eps)) * gq
    kf = (k * libdevice.rsqrt(mk + eps)) * gk
    x = _rn_div_f32(tl.sum(qf * kf, axis=0), sqrt_k) + tl.load(BETA).to(tl.float32)
    alpha = _rn_div_f32(1.0, 1.0 + libdevice.exp(-x))
    out_row = r.to(tl.int64) * D
    for d0 in range(0, D, BLOCK_D):
        dd = d0 + tl.arange(0, BLOCK_D)
        md = dd < D
        h = tl.load(H + t * s_h0 + b * s_h1 + dd, mask=md, other=0.0)
        o = tl.load(O + t * s_o0 + b * s_o1 + dd, mask=md, other=0.0).to(tl.float32)
        tl.store(OUT + out_row + dd, h + alpha * o, mask=md)


# ================================================================================================ launcher
def _check(h, q_raw, k_raw, o, q_gain, k_gain, beta):
    for name, t in (("h", h), ("q_raw", q_raw), ("k_raw", k_raw), ("o", o), ("q_gain", q_gain), ("k_gain", k_gain),
                    ("beta", beta)):
        require_cuda(t, f"lex_read_prefill ({name})")
    if h.dim() != 3 or h.dtype != torch.float32:
        raise RuntimeError(f"neuronspark.lex_read_prefill: h must be fp32 (T, B, D), got {h.dtype} {tuple(h.shape)}")
    T, B, D = (int(v) for v in h.shape)
    K = int(q_raw.shape[-1]) if q_raw.dim() == 3 else 0
    if K < 16 or K & (K - 1):
        raise RuntimeError(f"neuronspark.lex_read_prefill: the key width {K} must be a power of two >= 16")
    for name, t, n in (("q_raw", q_raw, K), ("k_raw", k_raw, K), ("o", o, D)):
        if tuple(t.shape) != (T, B, n) or t.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(f"neuronspark.lex_read_prefill: {name} must be bf16 / fp32 ({T}, {B}, {n}), got "
                               f"{t.dtype} {tuple(t.shape)}")
    for name, t in (("h", h), ("q_raw", q_raw), ("k_raw", k_raw), ("o", o)):
        if t.stride(-1) != 1:
            raise RuntimeError(f"neuronspark.lex_read_prefill: {name} needs a unit inner stride, got {tuple(t.stride())}")
    for name, t in (("q_gain", q_gain), ("k_gain", k_gain)):
        if t.dtype != torch.float32 or tuple(t.shape) != (K,) or not t.is_contiguous():
            raise RuntimeError(f"neuronspark.lex_read_prefill: {name} must be a contiguous fp32 ({K},) tensor")
    if beta.dtype != torch.float32 or beta.numel() != 1:
        raise RuntimeError("neuronspark.lex_read_prefill: beta must be a one-element fp32 tensor")
    return T, B, D, K


def _lex_read_prefill_fwd(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                          q_gain: torch.Tensor, k_gain: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    T, B, D, K = _check(h, q_raw, k_raw, o, q_gain, k_gain, beta)
    out = torch.empty((T, B, D), dtype=torch.float32, device=h.device)
    if T * B == 0:
        return out
    _lex_read_kernel[(T * B,)](h, q_raw, k_raw, o, q_gain, k_gain, beta.reshape(1), out, B, D,
                               int(h.stride(0)), int(h.stride(1)), int(q_raw.stride(0)), int(q_raw.stride(1)),
                               int(k_raw.stride(0)), int(k_raw.stride(1)), int(o.stride(0)), int(o.stride(1)),
                               float(eps), float(math.sqrt(K)), K=K, BLOCK_D=LEX_READ_BLOCK_D,
                               num_warps=LEX_READ_WARPS, enable_fp_fusion=False)
    return out


@torch.library.custom_op("neuronspark::lex_read_prefill", mutates_args=())
def _lex_read_prefill_op(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                         q_gain: torch.Tensor, k_gain: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    return _lex_read_prefill_fwd(h, q_raw, k_raw, o, q_gain, k_gain, beta, eps)


@_lex_read_prefill_op.register_fake
def _lex_read_prefill_fake(h, q_raw, k_raw, o, q_gain, k_gain, beta, eps):
    return h.new_empty(tuple(h.shape), dtype=torch.float32)


def lex_read_prefill(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                     q_gain: torch.Tensor, k_gain: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """(T, B, D) fp32 contiguous ``h + alpha * f32(o)`` with the lexical gate alpha of every row.

    ``h`` (T, B, D) fp32; ``q_raw`` / ``k_raw`` (T, B, K) and ``o`` (T, B, D) the bf16 projections (unit inner
    strides; any (t, b) strides); ``q_gain`` / ``k_gain`` (K,) fp32 zero-centred norm gains; ``beta`` one-element fp32
    gate bias; ``eps`` the norms' eps (1e-6)."""
    return torch.ops.neuronspark.lex_read_prefill(h, q_raw, k_raw, o, q_gain, k_gain, beta, float(eps))


# ================================================================================================ reference
def lex_read_prefill_ref(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor,
                         q_gain: torch.Tensor, k_gain: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """The hippocampal layer's torch chain of the read after its projections (same arguments and output as
    :func:`lex_read_prefill`): the zero-centred-gain RMS norms of q / k in fp32, the scaled dot plus the bias through
    a sigmoid, and the fp32 residual ``h + (alpha * f32(o))``."""
    K = int(q_raw.shape[-1])
    qf = q_raw.float()
    qf = qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + q_gain.float())
    kf = k_raw.float()
    kf = kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + k_gain.float())
    alpha = torch.sigmoid((qf * kf).sum(dim=-1, keepdim=True) / math.sqrt(K) + beta.float())
    return h + (alpha * o.float()).to(h.dtype)
