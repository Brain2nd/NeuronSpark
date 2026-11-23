"""Decode-step lexical read of the hippocampal layer (T = 1): the gate, the residual addition and the frozen norm of
the new stream in one kernel, one program per row.

After the three bf16 projections ``q_raw = bf16(hn @ lex_q^T)`` (B, K), ``k_raw = bf16(m @ lex_k^T)`` (B, K) and
``o = bf16(m @ lex_o^T)`` (B, D), per row::

    qf    = (f(q_raw) rsqrt(mean(f(q_raw)^2) + eps)) (1 + g_q)             fp32, K wide
    kf    = (f(k_raw) rsqrt(mean(f(k_raw)^2) + eps)) (1 + g_k)
    alpha = sigmoid(sum(qf kf) / sqrt(K) + beta)                           fp32 scalar per row
    h'    = h + alpha f(o)                                                 fp32 (B, D): the stream after the read
    z     = h' rsqrt(mean_c h'^2 + eps_z)                                  fp32 (B, D): the layer's frozen norm

and the bf16 roundings of ``h'`` and ``z`` (the operands of the layer's bf16 projections of the stream and of its
norm).  f(.) is the exact widening to fp32.  These are the operations of the torch expression chain of the step
(``lex_read_step_ref``: the lexical read of :mod:`neuronspark.ops.triton.lex_read`, then the RMS norm and the two
casts); the row sums are formed in another order.  Tests: the per-operator tolerance rule of
:mod:`neuronspark.utils.testing` against the reference.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

import math
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....neuronspark.ops.triton import require_cuda
from ....neuronspark.ops.triton._ptx import _add_rn, _mul_rn, _rn_div_f32





__all__ = ["LEX_READ_STEP_WARPS", "lex_read_step", "lex_read_step_ref"]

# ================================================================================================ fixed constants
#: one program per row over the whole width (a power-of-two block with a mask), warps
LEX_READ_STEP_WARPS = 8


# ================================================================================================ kernel
@triton.jit
def _lex_read_step_kernel(H, Q, KR, O, GQ, GK, BETA, HO, HB, Z, ZB, s_h, s_q, s_k, s_o, D, eps, eps_z, sqrt_k,
                          K: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    kk = tl.arange(0, K)
    q = tl.load(Q + b * s_q + kk).to(tl.float32)
    k = tl.load(KR + b * s_k + kk).to(tl.float32)
    gq = 1.0 + tl.load(GQ + kk)
    gk = 1.0 + tl.load(GK + kk)
    mq = _rn_div_f32(tl.sum(q * q, axis=0), K * 1.0)
    mk = _rn_div_f32(tl.sum(k * k, axis=0), K * 1.0)
    qf = (q * libdevice.rsqrt(mq + eps)) * gq
    kf = (k * libdevice.rsqrt(mk + eps)) * gk
    x = _rn_div_f32(tl.sum(qf * kf, axis=0), sqrt_k) + tl.load(BETA)
    alpha = _rn_div_f32(1.0, 1.0 + libdevice.exp(-x))
    dd = tl.arange(0, BD)
    m = dd < D
    zero = tl.zeros([BD], tl.float32)
    h = tl.load(H + b * s_h + dd, mask=m, other=0.0)
    o = tl.load(O + b * s_o + dd, mask=m, other=0.0).to(tl.float32)
    hn = _add_rn(h, _mul_rn(zero + alpha, o))
    tl.store(HO + b * D + dd, hn, mask=m)
    tl.store(HB + b * D + dd, hn.to(tl.bfloat16), mask=m)
    z = _mul_rn(hn, zero + libdevice.rsqrt(tl.sum(hn * hn, axis=0) / D + eps_z))
    tl.store(Z + b * D + dd, z, mask=m)
    tl.store(ZB + b * D + dd, z.to(tl.bfloat16), mask=m)


# ================================================================================================ launcher
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.lex_read_step: " + msg)


def _lex_read_step_fwd(h, q_raw, k_raw, o, q_gain, k_gain, beta, eps: float, eps_z: float):
    require_cuda(h, "lex_read_step")
    _need(h.dim() == 2 and h.dtype == torch.float32 and h.stride(1) == 1,
          f"h must be fp32 (B, D) with unit inner stride, got {h.dtype} {tuple(h.shape)}")
    B, D = (int(s) for s in h.shape)
    dev = h.device
    K = int(q_raw.shape[-1]) if q_raw.dim() == 2 else 0
    _need(K >= 16 and K & (K - 1) == 0, f"the key width {K} must be a power of two >= 16")
    for name, t, n in (("q_raw", q_raw, K), ("k_raw", k_raw, K), ("o", o, D)):
        _need(tuple(t.shape) == (B, n) and t.dtype == torch.bfloat16 and t.stride(1) == 1 and t.device == dev,
              f"{name} must be bf16 ({B}, {n}) with unit inner stride, got {t.dtype} {tuple(t.shape)}")
    for name, t in (("q_gain", q_gain), ("k_gain", k_gain)):
        _need(t.dtype == torch.float32 and tuple(t.shape) == (K,) and t.is_contiguous() and t.device == dev,
              f"{name} must be a contiguous fp32 ({K},) tensor")
    _need(beta.dtype == torch.float32 and tuple(beta.shape) == (1,) and beta.device == dev,
          "beta must be a (1,) fp32 tensor")
    ho = torch.empty((B, D), dtype=torch.float32, device=dev)
    hb = torch.empty((B, D), dtype=torch.bfloat16, device=dev)
    z = torch.empty((B, D), dtype=torch.float32, device=dev)
    zb = torch.empty((B, D), dtype=torch.bfloat16, device=dev)
    _lex_read_step_kernel[(B,)](h, q_raw, k_raw, o, q_gain, k_gain, beta, ho, hb, z, zb, int(h.stride(0)),
                                int(q_raw.stride(0)), int(k_raw.stride(0)), int(o.stride(0)), D, float(eps),
                                float(eps_z), float(math.sqrt(K)), K=K, BD=triton.next_power_of_2(D),
                                num_warps=LEX_READ_STEP_WARPS, enable_fp_fusion=False)
    return ho, hb, z, zb


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::lex_read_step", mutates_args=())
def _lex_read_step_op(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor, q_gain: torch.Tensor,
                      k_gain: torch.Tensor, beta: torch.Tensor, eps: float,
                      eps_z: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _lex_read_step_fwd(h, q_raw, k_raw, o, q_gain, k_gain, beta, eps, eps_z)


@_lex_read_step_op.register_fake
def _lex_read_step_fake(h, q_raw, k_raw, o, q_gain, k_gain, beta, eps, eps_z):
    shape = tuple(h.shape)
    return (h.new_empty(shape, dtype=torch.float32), h.new_empty(shape, dtype=torch.bfloat16),
            h.new_empty(shape, dtype=torch.float32), h.new_empty(shape, dtype=torch.bfloat16))


# ================================================================================================ public function
def lex_read_step(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor, q_gain: torch.Tensor,
                  k_gain: torch.Tensor, beta: torch.Tensor, eps: float,
                  eps_z: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The lexical read of a decode step and the frozen norm of its result (module docstring).

    h (B, D) fp32: the stream; q_raw / k_raw (B, K) and o (B, D) bf16: the three projections (unit inner strides);
    q_gain / k_gain (K,) fp32: the zero-centred norm gains; beta (1,) fp32: the gate bias; eps: the eps of the two
    key norms; eps_z: the eps of the frozen norm.

    Returns (h' (B, D) fp32, h' (B, D) bf16, z (B, D) fp32, z (B, D) bf16)."""
    return _KERNEL_OPS.lex_read_step(h, q_raw, k_raw, o, q_gain, k_gain, beta, float(eps), float(eps_z))


# ================================================================================================ pure-torch reference
@torch.no_grad()
def lex_read_step_ref(h: torch.Tensor, q_raw: torch.Tensor, k_raw: torch.Tensor, o: torch.Tensor, q_gain: torch.Tensor,
                      k_gain: torch.Tensor, beta: torch.Tensor, eps: float,
                      eps_z: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The torch expression chain of the step: the zero-centred-gain RMS norms of q / k in fp32, the scaled dot plus
    the bias through a sigmoid, the fp32 residual ``h + alpha f(o)``, its RMS norm, and the two bf16 casts.  Same
    arguments and outputs as :func:`lex_read_step`."""
    K = int(q_raw.shape[-1])
    with torch.autocast("cuda", enabled=False):
        qf = q_raw.float()
        qf = qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + q_gain.float())
        kf = k_raw.float()
        kf = kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + eps) * (1.0 + k_gain.float())
        alpha = torch.sigmoid((qf * kf).sum(dim=-1, keepdim=True) / math.sqrt(K) + beta.float())
        hn = h + (alpha * o.float()).to(h.dtype)
        z = hn * torch.rsqrt(hn.pow(2).mean(dim=-1, keepdim=True) + eps_z)
    return hn, hn.to(torch.bfloat16), z, z.to(torch.bfloat16)
