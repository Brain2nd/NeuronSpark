"""Decode-step residual operators of the hippocampal layer (T = 1): the read of the attention residual into the stream
and the layer's tail, one kernel each, one program per row.

``read_gain_step``  from the output projection ``r`` (B, D) bf16 of the attention sub-block to the FFN's input, per row::

        r_c   = bf16(r - bf16(mean_c r))                              the centring (bf16 arithmetic)
        pi    = exp(scale max - logaddexp(sink, lse))                  copy confidence (sink = scale tau of the layer)
        c     = sigmoid((z . w + b) + (f(r_c) rsqrt(mean_c f(r_c)^2 + eps_r)) . v)        novelty
        h'    = h + bf16(f(r_c) (1 + (kappa pi) c))                    the read gain and the fp32 residual addition
        x_hat = (h' rsqrt(mean_c h'^2 + eps)) gain                     the FFN's RMSNorm, fp32 and its bf16 rounding
        rho   = 2 sigmoid(-strength pi)                                the PKM copy gate

``pkm_tail_step``  the layer's last addition, per row::

        y_g   = bf16(f(y) rho)
        out   = h + bf16(ffn + bf16(y_g - bf16(mean_c y_g)))           bf16 arithmetic, fp32 residual addition

f(.) is the exact widening to fp32; every fp32 operation of the chains is rounded once (no FMA contraction).  These
are the operations of the torch expression chains of the step (``read_gain_step_ref`` / ``pkm_tail_step_ref``) at
their rounding points; the row sums (means, dot products) are formed in another order.  Tests: the per-operator
tolerance rule of :mod:`neuronspark.utils.testing` against the references.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (resolved by the kernels below through this module's globals)
    _add_rn,
    _mul_rn,
    _sub_rn,
)

__all__ = ["HIPPO_STEP_WARPS", "read_gain_step", "read_gain_step_ref", "pkm_tail_step", "pkm_tail_step_ref"]

# ================================================================================================ fixed constants
#: one program per row over the whole width (a power-of-two block with a mask), warps
HIPPO_STEP_WARPS = 8


# ================================================================================================ kernels
@triton.jit
def _bf16(x):
    """Round an fp32 value to the nearest bf16 (ties to even) and widen it back through the bit pattern (finite
    values)."""
    u = x.to(tl.int32, bitcast=True)
    lsb = (u >> 16) & 1
    u = u + 0x7FFF + lsb
    u = u & -65536                     # 0xFFFF0000 as a signed int32
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _read_gain_step_kernel(H, Z, R, CMAX, CLSE, SINK, STR, NW, NV, NB, GW, HO, XH, XB, RHO,
                           s_h, s_z, s_r, s_cmax, s_clse, D, kappa, scale, eps_r, eps_n, BD: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    dd = tl.arange(0, BD)
    m = dd < D
    zero = tl.zeros([BD], tl.float32)
    # ---- the centred residual (bf16 mean, bf16 subtraction)
    r = tl.load(R + b * s_r + dd, mask=m, other=0.0).to(tl.float32)
    mean = _bf16(tl.sum(r, axis=0) / D)
    rc = tl.where(m, _bf16(_sub_rn(r, zero + mean)), zero)
    # ---- copy confidence
    sink = tl.load(SINK)
    lse = tl.load(CLSE + b * s_clse)
    la = tl.maximum(sink, lse) + libdevice.log1p(libdevice.exp(-tl.abs(sink - lse)))
    pi = libdevice.exp(scale * tl.load(CMAX + b * s_cmax) - la)
    # ---- novelty
    z = tl.load(Z + b * s_z + dd, mask=m, other=0.0)
    logit = tl.sum(z * tl.load(NW + dd, mask=m, other=0.0), axis=0) + tl.load(NB)
    rf = rc * libdevice.rsqrt(tl.sum(rc * rc, axis=0) / D + eps_r)
    logit = logit + tl.sum(rf * tl.load(NV + dd, mask=m, other=0.0), axis=0)
    c = 1.0 / (1.0 + libdevice.exp(-logit))
    # ---- read gain and the residual addition
    f1 = 1.0 + (kappa * pi) * c
    hn = _add_rn(tl.load(H + b * s_h + dd, mask=m, other=0.0), _bf16(_mul_rn(rc, zero + f1)))
    tl.store(HO + b * D + dd, hn, mask=m)
    # ---- the FFN's pre-norm (fp32 and its bf16 rounding) and the PKM copy gate
    xh = _mul_rn(_mul_rn(hn, zero + libdevice.rsqrt(tl.sum(hn * hn, axis=0) / D + eps_n)),
                 tl.load(GW + dd, mask=m, other=0.0))
    tl.store(XH + b * D + dd, xh, mask=m)
    tl.store(XB + b * D + dd, xh.to(tl.bfloat16), mask=m)
    tl.store(RHO + b, 2.0 / (1.0 + libdevice.exp(tl.load(STR) * pi)))


@triton.jit
def _pkm_tail_step_kernel(H, F, Y, RHO, HO, s_h, s_f, s_y, D, BD: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    dd = tl.arange(0, BD)
    m = dd < D
    zero = tl.zeros([BD], tl.float32)
    y = tl.load(Y + b * s_y + dd, mask=m, other=0.0).to(tl.float32)
    yg = _bf16(_mul_rn(y, zero + tl.load(RHO + b)))
    mean = _bf16(tl.sum(yg, axis=0) / D)
    d = _bf16(_sub_rn(yg, zero + mean))
    ff = _bf16(_add_rn(tl.load(F + b * s_f + dd, mask=m, other=0.0).to(tl.float32), d))
    tl.store(HO + b * D + dd, _add_rn(tl.load(H + b * s_h + dd, mask=m, other=0.0), ff), mask=m)


# ================================================================================================ launchers
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.hippo_step: " + msg)


def _rows(name: str, t: torch.Tensor, B: int, D: int, dtype: torch.dtype) -> None:
    _need(tuple(t.shape) == (B, D) and t.dtype == dtype and t.stride(1) == 1,
          f"{name} must be {dtype} ({B}, {D}) with unit inner stride, got {t.dtype} {tuple(t.shape)}")


def _vec(name: str, t: torch.Tensor, n: int) -> None:
    _need(tuple(t.shape) == (n,) and t.dtype == torch.float32 and t.is_contiguous(),
          f"{name} must be a contiguous fp32 ({n},) tensor, got {t.dtype} {tuple(t.shape)}")


def _read_gain_step_fwd(h, z, r, cmax, clse, sink, strength, nov_w, nov_v, nov_b, norm_weight, kappa: float,
                        scale: float, eps_r: float, eps_n: float):
    require_cuda(h, "read_gain_step")
    _need(h.dim() == 2, f"h must be fp32 (B, D), got {tuple(h.shape)}")
    B, D = (int(s) for s in h.shape)
    _rows("h", h, B, D, torch.float32)
    _rows("z", z, B, D, torch.float32)
    _rows("r", r, B, D, torch.bfloat16)
    for name, t in (("the copy maximum", cmax), ("the copy logsumexp", clse)):
        _need(tuple(t.shape) == (B,) and t.dtype == torch.float32,
              f"{name} must be an fp32 ({B},) tensor, got {t.dtype} {tuple(t.shape)}")
    for name, t in (("sink", sink), ("strength", strength), ("nov_b", nov_b)):
        _vec(name, t, 1)
    for name, t in (("nov_w", nov_w), ("nov_v", nov_v), ("norm_weight", norm_weight)):
        _vec(name, t, D)
    dev = h.device
    ho = torch.empty((B, D), dtype=torch.float32, device=dev)
    xh = torch.empty((B, D), dtype=torch.float32, device=dev)
    xb = torch.empty((B, D), dtype=torch.bfloat16, device=dev)
    rho = torch.empty((B,), dtype=torch.float32, device=dev)
    _read_gain_step_kernel[(B,)](h, z, r, cmax, clse, sink, strength, nov_w, nov_v, nov_b, norm_weight, ho, xh, xb,
                                 rho, int(h.stride(0)), int(z.stride(0)), int(r.stride(0)), int(cmax.stride(0)),
                                 int(clse.stride(0)), D, float(kappa), float(scale), float(eps_r), float(eps_n),
                                 BD=triton.next_power_of_2(D), num_warps=HIPPO_STEP_WARPS, enable_fp_fusion=False)
    return ho, xh, xb, rho


def _pkm_tail_step_fwd(h, ffn, y, rho):
    require_cuda(h, "pkm_tail_step")
    _need(h.dim() == 2, f"h must be fp32 (B, D), got {tuple(h.shape)}")
    B, D = (int(s) for s in h.shape)
    _rows("h", h, B, D, torch.float32)
    _rows("ffn", ffn, B, D, torch.bfloat16)
    _rows("y", y, B, D, torch.bfloat16)
    _need(tuple(rho.shape) == (B,) and rho.dtype == torch.float32 and rho.is_contiguous(),
          f"rho must be a contiguous fp32 ({B},) tensor, got {rho.dtype} {tuple(rho.shape)}")
    ho = torch.empty((B, D), dtype=torch.float32, device=h.device)
    _pkm_tail_step_kernel[(B,)](h, ffn, y, rho, ho, int(h.stride(0)), int(ffn.stride(0)), int(y.stride(0)), D,
                                BD=triton.next_power_of_2(D), num_warps=HIPPO_STEP_WARPS, enable_fp_fusion=False)
    return ho


@torch.library.custom_op("neuronspark::read_gain_step", mutates_args=())
def _read_gain_step_op(h: torch.Tensor, z: torch.Tensor, r: torch.Tensor, cmax: torch.Tensor, clse: torch.Tensor,
                       sink: torch.Tensor, strength: torch.Tensor, nov_w: torch.Tensor, nov_v: torch.Tensor,
                       nov_b: torch.Tensor, norm_weight: torch.Tensor, kappa: float, scale: float, eps_r: float,
                       eps_n: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _read_gain_step_fwd(h, z, r, cmax, clse, sink, strength, nov_w, nov_v, nov_b, norm_weight, kappa, scale,
                               eps_r, eps_n)


@_read_gain_step_op.register_fake
def _read_gain_step_fake(h, z, r, cmax, clse, sink, strength, nov_w, nov_v, nov_b, norm_weight, kappa, scale, eps_r,
                         eps_n):
    shape = tuple(h.shape)
    return (h.new_empty(shape, dtype=torch.float32), h.new_empty(shape, dtype=torch.float32),
            h.new_empty(shape, dtype=torch.bfloat16), h.new_empty(shape[:1], dtype=torch.float32))


@torch.library.custom_op("neuronspark::pkm_tail_step", mutates_args=())
def _pkm_tail_step_op(h: torch.Tensor, ffn: torch.Tensor, y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    return _pkm_tail_step_fwd(h, ffn, y, rho)


@_pkm_tail_step_op.register_fake
def _pkm_tail_step_fake(h, ffn, y, rho):
    return h.new_empty(tuple(h.shape), dtype=torch.float32)


# ================================================================================================ public functions
def read_gain_step(h: torch.Tensor, z: torch.Tensor, r: torch.Tensor, cmax: torch.Tensor, clse: torch.Tensor,
                   sink: torch.Tensor, strength: torch.Tensor, nov_w: torch.Tensor, nov_v: torch.Tensor,
                   nov_b: torch.Tensor, norm_weight: torch.Tensor, kappa: float, scale: float, eps_r: float,
                   eps_n: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The attention residual read into the stream (module docstring).

    h (B, D) fp32: the stream after the lexical read; z (B, D) fp32: its frozen norm; r (B, D) bf16: the attention
    output projection (before the centring); cmax / clse (B,) fp32: the copy detector's maximum and logsumexp; sink
    (1,) fp32 = scale tau; strength (1,) fp32: the PKM copy gate's strength; nov_w / nov_v (D,), nov_b (1,) fp32: the
    novelty predictor; norm_weight (D,) fp32: the FFN norm's gain; kappa: the read gain; scale: the copy logit scale;
    eps_r / eps_n: the eps of the novelty rms and of the FFN norm.

    Returns (h' (B, D) fp32, x_hat (B, D) fp32, x_hat (B, D) bf16, rho (B,) fp32)."""
    return torch.ops.neuronspark.read_gain_step(h, z, r, cmax, clse, sink, strength, nov_w, nov_v, nov_b, norm_weight,
                                                float(kappa), float(scale), float(eps_r), float(eps_n))


def pkm_tail_step(h: torch.Tensor, ffn: torch.Tensor, y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """(B, D) fp32 ``h + (ffn + (y_g - mean_c(y_g)))`` with ``y_g = bf16(f(y) rho)``: h (B, D) fp32, ffn / y (B, D)
    bf16, rho (B,) fp32."""
    return torch.ops.neuronspark.pkm_tail_step(h, ffn, y, rho)


# ================================================================================================ pure-torch references
@torch.no_grad()
def read_gain_step_ref(h: torch.Tensor, z: torch.Tensor, r: torch.Tensor, cmax: torch.Tensor, clse: torch.Tensor,
                       sink: torch.Tensor, strength: torch.Tensor, nov_w: torch.Tensor, nov_v: torch.Tensor,
                       nov_b: torch.Tensor, norm_weight: torch.Tensor, kappa: float, scale: float, eps_r: float,
                       eps_n: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The torch expression chain of the step: the centring, the copy confidence, the novelty predictor, the read
    gain, the residual addition, the FFN's RMSNorm and the PKM copy gate.  Same arguments and outputs as
    :func:`read_gain_step`."""
    with torch.autocast("cuda", enabled=False):
        rc = r - r.mean(dim=-1, keepdim=True)                                       # bf16
        pi = torch.exp(scale * cmax - torch.logaddexp(torch.ones_like(clse) * sink, clse))
        logit = torch.matmul(z.float(), nov_w.float()) + nov_b.float()
        rf = rc.float()
        rf = rf * torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + eps_r)
        c = torch.sigmoid(logit + torch.matmul(rf, nov_v.float()))
        f = (kappa * pi) * c
        hn = h + (rc.float() * (1.0 + f).unsqueeze(-1)).to(rc.dtype)
        xh = norm_weight * (hn * torch.rsqrt(hn.pow(2).mean(dim=-1, keepdim=True) + eps_n))
        rho = 2.0 * torch.sigmoid(-strength.float() * pi)
    return hn, xh, xh.to(torch.bfloat16), rho


@torch.no_grad()
def pkm_tail_step_ref(h: torch.Tensor, ffn: torch.Tensor, y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """The torch lines of the layer's tail: ``h + (ffn + (y_g - y_g.mean(-1, keepdim=True)))``."""
    yg = (y.float() * rho.unsqueeze(-1)).to(y.dtype)
    return h + (ffn + (yg - yg.mean(dim=-1, keepdim=True)))
