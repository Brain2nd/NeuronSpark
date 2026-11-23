"""Unit half-queries and gained unit sub-keys of the product-key memory (training).

The PKM read (:mod:`neuronspark.modules.pkm`) scores every half-query against the sub-keys of its head and half::

    qh = normalize(f32(q).view(T, B, H, 2, dh))            unit half-queries, q = bf16(x_hat @ pkm_q^T)
    kg = normalize(pkm_subkeys) * beta[h, a]                gained unit sub-keys (H, 2, n, dh)
    s  = einsum("tbhad,hand->tbhan", qh, kg)                beta = sqrt(dh) exp(pkm_log_gain).view(H, 2)

with ``normalize(x) = x / max(|x|, 1e-12)`` over the last dim.  The gain multiplies the keys, so the scores are the
gained cosines ``(qh . kh) beta`` with the product associated as ``qh . (kh beta)``.

Operators (training):

* ``pkm_unit_queries(q, heads, half)`` (:class:`PKMUnitQueriesFn`): q (T, B, heads * 2 * half) bf16 -> qh (T, B, heads,
  2, half) fp32 contiguous; backward from fp32 ``g`` of any strides: ``dq = bf16(g / n - x (g . x) / n^3)``.  One
  forward and one backward kernel, ``QRY_VECS`` half-queries per program.
* ``pkm_gained_keys(subkeys, log_gain)`` (:class:`PKMGainedKeysFn`): (H, 2, n, dh) fp32 and (2 H,) fp32 -> kg; backward
  ``dsub = gb / n - x (gb . x) / n^3`` with ``gb = g beta`` and ``dlog_gain[h, a] = sum_{n, d} g kg``.  One program per
  (head, half).

Numerics: fp32 register sums.  Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the
torch chains (:func:`pkm_unit_queries_ref`, :func:`pkm_gained_keys_ref`).
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE

import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ....neuronspark.ops.triton import require_cuda

__all__ = ["PKMUnitQueriesFn", "pkm_unit_queries", "pkm_unit_queries_ref",
           "PKMGainedKeysFn", "pkm_gained_keys", "pkm_gained_keys_ref", "NORM_EPS"]

# ================================================================================================ fixed constants
#: eps of F.normalize (the clamp of the norm)
NORM_EPS = 1e-12
#: half-queries: half-queries per program (a power of two), warps
QRY_VECS = 64
QRY_WARPS = 4
#: sub-keys: one (head, half) per program, warps
KEY_WARPS = 4


# ================================================================================================ kernels
@triton.jit
def _pq_fwd_kernel(Q, QH, R, B, G, DH, qs0, qs1, eps, NV: tl.constexpr, DB: tl.constexpr):
    """qh[r, g, :] = x / max(|x|, eps), x = f32(q[r, g * DH : (g + 1) * DH]), for NV of the R * G half-queries."""
    gi = tl.program_id(0).to(tl.int64) * NV + tl.arange(0, NV)
    r = gi // G
    g = gi - r * G
    t = r // B
    b = r - t * B
    d = tl.arange(0, DB)
    m = (r < R)[:, None] & (d < DH)[None, :]
    x = tl.load(Q + (t * qs0 + b * qs1 + g * DH)[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32)
    inv = 1.0 / tl.maximum(tl.sqrt(tl.sum(x * x, axis=1)), eps)
    tl.store(QH + (gi * DH)[:, None] + d[None, :], x * inv[:, None], mask=m)


@triton.jit
def _pq_bwd_kernel(GQ, Q, DQ, R, B, G, DH, gs0, gs1, gs2, gs3, gs4, qs0, qs1, ds0, ds1, eps,
                   NV: tl.constexpr, DB: tl.constexpr):
    """dq = bf16(g / n - x (g . x) / n^3) written through dq's strides; g read through its five strides."""
    gi = tl.program_id(0).to(tl.int64) * NV + tl.arange(0, NV)
    r = gi // G
    g = gi - r * G
    t = r // B
    b = r - t * B
    hh = g // 2
    aa = g - hh * 2
    d = tl.arange(0, DB)
    m = (r < R)[:, None] & (d < DH)[None, :]
    x = tl.load(Q + (t * qs0 + b * qs1 + g * DH)[:, None] + d[None, :], mask=m, other=0.0).to(tl.float32)
    gr = tl.load(GQ + (t * gs0 + b * gs1 + hh * gs2 + aa * gs3)[:, None] + d.to(tl.int64)[None, :] * gs4, mask=m,
                 other=0.0)
    n = tl.sqrt(tl.sum(x * x, axis=1))
    inv = 1.0 / tl.maximum(n, eps)
    proj = tl.where(n >= eps, tl.sum(gr * x, axis=1) * inv * inv * inv, 0.0)
    tl.store(DQ + (t * ds0 + b * ds1 + g * DH)[:, None] + d[None, :],
             (gr * inv[:, None] - x * proj[:, None]).to(tl.bfloat16), mask=m)


@triton.jit
def _pkey_fwd_kernel(SK, LG, KG, N, DH, sqrt_dh, eps, NB: tl.constexpr, DB: tl.constexpr):
    """kg[ha, n, :] = x / max(|x|, eps) * sqrt(dh) exp(log_gain[ha]) for one (head, half) ha."""
    ha = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, NB)
    d = tl.arange(0, DB)
    m = (rows < N)[:, None] & (d < DH)[None, :]
    off = ha * N * DH + rows[:, None] * DH + d[None, :]
    x = tl.load(SK + off, mask=m, other=0.0)
    inv = 1.0 / tl.maximum(tl.sqrt(tl.sum(x * x, axis=1)), eps)
    beta = sqrt_dh * tl.exp(tl.load(LG + ha))
    tl.store(KG + off, x * (inv * beta)[:, None], mask=m)


@triton.jit
def _pkey_bwd_kernel(GK, SK, LG, DSK, DLG, N, DH, gs0, gs1, gs2, gs3, sqrt_dh, eps,
                     NB: tl.constexpr, DB: tl.constexpr):
    """dsub = gb / n - x (gb . x) / n^3 with gb = g beta; dlog_gain[ha] = sum g kg."""
    ha = tl.program_id(0).to(tl.int64)
    hh = ha // 2
    aa = ha - hh * 2
    rows = tl.arange(0, NB)
    d = tl.arange(0, DB)
    m = (rows < N)[:, None] & (d < DH)[None, :]
    off = ha * N * DH + rows[:, None] * DH + d[None, :]
    x = tl.load(SK + off, mask=m, other=0.0)
    g = tl.load(GK + (hh * gs0 + aa * gs1) + rows.to(tl.int64)[:, None] * gs2 + d.to(tl.int64)[None, :] * gs3, mask=m,
                other=0.0)
    n = tl.sqrt(tl.sum(x * x, axis=1))
    inv = 1.0 / tl.maximum(n, eps)
    beta = sqrt_dh * tl.exp(tl.load(LG + ha))
    gx = tl.sum(g * x, axis=1)                                                       # [NB]
    tl.store(DLG + ha, tl.sum(gx * inv, axis=0) * beta)
    proj = tl.where(n >= eps, gx * inv * inv * inv, 0.0)
    tl.store(DSK + off, (g * inv[:, None] - x * proj[:, None]) * beta, mask=m)


# ================================================================================================ half-queries
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.pkm_query: " + msg)


def _dest_like(t: torch.Tensor) -> torch.Tensor:
    """An uninitialised tensor like ``t`` that the kernels write row by row: the layout torch gives a tensor like t
    when its inner stride is 1 (a dense t: t's own layout), row-major otherwise (a t whose strides are not dense)."""
    d = torch.empty_like(t)
    return d if d.stride(-1) == 1 else torch.empty(tuple(t.shape), dtype=t.dtype, device=t.device)


def _pow2(n: int) -> int:
    return int(triton.next_power_of_2(int(n)))


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_unit_queries_fwd", mutates_args=())
def _pkm_unit_queries_fwd(q: torch.Tensor, heads: int, half: int) -> torch.Tensor:
    T, B = int(q.shape[0]), int(q.shape[1])
    G = 2 * int(heads)
    qh = torch.empty((T, B, int(heads), 2, int(half)), dtype=torch.float32, device=q.device)
    R = T * B
    if R:
        _pq_fwd_kernel[(triton.cdiv(R * G, QRY_VECS),)](q, qh, R, B, G, int(half), int(q.stride(0)), int(q.stride(1)),
                                                        NORM_EPS, NV=QRY_VECS, DB=_pow2(half),
                                                        num_warps=QRY_WARPS)
    return qh


@_pkm_unit_queries_fwd.register_fake
def _pkm_unit_queries_fwd_fake(q, heads, half):
    return q.new_empty((q.shape[0], q.shape[1], heads, 2, half), dtype=torch.float32)


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_unit_queries_bwd", mutates_args=())
def _pkm_unit_queries_bwd(g: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    T, B, H, _, DH = (int(v) for v in g.shape)
    dq = _dest_like(q)
    R = T * B
    if R:
        _pq_bwd_kernel[(triton.cdiv(R * 2 * H, QRY_VECS),)](
            g, q, dq, R, B, 2 * H, DH, *(int(s) for s in g.stride()), int(q.stride(0)), int(q.stride(1)),
            int(dq.stride(0)), int(dq.stride(1)), NORM_EPS, NV=QRY_VECS, DB=_pow2(DH), num_warps=QRY_WARPS)
    return dq


@_pkm_unit_queries_bwd.register_fake
def _pkm_unit_queries_bwd_fake(g, q):
    return q.new_empty(tuple(q.shape))


class PKMUnitQueriesFn(torch.autograd.Function):
    """qh (T, B, H, 2, dh) fp32 unit half-queries of the bf16 projection q (T, B, H * 2 * dh); backward to bf16."""

    @staticmethod
    def forward(ctx, q, heads, half):
        qh = _pkm_unit_queries_fwd(q, int(heads), int(half))
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q)
        return qh

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None, None
        (q,) = ctx.saved_tensors
        if g.dtype != torch.float32 or g.dim() != 5 or g.numel() != q.numel():
            raise RuntimeError(f"neuronspark.pkm_unit_queries backward: gradient {g.dtype} {tuple(g.shape)} for "
                               f"{tuple(q.shape)}")
        return _pkm_unit_queries_bwd(g, q), None, None


def pkm_unit_queries(q: torch.Tensor, heads: int, half: int) -> torch.Tensor:
    """``F.normalize(q.float().view(T, B, heads, 2, half), dim=-1)`` (T, B, heads, 2, half) fp32 contiguous.

    ``q`` (T, B, heads * 2 * half) bf16 with a unit inner stride (any (t, b) strides).  Differentiable in q (bf16
    gradient).  Training regime."""
    require_cuda(q, "pkm_unit_queries")
    heads, half = int(heads), int(half)
    G = 2 * heads
    _need(q.dim() == 3 and q.dtype == torch.bfloat16 and q.stride(-1) == 1 and int(q.shape[-1]) == G * half,
          f"q must be bf16 (T, B, {G * half}) with a unit inner stride, got {q.dtype} {tuple(q.shape)} "
          f"{tuple(q.stride())}")
    _need(heads >= 1 and half >= 1 and q.numel() < 2 ** 31, f"heads = {heads}, half = {half}, {q.numel()} elements")
    return PKMUnitQueriesFn.apply(q, heads, half)


def pkm_unit_queries_ref(q: torch.Tensor, heads: int, half: int) -> torch.Tensor:
    """The torch chain: the fp32 unit half-queries of the bf16 projection."""
    T, B = int(q.shape[0]), int(q.shape[1])
    return F.normalize(q.to(torch.float32).reshape(T, B, int(heads), 2, int(half)), dim=-1)


# ================================================================================================ sub-keys
@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_gained_keys_fwd", mutates_args=())
def _pkm_gained_keys_fwd(subkeys: torch.Tensor, log_gain: torch.Tensor) -> torch.Tensor:
    H, _, N, DH = (int(v) for v in subkeys.shape)
    kg = torch.empty((H, 2, N, DH), dtype=torch.float32, device=subkeys.device)
    _pkey_fwd_kernel[(2 * H,)](subkeys, log_gain, kg, N, DH, math.sqrt(DH), NORM_EPS, NB=_pow2(N), DB=_pow2(DH),
                               num_warps=KEY_WARPS)
    return kg


@_pkm_gained_keys_fwd.register_fake
def _pkm_gained_keys_fwd_fake(subkeys, log_gain):
    return subkeys.new_empty(tuple(subkeys.shape))


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_gained_keys_bwd", mutates_args=())
def _pkm_gained_keys_bwd(g: torch.Tensor, subkeys: torch.Tensor,
                         log_gain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    H, _, N, DH = (int(v) for v in subkeys.shape)
    dsk = torch.empty((H, 2, N, DH), dtype=torch.float32, device=subkeys.device)
    dlg = torch.empty((2 * H,), dtype=torch.float32, device=subkeys.device)
    _pkey_bwd_kernel[(2 * H,)](g, subkeys, log_gain, dsk, dlg, N, DH, *(int(s) for s in g.stride()), math.sqrt(DH),
                               NORM_EPS, NB=_pow2(N), DB=_pow2(DH), num_warps=KEY_WARPS)
    return dsk, dlg


@_pkm_gained_keys_bwd.register_fake
def _pkm_gained_keys_bwd_fake(g, subkeys, log_gain):
    return subkeys.new_empty(tuple(subkeys.shape)), log_gain.new_empty(tuple(log_gain.shape))


class PKMGainedKeysFn(torch.autograd.Function):
    """kg (H, 2, n, dh) = normalize(subkeys) sqrt(dh) exp(log_gain.view(H, 2)); gradients for both."""

    @staticmethod
    def forward(ctx, subkeys, log_gain):
        kg = _pkm_gained_keys_fwd(subkeys, log_gain)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(subkeys, log_gain)
        return kg

    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None
        subkeys, log_gain = ctx.saved_tensors
        if g.dtype != torch.float32 or tuple(g.shape) != tuple(subkeys.shape):
            raise RuntimeError(f"neuronspark.pkm_gained_keys backward: gradient {g.dtype} {tuple(g.shape)} for "
                               f"{tuple(subkeys.shape)}")
        dsk, dlg = _pkm_gained_keys_bwd(g, subkeys, log_gain)
        need = ctx.needs_input_grad
        return dsk if need[0] else None, dlg if need[1] else None


def pkm_gained_keys(subkeys: torch.Tensor, log_gain: torch.Tensor) -> torch.Tensor:
    """``F.normalize(subkeys, dim=-1) * (sqrt(dh) * exp(log_gain)).view(H, 2, 1, 1)`` (H, 2, n, dh) fp32 contiguous.

    ``subkeys`` (H, 2, n, dh) fp32 contiguous, ``log_gain`` (2 H,) fp32 contiguous.  Differentiable in both.  Training
    regime."""
    require_cuda(subkeys, "pkm_gained_keys")
    _need(subkeys.dim() == 4 and subkeys.dtype == torch.float32 and subkeys.is_contiguous()
          and int(subkeys.shape[1]) == 2, f"subkeys must be contiguous fp32 (H, 2, n, dh), got {subkeys.dtype} "
          f"{tuple(subkeys.shape)}")
    H, _, N, DH = (int(v) for v in subkeys.shape)
    _need(N >= 1 and DH >= 1, f"subkeys must be contiguous fp32 (H, 2, n, dh), got {tuple(subkeys.shape)}")
    _need(log_gain.dtype == torch.float32 and tuple(log_gain.shape) == (2 * H,) and log_gain.is_contiguous()
          and log_gain.device == subkeys.device, f"log_gain must be a contiguous fp32 ({2 * H},) tensor on the "
          "sub-keys' device")
    return PKMGainedKeysFn.apply(subkeys, log_gain)


def pkm_gained_keys_ref(subkeys: torch.Tensor, log_gain: torch.Tensor) -> torch.Tensor:
    """The torch chain: the unit sub-keys times beta = sqrt(dh) exp(log_gain).view(H, 2)."""
    H, _, _, DH = (int(v) for v in subkeys.shape)
    beta = math.sqrt(DH) * torch.exp(log_gain.to(torch.float32)).view(H, 2, 1, 1)
    return F.normalize(subkeys.to(torch.float32), dim=-1) * beta
