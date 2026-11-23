"""Decode-step operators of the product-key memory read (T = 1), around the top-k selection kernel
(:func:`neuronspark.ops.triton.pkm_topk.pkm_topk_softmax_t1`).

``pkm_scores_step``  the half-query scores of one token, one program per (row, head, half)::

        qh = f(q) / max(|f(q)|, 1e-12)                       the unit half-query (dh wide), fp32
        s  = kh @ qh                                         (n,) fp32 cosines with the n unit sub-keys kh (n, dh)

``pkm_value_step``  the read after the selection, one program per row::

        o = sum_m w_m V[u_m]                                 the weighted sum of the M selected value rows, fp32
        r = bf16(((o rsqrt(mean_c o^2 + eps)) gain) sigmoid(f(glog)))        ZCN, gate, one rounding to bf16 (the
                                                             operand of the output projection)

f(.) is the exact widening to fp32.  These are the operations of the torch expression chains of the step
(``pkm_scores_step_ref`` / ``pkm_value_step_ref``); the dot products and row sums are formed in another order.
Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the references.
"""
from __future__ import annotations; from ....neuronspark._namespace import NAMESPACE as _KERNEL_OP_NAMESPACE, OPS as _KERNEL_OPS

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ....neuronspark.ops.triton import require_cuda

__all__ = ["PKM_SCORES_WARPS", "PKM_VALUE_WARPS", "pkm_scores_step", "pkm_scores_step_ref", "pkm_value_step",
           "pkm_value_step_ref"]

# ================================================================================================ fixed constants
#: scores: one program per (row, head, half), warps; value read: one program per row over the whole value width, warps
#: (the M value rows are gathered from a table the step has not touched since its last token: the kernel's time is the
#: number of loads a thread issues one after the other, so the tile is spread over as many threads as a block has)
PKM_SCORES_WARPS = 4
PKM_VALUE_WARPS = 32


# ================================================================================================ kernels
@triton.jit
def _pkm_scores_step_kernel(Q, KH, S, s_q, HA, N, DH, DB: tl.constexpr, NB: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    ha = tl.program_id(1)
    dd = tl.arange(0, DB)
    dm = dd < DH
    q = tl.load(Q + b * s_q + ha * DH + dd, mask=dm, other=0.0).to(tl.float32)
    qh = q / tl.maximum(tl.sqrt(tl.sum(q * q, axis=0)), 1e-12)
    nn = tl.arange(0, NB)
    nm = nn < N
    k = tl.load(KH + (ha * N + nn)[:, None].to(tl.int64) * DH + dd[None, :], mask=nm[:, None] & dm[None, :], other=0.0)
    tl.store(S + (b * HA + ha) * N + nn, tl.sum(k * qh[None, :], axis=1), mask=nm)


@triton.jit
def _pkm_value_step_kernel(Wt, U, VALS, G1, GLOG, OUT, s_g, DV, M, eps, MB: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    dd = tl.arange(0, BD)
    m = dd < DV
    mm = tl.arange(0, MB)
    sel = mm < M
    w = tl.load(Wt + b * M + mm, mask=sel, other=0.0)
    u = tl.load(U + b * M + mm, mask=sel, other=0)
    # the M selected value rows as one (MB, BD) tile: a thread issues its few loads of the cold table at once
    rows = tl.load(VALS + (u * DV)[:, None] + dd[None, :], mask=sel[:, None] & m[None, :], other=0.0)
    acc = tl.sum(w[:, None] * rows, axis=0)
    r = (acc * libdevice.rsqrt(tl.sum(acc * acc, axis=0) / DV + eps)) * tl.load(G1 + dd, mask=m, other=0.0)
    g = tl.load(GLOG + b * s_g + dd, mask=m, other=0.0).to(tl.float32)
    r = r * (1.0 / (1.0 + libdevice.exp(-g)))
    tl.store(OUT + b * DV + dd, r.to(tl.bfloat16), mask=m)


# ================================================================================================ launchers
def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.pkm_step: " + msg)


def _pkm_scores_step_fwd(q: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
    require_cuda(q, "pkm_scores_step")
    _need(kh.dim() == 4 and kh.dtype == torch.float32 and kh.is_contiguous() and int(kh.shape[1]) == 2,
          f"kh must be a contiguous fp32 (H, 2, n, dh) tensor, got {kh.dtype} {tuple(kh.shape)}")
    H, _, n, dh = (int(s) for s in kh.shape)
    _need(q.dim() == 2 and int(q.shape[1]) == H * 2 * dh and q.dtype in (torch.bfloat16, torch.float32)
          and q.stride(1) == 1,
          f"q must be bf16 / fp32 (B, {H * 2 * dh}) with unit inner stride, got {q.dtype} {tuple(q.shape)}")
    B = int(q.shape[0])
    s = torch.empty((B, H, 2, n), dtype=torch.float32, device=q.device)
    _pkm_scores_step_kernel[(B, 2 * H)](q, kh, s, int(q.stride(0)), 2 * H, n, dh, DB=triton.next_power_of_2(dh),
                                        NB=triton.next_power_of_2(n), num_warps=PKM_SCORES_WARPS,
                                        enable_fp_fusion=False)
    return s


def _pkm_value_step_fwd(w: torch.Tensor, u: torch.Tensor, values: torch.Tensor, gain: torch.Tensor,
                        glog: torch.Tensor, eps: float) -> torch.Tensor:
    require_cuda(w, "pkm_value_step")
    _need(w.dim() == 2 and w.dtype == torch.float32 and w.is_contiguous(),
          f"w must be a contiguous fp32 (B, M) tensor, got {w.dtype} {tuple(w.shape)}")
    B, M = (int(s) for s in w.shape)
    _need(tuple(u.shape) == (B, M) and u.dtype == torch.int64 and u.is_contiguous(),
          f"u must be a contiguous int64 ({B}, {M}) tensor, got {u.dtype} {tuple(u.shape)}")
    _need(values.dim() == 2 and values.dtype == torch.float32 and values.is_contiguous(),
          f"the value table must be a contiguous fp32 (slots, dv) tensor, got {values.dtype} {tuple(values.shape)}")
    dv = int(values.shape[1])
    _need(tuple(gain.shape) == (dv,) and gain.dtype == torch.float32 and gain.is_contiguous(),
          f"gain must be a contiguous fp32 ({dv},) tensor")
    _need(tuple(glog.shape) == (B, dv) and glog.dtype == torch.bfloat16 and glog.stride(1) == 1,
          f"glog must be bf16 ({B}, {dv}) with unit inner stride, got {glog.dtype} {tuple(glog.shape)}")
    out = torch.empty((B, dv), dtype=torch.bfloat16, device=w.device)
    _pkm_value_step_kernel[(B,)](w, u, values, gain, glog, out, int(glog.stride(0)), dv, M, float(eps),
                                 MB=triton.next_power_of_2(M), BD=triton.next_power_of_2(dv),
                                 num_warps=PKM_VALUE_WARPS, enable_fp_fusion=False)
    return out


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_scores_step", mutates_args=())
def _pkm_scores_step_op(q: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
    return _pkm_scores_step_fwd(q, kh)


@_pkm_scores_step_op.register_fake
def _pkm_scores_step_fake(q, kh):
    return q.new_empty((q.shape[0], kh.shape[0], 2, kh.shape[2]), dtype=torch.float32)


@torch.library.custom_op(_KERNEL_OP_NAMESPACE + "::pkm_value_step", mutates_args=())
def _pkm_value_step_op(w: torch.Tensor, u: torch.Tensor, values: torch.Tensor, gain: torch.Tensor,
                       glog: torch.Tensor, eps: float) -> torch.Tensor:
    return _pkm_value_step_fwd(w, u, values, gain, glog, eps)


@_pkm_value_step_op.register_fake
def _pkm_value_step_fake(w, u, values, gain, glog, eps):
    return w.new_empty((w.shape[0], values.shape[1]), dtype=torch.bfloat16)


# ================================================================================================ public functions
def pkm_scores_step(q: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
    """(B, H, 2, n) fp32 cosines of the unit half-queries with the unit sub-keys: ``q`` (B, H * 2 * dh) bf16 (the
    query projection), ``kh`` (H, 2, n, dh) fp32 contiguous unit sub-keys."""
    return _KERNEL_OPS.pkm_scores_step(q, kh)


def pkm_value_step(w: torch.Tensor, u: torch.Tensor, values: torch.Tensor, gain: torch.Tensor, glog: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """(B, dv) bf16 gated ZCN of the weighted value sum: ``w`` (B, M) fp32 slot weights, ``u`` (B, M) int64 slot ids
    inside the table (the kernel gathers the rows on the device), ``values`` (slots, dv) fp32, ``gain`` (dv,) fp32 =
    1 + pkm_norm_gain, ``glog`` (B, dv) bf16 gate logits, ``eps`` the norm's eps."""
    return _KERNEL_OPS.pkm_value_step(w, u, values, gain, glog, float(eps))


# ================================================================================================ pure-torch references
@torch.no_grad()
def pkm_scores_step_ref(q: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
    """The torch chain: ``einsum("bhad,hand->bhan", normalize(f(q).view(B, H, 2, dh)), kh)`` in fp32."""
    H, _, n, dh = (int(s) for s in kh.shape)
    with torch.autocast("cuda", enabled=False):
        qh = F.normalize(q.to(torch.float32).view(int(q.shape[0]), H, 2, dh), dim=-1)
        return torch.einsum("bhad,hand->bhan", qh, kh)


@torch.no_grad()
def pkm_value_step_ref(w: torch.Tensor, u: torch.Tensor, values: torch.Tensor, gain: torch.Tensor, glog: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """The torch chain: ``F.embedding`` + the fp32 weighted sum, the ZCN, the sigmoid gate, the cast to bf16."""
    with torch.autocast("cuda", enabled=False):
        val = torch.einsum("bm,bmd->bd", w, F.embedding(u, values).to(torch.float32))
        r = val * torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + eps) * gain
        r = r * torch.sigmoid(glog.to(torch.float32))
    return r.to(torch.bfloat16)
