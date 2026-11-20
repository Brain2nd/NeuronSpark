"""The pool attention's read of a hippocampal layer (training) as one operator: the per-head preparation, the tagged
values, the last head's extra queries and the pool read.

``pool_attend(q, k, gate, v, gain, wbias, sparsity, xq, psi, cos, sin, cum, beta, qtok, ktok, orders, scale)`` ->
``(heads, xreads, q_all, k_all)``.  With H heads of width dk (channels ``[h dk, (h + 1) dk)``) and M = ``len(xq)`` extra
queries::

    k_h   = k[.., h] / max(|k[.., h]|, 1e-12)                                    unit keys
    q_h   = (q[.., h] / max(|q[.., h]|, 1e-12)) gain_h                           gained unit queries
    gv_h  = bf16(bf16(sigmoid(gate[.., h] + wbias_h - sparsity)) v[.., h])       tagged values
    qx_m  = normalize(rope(silu(xq_m))) (exp(clamp(psi[.., m], -3, 3)) gain_{H-1})    extra queries of the last head
    reads = pool_read(cat_h(q_h) ++ cat_m(qx_m), cat_h(k_h), cat_h(gv_h), cum, beta, qtok, ktok, orders, scale)

``reads`` (T, B, H + M, dk) bf16: the H heads' reads, then the extra queries' reads of the last head's keys, values,
clock and induction term (:mod:`neuronspark.ops.pool_read`).  They are returned as their consumers take them:
``heads`` (T, B, H dk) fp32, the H heads' reads widened and concatenated, and ``xreads`` (T, B, M dk) bf16, the extra
queries' reads (the operand of their output projection).  ``q_all`` (T, B, H + M, dk) and ``k_all`` (T, B, H, dk)
are the bf16 queries and keys the read took; they carry no gradient (the probe reads them).

Operands: ``q`` / ``k`` / ``v`` bf16 (T, B, H dk) (q and k after SiLU and RoPE), ``gate`` fp32 (T, B, H dk) write-gate
spikes, ``gain`` bf16 (H,) match gain, ``wbias`` fp32 (H,) write bias, ``sparsity`` the write sparsity, ``xq`` the M
bf16 (T, B, dk) rank projections, ``psi`` bf16 (T, B, M) log-gain projection, ``cos`` / ``sin`` the bf16 (T, dk / 2)
RoPE band of the last head, ``cum`` fp32 (T, B, H) clocks, ``beta`` (len(orders), H), ``qtok`` / ``ktok`` the int32
(B, T, max(orders)) id windows.  Gradients: q, k, v, every xq_m, psi (bf16), gate (fp32), gain (bf16), wbias, cum, beta
(fp32).

One autograd Function: the formation kernels of :mod:`neuronspark.ops.triton.pool_pack` write the read's operands in
its layouts, the Gluon read kernels follow; the backward hands the read's fp32 query / key gradients to the formation
backward without an autograd edge (nothing is rounded in between).  :func:`pool_attend_ref` is the definition: the
torch lines above and :func:`neuronspark.ops.pool_read.pool_read_ref`.  The rule of :mod:`neuronspark.utils.testing`.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from neuronspark.modules.rope import apply_rope_interleaved_ref
from neuronspark.ops.gluon.pool_read_fwd import pool_read_fwd_gluon
from neuronspark.ops.pool_read import NB, _by_head_row, _ordmask, pool_read_ref, read_backward
from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton.pool_pack import (
    NORM_EPS,
    PSI_CLAMP,
    fold_gain_bias,
    pack_heads_bwd,
    pack_heads_fwd,
    pack_xq_bwd,
    pack_xq_fwd,
    read_delta_split,
    read_unpack,
)

__all__ = ["pool_attend", "pool_attend_ref", "PoolAttendFn"]


class PoolAttendFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, gate, v, gain, wbias, psi, cos, sin, cum, beta, qtok, ktok, sparsity, orders, scale, *xq):
        T, B, D = (int(s) for s in q.shape)
        H, M = int(gain.shape[0]), len(xq)
        dk = D // H
        dev = q.device
        q16 = torch.empty((T, B, H + M, dk), dtype=torch.bfloat16, device=dev)
        k16 = torch.empty((T, B, H, dk), dtype=torch.bfloat16, device=dev)
        gv = torch.empty((T, B, H, dk), dtype=torch.bfloat16, device=dev)
        qn, kn = pack_heads_fwd(q, k, gate, v, gain, wbias, sparsity, q16, k16, gv)
        if M:
            pack_xq_fwd(xq, psi, gain, cos, sin, q16)
        n0, L = orders[0], len(orders)
        if orders == tuple(range(n0, n0 + L)):                               # beta per head and order slot
            slots = F.pad(beta.t(), (n0, NB - n0 - L))
        else:
            slots = beta.new_zeros((H, NB), dtype=torch.float32)
            for i, n in enumerate(orders):
                slots[:, n] = beta[i]
        cum_t = _by_head_row(cum)
        ktok_t = _by_head_row(ktok.permute(1, 0, 2))
        ordmask = _ordmask(orders)
        out, out32, lse = pool_read_fwd_gluon(q16, k16, gv, cum_t, ktok_t, qtok, slots, ordmask, scale)
        heads, xr = read_unpack(out, H)
        ctx.save_for_backward(q, k, gate, v, gain, wbias, psi, cos, sin, qn, kn, q16, k16, gv, cum_t, slots, qtok, ktok,
                              ktok_t, out32, lse, *xq)
        ctx.meta = (float(sparsity), tuple(orders), ordmask, float(scale), M)
        ctx.mark_non_differentiable(q16, k16)
        return heads, xr, q16, k16

    @staticmethod
    def backward(ctx, dheads, dxr, _dq16, _dk16):
        saved = ctx.saved_tensors
        (q, k, gate, v, gain, wbias, psi, cos, sin, qn, kn, q16, k16, gv, cum_t, slots, qtok, ktok, ktok_t, out32,
         lse) = saved[:21]
        xq = saved[21:]
        sparsity, orders, ordmask, scale, M = ctx.meta
        H = int(gain.shape[0])
        if dxr is None:                                      # (no extra queries, or their reads unused)
            dxr = torch.zeros((*out32.shape[:2], M * int(out32.shape[-1])), dtype=torch.bfloat16, device=out32.device)
        do16, delta = read_delta_split(dheads, dxr, out32, H)
        dq, dkk, dgv, dcum, dslots = read_backward(do16, delta, q16, k16, gv, cum_t, slots, qtok, ktok, ktok_t, lse,
                                                   ordmask, scale)
        oq, ok, ov, og, pg, pb = pack_heads_bwd(dq, dkk, dgv, q, k, gate, v, gain, wbias, sparsity, qn, kn)
        dpsi, px, dxs = None, None, ()
        if M:
            ox, dpsi, px = pack_xq_bwd(dq, xq, psi, gain, cos, sin)
            dxs = ox.unbind(0)
        d_gain, d_wbias = fold_gain_bias(pg, pb, px)
        n0, L = orders[0], len(orders)
        if orders == tuple(range(n0, n0 + L)):
            dbeta = dslots[:, n0:n0 + L].t()
        else:
            dbeta = torch.stack([dslots[:, n] for n in orders], dim=0)
        return (oq, ok, og, ov, d_gain, d_wbias, dpsi, None, None, dcum, dbeta, None, None, None, None, None, *dxs)


def _check(q, k, gate, v, gain, wbias, xq, psi, cos, sin, cum, beta, qtok, ktok, orders) -> Tuple[int, int, int, int]:
    require_cuda(q, "pool_attend")
    T, B, D = (int(s) for s in q.shape)
    if not (isinstance(gain, torch.Tensor) and gain.dim() == 1 and gain.dtype == torch.bfloat16):
        raise RuntimeError("neuronspark.pool_attend: gain must be a bf16 (H,) tensor")
    H = int(gain.shape[0])
    for name, x, dt in (("q", q, torch.bfloat16), ("k", k, torch.bfloat16), ("v", v, torch.bfloat16),
                        ("gate", gate, torch.float32)):
        if x.dtype != dt or tuple(x.shape) != (T, B, D):
            raise RuntimeError(f"neuronspark.pool_attend: {name} must be {dt} ({T}, {B}, {D}), got {x.dtype} "
                               f"{tuple(x.shape)}")
    if D % H or tuple(wbias.shape) != (H,) or wbias.dtype != torch.float32:
        raise RuntimeError(f"neuronspark.pool_attend: {H} heads over {D} channels, an fp32 ({H},) write bias")
    n_max = max(orders)
    if (tuple(cum.shape) != (T, B, H) or tuple(beta.shape) != (len(orders), H) or qtok.dtype != torch.int32
            or tuple(qtok.shape) != (B, T, n_max) or tuple(ktok.shape) != (B, T, n_max)):
        raise RuntimeError("neuronspark.pool_attend: cum (T, B, H), beta (orders, H) and int32 (B, T, max(orders)) id "
                           "windows required")
    if len(xq) and (psi is None or cos is None or sin is None):
        raise RuntimeError("neuronspark.pool_attend: extra queries need psi and the RoPE band")
    return T, B, H, D // H


def pool_attend(q: torch.Tensor, k: torch.Tensor, gate: torch.Tensor, v: torch.Tensor, gain: torch.Tensor,
                wbias: torch.Tensor, sparsity: float, xq: Sequence[torch.Tensor], psi: Optional[torch.Tensor],
                cos: Optional[torch.Tensor], sin: Optional[torch.Tensor], cum: torch.Tensor, beta: torch.Tensor,
                qtok: torch.Tensor, ktok: torch.Tensor, orders: Sequence[int], scale: float):
    """(heads (T, B, H dk) fp32, xreads (T, B, M dk) bf16, q_all (T, B, H + M, dk) bf16, k_all (T, B, H, dk) bf16): see
    the module docstring."""
    orders = tuple(int(n) for n in orders)
    xq = tuple(xq)
    _check(q, k, gate, v, gain, wbias, xq, psi, cos, sin, cum, beta, qtok, ktok, orders)
    return PoolAttendFn.apply(q, k, gate, v, gain.contiguous(), wbias.contiguous(), psi, cos, sin, cum.float(),
                              beta.float(), qtok.contiguous(), ktok.contiguous(), float(sparsity), orders,
                              float(scale), *xq)


def pool_attend_ref(q: torch.Tensor, k: torch.Tensor, gate: torch.Tensor, v: torch.Tensor, gain: torch.Tensor,
                    wbias: torch.Tensor, sparsity: float, xq: Sequence[torch.Tensor], psi: Optional[torch.Tensor],
                    cos: Optional[torch.Tensor], sin: Optional[torch.Tensor], cum: torch.Tensor, beta: torch.Tensor,
                    qtok: torch.Tensor, ktok: torch.Tensor, orders: Sequence[int], scale: float):
    """The definition: the torch lines of the head preparation, the tagged values and the extra queries, then
    :func:`neuronspark.ops.pool_read.pool_read_ref`."""
    H = int(gain.shape[0])
    dk = int(q.shape[-1]) // H
    qs, ks, gvs = [], [], []
    # the lines of the bf16 autocast forward: F.normalize of a bf16 slice has an fp32 norm and an fp32 result, and
    # its backward rounds the two terms of the slice's gradient and their sum to bf16
    with torch.autocast(device_type=q.device.type, dtype=torch.bfloat16):
        for h in range(H):
            sl = slice(h * dk, (h + 1) * dk)
            ks.append(F.normalize(k[..., sl], dim=-1, eps=NORM_EPS))
            qs.append(F.normalize(q[..., sl], dim=-1, eps=NORM_EPS) * gain[h])
            g = (gate[..., sl].float() + wbias[h]) - float(sparsity)
            gvs.append(torch.sigmoid(g).to(v.dtype) * v[..., sl])
        if len(xq):
            ps = psi.to(torch.float32).clamp(-PSI_CLAMP, PSI_CLAMP)
            c3, s3 = cos.unsqueeze(1), sin.unsqueeze(1)
            for m, x in enumerate(xq):
                r = F.normalize(apply_rope_interleaved_ref(F.silu(x), c3.to(x.dtype), s3.to(x.dtype)), dim=-1,
                                eps=NORM_EPS)
                g = torch.exp(ps[..., m:m + 1]) * gain[H - 1].to(torch.float32)
                qs.append(r * g.to(r.dtype))
    q_all = torch.stack(qs, dim=2)
    k_all = torch.stack(ks, dim=2)
    reads = pool_read_ref(q_all, k_all, torch.stack(gvs, dim=2), cum, beta, qtok, ktok, orders, scale)
    T, B = int(reads.shape[0]), int(reads.shape[1])
    heads = reads[:, :, :H].reshape(T, B, H * dk).to(torch.float32)
    xreads = reads[:, :, H:].reshape(T, B, len(xq) * dk)
    return heads, xreads, q_all.detach().to(torch.bfloat16), k_all.detach().to(torch.bfloat16)

