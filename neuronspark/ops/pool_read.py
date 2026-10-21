"""The hippocampal layers' pool read of the training regime as one operator per layer, without any T x T tensor.

``pool_read(q, k, gv, cum, beta, qtok, ktok, orders, scale)`` -> (T, B, HQ, dv) bf16.  H key heads (the pool heads)
and HQ >= H query heads: query head i < H reads key head i, query heads H .. HQ - 1 (the last head's extra queries)
read key head H - 1 (its keys, values, clock and induction term)::

    m_n[b, t, j]    = the order-n induction match: j <= t - 1, t >= n - 1, j >= n, x[t - i] == x[j - 1 - i] for i < n
    X[b, t, j, h]   = ((m_2 beta_2h + m_3 beta_3h) + m_4 beta_4h)                fp32 adds in order, from +0
    M[b, t, j, h]   = bf16((cum[t, b, h] - cum[j, b, h]) + X[b, t, j, h])         for j <= t (-inf above the diagonal)
    out[t, b, i]    = bf16(softmax_j(scale q16[t, b, i] . k16[j, b, h] + M[b, t, j, h]) @ gv16[:, b, h]),
                      h = min(i, H - 1)

with gradients for q, k, gv, cum and beta (the bias's rounding passes the gradient straight through; no intermediate
gradient is rounded).  :func:`pool_read_ref` is the definition.  The matches are formed per tile from the two
(B, T, n_max) id windows of :func:`neuronspark.modules.induction.induction_token_windows` (formed once per forward,
shared by the layers); the memory is linear in T.

Kernels (Gluon, Blackwell): the forward (:mod:`neuronspark.ops.gluon.pool_read_fwd`: online softmax over the key
tiles, the output in bf16 and fp32 and the row log-sum-exp), then the backward (:mod:`neuronspark.ops.gluon.
pool_read_bwd`): D = dO . O from the unrounded output, the values' gradient (key-major), the keys' gradient with the
clock's column sums and the beta sums (key-major), the queries' gradient (query-major).  The clock gradient is
``-(column sums)``: the row sums of dS vanish (softmax is invariant to a shift of a row) and are not formed.  No
atomics: the results do not depend on the scheduling.

Numerics: bf16 q, k, v with fp32 accumulation; P and dS enter their products as two bf16 pieces (16 significant bits);
the bias formed with the definition's fp32 operations and rounding point.  The per-operator rule of
:mod:`neuronspark.utils.testing` against :func:`pool_read_ref`.  (The previous materialising chain rounded the bf16
mask's gradient and, for the extra queries, each head's key gradient; those gradients are further from the definition
than the rule allows and are not the reference.)
"""
from __future__ import annotations

from typing import Sequence

import torch

from neuronspark.modules.induction import induction_window_matches
from neuronspark.ops.gluon.pool_read_bwd import pool_read_dk_gluon, pool_read_dq_gluon, pool_read_dv_gluon
from neuronspark.ops.gluon.pool_read_fwd import pool_read_fwd_gluon
from neuronspark.ops.triton import require_cuda

__all__ = ["pool_read", "pool_read_ref", "PoolReadFn", "MAX_ORDER"]

MAX_ORDER = 4
NB = MAX_ORDER + 1           # beta slots per head (slot n = order n; slot 0 unused)


def _ordmask(orders: Sequence[int]) -> int:
    mask = 0
    for n in orders:
        if not 1 <= int(n) <= MAX_ORDER:
            raise RuntimeError(f"neuronspark.pool_read: orders 1 .. {MAX_ORDER} supported, got {tuple(orders)}")
        mask |= 1 << int(n)
    return mask


def _by_head_row(x: torch.Tensor) -> torch.Tensor:
    """(T, B, n) -> (B n, T) contiguous: one row per (b, head) / (b, window entry)."""
    T, B, n = (int(s) for s in x.shape)
    return x.permute(1, 2, 0).reshape(B * n, T).contiguous()


class PoolReadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, gv, cum, beta_slots, qtok, ktok, ordmask, scale):
        q16 = q.to(torch.bfloat16).contiguous()
        k16 = k.to(torch.bfloat16).contiguous()
        cum_t = _by_head_row(cum)
        ktok_t = _by_head_row(ktok.permute(1, 0, 2))
        out, out32, lse = pool_read_fwd_gluon(q16, k16, gv, cum_t, ktok_t, qtok, beta_slots, ordmask, scale)
        ctx.save_for_backward(q16, k16, gv, cum_t, beta_slots, qtok, ktok, ktok_t, out32, lse)
        ctx.meta = (ordmask, float(scale), q.dtype, k.dtype)
        return out

    @staticmethod
    def backward(ctx, dout):
        q16, k16, gv, cum_t, beta_slots, qtok, ktok, ktok_t, out32, lse = ctx.saved_tensors
        ordmask, scale, q_dtype, k_dtype = ctx.meta
        do16 = dout.to(torch.bfloat16).contiguous()
        # D = sum_j P dP = dO . (P V) from the unrounded output (the bf16 output's rounding would bias every dS)
        delta = (do16.float() * out32).sum(-1).permute(1, 2, 0).contiguous()               # (B, HQ, T)
        qtok_t = _by_head_row(qtok.permute(1, 0, 2))
        dgv = pool_read_dv_gluon(q16, k16, do16, lse, cum_t, qtok_t, ktok, beta_slots, ordmask, scale,
                                 int(gv.shape[-1]))
        dk, dcum, dslots = pool_read_dk_gluon(q16, k16, gv, do16, lse, delta, cum_t, qtok_t, ktok, beta_slots,
                                              ordmask, scale)
        dq = pool_read_dq_gluon(q16, k16, gv, do16, lse, delta, cum_t, ktok_t, qtok, beta_slots, ordmask, scale)
        return dq.to(q_dtype), dk.to(k_dtype), dgv, dcum, dslots, None, None, None, None


def pool_read(q: torch.Tensor, k: torch.Tensor, gv: torch.Tensor, cum: torch.Tensor, beta: torch.Tensor,
              qtok: torch.Tensor, ktok: torch.Tensor, orders: Sequence[int], scale: float) -> torch.Tensor:
    """q (T, B, HQ, dk) fp32 / bf16, k (T, B, H, dk), gv (T, B, H, dv) bf16, cum (T, B, H) fp32, beta (len(orders),
    H) fp32, (qtok, ktok) the (B, T, max(orders)) int32 id windows of
    :func:`neuronspark.modules.induction.induction_token_windows` -> (T, B, HQ, dv) bf16 (see the module docstring)."""
    require_cuda(q, "pool_read")
    orders = tuple(int(n) for n in orders)
    T, B, HQ, dk = (int(s) for s in q.shape)
    H = int(k.shape[2])
    if (tuple(k.shape) != (T, B, H, dk) or tuple(gv.shape[:3]) != (T, B, H) or tuple(cum.shape) != (T, B, H)
            or HQ < H):
        raise RuntimeError(f"neuronspark.pool_read: shapes q {tuple(q.shape)} k {tuple(k.shape)} gv {tuple(gv.shape)} "
                           f"cum {tuple(cum.shape)}")
    n_max = max(orders)
    if (tuple(beta.shape) != (len(orders), H) or gv.dtype != torch.bfloat16 or qtok.dtype != torch.int32
            or tuple(qtok.shape) != (B, T, n_max) or tuple(ktok.shape) != (B, T, n_max)):
        raise RuntimeError("neuronspark.pool_read: beta (orders, H), bf16 values and int32 (B, T, max(orders)) id "
                           "windows required")
    # beta per head and order slot 0 .. MAX_ORDER, without a host-to-device copy (the operator runs inside captured
    # CUDA graphs)
    zero, pos = beta.new_zeros((H,), dtype=torch.float32), {n: i for i, n in enumerate(orders)}
    slots = torch.stack([beta[pos[n]].float() if n in pos else zero for n in range(NB)], dim=1)       # (H, NB)
    return PoolReadFn.apply(q, k, gv.contiguous(), cum.float(), slots, qtok.contiguous(), ktok.contiguous(),
                            _ordmask(orders), scale)


def pool_read_ref(q: torch.Tensor, k: torch.Tensor, gv: torch.Tensor, cum: torch.Tensor, beta: torch.Tensor,
                  qtok: torch.Tensor, ktok: torch.Tensor, orders: Sequence[int], scale: float) -> torch.Tensor:
    """The definition, every T x T tensor materialised, in float64: the forward values of the operator (q and k rounded
    to bf16, the bias ``bf16((cum_t - cum_j) + X)`` with X the fp32 ordered sum, the output rounded to bf16) with
    exact gradients (every rounding passes the gradient straight through; the gradients are delivered in the inputs'
    dtypes)."""
    T, B, HQ, _ = (int(s) for s in q.shape)
    H = int(k.shape[2])
    orders = tuple(int(n) for n in orders)
    ms = induction_window_matches(qtok, ktok, orders)                         # per order (B, T, T) bool
    tri = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()

    def rounded(x64: torch.Tensor, value: torch.Tensor) -> torch.Tensor:  # the forward value, the exact gradient
        return x64 + (value.double() - x64).detach()

    # every operand widened once, so that a key head's uses by several query heads accumulate in float64
    q64 = rounded(q.double(), q.detach().to(torch.bfloat16))
    k64 = rounded(k.double(), k.detach().to(torch.bfloat16))
    gv64, cum64, beta64 = gv.double(), cum.double(), beta.double()
    biases = []
    for h in range(H):
        x32 = torch.zeros(B, T, T, dtype=torch.float32, device=q.device)
        for n in range(len(orders)):
            x32 = x32 + beta[n, h].detach().float() * ms[n].float()
        c32 = cum[:, :, h].detach().float().t()
        b16 = ((c32[:, :, None] - c32[:, None, :]) + x32).to(torch.bfloat16)
        c = cum64[:, :, h].t()
        bias = (c[:, :, None] - c[:, None, :]) + sum(beta64[n, h] * ms[n].double() for n in range(len(orders)))
        biases.append(rounded(bias, b16))
    outs = []
    for i in range(HQ):
        h = min(i, H - 1)
        s = float(scale) * torch.bmm(q64[:, :, i].permute(1, 0, 2), k64[:, :, h].permute(1, 2, 0)) + biases[h]
        p = torch.softmax(s.masked_fill(~tri, float("-inf")), dim=-1)
        outs.append(torch.bmm(p, gv64[:, :, h].permute(1, 0, 2)).permute(1, 0, 2))
    return torch.stack(outs, dim=2).to(torch.bfloat16)
