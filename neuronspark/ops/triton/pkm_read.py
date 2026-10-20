"""PKM value read: the softmax-weighted sum of the selected value rows, ``o[r] = sum_m w[r, m] * V[u[r, m]]``.

Shapes (1.27B): per token r = (t, b) M = 4 heads x 16 slots = 64 (slot, weight) pairs into one shared value table V
(S = 128^2 = 16384 rows x D = 768, fp32).  No (T, B, M, D) gathered tensor is formed.

Operators (regime: public function -> what runs):

* train   :func:`pkm_value_read` -> :class:`PKMReadFn`.  Forward (custom op ``neuronspark::pkm_read_fwd``): exact
  fp32, ``o[r, :]`` = an fma.rn chain over m = 0 .. 63 in slot order from +0; 8 rows x 128 columns per program,
  8 warps, the gathers issued in groups of 8 slots (one [8, 128, 8] load per group, then the group's slots enter the
  chain one by one).  Backward (custom op ``neuronspark::pkm_read_bwd``): one stable sort of the R * M slot ids (int16
  keys: S <= 32767), segment offsets by searchsorted, then one program per (slot, chunk of 64 occurrences in sorted
  order) that forms both gradients from a single load of the slot's value row: grad_V partial = fma.rn chain of
  w[r, m] * grad_o[r, :] over the chunk's occurrences, and grad_w[r, m] = one tl.sum tree over 256 lanes of the per-lane
  sums of the three 256-column chunks' fp32 products grad_o[r, d] * V[s, d]; the next occurrence's operands are
  prefetched.  A second kernel sums each slot's chunk partials in order (add.rn from +0).
  Deterministic and capture-safe (static shapes, no host synchronisation).  Both gradients are always formed (one
  fused kernel); a gradient no input needs is dropped.
* prefill :func:`pkm_weighted_sum` (custom op ``neuronspark::pkm_weighted_sum``): the forward sum for one (t, b) row
  per program and 256 columns, a sequential fma.rn chain over the 64 pairs from +0.
* decode: the modules' torch gather + einsum in IEEE fp32 (no op here).

Numerics: exact fp32 products (no TF32 operand rounding) and fixed summation orders, an fp32 reduction-order
difference from the float64 references.  Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing`.

Pure-torch references (same signatures): :func:`pkm_value_read_ref`, :func:`pkm_weighted_sum_ref`
(``F.embedding`` + ``einsum("tbm,tbmd->tbd")``, evaluated in float64).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import _rn_fma_f32  # noqa: F401  (resolved by _pkm_wsum_kernel)

__all__ = [
    "PKMReadFn",
    "pkm_value_read",
    "pkm_value_read_ref",
    "pkm_weighted_sum",
    "pkm_weighted_sum_ref",
]

# ================================================================================================ fixed constants
# Launch configuration of the training and inference operators.
#: column chunk of the backward (grad_w dot chunks, grad_V partial rows); D must be a multiple (D <= 3 chunks)
COL_CHUNK = 256
#: training forward: rows per program, columns per program, warps, slots per gather group
FWD_ROWS = 8
FWD_COLS = 128
FWD_WARPS = 8
FWD_GROUP = 8
#: training backward: occurrences per grad_V partial, warps of the fused partial kernel
BWD_CHUNK = 64
BWD_WARPS = 1
#: grad_V partial-sum kernel: columns per program, warps
DVSUM_COLS = 256
DVSUM_WARPS = 4
#: largest table for which the slot sort runs on int16 keys (the same stable order as int64 keys)
SORT16_MAX_SLOTS = 32767
#: prefill weighted sum: columns per program, warps, pipeline stages
WSUM_COLS = 256
WSUM_WARPS = 4
WSUM_STAGES = 1


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.pkm_read: " + msg)


def _check(w: torch.Tensor, u: torch.Tensor, V: torch.Tensor, what: str) -> None:
    require_cuda(w, what)
    _need(u.device == w.device and V.device == w.device, "w, u and V must be CUDA tensors on one device")
    _need(w.dtype == torch.float32 and V.dtype == torch.float32 and u.dtype == torch.int64,
          f"fp32 w / V and int64 u required, got {w.dtype} {V.dtype} {u.dtype}")
    _need(w.dim() == 3 and tuple(u.shape) == tuple(w.shape) and V.dim() == 2 and int(V.shape[1]) % COL_CHUNK == 0,
          f"w / u (T, B, M) and V (S, D) with D % {COL_CHUNK} == 0 required, got {tuple(w.shape)} {tuple(u.shape)} "
          f"{tuple(V.shape)}")


# ================================================================================================ launchers (training)
def _read_fwd(w2: torch.Tensor, u2: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """w2 (R, M) fp32, u2 (R, M) int64 contiguous, V (S, D) -> o (R, D) fp32 contiguous."""
    R, M = (int(s) for s in w2.shape)
    D = int(V.shape[1])
    _need(D % FWD_COLS == 0 and M % FWD_GROUP == 0,
          f"D = {D} must be a multiple of {FWD_COLS} and M = {M} of {FWD_GROUP}")
    _need(V.is_contiguous(), "V must be contiguous")
    o = torch.empty((R, D), dtype=torch.float32, device=w2.device)
    if R:
        _pr_fwd2_kernel[(triton.cdiv(R, FWD_ROWS), D // FWD_COLS)](
            w2, u2, V, o, R, M=M, D=D, RB=FWD_ROWS, DB=FWD_COLS, MB=FWD_GROUP, num_warps=FWD_WARPS,
            enable_fp_fusion=False)
    return o


def _read_bwd(go2: torch.Tensor, w2: torch.Tensor, u2: torch.Tensor,
              V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """grad_w (R, M) and grad_V (S, D), fp32 contiguous, for the upstream gradient go2 (R, D)."""
    R, M = (int(s) for s in w2.shape)
    S, D = (int(s) for s in V.shape)
    _need(D % COL_CHUNK == 0 and D // COL_CHUNK <= 3, f"the fused backward covers D = 256, 512 or 768, got {D}")
    _need(V.is_contiguous(), "V must be contiguous")
    dev = go2.device
    go2 = go2.contiguous()
    flat = u2.reshape(-1)
    if S <= SORT16_MAX_SLOTS:
        su16, perm = torch.sort(flat.to(torch.int16), stable=True)
        su = su16.to(torch.int64)
    else:
        su, perm = torch.sort(flat, stable=True)
    offs = torch.searchsorted(su, torch.arange(S + 1, device=dev, dtype=torch.int64))
    cnt = offs[1:] - offs[:-1]
    nch = (cnt + (BWD_CHUNK - 1)) // BWD_CHUNK
    cst = torch.zeros(S + 1, dtype=torch.int64, device=dev)
    cst[1:] = torch.cumsum(nch, 0)
    qmax = (R * M + BWD_CHUNK - 1) // BWD_CHUNK + S
    slot = torch.searchsorted(cst[1:], torch.arange(qmax, device=dev, dtype=torch.int64), right=True)
    part = torch.empty((qmax, D), dtype=torch.float32, device=dev)
    gw = torch.empty((R, M), dtype=torch.float32, device=dev)
    gv = torch.empty((S, D), dtype=torch.float32, device=dev)
    _pr_dvw1pf_partial_kernel[(qmax,)](
        w2, go2, V, perm, offs, cst, slot, part, gw, S, M=M, D=D, CH=BWD_CHUNK, DB=COL_CHUNK, NC=D // COL_CHUNK,
        num_warps=BWD_WARPS, enable_fp_fusion=False)
    _pr_dv_sum_kernel[(S, D // DVSUM_COLS)](part, cst, gv, D=D, DB=DVSUM_COLS, num_warps=DVSUM_WARPS,
                                            enable_fp_fusion=False)
    return gw, gv


@torch.library.custom_op("neuronspark::pkm_read_fwd", mutates_args=())
def _pkm_read_fwd_op(w2: torch.Tensor, u2: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return _read_fwd(w2, u2, V)


@_pkm_read_fwd_op.register_fake
def _pkm_read_fwd_fake(w2, u2, V):
    return w2.new_empty((w2.shape[0], V.shape[1]), dtype=torch.float32)


@torch.library.custom_op("neuronspark::pkm_read_bwd", mutates_args=())
def _pkm_read_bwd_op(go2: torch.Tensor, w2: torch.Tensor, u2: torch.Tensor,
                     V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _read_bwd(go2, w2, u2, V)


@_pkm_read_bwd_op.register_fake
def _pkm_read_bwd_fake(go2, w2, u2, V):
    return (w2.new_empty(tuple(w2.shape), dtype=torch.float32),
            V.new_empty(tuple(V.shape), dtype=torch.float32))


class PKMReadFn(torch.autograd.Function):
    """o (T, B, D) fp32 from w (T, B, M), u (T, B, M), V (S, D); gradients for w and V."""

    @staticmethod
    def forward(ctx, w, u, V):
        T, B, M = (int(s) for s in w.shape)
        w2 = w.reshape(T * B, M).contiguous()
        u2 = u.reshape(T * B, M).contiguous()
        o = torch.ops.neuronspark.pkm_read_fwd(w2, u2, V)
        ctx.save_for_backward(w2, u2, V)
        ctx.shape = (T, B, M)
        return o.view(T, B, int(V.shape[1]))

    @staticmethod
    def backward(ctx, go):
        w2, u2, V = ctx.saved_tensors
        T, B, M = ctx.shape
        go2 = go.reshape(T * B, int(V.shape[1]))
        gw, gv = torch.ops.neuronspark.pkm_read_bwd(go2, w2, u2, V)
        return (gw.view(T, B, M) if ctx.needs_input_grad[0] else None), None, \
            (gv if ctx.needs_input_grad[2] else None)


def pkm_value_read(w: torch.Tensor, u: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """``einsum("tbm,tbmd->tbd", w, F.embedding(u, V))`` in exact fp32: ``w`` (T, B, M) fp32 slot weights, ``u``
    (T, B, M) int64 slot ids in [0, S), ``V`` (S, D) fp32 contiguous -> (T, B, D) fp32; differentiable in w and V."""
    _check(w, u, V, "pkm_value_read")
    if torch.is_grad_enabled() and (w.requires_grad or V.requires_grad):
        return PKMReadFn.apply(w, u, V)
    T, B, M = (int(s) for s in w.shape)
    o = torch.ops.neuronspark.pkm_read_fwd(w.reshape(T * B, M).contiguous(), u.reshape(T * B, M).contiguous(), V)
    return o.view(T, B, int(V.shape[1]))


# ================================================================================================ prefill
def _wsum(w: torch.Tensor, u: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    require_cuda(w, "pkm_weighted_sum")
    _need(w.dim() == 3 and V.dim() == 2, f"w (T, B, M) and V (S, dv) expected, got {tuple(w.shape)} {tuple(V.shape)}")
    T, B, M = (int(s) for s in w.shape)
    dv = int(V.shape[1])
    _need(w.dtype == torch.float32 and V.dtype == torch.float32 and u.dtype == torch.int64
          and tuple(u.shape) == (T, B, M) and u.device == w.device and V.device == w.device,
          f"pkm_weighted_sum: fp32 w {tuple(w.shape)}, int64 u of the same shape and an fp32 table on one device "
          f"required, got w {w.dtype}, u {u.dtype} {tuple(u.shape)}, V {V.dtype}")
    wc, uc, vc = w.contiguous(), u.contiguous(), V.contiguous()
    out = torch.empty((T, B, dv), dtype=torch.float32, device=w.device)
    if T * B:
        _pkm_wsum_kernel[(T * B, triton.cdiv(dv, WSUM_COLS))](wc, uc, vc, out, M=M, DV=dv, BLOCK_D=WSUM_COLS,
                                                              num_warps=WSUM_WARPS, num_stages=WSUM_STAGES,
                                                              enable_fp_fusion=False)
    return out


@torch.library.custom_op("neuronspark::pkm_weighted_sum", mutates_args=())
def _pkm_weighted_sum_op(w: torch.Tensor, u: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    return _wsum(w, u, V)


@_pkm_weighted_sum_op.register_fake
def _pkm_weighted_sum_fake(w, u, V):
    return w.new_empty((w.shape[0], w.shape[1], V.shape[1]), dtype=torch.float32)


def pkm_weighted_sum(w: torch.Tensor, u: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """``o[t, b] = sum_m w[t, b, m] * V[u[t, b, m]]`` (T, B, dv) fp32 for the prefill: ``w`` (T, B, M) fp32, ``u``
    (T, B, M) int64, ``V`` (S, dv) fp32.  No autograd."""
    return torch.ops.neuronspark.pkm_weighted_sum(w, u, V)


# ================================================================================================ pure-torch references
def pkm_value_read_ref(w: torch.Tensor, u: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Pure-torch reference of :func:`pkm_value_read`: ``einsum("tbm,tbmd->tbd", w, F.embedding(u, V))``
    evaluated in float64 and rounded once to fp32; differentiable (float64 gradients delivered in the inputs'
    dtypes)."""
    vals = F.embedding(u, V.to(torch.float64))
    return torch.einsum("tbm,tbmd->tbd", w.to(torch.float64), vals).to(torch.float32)


@torch.no_grad()
def pkm_weighted_sum_ref(w: torch.Tensor, u: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Pure-torch reference of :func:`pkm_weighted_sum` (the same float64 expression, no autograd)."""
    vals = F.embedding(u, V.to(torch.float64))
    return torch.einsum("tbm,tbmd->tbd", w.to(torch.float64), vals).to(torch.float32)


# ================================================================================================ kernels
@triton.jit
def _pr_fma(a, b, c):
    return tl.inline_asm_elementwise(asm="fma.rn.f32 $0, $1, $2, $3;", constraints="=f,f,f,f", args=[a, b, c],
                                     dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _pr_add(a, b):
    return tl.inline_asm_elementwise(asm="add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _pr_fwd2_kernel(W, U, V, O, R, M: tl.constexpr, D: tl.constexpr, RB: tl.constexpr, DB: tl.constexpr,
                    MB: tl.constexpr):
    """o[r, d] = fma.rn chain over the slots m = 0 .. M - 1 from +0 for RB rows and DB columns, the slot loop in groups
    of MB = 8: the 8 gathers of a group are issued as one [RB, DB, 8] load, then the group's slots enter the chain one
    by one in slot order (m = 4 i + 2 j + k split along the last axes)."""
    pr = tl.program_id(0)
    pd = tl.program_id(1)
    rows = pr * RB + tl.arange(0, RB)
    rm = rows < R
    rr = tl.where(rm, rows, 0).to(tl.int64)
    d = pd * DB + tl.arange(0, DB)
    mm = tl.arange(0, MB)
    acc = tl.zeros([RB, DB], dtype=tl.float32)
    for m0 in range(0, M, MB):
        u = tl.load(U + rr[:, None] * M + (m0 + mm)[None, :])                      # [RB, MB]
        w = tl.load(W + rr[:, None] * M + (m0 + mm)[None, :])
        v = tl.load(V + u[:, None, :] * D + d[None, :, None])                      # [RB, DB, MB]
        wb = w[:, None, :] + tl.zeros([RB, DB, MB], dtype=tl.float32)               # w + 0 broadcast
        vk0, vk1 = tl.split(tl.reshape(v, [RB, DB, 2, 2, 2]))                        # MB == 8, m = 4 i + 2 j + k
        wk0, wk1 = tl.split(tl.reshape(wb, [RB, DB, 2, 2, 2]))
        v00, v10 = tl.split(vk0)                                                     # (j 0 / 1, k 0): m 0,4 / 2,6
        v01, v11 = tl.split(vk1)                                                     # k 1: m 1,5 / 3,7
        w00, w10 = tl.split(wk0)
        w01, w11 = tl.split(wk1)
        a0, a4 = tl.split(v00)
        a2, a6 = tl.split(v10)
        a1, a5 = tl.split(v01)
        a3, a7 = tl.split(v11)
        b0, b4 = tl.split(w00)
        b2, b6 = tl.split(w10)
        b1, b5 = tl.split(w01)
        b3, b7 = tl.split(w11)
        acc = _pr_fma(b0, a0, acc)
        acc = _pr_fma(b1, a1, acc)
        acc = _pr_fma(b2, a2, acc)
        acc = _pr_fma(b3, a3, acc)
        acc = _pr_fma(b4, a4, acc)
        acc = _pr_fma(b5, a5, acc)
        acc = _pr_fma(b6, a6, acc)
        acc = _pr_fma(b7, a7, acc)
    tl.store(O + rr[:, None] * D + d[None, :], acc, mask=rm[:, None])


@triton.jit
def _pr_dvw1pf_partial_kernel(W, GO, V, PERM, OFFS, CST, SLOT, PART, GW, S, M: tl.constexpr, D: tl.constexpr,
                              CH: tl.constexpr, DB: tl.constexpr, NC: tl.constexpr):
    """Program q = (slot s, chunk j of CH occurrences of s in sorted order): the slot's value row is loaded once; for
    each occurrence p (row r = p // M) the grad_V partial accumulates fma.rn(w[p], grad_o[r, :], .) and grad_w[p] =
    tl.sum over the DB lanes of the per-lane sum of the NC column chunks' fp32 products grad_o[r, c] * V[s, c] (chunk 0
    first); the partial row is stored to PART[q].  The occurrence loop is software-prefetched: the permutation entry two occurrences ahead and
    the next occurrence's w / grad_o row are loaded before the current occurrence is consumed (the loads past the
    chunk's last occurrence re-read its last entry)."""
    q = tl.program_id(0)
    s = tl.load(SLOT + q)
    if s < S:
        j = q - tl.load(CST + s)
        i0 = tl.load(OFFS + s) + j * CH
        i1 = tl.minimum(i0 + CH, tl.load(OFFS + s + 1))
        il = i1 - 1
        d = tl.arange(0, DB)
        vrow = V + s.to(tl.int64) * D
        v0 = tl.load(vrow + (0 * DB + d))
        a0 = tl.zeros([DB], dtype=tl.float32)
        if NC > 1:
            v1 = tl.load(vrow + (1 * DB + d))
            a1 = tl.zeros([DB], dtype=tl.float32)
        if NC > 2:
            v2 = tl.load(vrow + (2 * DB + d))
            a2 = tl.zeros([DB], dtype=tl.float32)
        pn = tl.load(PERM + i0)
        pnn = tl.load(PERM + tl.minimum(i0 + 1, il))
        wn = tl.load(W + pn)
        rn = pn // M
        gn0 = tl.load(GO + rn * D + (0 * DB + d))
        if NC > 1:
            gn1 = tl.load(GO + rn * D + (1 * DB + d))
        if NC > 2:
            gn2 = tl.load(GO + rn * D + (2 * DB + d))
        for i in range(i0, i1):
            p = pn
            w = wn
            g0 = gn0
            if NC > 1:
                g1 = gn1
            if NC > 2:
                g2 = gn2
            pn = pnn
            pnn = tl.load(PERM + tl.minimum(i + 2, il))
            wn = tl.load(W + pn)
            rn = pn // M
            gn0 = tl.load(GO + rn * D + (0 * DB + d))
            if NC > 1:
                gn1 = tl.load(GO + rn * D + (1 * DB + d))
            if NC > 2:
                gn2 = tl.load(GO + rn * D + (2 * DB + d))
            a0 = _pr_fma(w + tl.zeros([DB], dtype=tl.float32), g0, a0)
            pd = g0 * v0
            if NC > 1:
                a1 = _pr_fma(w + tl.zeros([DB], dtype=tl.float32), g1, a1)
                pd = pd + g1 * v1
            if NC > 2:
                a2 = _pr_fma(w + tl.zeros([DB], dtype=tl.float32), g2, a2)
                pd = pd + g2 * v2
            tl.store(GW + p, tl.sum(pd, axis=0))
        prow = PART + q.to(tl.int64) * D
        tl.store(prow + (0 * DB + d), a0)
        if NC > 1:
            tl.store(prow + (1 * DB + d), a1)
        if NC > 2:
            tl.store(prow + (2 * DB + d), a2)


@triton.jit
def _pr_dv_sum_kernel(PART, CST, DV, D: tl.constexpr, DB: tl.constexpr):
    """grad_V[s, cols] = the slot's chunk partials q in [CST[s], CST[s + 1]) added in order from +0 (add.rn)."""
    s = tl.program_id(0)
    pd = tl.program_id(1)
    q0 = tl.load(CST + s)
    q1 = tl.load(CST + s + 1)
    d = pd * DB + tl.arange(0, DB)
    acc = tl.zeros([DB], dtype=tl.float32)
    for q in range(q0, q1):
        acc = _pr_add(acc, tl.load(PART + q * D + d))
    tl.store(DV + s.to(tl.int64) * D + d, acc)


@triton.jit
def _pkm_wsum_kernel(Wt, U, VALS, O, M: tl.constexpr, DV, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = d < DV
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for m in range(M):
        u = tl.load(U + row * M + m)
        wv = tl.load(Wt + row * M + m)
        v = tl.load(VALS + u * DV + d, mask=dm, other=0.0).to(tl.float32)
        acc = _rn_fma_f32(tl.broadcast_to(wv, (BLOCK_D,)), v, acc)
    tl.store(O + row * DV + d, acc, mask=dm)
