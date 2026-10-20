"""The additive mask of the hippocampal layers' pool attention (training).

Operators (registry names):

* ``pool_induction_term(beta, ind)``: the (B, T, T) fp32 induction term of one head,
  ``X = ((m_0 b_0 + m_1 b_1) + m_2 b_2) ...`` from the bool (B, T, T) match matrices ``ind`` (orders 2, 3, 4) and the
  head's fp32 (orders,) slice ``beta`` of ``pool_ind_beta``; backward ``d beta_n = sum(dX m_n)``.
* ``pool_bias_mask(cum, X, T, B)``: the (B, 1, T, T) bf16 additive mask
  ``bf16((j <= t ? cum[t, b] - cum[j, b] : -inf) + X[b, t, j])`` from the head's fp32 (T, B) clock ``cum`` (the
  cumulative eligibility trace) and its induction term ``X`` (fp32 (B, T, T) or None); backward: X's gradient
  ``float(dM)`` and the clock's gradient ``cols + rows`` of the masked gradient.

Numerics contract
-----------------
Forward: ``mul.rn`` / ``add.rn`` / ``sub.rn`` in torch's order (no FMA), the bools as 0.0 / 1.0, one
``cvt.rn.bf16.f32`` of the fp32 mask -- the rounding points of the torch chain followed by the bf16 autocast's cast
of the mask (``enable_fp_fusion=False``).  Backward: the reductions are formed in the kernels -- the induction term's full sums as
per-program partial sums (one small torch sum), and the clock's gradient as per-tile row / column partial sums of the
masked gradient plus one finish kernel (``-(column sum) + row sum``) -- fp32 reduction-order class.

Launchers are ``torch.library`` custom ops (``neuronspark::pool_induction_*``, ``::pool_mask_*``) with fake
implementations.  References: :func:`pool_induction_term_ref` and :func:`pool_bias_mask_ref` (the torch chains of the
induction term and of the pool's mask).
"""
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton._ptx import (  # noqa: F401  (jit helpers resolved through this module's globals)
    _rn_add_f32 as _p_add,
    _rn_mul_f32 as _p_mul,
    _rn_sub_f32 as _p_sub,
    _sr_bf16_to_f32 as _p_w,
    _sr_f32_to_bf16 as _p_n,
    _sr_neg as _p_neg,
)

__all__ = ["PoolInductionTermFn", "pool_induction_term", "pool_induction_term_ref", "PoolBiasMaskFn",
           "pool_bias_mask", "pool_bias_mask_ref"]

# ------------------------------------------------------------------------------------------------ fixed constants
IND_BLOCK, IND_WARPS = 2048, 4         # induction term (flat), forward and reduced backward
MASK_BT, MASK_BJ, MASK_WARPS = 16, 256, 4   # mask tiles (query rows x key columns)
FIN_BR, FIN_WARPS = 64, 4              # clock-gradient finish kernel
MAX_ORDERS = 4


# ================================================================================================ kernels
@triton.jit
def _p_mf(m):
    """bool -> fp32 as ATen converts it inside the mul (static_cast<float>(bool)): 1.0 / 0.0."""
    return tl.where(m != 0, 1.0, 0.0).to(tl.float32)


@triton.jit
def _ind_fwd_kernel(M0, M1, M2, M3, BETA, X, N, sbeta, NORD: tl.constexpr, BLOCK: tl.constexpr):
    """X = m0 * b0 (+ m1 * b1) (+ m2 * b2) (+ m3 * b3), left to right; m's contiguous bool, X contiguous fp32."""
    lin = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    msk = lin < N
    b0 = tl.load(BETA)
    x = _p_mul(_p_mf(tl.load(M0 + lin, mask=msk, other=0).to(tl.int32)), b0)
    if NORD > 1:
        b1 = tl.load(BETA + sbeta)
        x = _p_add(x, _p_mul(_p_mf(tl.load(M1 + lin, mask=msk, other=0).to(tl.int32)), b1))
    if NORD > 2:
        b2 = tl.load(BETA + 2 * sbeta)
        x = _p_add(x, _p_mul(_p_mf(tl.load(M2 + lin, mask=msk, other=0).to(tl.int32)), b2))
    if NORD > 3:
        b3 = tl.load(BETA + 3 * sbeta)
        x = _p_add(x, _p_mul(_p_mf(tl.load(M3 + lin, mask=msk, other=0).to(tl.int32)), b3))
    tl.store(X + lin, x, mask=msk)


@triton.jit
def _ind_red_kernel(G, M0, M1, M2, M3, PART, N, NORD: tl.constexpr, BLOCK: tl.constexpr):
    """PART[pid, n] = sum over the program's block of g * m_n (the products as ATen's mul of the fp32 gradient by the
    bool matrix)."""
    pid = tl.program_id(0)
    lin = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    msk = lin < N
    g = tl.load(G + lin, mask=msk, other=0.0)
    base = pid.to(tl.int64) * NORD
    tl.store(PART + base, tl.sum(_p_mul(g, _p_mf(tl.load(M0 + lin, mask=msk, other=0).to(tl.int32))), axis=0))
    if NORD > 1:
        tl.store(PART + base + 1, tl.sum(_p_mul(g, _p_mf(tl.load(M1 + lin, mask=msk, other=0).to(tl.int32))), axis=0))
    if NORD > 2:
        tl.store(PART + base + 2, tl.sum(_p_mul(g, _p_mf(tl.load(M2 + lin, mask=msk, other=0).to(tl.int32))), axis=0))
    if NORD > 3:
        tl.store(PART + base + 3, tl.sum(_p_mul(g, _p_mf(tl.load(M3 + lin, mask=msk, other=0).to(tl.int32))), axis=0))


@triton.jit
def _mask_red_kernel(DM, G, ROWP, COLP, T, NJ, NT, WRITE_G: tl.constexpr, IN_BF16: tl.constexpr,
                     BT: tl.constexpr, BJ: tl.constexpr):
    """G = float(dM) (when WRITE_G); gm = masked_fill(G, j > t, 0); ROWP[b, t, pj] = sum over the tile's j of gm;
    COLP[b, j, pt] = sum over the tile's t of gm (ROWP (B, T, NJ), COLP (B, T, NT) contiguous)."""
    pt = tl.program_id(0)
    pj = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    t = pt * BT + tl.arange(0, BT)
    j = pj * BJ + tl.arange(0, BJ)
    mt = t < T
    mj = j < T
    m2 = mt[:, None] & mj[None, :]
    off = b * T * T + t.to(tl.int64)[:, None] * T + j.to(tl.int64)[None, :]
    if IN_BF16:
        g = _p_w(tl.load(DM + off, mask=m2, other=0.0))
    else:
        g = tl.load(DM + off, mask=m2, other=0.0)
    if WRITE_G:
        tl.store(G + off, g, mask=m2)
    gm = tl.where(j[None, :] <= t[:, None], g, 0.0)
    tl.store(ROWP + (b * T + t.to(tl.int64)) * NJ + pj, tl.sum(gm, axis=1), mask=mt)
    tl.store(COLP + (b * T + j.to(tl.int64)) * NT + pt, tl.sum(gm, axis=0), mask=mj)


@triton.jit
def _cum_fin_kernel(ROWP, COLP, OUT, T, B, NJ, NT, BR: tl.constexpr, BNJ: tl.constexpr, BNT: tl.constexpr):
    """OUT[t, b] = (-(sum_pt COLP[b, t, pt])) + sum_pj ROWP[b, t, pj] (OUT (T, B) contiguous)."""
    e = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    m = e < T * B
    t = e // B
    b = e - t * B
    row = b * T + t
    kj = tl.arange(0, BNJ)
    r = tl.sum(tl.load(ROWP + row[:, None] * NJ + kj[None, :], mask=m[:, None] & (kj < NJ)[None, :], other=0.0), axis=1)
    kt = tl.arange(0, BNT)
    c = tl.sum(tl.load(COLP + row[:, None] * NT + kt[None, :], mask=m[:, None] & (kt < NT)[None, :], other=0.0), axis=1)
    tl.store(OUT + e, _p_add(_p_neg(c), r), mask=m)


@triton.jit
def _mask_fwd_kernel(CUM, X, OUT, T, sct, scb, HAS_X: tl.constexpr, OUT_BF16: tl.constexpr,
                     BT: tl.constexpr, BJ: tl.constexpr):
    """OUT[b, 0, t, j] = cast((j <= t ? cum[t, b] - cum[j, b] : -inf) + X[b, t, j]); OUT / X contiguous."""
    pt = tl.program_id(0)
    pj = tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    t = pt * BT + tl.arange(0, BT)
    j = pj * BJ + tl.arange(0, BJ)
    mt = t < T
    mj = j < T
    ct = tl.load(CUM + t.to(tl.int64) * sct + b * scb, mask=mt, other=0.0)
    cj = tl.load(CUM + j.to(tl.int64) * sct + b * scb, mask=mj, other=0.0)
    v = _p_sub(ct[:, None], cj[None, :])
    v = tl.where(j[None, :] <= t[:, None], v, float("-inf"))
    off = b * T * T + t.to(tl.int64)[:, None] * T + j.to(tl.int64)[None, :]
    m2 = mt[:, None] & mj[None, :]
    if HAS_X:
        v = _p_add(v, tl.load(X + off, mask=m2, other=0.0))
    if OUT_BF16:
        tl.store(OUT + off, _p_n(v), mask=m2)
    else:
        tl.store(OUT + off, v, mask=m2)


# ================================================================================================ induction term
def _pad_orders(ms: Sequence[torch.Tensor]):
    return list(ms) + [ms[0]] * (MAX_ORDERS - len(ms))


@torch.library.custom_op("neuronspark::pool_induction_fwd", mutates_args=(),
                         schema="(Tensor beta, Tensor[] ms) -> Tensor")
def _ind_fwd(beta, ms):
    m0 = ms[0]
    N = m0.numel()
    x = torch.empty(tuple(m0.shape), dtype=torch.float32, device=m0.device)
    mm = _pad_orders(ms)
    _ind_fwd_kernel[(triton.cdiv(N, IND_BLOCK),)](mm[0], mm[1], mm[2], mm[3], beta, x, N, beta.stride(0),
                                                  NORD=len(ms), BLOCK=IND_BLOCK, num_warps=IND_WARPS,
                                                  enable_fp_fusion=False)
    return x


@_ind_fwd.register_fake
def _ind_fwd_fake(beta, ms):
    return torch.empty(tuple(ms[0].shape), dtype=torch.float32, device=ms[0].device)


@torch.library.custom_op("neuronspark::pool_induction_bwd", mutates_args=(),
                         schema="(Tensor g, Tensor[] ms) -> Tensor")
def _ind_bwd(g, ms):
    """the (K,) fp32 gradient of beta: per-program partial sums of g * m_n, then one torch sum."""
    m0 = ms[0]
    N = m0.numel()
    nprog = triton.cdiv(N, IND_BLOCK)
    part = torch.empty((nprog, len(ms)), dtype=torch.float32, device=m0.device)
    mm = _pad_orders(ms)
    _ind_red_kernel[(nprog,)](g, mm[0], mm[1], mm[2], mm[3], part, N, NORD=len(ms), BLOCK=IND_BLOCK,
                              num_warps=IND_WARPS, enable_fp_fusion=False)
    return part.sum(0)


@_ind_bwd.register_fake
def _ind_bwd_fake(g, ms):
    return torch.empty((len(ms),), dtype=torch.float32, device=g.device)


class PoolInductionTermFn(torch.autograd.Function):
    """sum_n beta[n] * m_n in fp32 (B, T, T); only the bool matrices are saved; nothing flows into them."""

    @staticmethod
    def forward(ctx, beta, *ms):
        ctx.save_for_backward(*ms)
        return _ind_fwd(beta, list(ms))

    @staticmethod
    def backward(ctx, grad_out):
        ms = ctx.saved_tensors
        g = grad_out
        if g.dtype != torch.float32 or tuple(g.shape) != tuple(ms[0].shape) or not g.is_contiguous():
            raise RuntimeError("neuronspark.pool_induction_term backward: the incoming gradient must be a contiguous "
                               f"fp32 (B, T, T) tensor, got {g.dtype} {tuple(g.shape)} strides {g.stride()}")
        return (_ind_bwd(g, list(ms)),) + tuple(None for _ in ms)


def _check_ind(beta: torch.Tensor, ms: Sequence[torch.Tensor]) -> None:
    if not ms or len(ms) > MAX_ORDERS:
        raise RuntimeError(f"neuronspark.pool_induction_term: 1..{MAX_ORDERS} match matrices, got {len(ms)}")
    m0 = ms[0]
    for m in ms:
        if (not isinstance(m, torch.Tensor) or m.dtype != torch.bool or not m.is_cuda or m.dim() != 3
                or m.shape != m0.shape or m.device != m0.device or not m.is_contiguous() or m.requires_grad):
            raise RuntimeError("neuronspark.pool_induction_term: the match matrices must be contiguous CUDA bool "
                               f"tensors of one (B, T, T) shape, got {getattr(m, 'dtype', None)} "
                               f"{tuple(getattr(m, 'shape', ()))}")
    if m0.shape[1] != m0.shape[2] or m0.numel() == 0:
        raise RuntimeError(f"neuronspark.pool_induction_term: (B, T, T) non-empty matrices expected, got "
                           f"{tuple(m0.shape)}")
    if (not isinstance(beta, torch.Tensor) or beta.dtype != torch.float32 or beta.dim() != 1
            or int(beta.shape[0]) != len(ms) or beta.device != m0.device):
        raise RuntimeError(f"neuronspark.pool_induction_term: beta must be an fp32 ({len(ms)},) tensor on the matrices' "
                           f"device, got {getattr(beta, 'dtype', None)} {tuple(getattr(beta, 'shape', ()))}")


def pool_induction_term(beta: torch.Tensor, ind) -> torch.Tensor:
    """The (B, T, T) fp32 additive induction term of one head: ``beta`` the head's fp32 (orders,) coefficients (any
    stride, e.g. a column of ``pool_ind_beta.view(orders, H)``), ``ind`` the sequence of bool (B, T, T) contiguous
    match matrices (one per order).  Training regime."""
    ms = tuple(ind)
    _check_ind(beta, ms)
    return PoolInductionTermFn.apply(beta, *ms)


def pool_induction_term_ref(beta: torch.Tensor, ind) -> torch.Tensor:
    ms = tuple(ind)
    out = ms[0].to(torch.float32) * beta[0]
    for n in range(1, len(ms)):
        out = out + ms[n].to(torch.float32) * beta[n]
    return out


# ================================================================================================ mask
def _mask_grid(T: int, B: int):
    return (triton.cdiv(T, MASK_BT), triton.cdiv(T, MASK_BJ), B)


@torch.library.custom_op("neuronspark::pool_mask_fwd", mutates_args=(), schema="(Tensor cum, Tensor? x) -> Tensor")
def _mask_fwd(cum, x):
    T, B = (int(v) for v in cum.shape)
    out = torch.empty((B, 1, T, T), dtype=torch.bfloat16, device=cum.device)
    _mask_fwd_kernel[_mask_grid(T, B)](cum, x if x is not None else out, out, T, cum.stride(0), cum.stride(1),
                                       HAS_X=x is not None, OUT_BF16=True, BT=MASK_BT, BJ=MASK_BJ,
                                       num_warps=MASK_WARPS, enable_fp_fusion=False)
    return out


@_mask_fwd.register_fake
def _mask_fwd_fake(cum, x):
    T, B = (int(v) for v in cum.shape)
    return torch.empty((B, 1, T, T), dtype=torch.bfloat16, device=cum.device)


@torch.library.custom_op("neuronspark::pool_mask_bwd", mutates_args=(),
                         schema="(Tensor dm, bool want_g) -> (Tensor, Tensor)")
def _mask_bwd(dm, want_g):
    """(G or an empty tensor, the clock's (T, B) gradient): one tile kernel + one finish kernel."""
    B, T = int(dm.shape[0]), int(dm.shape[2])
    dev = dm.device
    g = (torch.empty((B, T, T), dtype=torch.float32, device=dev) if want_g
         else torch.empty((0,), dtype=torch.float32, device=dev))
    NT, NJ = triton.cdiv(T, MASK_BT), triton.cdiv(T, MASK_BJ)
    rowp = torch.empty((B, T, NJ), dtype=torch.float32, device=dev)
    colp = torch.empty((B, T, NT), dtype=torch.float32, device=dev)
    _mask_red_kernel[_mask_grid(T, B)](dm, g if want_g else rowp, rowp, colp, T, NJ, NT, WRITE_G=want_g,
                                       IN_BF16=dm.dtype == torch.bfloat16, BT=MASK_BT, BJ=MASK_BJ,
                                       num_warps=MASK_WARPS, enable_fp_fusion=False)
    gcum = torch.empty((T, B), dtype=torch.float32, device=dev)
    _cum_fin_kernel[(triton.cdiv(T * B, FIN_BR),)](rowp, colp, gcum, T, B, NJ, NT, BR=FIN_BR,
                                                   BNJ=int(triton.next_power_of_2(NJ)),
                                                   BNT=int(triton.next_power_of_2(NT)),
                                                   num_warps=FIN_WARPS, enable_fp_fusion=False)
    return g, gcum


@_mask_bwd.register_fake
def _mask_bwd_fake(dm, want_g):
    B, T = int(dm.shape[0]), int(dm.shape[2])
    g = (torch.empty((B, T, T), dtype=torch.float32, device=dm.device) if want_g
         else torch.empty((0,), dtype=torch.float32, device=dm.device))
    return g, torch.empty((T, B), dtype=torch.float32, device=dm.device)


class PoolBiasMaskFn(torch.autograd.Function):
    """The (B, 1, T, T) bf16 mask from the clock ``cum`` (T, B) fp32 and the optional (B, T, T) fp32 induction term."""

    @staticmethod
    def forward(ctx, cum, x):
        T, B = (int(v) for v in cum.shape)
        ctx.TB = (T, B)
        ctx.has_x = x is not None
        ctx.set_materialize_grads(False)
        return _mask_fwd(cum, x)

    @staticmethod
    def backward(ctx, dm):
        if dm is None:
            return None, None
        T, B = ctx.TB
        if dm.dtype != torch.bfloat16 or tuple(dm.shape) != (B, 1, T, T) or not dm.is_contiguous():
            raise RuntimeError("neuronspark.pool_bias_mask backward: the incoming gradient must be a contiguous "
                               f"(B, 1, T, T) bf16 tensor, got {dm.dtype} {tuple(dm.shape)} strides {dm.stride()}")
        need_x = ctx.has_x and ctx.needs_input_grad[1]
        g, gcum = _mask_bwd(dm, need_x)
        return (gcum if ctx.needs_input_grad[0] else None), (g if need_x else None)


def pool_bias_mask(cum: torch.Tensor, X: Optional[torch.Tensor], T: int, B: int) -> torch.Tensor:
    """The (B, 1, T, T) bf16 additive mask of one pool head: ``cum`` the fp32 CUDA (T, B) clock, ``X`` the contiguous
    fp32 (B, T, T) induction term (or None).  Training regime (the dtype the bf16 autocast hands the attention)."""
    T, B = int(T), int(B)
    require_cuda(cum, "pool_bias_mask")
    if cum.dtype != torch.float32 or cum.dim() != 2 or tuple(cum.shape) != (T, B) or T < 1 or B < 1:
        raise RuntimeError(f"neuronspark.pool_bias_mask: the clock must be an fp32 CUDA (T, B) = ({T}, {B}) tensor, got "
                           f"{cum.dtype} {tuple(cum.shape)}")
    if X is not None:
        if (not isinstance(X, torch.Tensor) or X.dtype != torch.float32 or X.device != cum.device
                or tuple(X.shape) != (B, T, T) or not X.is_contiguous()):
            raise RuntimeError("neuronspark.pool_bias_mask: the induction term must be a contiguous fp32 (B, T, T) "
                               f"tensor on the clock's device, got {getattr(X, 'dtype', None)} "
                               f"{tuple(getattr(X, 'shape', ()))}")
    return PoolBiasMaskFn.apply(cum, X)


def pool_bias_mask_ref(cum: torch.Tensor, X: Optional[torch.Tensor], T: int, B: int) -> torch.Tensor:
    """The pool's torch lines: the trace bias, the causal -inf fill, the induction term, the cast to bf16."""
    bias = (cum.unsqueeze(1) - cum.unsqueeze(0)).permute(2, 0, 1)             # (B, T, T)
    causal = torch.ones(int(T), int(T), device=cum.device, dtype=torch.bool).tril()
    bias = bias.masked_fill(~causal, float("-inf"))
    if X is not None:
        bias = bias + X
    return bias.unsqueeze(1).to(torch.bfloat16)
