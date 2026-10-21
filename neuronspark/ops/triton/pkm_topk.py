"""PKM product-key top-k: the 16 best of the 128 x 128 product slots of every (token, head) row.

For each row (t, b, h) with half-query scores s1, s2 (n = 128 each, fp32):

    t1, i1 = topk(s1, 16);  t2, i2 = topk(s2, 16)               (largest, sorted)
    cand   = (t1[:, None] + t2[None, :]).flatten()               256 fp32 sums (one rounding each)
    sc, j  = topk(cand, 16);  ii = i1[j // 16];  jj = i2[j % 16];  u = ii * 128 + jj

where the selection follows torch.topk's semantics (PyTorch 2.9.1, CUDA): the selected set is the 16 largest
elements in the radix order of ``TopKTypeConfig<float>`` (an order of the bit patterns: -0 below +0, every NaN above
+inf, all NaNs equal); the pre-sort slot order is "strictly above the 16th key, in index order, then the equal keys, in index order"; the 16
slots are sorted by the fixed 32-slot bitonic network of ``SortUtils.cuh`` (``bitonicSortKVInPlace`` with
``GTOp<float, true>``: NaN greater, otherwise ``>``; -0 == +0; ties among equal keys are resolved by the network, not by
index).  No floating-point reduction is involved in the selection; the only arithmetic is the fp32 candidate add.

Operators (regime: public function -> what runs):

* train   :func:`pkm_topk_train` (s (T, B, H, 2, 128) fp32, any strides) -> :class:`PKMTopKFn`.  Forward: custom op
  ``neuronspark::pkm_topk_train_fwd``, 4 rows per program, 2 warps: composite int64 keys (radix key << 32 |
  (n - 1 - index)) through tl.topk give the selected set, a tl.sort the slot order, then the network compare-swap by
  compare-swap on the slot arrays.  The candidate top-k is taken over the 50 non-dominated pairs (i, j) with
  (i + 1)(j + 1) <= 16 (exact: with sorted, finite, non -0 halves every other candidate has at least 16 candidates at
  or above it by value and before it by index); a program with a non-finite or -0 half value runs all 256.
  Backward: custom op ``neuronspark::pkm_topk_train_bwd``, 2 rows per program, 2 warps: g_t1[a] / g_t2[c] = the sums
  of g_sc over the selected candidates of row a / column c (masked (BR, 16, 16) sums: fp32 reduction-order class
  against autograd's), the (2, 128) gradient row zero-filled, then the 16 + 16 sums scatter-stored at i1 / i2 (each
  value ``sum + 0.0``, every other slot +0.0).
* prefill :func:`pkm_topk_prefill` (s1, s2 (T, B, H, n) fp32 with unit inner stride): custom op
  ``neuronspark::pkm_topk_prefill``, one program per row with both halves batched as (2, n), 1 warp.  Each half's top-16
  is the first 16 of its composite int64 keys (radix key << 32 | (2^32 - 1 - index)) sorted descending; when no two of
  the 16 compare equal under ``GTOp`` (==, -0 == +0, NaN == NaN) the network's output is that order, so the values
  (re-read by index) and indices come from the sort.  The candidate top-k is taken over the 50 non-dominated pairs,
  sorted the same way.  A row with such a tie in either step, or a non-finite half value, runs the row-batched
  selection (sort, index-order prefix counts, compaction, the 32-slot network) instead.
* decode  :func:`pkm_topk_softmax_t1` (raw scores (B, H, 2, n) and the (H, 2) score gain): custom op
  ``neuronspark::pkm_topk_softmax_t1``: the gain multiply at the load, the same exact selection and the softmax of the
  16 selected scores (max, libdevice exp, 16-term sum, division: the sum is the only reordered arithmetic) in one
  kernel, 4 warps.  :func:`pkm_beta_table` forms the gain table.

Outputs: sc (and the decode weights) fp32, u / ii / jj int64, each (..., 16) contiguous.  The prefill and decode
kernels accept n in {16, 32, 64, 128, 256}; training is n = 128 only.  Tests: the per-operator tolerance rule of
:mod:`neuronspark.utils.testing`.

Pure-torch references (same signatures): :func:`pkm_topk_train_ref`, :func:`pkm_topk_prefill_ref`,
:func:`pkm_topk_softmax_t1_ref` (three torch.topk calls, the candidate add, the gathers).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from neuronspark.ops.triton import require_cuda

__all__ = [
    "N_SUB",
    "TOPK",
    "PKMTopKFn",
    "pkm_topk_train",
    "pkm_topk_train_ref",
    "pkm_topk_prefill",
    "pkm_topk_prefill_ref",
    "pkm_topk_softmax_t1",
    "pkm_topk_softmax_t1_ref",
    "pkm_beta_table",
]

# ================================================================================================ fixed constants
# Shapes of the default configuration and the launch configuration of the training and inference operators.
#: sub-keys per half (n) and slots selected per head (k)
N_SUB = 128
TOPK = 16
#: training forward: rows (t, b, h) per program, warps; the pruned candidate table (50 pairs, padded to 64 slots)
TRAIN_ROWS = 4
TRAIN_WARPS = 2
PRUNE_SLOTS = 64
PRUNE_PAIRS = 50
#: training backward (zero-fill + scatter-store form): rows per program, warps
BWD_ROWS = 2
BWD_WARPS = 2
#: prefill exact kernel: one row per program, 1 warp
PREFILL_WARPS = 1
#: decode kernel (scaled top-k + softmax): one row per program, 4 warps
DECODE_WARPS = 4
#: half widths supported by the prefill / decode exact kernels
EXACT_N = (16, 32, 64, 128, 256)

# documented first-call cache: the (64,) int32 pruned-pair table per device (built eagerly on the first call)
_PIX: Dict[Tuple[str, int], torch.Tensor] = {}


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError("neuronspark.ops.pkm_topk: " + msg)


def _pix(dev: torch.device) -> torch.Tensor:
    """(64,) int32: i * 16 + j of the pairs (i, j) with (i + 1)(j + 1) <= 16 in (i, j) order, -1 padding."""
    key = (dev.type, -1 if dev.index is None else int(dev.index))
    t = _PIX.get(key)
    if t is None:
        v = [i * TOPK + j for i in range(TOPK) for j in range(TOPK) if (i + 1) * (j + 1) <= TOPK]
        _need(len(v) == PRUNE_PAIRS, "pruned candidate table size")
        t = torch.tensor(v + [-1] * (PRUNE_SLOTS - len(v)), dtype=torch.int32, device=dev)
        _PIX[key] = t
    return t


# ================================================================================================ launchers (training)
def _check_train(s: torch.Tensor, k: int) -> None:
    require_cuda(s, "pkm_topk_train")
    _need(s.dtype == torch.float32 and s.dim() == 5 and int(s.shape[3]) == 2,
          f"s must be an fp32 (T, B, H, 2, n) tensor, got {tuple(s.shape)} {s.dtype}")
    _need(int(s.shape[4]) == N_SUB and int(k) == TOPK,
          f"the training kernel covers n = {N_SUB}, k = {TOPK}, got n = {int(s.shape[4])}, k = {int(k)}")
    T, B, H = int(s.shape[0]), int(s.shape[1]), int(s.shape[2])
    _need(T * B * H < 2 ** 31 // TOPK, "too many rows")


def _topk_train_fwd(s: torch.Tensor):
    """(sc, u, ii, jj, i1, i2, j), each (T, B, H, 16) contiguous: the outputs and the backward's indices."""
    _check_train(s, TOPK)
    T, B, H = int(s.shape[0]), int(s.shape[1]), int(s.shape[2])
    R = T * B * H
    dev = s.device
    sc = torch.empty((T, B, H, TOPK), dtype=torch.float32, device=dev)
    u, ii, jj, i1, i2, j = (torch.empty((T, B, H, TOPK), dtype=torch.int64, device=dev) for _ in range(6))
    nd = (TRAIN_ROWS * 2 * TOPK).bit_length() - 1
    st = s.stride()
    if R:
        _pkm_topk_kernel[(triton.cdiv(R, TRAIN_ROWS),)](
            s, sc, u, ii, jj, i1, i2, j, R, B, H, st[0], st[1], st[2], st[3], st[4], _pix(dev),
            NL=N_SUB, KK=TOPK, BR=TRAIN_ROWS, ND=nd, NP=PRUNE_SLOTS, num_warps=TRAIN_WARPS, enable_fp_fusion=False)
    return sc, u, ii, jj, i1, i2, j


def _topk_train_bwd(g_sc: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor, j: torch.Tensor,
                    s_shape: Tuple[int, ...]) -> torch.Tensor:
    """The (T, B, H, 2, n) fp32 contiguous gradient of s from the gradient of sc."""
    T, B, H, two, n = (int(v) for v in s_shape)
    _need(two == 2 and n == N_SUB and tuple(g_sc.shape) == (T, B, H, TOPK) and g_sc.dtype == torch.float32,
          f"backward shapes {tuple(s_shape)} / {tuple(g_sc.shape)} {g_sc.dtype} outside the kernel's scope")
    R = T * B * H
    g2 = g_sc.reshape(R, TOPK)
    j2, a2, b2 = j.reshape(R, TOPK), i1.reshape(R, TOPK), i2.reshape(R, TOPK)
    for t_ in (g2, j2, a2, b2):
        _need(t_.stride(1) == 1, "unit last stride expected for the gradient and the saved indices")
    out = torch.empty((T, B, H, two, n), dtype=torch.float32, device=g_sc.device)
    if R:
        _pkm_topk_bwd3_kernel[(triton.cdiv(R, BWD_ROWS),)](
            g2, j2, a2, b2, out, R, g2.stride(0), j2.stride(0), a2.stride(0), b2.stride(0),
            NL=N_SUB, KK=TOPK, BR=BWD_ROWS, num_warps=BWD_WARPS)
    return out


@torch.library.custom_op("neuronspark::pkm_topk_train_fwd", mutates_args=())
def _pkm_topk_train_fwd_op(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor, torch.Tensor]:
    return _topk_train_fwd(s)


@_pkm_topk_train_fwd_op.register_fake
def _pkm_topk_train_fwd_fake(s):
    shp = (s.shape[0], s.shape[1], s.shape[2], TOPK)
    return (s.new_empty(shp, dtype=torch.float32),) + tuple(s.new_empty(shp, dtype=torch.int64) for _ in range(6))


@torch.library.custom_op("neuronspark::pkm_topk_train_bwd", mutates_args=())
def _pkm_topk_train_bwd_op(g_sc: torch.Tensor, i1: torch.Tensor, i2: torch.Tensor, j: torch.Tensor,
                           s_shape: List[int]) -> torch.Tensor:
    return _topk_train_bwd(g_sc, i1, i2, j, tuple(s_shape))


@_pkm_topk_train_bwd_op.register_fake
def _pkm_topk_train_bwd_fake(g_sc, i1, i2, j, s_shape):
    return g_sc.new_empty(tuple(s_shape), dtype=torch.float32)


class PKMTopKFn(torch.autograd.Function):
    """(sc, u, ii, jj) of the product top-k; the gradient reaches s through sc only (u / ii / jj non-differentiable)."""

    @staticmethod
    def forward(ctx, s, k: int):
        sc, u, ii, jj, i1, i2, j = torch.ops.neuronspark.pkm_topk_train_fwd(s)
        ctx.save_for_backward(i1, i2, j)
        ctx.s_shape = [int(v) for v in s.shape]
        ctx.mark_non_differentiable(u, ii, jj)
        return sc, u, ii, jj

    @staticmethod
    def backward(ctx, g_sc, g_u, g_ii, g_jj):
        if g_sc is None:
            return None, None
        i1, i2, j = ctx.saved_tensors
        return torch.ops.neuronspark.pkm_topk_train_bwd(g_sc, i1, i2, j, ctx.s_shape), None


def pkm_topk_train(s: torch.Tensor, k: int = TOPK) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sc fp32, u, ii, jj int64), each (T, B, H, 16) contiguous, from the scaled half scores ``s`` (T, B, H, 2, 128)
    fp32 (any strides); differentiable in ``s`` through ``sc``."""
    _check_train(s, k)
    if torch.is_grad_enabled() and s.requires_grad:
        return PKMTopKFn.apply(s, int(k))
    sc, u, ii, jj, _i1, _i2, _j = torch.ops.neuronspark.pkm_topk_train_fwd(s)
    return sc, u, ii, jj


# ================================================================================================ prefill
def _exact_supported(s1: torch.Tensor, s2: torch.Tensor, k: int) -> bool:
    n = int(s1.shape[-1]) if s1.dim() else 0
    return (s1.dim() == 4 and s2.dim() == 4 and tuple(s1.shape) == tuple(s2.shape) and int(s1.shape[0]) >= 1
            and s1.is_cuda and s2.is_cuda and s1.device == s2.device and s1.dtype == torch.float32
            and s2.dtype == torch.float32 and int(k) == TOPK and n in EXACT_N)


def _topk_prefill(s1: torch.Tensor, s2: torch.Tensor, k: int):
    _need(_exact_supported(s1, s2, k),
          f"prefill top-k: unsupported operands {tuple(s1.shape)} {s1.dtype} / {tuple(s2.shape)} {s2.dtype}, k {k} "
          f"(CUDA fp32 (T, B, H, n) halves of one shape, n in {EXACT_N}, k = {TOPK})")
    T, B, H, n = (int(v) for v in s1.shape)
    if s1.stride(-1) != 1:
        s1 = s1.contiguous()
    if s2.stride(-1) != 1:
        s2 = s2.contiguous()
    shp = (T, B, H, int(k))
    dev = s1.device
    sc = torch.empty(shp, dtype=torch.float32, device=dev)
    u = torch.empty(shp, dtype=torch.int64, device=dev)
    ii = torch.empty(shp, dtype=torch.int64, device=dev)
    jj = torch.empty(shp, dtype=torch.int64, device=dev)
    R = T * B * H
    if R:
        _product_topk_prefill_kernel[(R,)](
            s1, s2, sc, u, ii, jj, _pix(dev), s1.stride(0), s1.stride(1), s1.stride(2), s2.stride(0), s2.stride(1),
            s2.stride(2), B, H, n, N=n, num_warps=PREFILL_WARPS)
    return sc, u, ii, jj


@torch.library.custom_op("neuronspark::pkm_topk_prefill", mutates_args=())
def _pkm_topk_prefill_op(s1: torch.Tensor, s2: torch.Tensor,
                         k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _topk_prefill(s1, s2, k)


@_pkm_topk_prefill_op.register_fake
def _pkm_topk_prefill_fake(s1, s2, k):
    shp = (*s1.shape[:-1], int(k))
    return (s1.new_empty(shp), s1.new_empty(shp, dtype=torch.int64), s1.new_empty(shp, dtype=torch.int64),
            s1.new_empty(shp, dtype=torch.int64))


def pkm_topk_prefill(s1: torch.Tensor, s2: torch.Tensor,
                     k: int = TOPK) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sc fp32, u, ii, jj int64), each (T, B, H, 16) contiguous, from the scaled half scores ``s1`` / ``s2``
    (T, B, H, n) fp32 (strided views are fine; a non-unit inner stride is made contiguous).  No autograd."""
    return torch.ops.neuronspark.pkm_topk_prefill(s1, s2, int(k))


# ================================================================================================ decode
def _topk_softmax_t1(s_raw: torch.Tensor, beta: torch.Tensor, k: int):
    require_cuda(s_raw, "pkm_topk_softmax_t1")
    _need(s_raw.dim() == 4, f"s_raw must be (B, H, 2, n), got {tuple(s_raw.shape)}")
    B, H, two, n = (int(v) for v in s_raw.shape)
    _need(two == 2 and s_raw.dtype == torch.float32 and s_raw.stride(3) == 1,
          f"s_raw must be fp32 (B, H, 2, n) with unit inner stride, got {tuple(s_raw.shape)} {s_raw.dtype} "
          f"{tuple(s_raw.stride())}")
    _need(int(k) == TOPK and n in EXACT_N, f"k {k} / n {n} outside the kernel (k = {TOPK}, n in {EXACT_N})")
    _need(tuple(beta.shape) == (H, 2) and beta.dtype == torch.float32 and beta.is_contiguous()
          and beta.device == s_raw.device, f"beta must be a contiguous fp32 ({H}, 2) tensor, got {tuple(beta.shape)}")
    shp = (B, H, TOPK)
    dev = s_raw.device
    sc = torch.empty(shp, dtype=torch.float32, device=dev)
    w = torch.empty(shp, dtype=torch.float32, device=dev)
    u = torch.empty(shp, dtype=torch.int64, device=dev)
    ii = torch.empty(shp, dtype=torch.int64, device=dev)
    jj = torch.empty(shp, dtype=torch.int64, device=dev)
    if B * H:
        _pkm_topk_sm_kernel[(B * H,)](s_raw, beta, sc, w, u, ii, jj, _pix(dev), s_raw.stride(0), s_raw.stride(1),
                                      s_raw.stride(2), H, n, N=n, num_warps=DECODE_WARPS)
    return sc, w, u, ii, jj


@torch.library.custom_op("neuronspark::pkm_topk_softmax_t1", mutates_args=())
def _pkm_topk_softmax_t1_op(s_raw: torch.Tensor, beta: torch.Tensor, k: int
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _topk_softmax_t1(s_raw, beta, k)


@_pkm_topk_softmax_t1_op.register_fake
def _pkm_topk_softmax_t1_fake(s_raw, beta, k):
    shp = (s_raw.shape[0], s_raw.shape[1], int(k))
    return (s_raw.new_empty(shp), s_raw.new_empty(shp), s_raw.new_empty(shp, dtype=torch.int64),
            s_raw.new_empty(shp, dtype=torch.int64), s_raw.new_empty(shp, dtype=torch.int64))


def pkm_topk_softmax_t1(s_raw: torch.Tensor, beta: torch.Tensor, k: int = TOPK
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sc, w fp32, u, ii, jj int64), each (B, H, 16) contiguous, for one decode token: ``s_raw`` (B, H, 2, n) fp32
    the unit sub-key cosines (unit inner stride), ``beta`` (H, 2) fp32 contiguous the score gain
    (:func:`pkm_beta_table`); ``s = s_raw * beta`` (one fp32 multiply), the product top-k of s and ``w = softmax(sc)``."""
    return torch.ops.neuronspark.pkm_topk_softmax_t1(s_raw, beta, int(k))


def pkm_beta_table(pkm_log_gain: torch.Tensor, heads: int = 4, half_dim: int = 64) -> torch.Tensor:
    """(heads, 2) fp32 contiguous score gain ``sqrt(half_dim) * exp(pkm_log_gain)``, formed as the PKM read forms it
    (``math.sqrt(dh) * torch.exp(log_gain.float()).view(H, 2)``), for :func:`pkm_topk_softmax_t1`."""
    return (math.sqrt(half_dim) * torch.exp(pkm_log_gain.to(torch.float32))).view(int(heads), 2).contiguous()


# ================================================================================================ pure-torch references
def _product_topk_torch(s1: torch.Tensor, s2: torch.Tensor, k: int):
    """The product-key top-k in torch: three torch.topk calls, the candidate add, the gathers."""
    n = s1.shape[-1]
    t1, i1 = torch.topk(s1, k, dim=-1)
    t2, i2 = torch.topk(s2, k, dim=-1)
    cand = (t1.unsqueeze(-1) + t2.unsqueeze(-2)).flatten(-2)
    sc, j = torch.topk(cand, k, dim=-1)
    ii = torch.gather(i1, -1, torch.div(j, k, rounding_mode="floor"))
    jj = torch.gather(i2, -1, torch.remainder(j, k))
    return sc, ii * n + jj, ii, jj


def pkm_topk_train_ref(s: torch.Tensor, k: int = TOPK):
    """Pure-torch reference of :func:`pkm_topk_train` (differentiable through torch.topk / gather)."""
    return _product_topk_torch(s[..., 0, :], s[..., 1, :], int(k))


def pkm_topk_prefill_ref(s1: torch.Tensor, s2: torch.Tensor, k: int = TOPK):
    """Pure-torch reference of :func:`pkm_topk_prefill`."""
    return _product_topk_torch(s1, s2, int(k))


def pkm_topk_softmax_t1_ref(s_raw: torch.Tensor, beta: torch.Tensor, k: int = TOPK):
    """Pure-torch reference of :func:`pkm_topk_softmax_t1`: s = s_raw * beta, the product top-k, torch.softmax of the
    selected scores."""
    H = int(s_raw.shape[1])
    s = s_raw * beta.view(1, H, 2, 1)
    sc, u, ii, jj = _product_topk_torch(s[..., 0, :], s[..., 1, :], int(k))
    w = torch.softmax(sc, dim=-1)
    return sc, w, u, ii, jj


# ================================================================================================ kernels
# Training selection network.
@triton.jit
def _pk_add(a, b):
    return tl.inline_asm_elementwise(asm="add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def _pk_skey(bits):
    """int32 bits of an fp32 value -> int32 key ordered as TopKTypeConfig<float>::convert (as unsigned), i.e. the
    convert result with its top bit flipped: non-NaN x >= 0: bits; x < 0: bits ^ 0x7fffffff; every NaN 0x7fffffff."""
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    k = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    return tl.where(nan, 0x7FFFFFFF, k)

@triton.jit
def _pk_slot_order(skey, NL: tl.constexpr, KK: tl.constexpr, BR: tl.constexpr):
    """(BR, NL) int32 radix keys -> (BR, KK) int32 element indices in gatherTopK's output slot order."""
    idx = tl.arange(0, NL)[None, :]
    comp = (skey.to(tl.int64) << 32) | (NL - 1 - idx).to(tl.int64)
    top = tl.topk(comp, KK)                                   # (BR, KK) descending, unique keys
    sk = (top >> 32).to(tl.int32)
    ix = (NL - 1) - (top - ((top >> 32) << 32)).to(tl.int32)
    kth = tl.min(sk, axis=1)
    key2 = tl.where(sk == kth[:, None], 4096, 0) + ix
    key2 = tl.sort(key2)
    return key2 & 4095

@triton.jit
def _pk_slot_order_ix(skey, ix, NIX: tl.constexpr, KK: tl.constexpr, BR: tl.constexpr):
    """_pk_slot_order over a SUBSET of an NIX-wide array: (BR, NP) int32 radix keys of the subset's elements and
    their (BR, NP) int32 indices in the full array -> (BR, KK) full-array indices in gatherTopK's slot order (the k
    largest by (key desc, index asc); slot order = above the k-th key first, equal ones after, each by index)."""
    comp = (skey.to(tl.int64) << 32) | (NIX - 1 - ix).to(tl.int64)
    top = tl.topk(comp, KK)
    sk = (top >> 32).to(tl.int32)
    sx = (NIX - 1) - (top - ((top >> 32) << 32)).to(tl.int32)
    kth = tl.min(sk, axis=1)
    key2 = tl.where(sk == kth[:, None], 4096, 0) + sx
    key2 = tl.sort(key2)
    return key2 & 4095

@triton.jit
def _pk_cas(kb, vb, ok, pos, J: tl.constexpr, SIZE: tl.constexpr, ND: tl.constexpr):
    """one compare-swap layer of SortUtils.cuh bitonicSort on hypercube-shaped slot arrays: stride 2^J, dir =
    (p & SIZE) != 0 (SIZE = 0: the final merge, dir = false)."""
    ax: tl.constexpr = ND - 1 - J
    pk = kb ^ tl.xor_sum(kb, ax, keep_dims=True)
    pv = vb ^ tl.xor_sum(vb, ax, keep_dims=True)
    po = ok ^ tl.xor_sum(ok, ax, keep_dims=True)
    right = ((pos >> J) & 1) != 0
    ka = tl.where(right, pk, kb).to(tl.float32, bitcast=True)
    kbb = tl.where(right, kb, pk).to(tl.float32, bitcast=True)
    oka = tl.where(right, po, ok) != 0
    okb = tl.where(right, ok, po) != 0
    gt = ((ka != ka) & (kbb == kbb)) | (ka > kbb)
    sw = (gt & oka) | (~okb)
    if SIZE == 0:
        do = ~sw
    else:
        do = sw == ((pos & SIZE) != 0)
    kb = tl.where(do, pk, kb)
    vb = tl.where(do, pv, vb)
    ok = tl.where(do, po, ok)
    return kb, vb, ok

@triton.jit
def _pk_bitonic(kbits, vidx, KK: tl.constexpr, BR: tl.constexpr):
    """(BR, KK) int32 value bits and int32 indices in slot order -> torch's bitonicSortKVInPlace result (first KK
    slots of the 32-slot network)."""
    P2: tl.constexpr = 2 * KK
    zk = tl.zeros([BR, KK], dtype=tl.int32)
    kb = tl.reshape(tl.permute(tl.join(kbits, zk), (0, 2, 1)), (BR, P2))
    vb = tl.reshape(tl.permute(tl.join(vidx, zk), (0, 2, 1)), (BR, P2))
    ok = tl.reshape(tl.permute(tl.join(zk + 1, zk), (0, 2, 1)), (BR, P2))
    pos = tl.zeros([BR, P2], dtype=tl.int32) + tl.arange(0, P2)[None, :]
    return kb, vb, ok, pos

@triton.jit
def _pk_network(kb, vb, ok, pos, ND: tl.constexpr):
    # size 2
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 0, 2, ND)
    # size 4
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 1, 4, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 0, 4, ND)
    # size 8
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 2, 8, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 1, 8, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 0, 8, ND)
    # size 16
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 3, 16, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 2, 16, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 1, 16, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 0, 16, ND)
    # final merge, dir = false
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 4, 0, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 3, 0, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 2, 0, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 1, 0, ND)
    kb, vb, ok = _pk_cas(kb, vb, ok, pos, 0, 0, ND)
    return kb, vb

@triton.jit
def _pk_sorted16(kbits, vidx, KK: tl.constexpr, BR: tl.constexpr, ND: tl.constexpr):
    kb, vb, ok, pos = _pk_bitonic(kbits, vidx, KK, BR)
    shape: tl.constexpr = [2] * ND
    kb = tl.reshape(kb, shape)
    vb = tl.reshape(vb, shape)
    ok = tl.reshape(ok, shape)
    pos = tl.reshape(pos, shape)
    kb, vb = _pk_network(kb, vb, ok, pos, ND)
    kb = tl.reshape(kb, (BR, 2, KK))
    vb = tl.reshape(vb, (BR, 2, KK))
    k0, k1 = tl.split(tl.permute(kb, (0, 2, 1)))
    v0, v1 = tl.split(tl.permute(vb, (0, 2, 1)))
    return k0, v0

@triton.jit
def _pkm_topk_kernel(S, SC, U, II, JJ, I1, I2, JO, R, B, H, s_t, s_b, s_h, s_a, s_n, PIX,
                     NL: tl.constexpr, KK: tl.constexpr, BR: tl.constexpr, ND: tl.constexpr,
                     NP: tl.constexpr = 64):
    """BR rows (t, b, h) of s: the half top-16s, the pruned candidate top-16 (all 256 candidates in a program with a
    non-finite or -0 half value), the gathers and u; also the backward's i1 / i2 / j."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    rmask = rows < R
    rr = tl.where(rmask, rows, 0)
    h = rr % H
    b = (rr // H) % B
    t = rr // (H * B)
    base = t.to(tl.int64) * s_t + b.to(tl.int64) * s_b + h.to(tl.int64) * s_h          # (BR,)
    col = tl.arange(0, NL)
    # ---- the two half top-ks
    x1 = tl.load(S + base[:, None] + col[None, :].to(tl.int64) * s_n)                  # (BR, NL) fp32
    x2 = tl.load(S + base[:, None] + s_a + col[None, :].to(tl.int64) * s_n)
    o1 = _pk_slot_order(_pk_skey(x1.to(tl.int32, bitcast=True)), NL, KK, BR)            # (BR, KK) indices
    o2 = _pk_slot_order(_pk_skey(x2.to(tl.int32, bitcast=True)), NL, KK, BR)
    g1 = tl.load(S + base[:, None] + o1.to(tl.int64) * s_n)                             # slot-order values
    g2 = tl.load(S + base[:, None] + s_a + o2.to(tl.int64) * s_n)
    tb1, ti1 = _pk_sorted16(g1.to(tl.int32, bitcast=True), o1, KK, BR, ND)
    tb2, ti2 = _pk_sorted16(g2.to(tl.int32, bitcast=True), o2, KK, BR, ND)
    t1 = tb1.to(tl.float32, bitcast=True)
    t2 = tb2.to(tl.float32, bitcast=True)
    # ---- candidates and their top-k: over the NP-slot table PIX of the (i, j) with (i + 1)(j + 1) <= k (50 of 256;
    # -1 = unused slot).  With t1 / t2 sorted descending and every value finite and not -0, the (i + 1)(j + 1) - 1 >= k
    # candidates (i' <= i, j' <= j) rank at or above cand(i, j) by value (RN addition is monotonic; no NaN / -0 / inf
    # among the sums) and strictly before it by index, so a pruned candidate is never among torch's k (value desc,
    # index asc); a program with a non-finite or -0 value runs the full 256.
    b1 = t1.to(tl.int32, bitcast=True)
    b2 = t2.to(tl.int32, bitcast=True)
    bad1 = ((b1 & 0x7F800000) == 0x7F800000) | (b1 == -2147483648)
    bad2 = ((b2 & 0x7F800000) == 0x7F800000) | (b2 == -2147483648)
    nbad = tl.sum(tl.sum(bad1.to(tl.int32), axis=1), axis=0) + tl.sum(tl.sum(bad2.to(tl.int32), axis=1), axis=0)
    if nbad == 0:
        pix = tl.load(PIX + tl.arange(0, NP))
        pv = pix >= 0
        pix2 = tl.broadcast_to(tl.where(pv, pix, 0)[None, :], (BR, NP))
        pa = tl.gather(t1, pix2 >> 4, 1)
        pb = tl.gather(t2, pix2 & 15, 1)
        pk_ = _pk_skey(_pk_add(pa, pb).to(tl.int32, bitcast=True))
        pk_ = tl.where(tl.broadcast_to(pv[None, :], (BR, NP)), pk_, -2147483648)
        oc = _pk_slot_order_ix(pk_, pix2, KK * KK, KK, BR)
    else:
        cand = tl.reshape(_pk_add(t1[:, :, None], t2[:, None, :]), (BR, KK * KK))
        oc = _pk_slot_order(_pk_skey(cand.to(tl.int32, bitcast=True)), KK * KK, KK, BR)
    ca = tl.gather(t1, oc >> 4, 1)
    cb = tl.gather(t2, oc & 15, 1)
    gc = _pk_add(ca, cb)
    scb, jj_ = _pk_sorted16(gc.to(tl.int32, bitcast=True), oc, KK, BR, ND)
    iia = tl.gather(ti1, jj_ >> 4, 1).to(tl.int64)
    jja = tl.gather(ti2, jj_ & 15, 1).to(tl.int64)
    oo = rr.to(tl.int64)[:, None] * KK + tl.arange(0, KK)[None, :]
    om = rmask[:, None]
    tl.store(SC + oo, scb.to(tl.float32, bitcast=True), mask=om)
    tl.store(U + oo, iia * NL + jja, mask=om)
    tl.store(II + oo, iia, mask=om)
    tl.store(JJ + oo, jja, mask=om)
    tl.store(I1 + oo, ti1.to(tl.int64), mask=om)
    tl.store(I2 + oo, ti2.to(tl.int64), mask=om)
    tl.store(JO + oo, jj_.to(tl.int64), mask=om)


@triton.jit
def _pkm_topk_bwd3_kernel(GSC, J, I1, I2, OUT, R, gss, js, i1s, i2s, NL: tl.constexpr, KK: tl.constexpr,
                          BR: tl.constexpr):
    """Rows r of (T, B, H): the (2, NL) output row zero-filled, then OUT[r, 0, i1[a]] = g_t1[a] + 0,
    OUT[r, 1, i2[a]] = g_t2[a] + 0 (scatter stores at distinct slots, after a CTA barrier); g_t1[a] / g_t2[c] = the
    masked (BR, KK, KK) row sums of g_sc over the candidates j with j // KK == a / j % KK == c."""
    rr = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    rm = rr < R
    q = tl.arange(0, KK)
    m2 = rm[:, None]
    g = tl.load(GSC + rr[:, None] * gss + q[None, :], mask=m2, other=0.0)               # (BR, KK)
    jv = tl.load(J + rr[:, None] * js + q[None, :], mask=m2, other=0).to(tl.int32)
    jr = jv // KK
    jc = jv - jr * KK
    a = tl.arange(0, KK)
    gt1 = tl.sum(tl.where(jr[:, None, :] == a[None, :, None], g[:, None, :], 0.0), axis=2)   # (BR, KK) indexed by a
    gt2 = tl.sum(tl.where(jc[:, None, :] == a[None, :, None], g[:, None, :], 0.0), axis=2)
    i1v = tl.load(I1 + rr[:, None] * i1s + a[None, :], mask=m2, other=0).to(tl.int32)     # slot of half-row a
    i2v = tl.load(I2 + rr[:, None] * i2s + a[None, :], mask=m2, other=0).to(tl.int32)
    p = tl.arange(0, 2 * NL)
    ob = rr[:, None] * (2 * NL)
    tl.store(OUT + (ob + p[None, :]), tl.zeros([BR, 2 * NL], dtype=tl.float32), mask=m2)   # +0.0 everywhere
    tl.debug_barrier()
    tl.store(OUT + (ob + i1v), gt1 + 0.0, mask=m2)
    tl.store(OUT + (ob + NL + i2v), gt2 + 0.0, mask=m2)


# Exact selection of the prefill and decode kernels.
@triton.jit
def _order_key(x):
    # TopKTypeConfig<float>::convert (uint32) as a signed int32 of the same order (key ^ 0x80000000): x >= +0.0 -> its
    # bits; x < 0 (incl. -0.0) -> bits ^ 0x7FFFFFFF; NaN -> 0x7FFFFFFF (the largest, as torch's 0xFFFFFFFF)
    b = x.to(tl.int32, bitcast=True)
    k = tl.where(b >= 0, b, b ^ 0x7FFFFFFF)
    return tl.where(x == x, k, 0x7FFFFFFF)


@triton.jit
def _select_slots_rows(x, R: tl.constexpr, N: tl.constexpr, K: tl.constexpr, P2: tl.constexpr):
    # _select_slots of every row of x (R, N), along axis 1
    key = _order_key(x)
    srt = tl.sort(key, 1, True)
    offs = tl.arange(0, N)[None, :]
    kth = tl.sum(tl.where(offs == K - 1, srt, 0), axis=1)
    gt = (key > kth[:, None]).to(tl.int32)
    eq = (key == kth[:, None]).to(tl.int32)
    n_gt = tl.sum(gt, axis=1)
    gt_rank = tl.cumsum(gt, axis=1) - gt
    eq_rank = tl.cumsum(eq, axis=1) - eq
    take_eq = (eq != 0) & (eq_rank < K - n_gt[:, None])
    return tl.where(gt != 0, gt_rank, tl.where(take_eq, n_gt[:, None] + eq_rank, P2))


@triton.jit
def _compact_rows(x, slot, R: tl.constexpr, N: tl.constexpr, P2: tl.constexpr):
    s = tl.arange(0, P2)[None, :, None]
    m = slot[:, None, :] == s                                             # (R, P2, N)
    bits = x.to(tl.int32, bitcast=True)
    vb = tl.sum(tl.where(m, bits[:, None, :], 0), axis=2)
    vi = tl.sum(tl.where(m, tl.arange(0, N)[None, None, :], 0), axis=2)
    return vb.to(tl.float32, bitcast=True), vi


@triton.jit
def _bstage_rows(k, v, ok, R: tl.constexpr, STRIDE: tl.constexpr, SIZE: tl.constexpr, FINAL: tl.constexpr,
                 P2: tl.constexpr):
    # _bstage on every row of (R, P2)
    G: tl.constexpr = P2 // (2 * STRIDE)
    ka, kb = tl.split(tl.permute(tl.reshape(k, (R, G, 2, STRIDE)), (0, 1, 3, 2)))
    va, vb = tl.split(tl.permute(tl.reshape(v, (R, G, 2, STRIDE)), (0, 1, 3, 2)))
    oa, ob = tl.split(tl.permute(tl.reshape(ok, (R, G, 2, STRIDE)), (0, 1, 3, 2)))
    tid = tl.arange(0, G)[None, :, None] * STRIDE + tl.arange(0, STRIDE)[None, None, :]
    if FINAL:
        dirn = tid < 0
    else:
        dirn = (tid & (SIZE // 2)) != 0
    comp = ((ka != ka) & (kb == kb)) | (ka > kb)
    swap = (comp & (oa != 0)) | (ob == 0)
    ex = swap == dirn
    nka = tl.where(ex, kb, ka)
    nkb = tl.where(ex, ka, kb)
    nva = tl.where(ex, vb, va)
    nvb = tl.where(ex, va, vb)
    noa = tl.where(ex, ob, oa)
    nob = tl.where(ex, oa, ob)
    k = tl.reshape(tl.permute(tl.join(nka, nkb), (0, 1, 3, 2)), (R, P2))
    v = tl.reshape(tl.permute(tl.join(nva, nvb), (0, 1, 3, 2)), (R, P2))
    ok = tl.reshape(tl.permute(tl.join(noa, nob), (0, 1, 3, 2)), (R, P2))
    return k, v, ok


@triton.jit
def _bitonic32_rows(k, v, ok, R: tl.constexpr):
    k, v, ok = _bstage_rows(k, v, ok, R, 1, 2, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 2, 4, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 1, 4, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 4, 8, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 2, 8, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 1, 8, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 8, 16, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 4, 16, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 2, 16, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 1, 16, False, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 16, 32, True, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 8, 32, True, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 4, 32, True, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 2, 32, True, 32)
    k, v, ok = _bstage_rows(k, v, ok, R, 1, 32, True, 32)
    return k, v, ok


@triton.jit
def _first16_rows(x, R: tl.constexpr):
    a, _b = tl.split(tl.permute(tl.reshape(x, (R, 2, 16)), (0, 2, 1)))
    return a


@triton.jit
def _topk16_rows(x, R: tl.constexpr, N: tl.constexpr):
    # torch.topk(x, 16) of every row of x (R, N): (R, 16) values / int32 indices
    slot = _select_slots_rows(x, R, N, 16, 32)
    kv, iv = _compact_rows(x, slot, R, N, 32)
    ok = ((tl.arange(0, 32)[None, :] < 16) & (tl.arange(0, R)[:, None] >= 0)).to(tl.int32)
    kv, iv, ok = _bitonic32_rows(kv, iv, ok, R)
    return _first16_rows(kv, R), _first16_rows(iv, R)


@triton.jit
def _pack_desc(key, idx):
    # composite int64 key: descending order = radix key descending, then index ascending
    return (key.to(tl.int64) << 32) | (4294967295 - idx.to(tl.int64))


@triton.jit
def _unpack_idx(p):
    return (4294967295 - (p & 4294967295)).to(tl.int32)


@triton.jit
def _first16_of(x, R: tl.constexpr, N: tl.constexpr):
    # x (R, N) -> (R, 16): the first 16 columns (one nonzero term per sum)
    G: tl.constexpr = N // 16
    x3 = tl.reshape(x, (R, G, 16))
    g = tl.arange(0, G)[None, :, None]
    return tl.sum(tl.where(g == 0, x3, 0), axis=1)


@triton.jit
def _net_ties(v, R: tl.constexpr):
    # (R,) int32: 1 when two of the row's 16 values compare equal under GTOp (==, or both NaN)
    a = v[:, :, None]
    b = v[:, None, :]
    eq = ((a == b) | ((a != a) & (b != b))).to(tl.int32)
    return (tl.sum(tl.sum(eq, axis=2), axis=1) > 16).to(tl.int32)


@triton.jit
def _product16(x, hb, beta, PIX, N: tl.constexpr, HAS_BETA: tl.constexpr):
    """the product top-16 of one row from its (2, N) half scores ``x`` (row 0 = s1, row 1 = s2; ``hb`` (2, 1) the rows'
    pointers, ``x`` = the loaded rows [* ``beta`` (2, 1)]): (scores fp32, ii int32, jj int32), each (16,)"""
    offs = tl.arange(0, N)
    srt = tl.sort(_pack_desc(_order_key(x), offs[None, :]), 1, True)
    ti = _unpack_idx(_first16_of(srt, 2, N))                              # (2, 16) int32
    tv = tl.load(hb + ti)                                                 # the values' own bits
    if HAS_BETA:
        tv = tv * beta                                                    # the same fp32 multiply as x's
    bad = tl.max(_net_ties(tv, 2), axis=0)
    fin = (tv == tv) & (tl.abs(tv) < float("inf"))
    bad = tl.maximum(bad, 1 - tl.min(tl.min(fin.to(tl.int32), axis=1), axis=0))
    t1, t2 = tl.split(tl.permute(tv, (1, 0)))                             # (16,) each
    i1, i2 = tl.split(tl.permute(ti, (1, 0)))
    # the 50 non-dominated candidates (PIX: j = r * 16 + c, -1 = padding)
    jc = tl.load(PIX + tl.arange(0, 64))
    rc = tl.where(jc >= 0, jc // 16, 0)
    cc = tl.where(jc >= 0, jc % 16, 0)
    sel16 = tl.arange(0, 16)[None, :]
    b1 = tl.sum(tl.where(rc[:, None] == sel16, t1.to(tl.int32, bitcast=True)[None, :], 0), axis=1).to(tl.float32, bitcast=True)
    b2 = tl.sum(tl.where(cc[:, None] == sel16, t2.to(tl.int32, bitcast=True)[None, :], 0), axis=1).to(tl.float32, bitcast=True)
    cand = b1 + b2                                                        # the fp32 candidate add
    ck = tl.where(jc >= 0, _order_key(cand), -2147483648)
    cs = tl.sort(_pack_desc(ck, tl.where(jc >= 0, jc, 0)), 0, True)
    j = tl.reshape(_unpack_idx(_first16_of(tl.reshape(cs, (1, 64)), 1, 64)), (16,))
    jr = j // 16
    jcol = j % 16
    s1v = tl.sum(tl.where(jr[:, None] == sel16, t1.to(tl.int32, bitcast=True)[None, :], 0), axis=1).to(tl.float32, bitcast=True)
    s2v = tl.sum(tl.where(jcol[:, None] == sel16, t2.to(tl.int32, bitcast=True)[None, :], 0), axis=1).to(tl.float32, bitcast=True)
    sc = s1v + s2v
    bad = tl.maximum(bad, tl.max(_net_ties(tl.reshape(sc, (1, 16)), 1), axis=0))
    ii = tl.sum(tl.where(jr[:, None] == sel16, i1[None, :], 0), axis=1)
    jj = tl.sum(tl.where(jcol[:, None] == sel16, i2[None, :], 0), axis=1)
    if bad != 0:
        # the row-batched selection (ties / non-finite values)
        vv, vi = _topk16_rows(x, 2, N)
        u1, u2 = tl.split(tl.permute(vv, (1, 0)))
        k1, k2 = tl.split(tl.permute(vi, (1, 0)))
        cnd = tl.reshape(u1[:, None] + u2[None, :], (1, 256))
        scv, jv = _topk16_rows(cnd, 1, 256)
        sc = tl.reshape(scv, (16,))
        j2 = tl.reshape(jv, (16,))
        ii = tl.sum(tl.where((j2 // 16)[:, None] == sel16, k1[None, :], 0), axis=1)
        jj = tl.sum(tl.where((j2 % 16)[:, None] == sel16, k2[None, :], 0), axis=1)
    return sc, ii, jj


@triton.jit
def _product_topk_prefill_kernel(S1, S2, SC, U, II, JJ, PIX, s1_t, s1_b, s1_h, s2_t, s2_b, s2_h, B, H, n_sub,
                                 N: tl.constexpr):
    # the product top-k of one (t, b, h) row, the two halves as a (2, N) batch
    row = tl.program_id(0)
    h = row % H
    b = (row // H) % B
    t = row // (H * B)
    offs = tl.arange(0, N)
    half = tl.arange(0, 2)[:, None]
    base1 = S1 + t * s1_t + b * s1_b + h * s1_h
    base2 = S2 + t * s2_t + b * s2_b + h * s2_h
    hb = tl.where(half == 0, base1, base2)                                # (2, 1)
    x = tl.load(hb + offs[None, :])                                       # (2, N): row 0 = s1, row 1 = s2
    sc, ii, jj = _product16(x, hb, x, PIX, N, False)
    ii64 = ii.to(tl.int64)
    jj64 = jj.to(tl.int64)
    o = row * 16 + tl.arange(0, 16)
    tl.store(SC + o, sc)
    tl.store(U + o, ii64 * n_sub + jj64)
    tl.store(II + o, ii64)
    tl.store(JJ + o, jj64)


@triton.jit
def _pkm_topk_sm_kernel(S, BETA, SC, W, U, II, JJ, PIX, s_b, s_h, s_a, H, n_sub, N: tl.constexpr):
    """_product_topk_prefill_kernel for one (b, h) row of the raw (B, H, 2, N) scores: the two halves scaled by
    beta[h, a] at the load (the `s * beta` fp32 multiply of the PKM read), the exact half / product top-16s, and the
    softmax of the 16 selected scores (fp32: max, libdevice exp, sum, division)"""
    row = tl.program_id(0)
    h = row % H
    b = row // H
    offs = tl.arange(0, N)
    half = tl.arange(0, 2)[:, None]
    hb = S + b * s_b + h * s_h + half * s_a                                  # (2, 1): row 0 = half 0, row 1 = half 1
    x = tl.load(hb + offs[None, :])                                          # (2, N)
    beta = tl.load(BETA + h * 2 + half)                                      # (2, 1) fp32
    x = x * beta
    sc, ii, jj = _product16(x, hb, beta, PIX, N, True)
    ii64 = ii.to(tl.int64)
    jj64 = jj.to(tl.int64)
    m = tl.max(sc, axis=0)
    e = libdevice.exp(sc - m)                                                # libdevice exp, fp32 sum, division
    w = e / tl.sum(e, axis=0)
    o = row * 16 + tl.arange(0, 16)
    tl.store(SC + o, sc)
    tl.store(W + o, w)
    tl.store(U + o, ii64 * n_sub + jj64)
    tl.store(II + o, ii64)
    tl.store(JJ + o, jj64)
