"""Session copy-detector statistics of a prefill call in one kernel.

An inference call of T tokens first writes its unit operands u (the normalised order-1 slice of the lexical vector,
:mod:`neuronspark.modules.copy_detector`) into the copy ring ``ring`` (B, W, d) fp32 at the slots ``(p0 + t) % W``,
then scores every query of the call against the whole ring::

    end           = p0 + T - 1                                  the row position of the call's last token
    pos(w)        = end - ((end - w) mod W)                     the row position held by slot w (< 0: never written)
    start_b       = p0 - n_b                                    the row's first event (n_b = tokens of row b before the
                                                                call; 0 for every row of a cache started with the batch)
    valid[b,t,w]  = start_b <= pos(w) <= p0 + t - 2             the strict past (events at s <= t - 2)
    s[b,t,w]      = <ring[b, (p0 + t) % W], ring[b, w]>         fp32
    maximum[t,b]  = max_w s[b,t,w]               over valid w   (-inf without a valid slot)
    lse[t,b]      = logsumexp_w 32 s[b,t,w]      over valid w   (-inf without a valid slot)

The torch chain (:func:`copy_stats_prefill_ref`) materialises the (B, T, W) fp32 scores and walks them for the mask,
the max, the scale and the logsumexp.  Here one program holds ``COPY_BLOCK_T`` queries of one row and walks the ring in
tiles of ``COPY_BLOCK_W`` slots: the scores of a tile are one tensor-core dot over d (chunks of 32 channels, or 16 when
d is not a multiple of 32) at ``tf32x3`` input precision (three TF32 products, close to IEEE fp32), the invalid slots
are set to -inf, and the row statistics are kept online (running max m, running sum of exp(32 (s - m))).  A tile that
holds no valid slot for any query of the block is skipped, so a call that starts its rows reads the lower triangle
only.  32 is a power of two, so ``32 maximum`` is the max of the scaled scores exactly as in the chain; the sums run
in another order and ``tl.exp`` is the approximate exponential.

Operator: :func:`copy_stats_prefill` (custom op ``neuronspark::copy_stats_prefill``), pure-torch reference
:func:`copy_stats_prefill_ref` (same signature).  The ring pointer ``p0`` and the row counts are read on the device
(CUDA-graph capture safe).  Tests: the per-operator tolerance rule of :mod:`neuronspark.utils.testing` against the
torch chain, on the finite statistics and on the confidence ``pi`` the hippocampal layers form from them.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from neuronspark.ops.triton import require_cuda

__all__ = [
    "COPY_BLOCK_T",
    "COPY_BLOCK_W",
    "COPY_WARPS",
    "COPY_STAGES",
    "COPY_DOT_PRECISION",
    "COPY_LOGIT_SCALE",
    "COPY_LAG",
    "copy_stats_prefill",
    "copy_stats_prefill_ref",
]

# ================================================================================================ fixed constants
#: queries per program, ring slots per tile, warps, pipeline stages
COPY_BLOCK_T = 32
COPY_BLOCK_W = 128
COPY_WARPS = 4
COPY_STAGES = 2
#: input precision of the score dot
COPY_DOT_PRECISION = "tf32x3"
#: logit scale of the scores
COPY_LOGIT_SCALE = 32.0
#: valid slots hold events at s <= t - COPY_LAG
COPY_LAG = 2


# ================================================================================================ kernel
@triton.jit
def _copy_stats_kernel(RING, MAXO, LSEO, P0, NPOS, T, W, B, s_rb, s_rw,
                       D: tl.constexpr, DC: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_W: tl.constexpr,
                       PRECISION: tl.constexpr, SCALE: tl.constexpr, LAG: tl.constexpr):
    """One program = BLOCK_T queries of row b; the ring in BLOCK_W tiles, the d axis in D // DC chunks of DC."""
    b = tl.program_id(0)
    pt = tl.program_id(1)
    p0 = tl.load(P0)                                                # int64: the ring pointer at the call's start
    rows = pt * BLOCK_T + tl.arange(0, BLOCK_T)
    rmask = rows < T
    dc = tl.arange(0, DC)
    slots = (p0 + rows) % W                                         # p0 >= 0: non-negative
    base = RING + b.to(tl.int64) * s_rb
    t_abs = p0 + rows                                               # the row position of each query
    t_hi = p0 + tl.minimum(pt * BLOCK_T + BLOCK_T, T) - 1           # the block's last query position
    end1 = p0 + T - 1                                               # the last written position
    start = p0 - tl.load(NPOS + b).to(tl.int64)                     # the row's first event
    m = tl.full((BLOCK_T,), float("-inf"), tl.float32)
    acc = tl.zeros((BLOCK_T,), tl.float32)
    for w0 in range(0, W, BLOCK_W):
        ww = w0 + tl.arange(0, BLOCK_W)
        wmask = ww < W
        r = (end1 - ww) % W
        r = tl.where(r < 0, r + W, r)                               # the non-negative remainder
        pos = end1 - r                                              # the position held by slot w (< 0: never written)
        lo = tl.min(tl.where(wmask & (pos >= start), pos, end1 + 1), axis=0)
        if lo <= t_hi - LAG:
            s = tl.zeros((BLOCK_T, BLOCK_W), tl.float32)
            for c in tl.static_range(D // DC):
                q = tl.load(base + slots[:, None] * s_rw + (c * DC + dc)[None, :], mask=rmask[:, None], other=0.0)
                k = tl.load(base + ww[:, None] * s_rw + (c * DC + dc)[None, :], mask=wmask[:, None], other=0.0)
                s = tl.dot(q, tl.trans(k), s, input_precision=PRECISION)        # (BLOCK_T, BLOCK_W) fp32
            valid = (wmask & (pos >= start))[None, :] & (pos[None, :] <= t_abs[:, None] - LAG)
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1))
            live = m_new > float("-inf")
            alpha = tl.where(live, tl.exp(SCALE * (m - m_new)), 0.0)
            pexp = tl.where(live[:, None], tl.exp(SCALE * (s - m_new[:, None])), 0.0)
            acc = acc * alpha + tl.sum(pexp, axis=1)
            m = m_new
    live = m > float("-inf")
    lse = tl.where(live, SCALE * m + tl.log(acc), float("-inf"))
    out = rows.to(tl.int64) * B + b
    tl.store(MAXO + out, m, mask=rmask)
    tl.store(LSEO + out, lse, mask=rmask)


# ================================================================================================ launcher
def _check(ring: torch.Tensor, ring_ptr: torch.Tensor, pos: torch.Tensor, T: int) -> Tuple[int, int, int]:
    for name, t in (("ring", ring), ("ring_ptr", ring_ptr), ("pos", pos)):
        require_cuda(t, f"copy_stats_prefill ({name})")
    if ring.dtype != torch.float32 or ring.dim() != 3 or ring.stride(-1) != 1:
        raise RuntimeError(f"neuronspark.copy_stats_prefill: the ring must be fp32 (B, W, d) with a unit inner stride, "
                           f"got {ring.dtype} {tuple(ring.shape)} strides {tuple(ring.stride())}")
    B, W, D = (int(s) for s in ring.shape)
    if ring_ptr.dtype != torch.int64 or ring_ptr.numel() != 1:
        raise RuntimeError(f"neuronspark.copy_stats_prefill: ring_ptr must be a (1,) int64 tensor, got "
                           f"{ring_ptr.dtype} {tuple(ring_ptr.shape)}")
    if pos.dtype != torch.int64 or tuple(pos.shape) != (B,) or not pos.is_contiguous():
        raise RuntimeError(f"neuronspark.copy_stats_prefill: pos must be a contiguous ({B},) int64 tensor, got "
                           f"{pos.dtype} {tuple(pos.shape)}")
    if not 1 <= int(T) <= W:
        raise RuntimeError(f"neuronspark.copy_stats_prefill: a call of {T} tokens for a ring of {W} slots")
    if D % 16:
        raise RuntimeError(f"neuronspark.copy_stats_prefill: the operand width {D} is not a multiple of 16")
    return B, W, D


def _copy_stats_prefill_fwd(ring: torch.Tensor, ring_ptr: torch.Tensor, pos: torch.Tensor,
                            T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, W, D = _check(ring, ring_ptr, pos, T)
    T = int(T)
    maximum = torch.empty((T, B), dtype=torch.float32, device=ring.device)
    lse = torch.empty((T, B), dtype=torch.float32, device=ring.device)
    dc = 32 if D % 32 == 0 else 16
    grid = (B, triton.cdiv(T, COPY_BLOCK_T))
    _copy_stats_kernel[grid](ring, maximum, lse, ring_ptr.reshape(1), pos, T, W, B, ring.stride(0), ring.stride(1),
                             D=D, DC=dc, BLOCK_T=COPY_BLOCK_T, BLOCK_W=COPY_BLOCK_W, PRECISION=COPY_DOT_PRECISION,
                             SCALE=COPY_LOGIT_SCALE, LAG=COPY_LAG, num_warps=COPY_WARPS, num_stages=COPY_STAGES)
    return maximum, lse


@torch.library.custom_op("neuronspark::copy_stats_prefill", mutates_args=())
def _copy_stats_prefill_op(ring: torch.Tensor, ring_ptr: torch.Tensor, pos: torch.Tensor,
                           T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return _copy_stats_prefill_fwd(ring, ring_ptr, pos, T)


@_copy_stats_prefill_op.register_fake
def _copy_stats_prefill_fake(ring, ring_ptr, pos, T):
    B = int(ring.shape[0])
    return (ring.new_empty((int(T), B), dtype=torch.float32), ring.new_empty((int(T), B), dtype=torch.float32))


def copy_stats_prefill(ring: torch.Tensor, ring_ptr: torch.Tensor, pos: torch.Tensor,
                       T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(maximum, lse)``, each (T, B) fp32 contiguous, of the call's T queries against the copy ring.

    ``ring`` (B, W, d) fp32 with a unit inner stride: the unit operands, the call's own already written at
    ``(ring_ptr + t) % W``; ``ring_ptr`` (1,) int64: the ring pointer at the start of the call; ``pos`` (B,) int64
    contiguous: the tokens each row consumed before the call (a row's events start at ``ring_ptr - pos``).  A query
    without a valid slot has ``maximum = lse = -inf``."""
    return torch.ops.neuronspark.copy_stats_prefill(ring, ring_ptr, pos, int(T))


# ================================================================================================ reference
@torch.no_grad()
def copy_stats_prefill_ref(ring: torch.Tensor, ring_ptr: torch.Tensor, pos: torch.Tensor,
                           T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The torch chain of the session detector (same arguments and outputs as :func:`copy_stats_prefill`): the
    call's operands gathered from the ring, the (B, T, W) fp32 einsum of scores (IEEE fp32 under the inference
    regime, autocast disabled), the strict-past mask from the row's first event, the max, and the logsumexp of the
    scaled scores padded with -inf to a multiple of 4 columns."""
    B, W, _ = (int(s) for s in ring.shape)
    T = int(T)
    dev = ring.device
    with torch.autocast(dev.type, enabled=False):
        steps = torch.arange(T, device=dev, dtype=torch.int64)
        slots = (ring_ptr.reshape(1) + steps) % W
        uq = ring.index_select(1, slots)                                  # (B, T, d): the call's unit operands
        end1 = ring_ptr.reshape(1) + (T - 1)
        w = torch.arange(W, device=dev, dtype=torch.int64)
        p = (end1 - torch.remainder(end1 - w, W)).view(1, 1, W)
        t_abs = (ring_ptr.reshape(1) + steps).view(1, T, 1)
        start = (ring_ptr.reshape(1) - pos.to(torch.int64)).view(B, 1, 1)
        valid = (p >= start) & (p <= t_abs - COPY_LAG)                    # (B, T, W)
        scores = torch.einsum("btd,bwd->btw", uq, ring)
        scores = scores.masked_fill(~valid, float("-inf"))
        maximum = scores.amax(-1)
        logits = COPY_LOGIT_SCALE * scores
        padding = (-W) % 4
        if padding:
            logits = F.pad(logits, (0, padding), value=float("-inf"))
        lse = torch.logsumexp(logits, dim=-1)
        return maximum.transpose(0, 1).contiguous(), lse.transpose(0, 1).contiguous()
