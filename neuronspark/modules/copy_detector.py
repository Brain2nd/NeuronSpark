"""Copy-detector statistics (model level) and the per-layer copy confidence ``pi`` (no gradient).

The detector asks, for every position t, how strongly the current token is a repeat whose continuation the row has
already seen.  Its operand is the order-1 slice of the model-level lexical vector, ``u_t = normalize(lex_t[0:96])``
(the exact-id table row of x_t, detached), so equal tokens have cosine 1 whatever the stream holds.  The source
positions are s <= t - 2 (source s proposes the continuation after x_{s-1})::

    score[t, s] = u_t . u_s                     (-inf outside s <= t - 2)
    maximum_t   = max_s score[t, s]
    lse_t       = logsumexp_s(32 score[t, s])   (the logits padded with -inf to a multiple of 4)
    pi_t        = exp(32 maximum_t - logaddexp(32 tau, lse_t))           tau = the layer's pkm_copy_tau

``pi`` is 0 at t < 2.  The operand is the same for the six hippocampal layers, so the statistics ``(maximum, lse)`` are
computed once per call (:func:`copy_stats_whole_row` / :func:`copy_stats_session`) and every layer applies
:func:`copy_pi` with its own calibrated ``tau``: the same operations on the same operands as a per-layer detector.

Numerics per regime: the score matrix is an fp32 einsum, so it runs in TF32 under the training regime
(``torch.backends.cuda.matmul.allow_tf32 = True``) and in IEEE fp32 under the inference regime; autocast is disabled
inside.  Inference keeps one ring of unit operands (:class:`~neuronspark.utils.cache.CopyState` ``u`` (B, W, 96)): a
call writes its operands at slots ``(ring_ptr + t) % W`` and scores its queries against every slot whose event lies at
a row position in ``[start_b, t - 2]`` (``start_b = ring_ptr - pos_b``, the row's first event: 0 for every row of a
cache started with the batch).  A prefill call (T >= 2) reduces the scores with the operator ``copy_stats_prefill``
(:mod:`neuronspark.ops.triton.copy_stats`: one kernel, online max / logsumexp over ring tiles, the score dot at
``tf32x3`` precision; ``impl="ref"`` takes its torch chain); a decode step (T == 1) runs the torch chain below.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple, Union

import torch
import torch.nn.functional as F

from neuronspark.modules.induction import ring_positions
from neuronspark.ops import namespace
from neuronspark.utils.cache import CopyState

__all__ = [
    "CopyStats",
    "copy_stats_whole_row",
    "copy_stats_session",
    "copy_pi",
    "allocate_copy_state",
    "COPY_LOGIT_SCALE",
    "COPY_LAG",
    "COPY_ORDER1_SLICE",
]

COPY_LOGIT_SCALE = 32.0          # logit scale of the source scores and of the sink 32 tau
COPY_LAG = 2                     # sources s <= t - 2
COPY_ORDER1_SLICE = (0, 96)      # the order-1 channels of the 1.27B lexical vector


class CopyStats(NamedTuple):
    """Per-position detector statistics of a call, each (T, B) fp32 contiguous."""

    maximum: torch.Tensor        # max_s score[t, s] (-inf where no source)
    lse: torch.Tensor            # logsumexp_s(32 score[t, s])


def _operand(lex: torch.Tensor, order1: Tuple[int, int]) -> torch.Tensor:
    if lex.dim() != 3:
        raise ValueError(f"copy detector: expected the lexical vector (T, B, M), got {tuple(lex.shape)}")
    s0, s1 = (int(v) for v in order1)
    if not 0 <= s0 < s1 <= int(lex.shape[-1]):
        raise ValueError(f"copy detector: order-1 slice {order1} outside the lexical width {int(lex.shape[-1])}")
    return lex.detach()[..., s0:s1]


def _stats(scores: torch.Tensor, valid: torch.Tensor, n_cols: int) -> CopyStats:
    """(B, T, S) scores masked by ``valid`` -> CopyStats (T, B)."""
    scores = scores.masked_fill(~valid, float("-inf"))
    maximum = scores.amax(-1)
    logits = COPY_LOGIT_SCALE * scores
    # keep the fp32 reduction rows 16-byte aligned (padding adds no probability mass)
    padding = (-int(n_cols)) % 4
    if padding:
        logits = F.pad(logits, (0, padding), value=float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    return CopyStats(maximum.transpose(0, 1).contiguous(), lse.transpose(0, 1).contiguous())


@torch.no_grad()
def copy_stats_whole_row(lex: torch.Tensor, order1: Tuple[int, int] = COPY_ORDER1_SLICE) -> CopyStats:
    """Whole row: ``lex`` (T, B, M) fp32 lexical vector -> :class:`CopyStats` of every position (sources within
    the row)."""
    z = _operand(lex, order1)
    with torch.autocast(z.device.type, enabled=False):
        u = F.normalize(z.float(), dim=-1)
        T = int(u.shape[0])
        scores = torch.einsum("tbd,sbd->bts", u, u)
        valid = torch.ones(T, T, device=z.device, dtype=torch.bool).tril(-COPY_LAG).unsqueeze(0)
        return _stats(scores, valid, T)


@torch.no_grad()
def copy_stats_session(lex_t: torch.Tensor, state: CopyState, ring_ptr: torch.Tensor, pos: torch.Tensor, W: int,
                       order1: Tuple[int, int] = COPY_ORDER1_SLICE, ops=None) -> CopyStats:
    """Inference call: ``lex_t`` (T, B, M) lexical vector of the call -> :class:`CopyStats` of the call's positions;
    the call's unit operands are written into ``state.u`` (B, W, d) at ``(ring_ptr + t) % W`` first.  ``ring_ptr``
    (1,) int64 and ``pos`` (B,) int64 are the counters at the start of the call (tokens consumed before it).  ``ops``:
    the operator namespace of the model (:func:`neuronspark.ops.namespace`; None = the kernels), whose
    ``copy_stats_prefill`` reduces the scores of a call with T >= 2."""
    ring = state.u
    B, Wr, d = (int(s) for s in ring.shape)
    if Wr != int(W):
        raise RuntimeError(f"copy_stats_session: the ring has {Wr} slots, the call says W = {W}")
    z = _operand(lex_t, order1)
    T = int(z.shape[0])
    if tuple(z.shape[1:]) != (B, d):
        raise RuntimeError(f"copy_stats_session: operand {tuple(z.shape)} for a ring ({B}, {Wr}, {d})")
    if T > Wr:
        raise RuntimeError(f"copy_stats_session: a call of {T} tokens exceeds the ring ({Wr} slots)")
    if tuple(ring_ptr.shape) != (1,) or ring_ptr.dtype != torch.int64:
        raise RuntimeError("copy_stats_session: ring_ptr must be a (1,) int64 device tensor")
    if tuple(pos.shape) != (B,):
        raise RuntimeError(f"copy_stats_session: pos must be ({B},), got {tuple(pos.shape)}")
    with torch.autocast(z.device.type, enabled=False):
        u = F.normalize(z.float(), dim=-1)                                # the whole row's normalisation, per row
        steps = torch.arange(T, device=z.device, dtype=torch.int64)
        slots = (ring_ptr + steps) % Wr
        ring.index_copy_(1, slots, u.transpose(0, 1))
        if T > 1:
            # prefill: the scores and their reduction in one operator (online max / logsumexp over ring tiles)
            o = namespace("kernel") if ops is None else ops
            maximum, lse = o.copy_stats_prefill(ring, ring_ptr, pos.to(torch.int64).contiguous(), T)
            return CopyStats(maximum, lse)
        uq = ring.index_select(1, slots)                                  # (B, T, d): the call's unit operands
        p = ring_positions(ring_ptr, T, Wr).view(1, 1, Wr)
        t_abs = (ring_ptr + steps).view(1, T, 1)
        start = (ring_ptr - pos.to(torch.int64)).view(B, 1, 1)
        valid = (p >= start) & (p <= t_abs - COPY_LAG)                    # (B, T, W)
        scores = torch.einsum("btd,bwd->btw", uq, ring)
        return _stats(scores, valid, Wr)


@torch.no_grad()
def copy_pi(stats: CopyStats, tau: Union[torch.Tensor, float]) -> torch.Tensor:
    """Copy confidence ``exp(32 maximum - logaddexp(32 tau, lse))`` (T, B) fp32 of one hippocampal layer."""
    lse = stats.lse
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(float(tau), dtype=torch.float32, device=lse.device)
    with torch.autocast(lse.device.type, enabled=False):
        sink = torch.ones_like(lse) * (COPY_LOGIT_SCALE * tau.float())
        return torch.exp(COPY_LOGIT_SCALE * stats.maximum - torch.logaddexp(sink, lse))


def allocate_copy_state(batch_size: int, max_seqlen: int, dim: int = COPY_ORDER1_SLICE[1] - COPY_ORDER1_SLICE[0],
                        device=None) -> CopyState:
    """The row-start ring of unit operands (B, W, dim), zero."""
    return CopyState.allocate(int(batch_size), int(max_seqlen), int(dim),
                              torch.device("cpu") if device is None else torch.device(device))
