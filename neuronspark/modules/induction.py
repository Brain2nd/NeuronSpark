"""Induction match matrices of the hippocampal layers' pool attention (model level, pure torch, bool).

Whole row (:func:`induction_match_matrices`): one (B, T, T) bool matrix per order n in ``orders`` (default
configuration: (2, 3, 4)),
ascending::

    m_n[b, t, j] = 1[j <= t - 1] * prod_{i = 0 .. n-1} 1[x_{t-i} == x_{j-1-i}]

with every compared index in range (an index before the row start never matches).  The matrices are nested
(m_4 is a subset of m_3, of m_2). Each hippocampal layer forms its additive pool term ``sum_n beta[n, h] m_n`` from them
with its own ``pool_ind_beta``.

Inference (:func:`induction_session_match`): the cache keeps a keys ring
(:class:`~neuronspark.utils.cache.InductionState`): slot ``w`` holds the ``n_max`` ids preceding the event written
there, ``keys[w, i] = x_{p(w)-1-i}``, and ``win`` the last ``n_max`` ids of the row (pad id ``V`` at the row start).  A
call on T ids writes its keys at slots ``(ring_ptr + t) % W`` first, then matches every query of the call against
every slot::

    m_n[b, t, w] = past(t, w) * prod_{i < n} 1[keys[w, i] == x_{t-i}] * 1[x_{t-i} != V]

where ``past`` keeps the slots whose event lies at a row position <= t - 1.  At written slots this is the whole-row
matrix (the same integer comparisons): with the ring started at slot 0, slot ``w`` of a row of length <= W is position
``w``.

The ring helpers :func:`ring_positions` / :func:`ring_strict_past` (device arithmetic only, CUDA-graph safe) are
shared with the copy detector.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch

from neuronspark.utils.cache import InductionState

__all__ = [
    "induction_match_matrices",
    "induction_session_match",
    "ring_positions",
    "ring_strict_past",
    "allocate_induction_state",
    "INDUCTION_ORDERS",
]

INDUCTION_ORDERS = (2, 3, 4)


def _orders(orders: Sequence[int]) -> Tuple[int, ...]:
    out = tuple(int(n) for n in orders)
    if not out or list(out) != sorted(set(out)) or out[0] < 1:
        raise ValueError(f"induction: orders must be strictly increasing positive integers, got {orders}")
    return out


def induction_match_matrices(ids: torch.Tensor, orders: Sequence[int] = INDUCTION_ORDERS,
                             pad: int = 151669) -> Tuple[torch.Tensor, ...]:
    """Whole row: ``ids`` (B, T) int -> one (B, T, T) bool match matrix per order (ascending); ``pad`` = the id
    outside the vocabulary (vocab_size).  Static shapes, no host sync."""
    x = ids
    if x.dim() != 2:
        raise ValueError(f"induction_match_matrices: ids must be (B, T), got {tuple(x.shape)}")
    B, T = (int(s) for s in x.shape)
    orders = _orders(orders)
    n_max = max(orders)
    past = torch.ones(T, T, device=x.device, dtype=torch.bool).tril(-1)                   # [t, j]: j <= t - 1
    shifted = [x]                                                                          # shifted[k][b, j] = x_{j-k}
    for k in range(1, n_max + 1):
        # pad then slice: width exactly T for any T >= 1 (T <= k: all pad, nothing matches)
        shifted.append(torch.cat([x.new_full((B, k), int(pad)), x], dim=1)[:, :T])
    _pos = torch.arange(T, device=x.device)

    def pos_ok(k: int) -> torch.Tensor:                                                    # (B, T): j - k in range
        return (_pos >= k).unsqueeze(0).expand(B, T)

    acc = past.unsqueeze(0).expand(B, T, T)
    out = []
    for n in range(1, n_max + 1):
        acc = acc & (shifted[n - 1].unsqueeze(2) == shifted[n].unsqueeze(1))              # x_{t-(n-1)} == x_{j-n}
        if n in orders:
            out.append(acc & pos_ok(n - 1).unsqueeze(2) & pos_ok(n).unsqueeze(1))
    return tuple(out)


def ring_positions(ptr0: torch.Tensor, T: int, W: int) -> torch.Tensor:
    """(W,) int64: the row position of the event held by every slot after a call of ``T`` tokens that started at the
    pointer ``ptr0`` ((1,) int64 device tensor); negative = never written."""
    end1 = ptr0 + (int(T) - 1)
    w = torch.arange(int(W), device=ptr0.device, dtype=torch.int64)
    return end1 - torch.remainder(end1 - w, int(W))


def ring_strict_past(ptr0: torch.Tensor, T: int, W: int, lag: int) -> torch.Tensor:
    """(T, W) bool: slot ``w`` holds an event at a row position <= t - ``lag`` of query tau (t = ptr0 + tau)."""
    pos = ring_positions(ptr0, T, W).view(1, int(W))
    t_abs = ptr0 + torch.arange(int(T), device=ptr0.device, dtype=torch.int64)
    return (pos >= 0) & (pos <= t_abs.view(int(T), 1) - int(lag))


def induction_session_match(ids: torch.Tensor, state: InductionState, ring_ptr: torch.Tensor, W: int,
                            orders: Sequence[int] = INDUCTION_ORDERS) -> Tuple[torch.Tensor, ...]:
    """Inference call: ``ids`` (B, T) -> one (B, T, W) bool match per order (ascending), the call's queries against
    every slot of the keys ring after this call's keys were written at ``(ring_ptr + t) % W``; ``state.keys`` /
    ``state.win`` are updated in place.  ``ring_ptr`` (1,) int64: the ring pointer at the start of the call."""
    keys, win = state.keys, state.win
    B, Wr, n = (int(s) for s in keys.shape)
    if Wr != int(W):
        raise RuntimeError(f"induction_session_match: the keys ring has {Wr} slots, the call says W = {W}")
    orders = _orders(orders)
    if n != max(orders):
        raise RuntimeError(f"induction_session_match: the keys ring holds {n} ids per slot, the orders need "
                           f"{max(orders)}")
    x = ids.to(torch.int64)
    if x.dim() != 2 or int(x.shape[0]) != B:
        raise RuntimeError(f"induction_session_match: token ids {tuple(x.shape)} for a keys ring of {B} rows")
    T = int(x.shape[1])
    if T > Wr:
        raise RuntimeError(f"induction_session_match: a call of {T} tokens exceeds the keys ring ({Wr} slots)")
    if tuple(ring_ptr.shape) != (1,) or ring_ptr.dtype != torch.int64:
        raise RuntimeError("induction_session_match: ring_ptr must be a (1,) int64 device tensor")
    pad = int(state.pad_id)
    ext = torch.cat([win, x], dim=1)                                      # (B, n + T): ext[:, n + tau] = x_{p0 + tau}
    keys_new = torch.stack([ext[:, n - 1 - i:n - 1 - i + T] for i in range(n)], dim=-1)   # (B, T, n): x_{t-1-i}
    qwin = torch.stack([ext[:, n - i:n - i + T] for i in range(n)], dim=-1)              # (B, T, n): x_{t-i}
    slots = (ring_ptr + torch.arange(T, device=x.device, dtype=torch.int64)) % Wr
    keys.index_copy_(1, slots, keys_new)
    win.copy_(ext[:, T:])
    past = ring_strict_past(ring_ptr, T, Wr, 1)                            # (T, W): the slot's event at p <= t - 1
    eq = (keys.unsqueeze(1) == qwin.unsqueeze(2)) & (qwin != pad).unsqueeze(2)          # (B, T, W, n)
    acc = past.unsqueeze(0).expand(B, T, Wr)
    out = []
    for k in range(1, n + 1):
        acc = acc & eq[..., k - 1]
        if k in orders:
            out.append(acc)
    return tuple(out)


def allocate_induction_state(batch_size: int, max_seqlen: int, orders: Sequence[int] = INDUCTION_ORDERS,
                             pad: int = 151669, device=None) -> InductionState:
    """The row-start keys ring (B, W, n_max) and window (B, n_max), filled with the pad id."""
    return InductionState.allocate(int(batch_size), int(max_seqlen), max(_orders(orders)), int(pad),
                                   torch.device("cpu") if device is None else torch.device(device))
