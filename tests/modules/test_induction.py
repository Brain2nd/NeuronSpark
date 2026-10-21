"""Induction matches: the whole-row id windows (as match matrices) and the session match against the keys ring.

* the matrices of the whole-row windows against a direct transcription of the definition (also on rows of 1 .. 5
  tokens);
* the session form (calls of several lengths from the row start) against the whole-row matrices at the written slots;
* the ring helpers (slot positions, strict-past mask).

The outputs are bool; they are compared with the per-operator rule of ``neuronspark.utils.testing`` (a flipped
element differs by 1, which fails the rule).
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.models.config_neuronspark import NeuronSparkConfig
from neuronspark.modules.induction import (INDUCTION_ORDERS, induction_session_match, induction_token_windows,
                                           induction_window_matches, ring_positions, ring_strict_past)
from neuronspark.utils.cache import InductionState
from neuronspark.utils.testing import assert_all_match, assert_match

pytestmark = pytest.mark.cuda

CFG = NeuronSparkConfig()
V = CFG.vocab_size
ORDERS = tuple(CFG.hippocampal_induction_bias_orders)
CHUNKS = (37, 1, 1, 128, 1, 64, 3)                      # 235 tokens: fits a 512-slot ring


def _ids(B: int, T: int, device, vocab: int = 6) -> torch.Tensor:
    """Ids from a small set (n-grams recur often)."""
    g = torch.Generator(device=device).manual_seed(5)
    return torch.randint(0, vocab, (B, T), generator=g, device=device)


def _whole(ids: torch.Tensor):
    """The whole-row matrices: the matches of the id windows."""
    return induction_window_matches(*induction_token_windows(ids, ORDERS, V), ORDERS)


def _definition(ids: torch.Tensor, orders, pad: int):
    """m_n[b, t, j] = [j <= t - 1] prod_{i < n} [x_{t-i} == x_{j-1-i}], indices in range (Python loops)."""
    x = ids.cpu().tolist()
    B, T = len(x), len(x[0])
    out = []
    for n in orders:
        m = torch.zeros((B, T, T), dtype=torch.bool)
        for b in range(B):
            for t in range(T):
                for j in range(t):
                    ok = True
                    for i in range(n):
                        if t - i < 0 or j - 1 - i < 0 or x[b][t - i] != x[b][j - 1 - i]:
                            ok = False
                            break
                    m[b, t, j] = ok
        out.append(m.to(ids.device))
    return tuple(out)


def _session(ids: torch.Tensor, chunks, W: int, device) -> dict:
    B = int(ids.shape[0])
    st = InductionState.allocate(B, W, max(ORDERS), V, device)
    ptr = torch.zeros((1,), dtype=torch.int64, device=device)
    out, off = {}, 0
    for i, T in enumerate(chunks):
        ms = induction_session_match(ids[:, off:off + T], st, ptr, W, ORDERS)
        for k, m in zip(ORDERS, ms):
            out[f"c{i}.m{k}"] = m
        ptr += T
        off += T
    return out


def test_whole_row_definition(device):
    ids = _ids(2, 40, device)
    new = _whole(ids)
    ref = _definition(ids, ORDERS, V)
    assert_all_match(list(new), list(ref), prefix="induction_definition")
    m2, m3, m4 = new
    assert not bool((m3 & ~m2).any()) and not bool((m4 & ~m3).any())      # nested orders


def test_whole_row_short_rows(device):
    for T in (1, 2, 3, 4, 5):
        ids = _ids(1, T, device)
        new = _whole(ids)
        assert all(tuple(m.shape) == (1, T, T) and m.dtype == torch.bool for m in new)
        assert_all_match(list(new), list(_definition(ids, ORDERS, V)), prefix=f"induction_T{T}")


def test_session_vs_whole_row(device):
    W = 512
    B = 2
    L = sum(CHUNKS)
    ids = _ids(B, L, device)
    whole = _whole(ids)
    new = _session(ids, CHUNKS, W, device)
    ref, off = {}, 0
    for i, T in enumerate(CHUNKS):
        for k, m in zip(ORDERS, whole):
            e = torch.zeros((B, T, W), dtype=torch.bool, device=device)
            e[..., :L] = m[:, off:off + T, :L]            # slot w holds row position w (ring started at slot 0)
            ref[f"c{i}.m{k}"] = e
        off += T
    assert_all_match(new, ref, prefix="induction_session_vs_whole_row")


def test_ring_helpers(device):
    W = 8
    ptr = torch.tensor([10], dtype=torch.int64, device=device)
    pos = ring_positions(ptr, 3, W)                       # after positions 10, 11, 12 were written at slots 2, 3, 4
    want = torch.tensor([8, 9, 10, 11, 12, 5, 6, 7], dtype=torch.int64, device=device)
    assert_match(pos.float(), want.float(), "ring_positions")
    past = ring_strict_past(ptr, 3, W, 1)
    assert_match(past, (want.view(1, W) <= (ptr + torch.arange(3, device=device)).view(3, 1) - 1) & (want >= 0),
                 "ring_strict_past")
    assert INDUCTION_ORDERS == ORDERS
