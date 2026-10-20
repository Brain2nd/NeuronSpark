"""Copy-detector statistics (model level) and the per-layer copy confidence ``pi``.

* inference session (IEEE fp32): ``copy_pi(copy_stats_session(...), tau)`` over calls of several lengths from the row
  start (prefill chunks and decode steps) against ``copy_pi(copy_stats_whole_row(lex), tau)`` at the same positions
  under the same regime;
* ``pi`` from the whole-row statistics (training regime): zero before the second token (no source position), in [0, 1].

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.models.config_neuronspark import NeuronSparkConfig
from neuronspark.modules.copy_detector import CopyStats, copy_pi, copy_stats_session, copy_stats_whole_row
from neuronspark.utils.cache import CopyState
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match

pytestmark = pytest.mark.cuda

CFG = NeuronSparkConfig()
M = CFG.lexical_dim
CHUNKS = (37, 1, 1, 128, 1, 64, 3)


def _lex(T: int, B: int, device) -> torch.Tensor:
    """A lexical vector (T, B, M) whose order-1 slice repeats exactly for repeated ids (cosine 1) and is perturbed at
    half of the positions (graded cosines)."""
    g = torch.Generator(device=device).manual_seed(9)
    table = torch.randn((12, M), generator=g, device=device)
    ids = torch.randint(0, 12, (T, B), generator=g, device=device)
    lex = table[ids]
    noisy = (torch.rand((T, B, 1), generator=g, device=device) < 0.5).float()
    return lex + noisy * 0.3 * torch.randn((T, B, M), generator=g, device=device)


def _session(lex: torch.Tensor, chunks, W: int, tau: float, device) -> dict:
    T_all, B, _ = (int(s) for s in lex.shape)
    st = CopyState.allocate(B, W, 96, device)
    ptr = torch.zeros((1,), dtype=torch.int64, device=device)
    pos = torch.zeros((B,), dtype=torch.int64, device=device)
    out, off = {}, 0
    with torch.no_grad(), regime("infer"):
        for i, T in enumerate(chunks):
            stats = copy_stats_session(lex[off:off + T], st, ptr, pos, W)
            out[f"pi{i}"] = copy_pi(stats, torch.tensor(tau, dtype=torch.float32, device=device))
            ptr += T
            pos += T
            off += T
    return out


def test_session_vs_whole_row(device):
    W, B, tau = 512, 3, 1.0
    lex = _lex(sum(CHUNKS), B, device)
    new = _session(lex, CHUNKS, W, tau, device)
    with torch.no_grad(), regime("infer"):
        whole = copy_pi(copy_stats_whole_row(lex), tau)
    ref, off = {}, 0
    for i, T in enumerate(CHUNKS):
        ref[f"pi{i}"] = whole[off:off + T]
        off += T
    assert_all_match(new, ref, prefix="copy_session_vs_whole_row")


def test_pi_properties(device):
    lex = _lex(64, 2, device)
    with torch.no_grad(), regime("train"):
        stats = copy_stats_whole_row(lex)
        pi = copy_pi(stats, 1.0)
    assert isinstance(stats, CopyStats)
    assert_match(pi[:2], torch.zeros_like(pi[:2]), "pi_before_two_tokens")      # no source position at t < 2
    assert bool(((pi >= 0) & (pi <= 1)).all())
