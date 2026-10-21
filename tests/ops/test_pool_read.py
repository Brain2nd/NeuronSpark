"""ops.pool_read: the fused pool read of a hippocampal layer (bias, induction matches and softmax attention
without T x T tensors) against its definition, :func:`pool_read_ref` (float64, the operator's forward rounding, exact
gradients): the output and the gradients of q, k, the tagged values, the clock and the beta, for pool heads alone and
with extra queries on the last head, at row lengths of whole and partial tiles; token ids from a small vocabulary so
that the induction matches are frequent.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuronspark.modules.induction import induction_token_windows
from neuronspark.ops.pool_read import pool_read, pool_read_ref
from neuronspark.utils.testing import assert_all_match

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

ORDERS = (2, 3, 4)
PAD = 1 << 20
SCALE = 16.0
# (T, B, H key heads, HQ query heads)
CASES = [(512, 2, 2, 2), (512, 2, 2, 6), (1000, 1, 1, 1), (776, 2, 3, 7), (300, 1, 8, 12), (4096, 1, 2, 3)]


def operands(T, B, H, HQ, device, dk=256, dv=256, vocab=64):
    q = F.normalize(torch.randn(T, B, HQ, dk, device=device), dim=-1) * 1.3
    k = F.normalize(torch.randn(T, B, H, dk, device=device), dim=-1)
    gv = torch.randn(T, B, H, dv, device=device).to(torch.bfloat16)
    cum = (-torch.rand(T, B, H, device=device) * 0.05).cumsum(0)
    beta = torch.rand(len(ORDERS), H, device=device) * 1.5
    ids = torch.randint(0, vocab, (B, T), device=device)
    go = torch.randn(T, B, HQ, dv, device=device).to(torch.bfloat16)
    return q, k, gv, cum, beta, ids, go


def _run(fn, o):
    q, k, gv, cum, beta, ids, go = o
    leaves = [x.clone().requires_grad_(True) for x in (q, k, gv, cum, beta)]
    out = fn(*leaves, *induction_token_windows(ids, ORDERS, PAD), ORDERS, SCALE)
    out.backward(go)
    return {"out": out.detach(), "dq": leaves[0].grad, "dk": leaves[1].grad, "dgv": leaves[2].grad,
            "dcum": leaves[3].grad, "dbeta": leaves[4].grad}


@pytest.mark.parametrize("T,B,H,HQ", CASES, ids=[f"T{t}_B{b}_H{h}_HQ{q}" for t, b, h, q in CASES])
def test_pool_read_vs_reference(device, T, B, H, HQ):
    o = operands(T, B, H, HQ, device)
    assert_all_match(_run(pool_read, o), _run(pool_read_ref, o), prefix=f"pool_read_T{T}_B{B}_H{H}_HQ{HQ}")


def test_pool_read_strided_operands(device):
    """Operands that are views of wider tensors (as projections hand them over) give the contiguous result."""
    o = operands(512, 2, 2, 6, device)
    q, k, gv, cum, beta, ids, go = o
    wide = lambda x: torch.cat([x, torch.zeros_like(x)], dim=-1)[..., :x.shape[-1]]   # noqa: E731
    strided = (wide(q), wide(k), wide(gv), torch.stack([cum, cum], dim=-1)[..., 0], beta, ids, go)
    assert not strided[0].is_contiguous() and not strided[3].is_contiguous()
    assert_all_match(_run(pool_read, strided), _run(pool_read, o), prefix="pool_read_strided")


def test_pool_read_in_cuda_graph(device):
    """Forward and backward captured in one CUDA graph (as the trainer runs it) replay the eager result."""
    q, k, gv, cum, beta, ids, go = operands(776, 2, 3, 7, device)
    eager = _run(pool_read, (q, k, gv, cum, beta, ids, go))
    qtok, ktok = induction_token_windows(ids, ORDERS, PAD)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, gv, cum, beta)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            pool_read(*leaves, qtok, ktok, ORDERS, SCALE).backward(go)
            for x in leaves:
                x.grad = None
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pool_read(*leaves, qtok, ktok, ORDERS, SCALE)
        out.backward(go)
    graph.replay()
    torch.cuda.synchronize()
    got = {"out": out.detach(), "dq": leaves[0].grad, "dk": leaves[1].grad, "dgv": leaves[2].grad,
           "dcum": leaves[3].grad, "dbeta": leaves[4].grad}
    for name in eager:
        assert torch.equal(got[name], eager[name]), name
