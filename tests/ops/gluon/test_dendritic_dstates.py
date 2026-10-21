"""ops.gluon.dendritic_dstates: the warp-specialised Gluon state gradients (dstates GEMM + state-passing backward) of
the dendritic-layer training scan.

* ``dstates_train`` on the forward's fp32 passed states (``chunk_states_train(.., passed_fp32=True)``) against the
  float64 reference, on the model's operands (B 8, T 1024, H 144, P 64, N 128, chunk 256), smaller shapes, the stress
  decays and partial last chunks;
* the passed-state output of ``chunk_states_train`` against its reference.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.gluon.dendritic_chunk_states import chunk_states_train
from neuronspark.ops.gluon.dendritic_dstates import dstates_train, dstates_train_ref
from neuronspark.ops.triton.dendritic_chunk import chunk_dt_cumsum
from neuronspark.ops.triton.dendritic_states import csp_fused_ref
from neuronspark.utils.testing import assert_all_match, assert_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
SHAPES = [(8, 1024), (2, 512), (3, 768), (1, 300), (2, 600)]


def operands(B, L, device, stress=False):
    xbc = rand_tensor((B, L, DI + 2 * N), torch.bfloat16, device)
    if stress:
        dt_raw = rand_tensor((B, L, H), torch.float32, device).to(torch.bfloat16)
        A = -torch.exp(rand_tensor((H,), torch.float32, device) * 0.5 + 2.5)
    else:
        dt_raw = (rand_tensor((B, L, H), torch.float32, device) * 0.5 - 2.0).to(torch.bfloat16)
        A = -torch.exp(rand_tensor((H,), torch.float32, device) * 0.5)
    dt_bias = rand_tensor((H,), torch.float32, device) * 0.5
    x = xbc[..., :DI].unflatten(-1, (H, P))
    Bm = xbc[..., DI:DI + N].unflatten(-1, (1, N))
    Cm = xbc[..., DI + N:].unflatten(-1, (1, N))
    dA, dt = chunk_dt_cumsum(dt_raw, A, CS, dt_bias=dt_bias)
    dout = rand_tensor((B, L, H, P), torch.bfloat16, device)
    return x, Bm, Cm, dt, dA, dout


@pytest.mark.parametrize("stress", [False, True], ids=["main", "stress"])
@pytest.mark.parametrize("B,L", SHAPES, ids=[f"B{b}_L{l}" for b, l in SHAPES])
def test_dstates_vs_reference(device, B, L, stress):
    x, Bm, Cm, dt, dA, dout = operands(B, L, device, stress)
    with torch.no_grad():
        passed, _ = chunk_states_train(Bm, x, dt, dA, torch.bfloat16, passed_fp32=True)
        dst, ws, nv, conv = dstates_train(passed, dA, dout, Cm, torch.bfloat16)
        rdst, rws, rnv, rconv = dstates_train_ref(passed, dA, dout, Cm, torch.bfloat16)
    assert nv == rnv
    new = {"dstates": dst, "ws": ws, "conv": conv}
    assert_all_match(new, {"dstates": rdst, "ws": rws, "conv": rconv}, prefix=f"vs_ref_B{B}_L{L}")


@pytest.mark.parametrize("B,L", [(8, 1024), (1, 300)], ids=["B8_L1024", "B1_L300"])
def test_passed_states_output(device, B, L):
    x, Bm, Cm, dt, dA, dout = operands(B, L, device)
    with torch.no_grad():
        passed, states = chunk_states_train(Bm, x, dt, dA, torch.float32, passed_fp32=True)
        _, rstates = csp_fused_ref(Bm, x, dt, dA, torch.float32)
    assert_match(passed, rstates, f"passed_fp32_B{B}_L{L}")
    assert torch.equal(passed, states)                                 # the same fp32 values in both outputs
