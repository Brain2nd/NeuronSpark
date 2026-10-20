"""ops.triton.dendritic_scan: the TMA chunk-scan forward (``chunk_scan_fwd_tma``, prefill) against its float64 reference
at B 1 x L 512 and a partial-chunk length (the forward at the prefill shapes is checked against this kernel in
``tests/ops/test_dendritic_prefill.py`` and ``tests/ops/gluon/test_dendritic_chunk_scan_fwd.py``; the backward's dda
pieces in ``tests/ops/gluon/test_dendritic_dda.py``).

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.mamba import BMM_CHUNK_FWD, CHUNK_CUMSUM_FWD_TRAIN, _bmm_chunk_fwd, _chunk_cumsum_fwd
from neuronspark.ops.triton.dendritic_scan import (
    chunk_scan_fwd_tma,
    chunk_scan_fwd_tma_ref,
)
from neuronspark.ops.gluon.dendritic_chunk_states import chunk_states_train
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
REF_SHAPES = [(1, 512), (1, 300)]


def operands(B, L, device):
    """conv-output views of one (B, L, DI + 2N) projection, dt / dA from the chunk cumsum, CB and the passed states,
    plus a dout (scale 1e-2) and dstates (scale 0.1)"""
    xbc = rand_tensor((B, L, DI + 2 * N), torch.bfloat16, device)
    dt_raw = (rand_tensor((B, L, H), torch.float32, device) * 0.5 - 2.0).to(torch.bfloat16)
    A = -torch.exp(rand_tensor((H,), torch.float32, device) * 0.5)
    dt_bias = rand_tensor((H,), torch.float32, device) * 0.5
    D = rand_tensor((H,), torch.float32, device) * 0.5
    x = xbc[..., :DI].unflatten(-1, (H, P))
    Bm = xbc[..., DI:DI + N].unflatten(-1, (1, N))
    Cm = xbc[..., DI + N:].unflatten(-1, (1, N))
    dA, dt = _chunk_cumsum_fwd(dt_raw, A, CS, dt_bias=dt_bias, dt_softplus=True, dt_limit=(0.0, float("inf")),
                               config=CHUNK_CUMSUM_FWD_TRAIN)
    CB = _bmm_chunk_fwd(Cm, Bm, CS, seq_idx=None, output_dtype=torch.float32, config=BMM_CHUNK_FWD)
    _, states = chunk_states_train(Bm, x, dt, dA, torch.bfloat16)
    nc = dA.shape[2]
    dout = (rand_tensor((B, L, H, P), torch.float32, device) * 1e-2).to(torch.bfloat16)
    dstates = (rand_tensor((B, nc, H, P, N), torch.float32, device) * 0.1).to(torch.bfloat16)
    return {"xbc": xbc, "x": x, "B": Bm, "C": Cm, "D": D, "dA": dA, "dt": dt, "CB": CB, "states": states,
            "dout": dout, "dstates": dstates}


@pytest.mark.parametrize("B,L", REF_SHAPES, ids=[f"B{b}_L{l}" for b, l in REF_SHAPES])
def test_fwd_vs_reference(device, B, L):
    o = operands(B, L, device)
    args = (o["CB"], o["x"], o["dt"], o["dA"], o["C"], o["states"], o["D"])
    with torch.no_grad():
        assert_all_match({"out": chunk_scan_fwd_tma(*args)}, {"out": chunk_scan_fwd_tma_ref(*args)}, prefix="fwd_ref")
