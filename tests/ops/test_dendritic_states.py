"""ops.triton.dendritic_states: the dB / dC pieces of the SSD backward (``db_tma`` / ``dc_tma``), on chunk states from
the Gluon forward (``ops.gluon.dendritic_chunk_states``).

Each against its float64 reference ``*_ref`` (after the head-split / tile reductions) at B 1 x L 512 and a
partial-chunk length (H 144, P 64, N 128, chunk 256).

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.  The workspace slots the kernels leave
unwritten (db slot l = 0, the blocks past n_valid / n tiles) are excluded.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.dendritic_chunk import chunk_cb, chunk_dt_cumsum
from neuronspark.ops.gluon.dendritic_chunk_states import chunk_states_train
from neuronspark.ops.triton.dendritic_states import (
    db_tma,
    db_tma_ref,
    dc_tma,
    dc_tma_ref,
)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
REF_SHAPES = [(1, 512), (1, 300)]


def operands(B, L, device):
    """the conv output (B, L, 9472) bf16 and its x / B / C views, dt_raw bf16, A, dt_bias, the chunk cumsum (training
    configuration), CB and the chunk states, plus a dout (scale 1e-2) and dstates (scale 0.1)"""
    xbc = rand_tensor((B, L, DI + 2 * N), torch.bfloat16, device)
    dt_raw = (rand_tensor((B, L, H), torch.float32, device) * 0.5 - 2.0).to(torch.bfloat16)
    A = -torch.exp(rand_tensor((H,), torch.float32, device) * 0.5)
    dt_bias = rand_tensor((H,), torch.float32, device) * 0.5
    o = {"xbc": xbc, "dt_raw": dt_raw, "A": A, "dt_bias": dt_bias}
    o.update(views(o))
    o["dA"], o["dt"] = chunk_dt_cumsum(dt_raw, A, CS, dt_bias=dt_bias)
    o["CB"] = chunk_cb(o["C"], o["B"], CS)
    _, o["states"] = chunk_states_train(o["B"], o["x"], o["dt"], o["dA"], torch.bfloat16)
    nc = o["dA"].shape[2]
    o["dout"] = (rand_tensor((B, L, H, P), torch.float32, device) * 1e-2).to(torch.bfloat16)
    o["dstates"] = (rand_tensor((B, nc, H, P, N), torch.float32, device) * 0.1).to(torch.bfloat16)
    return o


def views(o):
    xbc = o["xbc"]
    return {"x": xbc[..., :DI].unflatten(-1, (H, P)), "B": xbc[..., DI:DI + N].unflatten(-1, (1, N)),
            "C": xbc[..., DI + N:].unflatten(-1, (1, N))}


def _run_dbdc(o):
    dB, wsb, nvb = db_tma(o["x"], o["dt"], o["dA"], o["dstates"], o["B"], 1)
    dC, wsc, nvc = dc_tma(o["states"], o["dA"], o["dout"], o["C"], 1)
    return {"dB_parts": dB, "ws_db": wsb[..., 1:, :nvb].contiguous(), "dC_parts": dC,
            "ws_dc": wsc[..., :nvc].contiguous()}


@pytest.mark.parametrize("B,L", REF_SHAPES, ids=[f"B{b}_L{l}" for b, l in REF_SHAPES])
def test_db_dc_vs_reference(device, B, L):
    o = operands(B, L, device)
    new = _run_dbdc(o)
    dB_ref, nxt_ref = db_tma_ref(o["x"], o["dt"], o["dA"], o["dstates"], o["B"], 1)
    dC_ref, prv_ref = dc_tma_ref(o["states"], o["dA"], o["dout"], o["C"], 1)
    got = {"dB": new["dB_parts"].sum(2), "ddA_next": new["ws_db"].sum(-1), "dC": new["dC_parts"].sum(2),
           "ddA_prev": new["ws_dc"].sum(-1)}
    ref = {"dB": dB_ref, "ddA_next": nxt_ref[..., 1:], "dC": dC_ref, "ddA_prev": prv_ref}
    assert_all_match(got, ref, prefix="db_dc_ref")
