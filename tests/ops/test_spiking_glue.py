"""ops.triton.spiking_glue: the SSD backward driver ``bwd_store_glue`` of the spiking-layer call (store-not-recompute)
and its pieces ``splits_sum`` / ``ddA_combine`` / ``chunk_cumsum_bwd``.

* against ``bwd_store_glue_ref`` (the same backward sequence in pure torch on the same kept tensors, the kernels' bf16
  operand rounding points reproduced) at B 1 x L 512 and B 2 x L 512: dx, ddt (written into the projection-gradient
  view), dA, dB, dC (written into the conv-input-gradient views), dD, ddt_bias;
* the whole-chunk contract of the training regime (a partial chunk raises: the chunk-scan kernels are whole-chunk
  Gluon kernels);
* ``splits_sum`` against torch sums.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.mamba import BMM_CHUNK_FWD, CHUNK_CUMSUM_FWD_TRAIN, _bmm_chunk_fwd, _chunk_cumsum_fwd
from neuronspark.ops.triton.spiking_glue import bwd_store_glue, bwd_store_glue_ref, splits_sum
from neuronspark.ops.gluon.spiking_chunk_states import chunk_states_train
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
C_CONV = DI + 2 * N
WIDTH = 2 * DI + 2 * N + H
REF_SHAPES = [(1, 512), (2, 512)]


def operands(B, L, device):
    """forward tensors of one spiking-layer call (conv output views, the dt slice of a projection, the chunk cumsum, CB,
    the fp32
    passed states) and the incoming gradient dout (scale 1e-2)"""
    proj = rand_tensor((B, L, WIDTH), torch.bfloat16, device)
    xbc = rand_tensor((B, L, C_CONV), torch.bfloat16, device)
    proj[..., DI + C_CONV:] = (rand_tensor((B, L, H), torch.float32, device) * 0.5 - 2.0).to(torch.bfloat16)
    A = -torch.exp(rand_tensor((H,), torch.float32, device) * 0.5)
    dt_bias = rand_tensor((H,), torch.float32, device) * 0.5
    D = rand_tensor((H,), torch.float32, device) * 0.5
    o = {"proj": proj, "xbc": xbc, "A": A, "dt_bias": dt_bias, "D": D}
    v = views(o)
    o["dA"], o["dt"] = _chunk_cumsum_fwd(v["dt_raw"], A, CS, dt_bias=dt_bias, dt_softplus=True,
                                         dt_limit=(0.0, float("inf")), config=CHUNK_CUMSUM_FWD_TRAIN)
    o["CB"] = _bmm_chunk_fwd(v["C"], v["B"], CS, seq_idx=None, output_dtype=torch.float32, config=BMM_CHUNK_FWD)
    o["states_passed"], _ = chunk_states_train(v["B"], v["x"], o["dt"], o["dA"], torch.bfloat16, passed_fp32=True)
    o["dout"] = (rand_tensor((B, L, H, P), torch.float32, device) * 1e-2).to(torch.bfloat16)
    return o


def views(o):
    xbc, proj = o["xbc"], o["proj"]
    return {"x": xbc[..., :DI].unflatten(-1, (H, P)), "B": xbc[..., DI:DI + N].unflatten(-1, (1, N)),
            "C": xbc[..., DI + N:].unflatten(-1, (1, N)), "dt_raw": proj[..., DI + C_CONV:]}


def _run(o):
    v = views(o)
    B, L = o["xbc"].shape[:2]
    dproj = torch.empty((B, L, WIDTH), dtype=torch.bfloat16, device=o["xbc"].device)
    dxBC = torch.empty((B, L, C_CONV), dtype=torch.bfloat16, device=o["xbc"].device)
    ddt_buf = dproj[..., DI + C_CONV:]
    dx_buf = dxBC[..., :DI].unflatten(-1, (H, P))
    dB_buf = dxBC[..., DI:DI + N].unflatten(-1, (1, N))
    dC_buf = dxBC[..., DI + N:].unflatten(-1, (1, N))
    dx, ddt, dA, dB, dC, dD, ddt_bias = bwd_store_glue(
        o["dout"], v["x"], v["dt_raw"], o["A"], v["B"], v["C"], CS, o["D"], o["dt_bias"], dx_buf, ddt_buf, dB_buf, dC_buf,
        o["dA"], o["dt"], o["CB"], o["states_passed"])
    assert dx.data_ptr() == dx_buf.data_ptr() and ddt.data_ptr() == ddt_buf.data_ptr()
    assert dB.data_ptr() == dB_buf.data_ptr() and dC.data_ptr() == dC_buf.data_ptr()
    return {"dx": dx_buf, "ddt": ddt_buf, "dA": dA, "dB": dB_buf, "dC": dC_buf, "dD": dD, "ddt_bias": ddt_bias}


@pytest.mark.parametrize("B,L", REF_SHAPES, ids=[f"B{b}_L{l}" for b, l in REF_SHAPES])
def test_glue_vs_reference(device, B, L):
    o = operands(B, L, device)
    v = views(o)
    new = _run(o)
    ref = bwd_store_glue_ref(o["dout"], v["x"], v["dt_raw"], o["A"], v["B"], v["C"], CS, o["D"], o["dt_bias"],
                             None, None, None, None, o["dA"], o["dt"], o["CB"], o["states_passed"])
    names = ("dx", "ddt", "dA", "dB", "dC", "dD", "ddt_bias")
    assert_all_match({k: new[k].contiguous() for k in names}, dict(zip(names, ref)), prefix="glue_ref")


def test_glue_whole_chunks_only(device):
    o = operands(1, 300, device)
    with pytest.raises(RuntimeError, match="full chunks only"):
        _run(o)


def test_splits_sum(device):
    a = rand_tensor((2, 1024, 7, 1, 128), torch.float32, device)
    b = rand_tensor((2, 4, 3, 1, 256, 256), torch.float32, device)
    sa, sb = splits_sum((a, 2), (b, 2))
    assert_all_match({"a": sa, "b": sb}, {"a": a.sum(2), "b": b.sum(2)}, prefix="splits_sum")
