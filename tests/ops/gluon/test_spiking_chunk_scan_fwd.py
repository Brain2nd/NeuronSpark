"""ops.gluon.spiking_chunk_scan_fwd: the warp-specialised Gluon chunk-scan forward of the spiking-layer training regime.

* against the TMA chunk-scan forward ``chunk_scan_fwd_tma`` (the stock kernel's arithmetic), on the model's operands
  (B 8, T 1024, H 144, P 64, N 128, chunk 256) and smaller batches / lengths: the main set and the stress set whose
  decays reach the subnormal range (the Gluon kernel's ``ex2.approx.ftz`` flushes results below 2^-126);
* against the float64 reference on a smaller shape;
* the full-chunk contract (a partial chunk raises).

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.gluon.spiking_chunk_scan_fwd import chunk_scan_fwd_gluon, chunk_scan_fwd_gluon_ref
from neuronspark.ops.triton.mamba import BMM_CHUNK_FWD, CHUNK_CUMSUM_FWD_TRAIN, _bmm_chunk_fwd, _chunk_cumsum_fwd
from neuronspark.ops.triton.spiking_scan import chunk_scan_fwd_tma
from neuronspark.ops.gluon.spiking_chunk_states import chunk_states_train
from neuronspark.utils.testing import assert_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
SHAPES = [(8, 1024), (2, 512), (1, 256), (3, 768)]


def operands(B, L, device, stress=False):
    """conv-output views of one (B, L, DI + 2N) projection, dt / dA from the chunk cumsum, CB, and the passed states"""
    xbc = rand_tensor((B, L, DI + 2 * N), torch.bfloat16, device)
    if stress:
        dt_raw = rand_tensor((B, L, H), torch.float32, device).to(torch.bfloat16)
        A = -torch.exp(rand_tensor((H,), torch.float32, device) * 0.5 + 2.5)
    else:
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
    return {"xbc": xbc, "CB": CB, "x": x, "dt": dt, "dA": dA, "C": Cm, "states": states, "D": D}


def _args(o):
    return (o["CB"], o["x"], o["dt"], o["dA"], o["C"], o["states"], o["D"])


@pytest.mark.parametrize("stress", [False, True], ids=["main", "stress"])
@pytest.mark.parametrize("B,L", SHAPES, ids=[f"B{b}_L{l}" for b, l in SHAPES])
def test_vs_tma_forward(device, B, L, stress):
    o = operands(B, L, device, stress)
    with torch.no_grad():
        out = chunk_scan_fwd_gluon(*_args(o))
        ref = chunk_scan_fwd_tma(*_args(o))
    assert out.dtype == torch.bfloat16 and out.is_contiguous() and out.shape == (B, L, H, P)
    assert_match(out, ref, f"gluon_vs_tma_B{B}_L{L}")


def test_vs_reference(device):
    o = operands(1, 512, device)
    with torch.no_grad():
        assert_match(chunk_scan_fwd_gluon(*_args(o)), chunk_scan_fwd_gluon_ref(*_args(o)), "gluon_vs_ref")


def test_full_chunks_only(device):
    o = operands(1, 300, device)
    with pytest.raises(RuntimeError):
        chunk_scan_fwd_gluon(*_args(o))
