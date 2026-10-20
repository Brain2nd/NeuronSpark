"""ops.gluon.sd_dda: the warp-specialised Gluon intra-chunk dCB / ddA pieces of the SD training scan's backward.

``dda_gluon`` -- its dCB split partials summed over the splits and its V grid assembled by ``dda_asm`` -- against the
float64 reference ``dda_ref``, on the 1.27B operands (B 8, T 1024, H 144, P 64, N 128, chunk 256), smaller shapes and
stress decays (steep diagonal blocks take the per-element exponent); the call contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.gluon.sd_dda import NSPLIT, dda_gluon
from neuronspark.ops.triton.mamba import BMM_CHUNK_FWD, CHUNK_CUMSUM_FWD_TRAIN, _bmm_chunk_fwd, _chunk_cumsum_fwd
from neuronspark.ops.triton.sd_scan import dda_asm, dda_ref
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
SHAPES = [(8, 1024), (2, 512), (1, 256), (3, 768)]


def operands(B, L, device, stress=False):
    """x / B / C views of one conv output, dt / dA from the chunk cumsum (training configuration), CB, and dout
    (scale 1e-2) as the gated norm's input gradient; ``stress``: larger step sizes and decays."""
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
    dA, dt = _chunk_cumsum_fwd(dt_raw, A, CS, dt_bias=dt_bias, dt_softplus=True, dt_limit=(0.0, float("inf")),
                               config=CHUNK_CUMSUM_FWD_TRAIN)
    CB = _bmm_chunk_fwd(Cm, Bm, CS, seq_idx=None, output_dtype=torch.float32, config=BMM_CHUNK_FWD)
    dout = (rand_tensor((B, L, H, P), torch.float32, device) * 1e-2).to(torch.bfloat16)
    return x, dt, dA, dout, CB


@pytest.mark.parametrize("stress", [False, True], ids=["main", "stress"])
@pytest.mark.parametrize("B,L", SHAPES, ids=[f"B{b}_L{l}" for b, l in SHAPES])
def test_dda_vs_reference(device, B, L, stress):
    x, dt, dA, dout, CB = operands(B, L, device, stress)
    with torch.no_grad():
        dcbp, v = dda_gluon(x, dt, dA, dout, CB)
        rdcb, rdda = dda_ref(x, dt, dA, dout, CB)
    assert dcbp.shape == (B, L // CS, NSPLIT, 1, CS, CS)
    assert_all_match({"dCB": dcbp.sum(2), "ddA_intra": dda_asm(v)}, {"dCB": rdcb, "ddA_intra": rdda},
                     prefix=f"vs_ref_B{B}_L{L}")


def test_refusals(device):
    x, dt, dA, dout, CB = operands(1, 512, device)
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            dda_gluon(x[:, :300], dt, dA, dout[:, :300], CB)                         # not full chunks
        with pytest.raises(RuntimeError):
            dda_gluon(x.float(), dt, dA, dout, CB)                                   # fp32 x
        with pytest.raises(RuntimeError):
            dda_gluon(x, dt, dA, dout, CB.to(torch.bfloat16))                        # bf16 CB
        with pytest.raises(RuntimeError):
            dda_gluon(x, dt.transpose(2, 3).contiguous().transpose(2, 3), dA, dout, CB)   # strided dt
