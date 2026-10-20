"""ops.gluon.spiking_chunk_scan_dx: the warp-specialised Gluon chunk-scan input backward (dx, ddt, dD) of the
spiking-layer training regime.

* against the float64 reference ``chunk_scan_dx_ref`` (the FTZ decay and the bf16 rounding point of the K operand
  reproduced), on the model's operands (B 8, T 1024, H 144, P 64, N 128, chunk 256) and smaller batches / lengths: the
  main set and the stress set whose decays reach the subnormal range (``ex2.approx.ftz`` flushes results below
  2^-126);
* the full-chunk contract (a partial chunk raises).

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.gluon.spiking_chunk_scan_dx import chunk_scan_dx_gluon, chunk_scan_dx_gluon_ref
from neuronspark.ops.triton.mamba import BMM_CHUNK_FWD, CHUNK_CUMSUM_FWD_TRAIN, _bmm_chunk_fwd, _chunk_cumsum_fwd
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
SHAPES = [(8, 1024), (2, 512), (1, 256), (3, 768)]


def operands(B, L, device, stress=False):
    """conv-output views of one (B, L, DI + 2N) projection, dt / dA from the chunk cumsum, CB, the output gradient and
    the state gradients"""
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
    dout = rand_tensor((B, L, H, P), torch.bfloat16, device)
    dstates = rand_tensor((B, L // CS, H, P, N), torch.bfloat16, device)
    return (x, dt, dA, Bm, CB, dout, dstates, D)


@pytest.mark.parametrize("stress", [False, True], ids=["main", "stress"])
@pytest.mark.parametrize("B,L", SHAPES, ids=[f"B{b}_L{l}" for b, l in SHAPES])
def test_vs_reference(device, B, L, stress):
    args = operands(B, L, device, stress)
    with torch.no_grad():
        dx, ddt, dD = chunk_scan_dx_gluon(*args)
        rdx, rddt, rdD = chunk_scan_dx_gluon_ref(*args)
    assert dx.dtype == torch.bfloat16 and dx.is_contiguous() and dx.shape == (B, L, H, P)
    assert ddt.dtype == torch.float32 and ddt.shape == rddt.shape and dD.shape == (H,)
    assert_all_match({"dx": dx, "ddt": ddt, "dD": dD}, {"dx": rdx, "ddt": rddt, "dD": rdD}, prefix=f"gluon_vs_ref_B{B}_L{L}")


def test_full_chunks_only(device):
    args = operands(1, 300, device)
    with pytest.raises(RuntimeError):
        chunk_scan_dx_gluon(*args)
