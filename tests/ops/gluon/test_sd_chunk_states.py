"""ops.gluon.sd_chunk_states: the warp-specialised Gluon chunk states + state passing of the SD scan.

* the training form ``chunk_states_train`` (states_pre fp32, states bf16 / fp32) and the prefill form
  ``chunk_states_prefill`` (from a carried fp32 / bf16 initial state: states, final state) against the float64
  references, on the model's operands (B 8, T 1024, H 144, P 64, N 128, chunk 256) and smaller batches / lengths: the
  main set and the stress set whose decays reach the subnormal range;
* a partial last chunk (rows past the sequence weigh 0).

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.gluon.sd_chunk_states import (chunk_states_prefill, chunk_states_prefill_ref, chunk_states_train,
                                                   chunk_states_train_ref)
from neuronspark.ops.triton.mamba import CHUNK_CUMSUM_FWD_TRAIN, _chunk_cumsum_fwd
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N, CS = 144, 64, 128, 256
DI = H * P
SHAPES = [(8, 1024), (2, 512), (1, 256), (3, 768)]


def operands(B, L, device, stress=False):
    """conv-output views of one (B, L, DI + 2N) projection and dt / dA from the chunk cumsum"""
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
    dA, dt = _chunk_cumsum_fwd(dt_raw, A, CS, dt_bias=dt_bias, dt_softplus=True, dt_limit=(0.0, float("inf")),
                               config=CHUNK_CUMSUM_FWD_TRAIN)
    return Bm, x, dt, dA


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("stress", [False, True], ids=["main", "stress"])
@pytest.mark.parametrize("B,L", SHAPES, ids=[f"B{b}_L{l}" for b, l in SHAPES])
def test_train_vs_reference(device, B, L, stress, out_dtype):
    args = operands(B, L, device, stress)
    with torch.no_grad():
        pre, st = chunk_states_train(*args, out_dtype)
        rpre, rst = chunk_states_train_ref(*args, out_dtype)
    assert pre.dtype == torch.float32 and st.dtype == out_dtype and st.shape == (B, L // CS, H, P, N)
    assert_all_match({"states_pre": pre, "states": st}, {"states_pre": rpre, "states": rst}, prefix=f"train_B{B}_L{L}")


@pytest.mark.parametrize("init_dtype", [torch.float32, torch.bfloat16], ids=["init_fp32", "init_bf16"])
@pytest.mark.parametrize("stress", [False, True], ids=["main", "stress"])
@pytest.mark.parametrize("B,L", SHAPES, ids=[f"B{b}_L{l}" for b, l in SHAPES])
def test_prefill_vs_reference(device, B, L, stress, init_dtype):
    args = operands(B, L, device, stress)
    init = rand_tensor((B, H, P, N), torch.float32, device).to(init_dtype)
    with torch.no_grad():
        st, fin = chunk_states_prefill(*args, init, torch.bfloat16)
        rst, rfin = chunk_states_prefill_ref(*args, init, torch.bfloat16)
    assert fin.dtype == torch.float32 and fin.shape == (B, H, P, N)
    assert_all_match({"states": st, "final": fin}, {"states": rst, "final": rfin}, prefix=f"prefill_B{B}_L{L}")


def test_time_slices(device):
    """x / B / dt views of a time slice of a longer projection (batch stride > seqlen x row stride)"""
    Bm, x, dt, dA = operands(2, 1024, device)
    xs, Bs = x[:, 256:768], Bm[:, 256:768]
    dts, dAs = dt[:, :, 1:3].contiguous(), dA[:, :, 1:3].contiguous()
    init = rand_tensor((2, H, P, N), torch.float32, device)
    with torch.no_grad():
        st, fin = chunk_states_prefill(Bs, xs, dts, dAs, init, torch.bfloat16)
        rst, rfin = chunk_states_prefill_ref(Bs, xs, dts, dAs, init, torch.bfloat16)
    assert_all_match({"states": st, "final": fin}, {"states": rst, "final": rfin}, prefix="time_slice")


def test_prefill_from_zero(device):
    args = operands(2, 512, device)
    zero = torch.zeros((2, H, P, N), dtype=torch.float32, device=device)
    with torch.no_grad():
        st, fin = chunk_states_prefill(*args, None, torch.bfloat16)
        rst, rfin = chunk_states_prefill_ref(*args, zero, torch.bfloat16)
    assert_all_match({"states": st, "final": fin}, {"states": rst, "final": rfin}, prefix="prefill_zero")


@pytest.mark.parametrize("B,L", [(1, 300), (3, 1000), (2, 65), (8, 128), (8, 2), (4, 33)],
                         ids=["B1_L300", "B3_L1000", "B2_L65", "B8_L128", "B8_L2", "B4_L33"])
def test_partial_last_chunk(device, B, L):
    args = operands(B, L, device)
    init = rand_tensor((B, H, P, N), torch.float32, device)
    with torch.no_grad():
        pre, st = chunk_states_train(*args, torch.bfloat16)
        rpre, rst = chunk_states_train_ref(*args, torch.bfloat16)
        pst, pfin = chunk_states_prefill(*args, init, torch.bfloat16)
        rpst, rpfin = chunk_states_prefill_ref(*args, init, torch.bfloat16)
    assert_all_match({"states_pre": pre, "states": st, "p_states": pst, "p_final": pfin},
                     {"states_pre": rpre, "states": rst, "p_states": rpst, "p_final": rpfin}, prefix=f"partial_B{B}_L{L}")
