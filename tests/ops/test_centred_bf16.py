"""The prefill centred inputs of the value-residual gate and the centred eta gate, written in bf16.

In a prefill call the memory layer forms the strictly-past centred input of the value-residual gate and of the
centred PLIF neuron's eta projection with ``centred_running(..., out_dtype=torch.bfloat16)``: their only consumer is a
bf16 linear, whose input cast rounds the fp32 value to bf16 exactly as the kernel's store does.  Checked under the
inference regime at the 1.27B shapes (x (T, B, 1536) fp32, the carried sum (B, 1536) fp32, the counts (B,) int64 of
rows at their start and inside a row; the projections (16, 1536) = eta_V and (6, 1536) = pool_vr_gate): the bf16 path
(kernel) against the fp32 path of the torch chain ``centred_running_ref`` followed by the same linear -- the linear's
output and the advanced carried sum -- by the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
T x B in {2 x 1, 16 x 8, 128 x 8, 1024 x 1, 1024 x 8, 2048 x 8, 896 x 8}.
"""
import pytest
import torch

from neuronspark.ops.linear import linear
from neuronspark.ops.triton.cumsum import centred_running, centred_running_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match

pytestmark = pytest.mark.cuda

C = 1536
SHAPES = ((2, 1), (16, 8), (128, 8), (1024, 1), (1024, 8), (2048, 8), (896, 8))
SEEDS = (0, 1)


def _inputs(T: int, B: int, R: int, seed: int, device):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = (torch.randn((T, B, C), generator=g) * 3.0).to(device)
    s = (torch.randn((B, C), generator=g) * 40.0).to(device)
    cnt = torch.tensor([0 if b % 2 == 0 else 37 + 100 * b for b in range(B)], dtype=torch.int64, device=device)
    w = (torch.randn((R, C), generator=g) * 0.05).to(device)
    return x, s, cnt, w


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("R", (16, 6), ids=["eta_V", "pool_vr_gate"])
@pytest.mark.parametrize("T,B", SHAPES, ids=[f"t{t}_b{b}" for t, b in SHAPES])
def test_centred_bf16_vs_fp32_chain(device, T, B, R, seed):
    x, s, cnt, w = _inputs(T, B, R, 1000 * seed + T + B + R, device)
    s_new, s_ref = s.clone(), s.clone()
    with torch.no_grad(), regime("infer"):
        xc_new = centred_running(x, s_new, cnt, torch.bfloat16)
        xc_ref = centred_running_ref(x, s_ref, cnt, torch.float32)
        assert xc_new.dtype == torch.bfloat16 and tuple(xc_new.shape) == (T, B, C)
        y_new = linear(xc_new, w)
        y_ref = linear(xc_ref, w)
    assert_all_match({"y": y_new, "xc": xc_new, "run_sum": s_new}, {"y": y_ref, "xc": xc_ref.to(torch.bfloat16),
                                                                    "run_sum": s_ref}, prefix="centred_bf16")
