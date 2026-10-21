"""ops.triton.linear_small_n: the inference product of 2 to 32 rows and fewer than 16 output features.

* the kernel against the reference ``linear_small_n_ref`` (float64 products of the bf16 operands, one bf16 rounding) on
  rows 2 / 8 / 16 / 32, output features 1 / 2 / 6 / 8 / 15 and K 768 / 1024 / 1536;
* ``ops.linear``'s inference path on the decode shapes (B, 1, K) of the value-residual gate (6, 1536): the routed result
  against ``linear_ref``; one row and 16 output features stay on cuBLAS (``takes`` refuses them).

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.linear import linear, linear_ref
from neuronspark.ops.triton.linear_small_n import linear_small_n, linear_small_n_ref, takes
from neuronspark.utils.testing import assert_match, rand_tensor

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("K", [768, 1024, 1536])
@pytest.mark.parametrize("N", [1, 2, 6, 8, 15])
@pytest.mark.parametrize("M", [2, 8, 16, 32])
def test_kernel_vs_reference(device, M, N, K):
    x = rand_tensor((M, K), torch.bfloat16, device)
    w = (rand_tensor((N, K), torch.float32, device) / K ** 0.5).to(torch.bfloat16)
    assert takes(x, w)
    assert_match(linear_small_n(x, w), linear_small_n_ref(x, w), f"small_n_M{M}_N{N}_K{K}")


@pytest.mark.parametrize("B", [2, 8, 32])
def test_linear_routes_decode_products(device, B):
    x = rand_tensor((B, 1, 1536), torch.float32, device)
    w = rand_tensor((6, 1536), torch.float32, device) * 0.02
    with torch.no_grad():
        y = linear(x, w)
    assert y.shape == (B, 1, 6) and y.dtype == torch.bfloat16
    assert takes(x.to(torch.bfloat16).reshape(B, 1536), w.to(torch.bfloat16))
    assert_match(y, linear_ref(x, w), f"linear_vr_gate_B{B}")


def test_outside_the_form_stays_on_cublas(device):
    w = rand_tensor((6, 1536), torch.float32, device).to(torch.bfloat16)
    assert not takes(rand_tensor((1, 1536), torch.bfloat16, device), w)                       # one row
    assert not takes(rand_tensor((8, 1536), torch.bfloat16, device), rand_tensor((16, 1536), torch.bfloat16, device))
    assert not takes(rand_tensor((64, 1536), torch.bfloat16, device), w)                      # more than 32 rows
    assert not takes(rand_tensor((8, 1000), torch.bfloat16, device), rand_tensor((6, 1000), torch.bfloat16, device))
    x = rand_tensor((1, 1, 1536), torch.float32, device)
    with torch.no_grad():
        assert_match(linear(x, w.float()), linear_ref(x, w.float()), "linear_one_row")
