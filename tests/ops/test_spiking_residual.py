"""ops.triton.spiking_residual: the spiking-layer prefill residual addition ``h + upd`` of the transposed mixer output.

* ``spiking_res_add`` against its reference ``spiking_res_add_ref`` (``h + upd``, the ATen addition) in the layer's
  prefill layout: ``h`` (T, B, 1536) fp32 contiguous (T-major), ``upd`` the (B, T, 1536) bf16 GEMM output viewed as
  (T, B, 1536) by a transpose; T x B in {2 x 1, 16 x 1, 128 x 8, 1024 x 1, 1024 x 8, 2048 x 1, 2048 x 8, 896 x 8},
  two random sets each (h 4 randn, upd 2 randn rounded to bf16);
* other T / B strides of both operands (a B-major ``h``, a contiguous ``upd``), an output that is a new contiguous
  fp32 tensor, inputs unchanged;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.spiking_residual import spiking_res_add, spiking_res_add_ref
from neuronspark.utils.testing import assert_all_match, assert_match

pytestmark = pytest.mark.cuda

D = 1536
SHAPES = [(2, 1), (16, 1), (128, 8), (1024, 1), (1024, 8), (2048, 1), (2048, 8), (896, 8)]     # (T, B)
SEEDS = [0, 1]


def make(T, B, seed, device):
    """h (T, B, D) fp32 contiguous and upd = the (B, T, D) bf16 tensor viewed as (T, B, D), drawn on the CPU from a
    generator seeded per (seed, T, B)."""
    g = torch.Generator(device="cpu").manual_seed(100 * seed + T + B)
    h = (torch.randn((T, B, D), generator=g) * 4.0).to(device)
    upd = (torch.randn((B, T, D), generator=g) * 2.0).to(torch.bfloat16).to(device).transpose(0, 1)
    return h, upd


@pytest.mark.parametrize("T,B", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_res_add_vs_reference(device, T, B, seed):
    h, upd = make(T, B, seed, device)
    h0, u0 = h.clone(), upd.clone()
    with torch.no_grad():
        out = spiking_res_add(h, upd)
    ref = spiking_res_add_ref(h, upd)
    assert out.dtype == torch.float32 and out.is_contiguous() and tuple(out.shape) == (T, B, D)
    assert out.data_ptr() != h.data_ptr()
    assert_match(out, ref, f"res_add_T{T}_B{B}_s{seed}")
    assert_all_match({"h": h, "upd": upd}, {"h": h0, "upd": u0}, prefix="unchanged")


@pytest.mark.parametrize("T,B", [(128, 8), (896, 8)])
def test_res_add_other_strides(device, T, B):
    """A B-major h (the transposed view of a contiguous (B, T, D) tensor) and a contiguous upd."""
    g = torch.Generator(device="cpu").manual_seed(7 + T + B)
    h = (torch.randn((B, T, D), generator=g) * 4.0).to(device).transpose(0, 1)
    upd_t = (torch.randn((B, T, D), generator=g) * 2.0).to(torch.bfloat16).to(device).transpose(0, 1)
    upd_c = upd_t.contiguous()
    with torch.no_grad():
        for name, u in (("upd_transposed", upd_t), ("upd_contiguous", upd_c)):
            out = spiking_res_add(h, u)
            assert out.is_contiguous()
            assert_match(out, spiking_res_add_ref(h, u), f"res_add_bmajor_h_{name}_T{T}_B{B}")


def test_res_add_refusals(device):
    h, upd = make(16, 2, 0, device)
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            spiking_res_add(h, upd.float())                                     # fp32 upd
        with pytest.raises(RuntimeError):
            spiking_res_add(h.to(torch.bfloat16), upd)                          # bf16 h
        with pytest.raises(RuntimeError):
            spiking_res_add(h[:8], upd)                                         # shape mismatch
        upd_strided = torch.zeros((16, 2, D, 2), dtype=torch.bfloat16, device=device)[..., 0]
        with pytest.raises(RuntimeError):
            spiking_res_add(h, upd_strided)                                     # non-unit last stride
        with pytest.raises(RuntimeError):
            spiking_res_add(h.cpu(), upd.cpu())                                 # CPU tensors
    h_req = h.clone().requires_grad_(True)
    with pytest.raises(RuntimeError):
        spiking_res_add(h_req, upd)                                             # autograd requested
