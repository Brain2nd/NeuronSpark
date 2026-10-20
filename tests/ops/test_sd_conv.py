"""ops.triton.sd_conv: the SD causal depthwise conv (width 4) + SiLU.

* training form (``sd_conv_silu``) on the projection-slice layout (row stride 18832) against the pure-torch reference
  ``sd_conv_silu_ref`` (autograd): output, input gradient, weight and bias gradients;
* prefill form (``sd_conv_silu_prefill``, the carried tail as initial states) against the pure-torch reference
  ``sd_conv_silu_prefill_ref`` at b in {1, 8} and T in {2, 3, 63, 64, 256, 1024};
* consistency: a prefill from a zero tail equals the training form at the same positions; two chained prefills equal
  one prefill;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.sd_conv import (
    sd_conv_silu,
    sd_conv_silu_prefill,
    sd_conv_silu_prefill_ref,
    sd_conv_silu_ref,
)
from neuronspark.utils.testing import assert_all_match, assert_match, rand_tensor

pytestmark = pytest.mark.cuda

DI, N, H = 9216, 128, 144
C = DI + 2 * N                          # 9472 conv channels
WIDTH = 2 * DI + 2 * N + H              # 18832 projection columns
TRAIN_SHAPES = [(8, 1024), (2, 256), (1, 77)]
PREFILL_B = [1, 8]
PREFILL_T = [2, 3, 63, 64, 256, 1024]


def _proj_slice(B, T, device, scale=1.0):
    """the x|B|C slice of a (B, T, 18832) bf16 projection (row stride 18832)"""
    proj = (rand_tensor((B, T, WIDTH), torch.float32, device) * scale).to(torch.bfloat16)
    return proj[..., DI:DI + C]


def _params(device):
    w = rand_tensor((C, 4), torch.float32, device) * 0.3
    b = rand_tensor((C,), torch.float32, device) * 0.1
    return w, b


def _run_train(fn, xBC, w, b, dout):
    x = xBC.detach().requires_grad_(True)              # a leaf with the projection-slice strides
    wg = w.detach().clone().requires_grad_(True)
    bg = b.detach().clone().requires_grad_(True)
    out = fn(x, wg, bg)
    out.backward(dout)
    return {"out": out.detach(), "dx": x.grad, "dw": wg.grad, "db": bg.grad}


@pytest.mark.parametrize("B,T", TRAIN_SHAPES, ids=[f"B{b}_T{t}" for b, t in TRAIN_SHAPES])
def test_conv_vs_reference(device, B, T):
    xBC = _proj_slice(B, T, device)
    w, b = _params(device)
    dout = rand_tensor((B, T, C), torch.bfloat16, device)
    new = _run_train(sd_conv_silu, xBC, w, b, dout)
    ref = _run_train(sd_conv_silu_ref, xBC, w, b, dout)
    assert_all_match(new, ref, prefix=f"conv_ref_B{B}_T{T}")


def _tail(B, device):
    """a carried tail: bf16-exact fp32 values (the conv inputs of the previous call)"""
    return rand_tensor((B, C, 3), torch.float32, device).to(torch.bfloat16).float()


@pytest.mark.parametrize("B", PREFILL_B)
@pytest.mark.parametrize("T", PREFILL_T)
def test_prefill_vs_reference(device, B, T):
    xBC = _proj_slice(B, T, device)
    w, b = _params(device)
    tail = _tail(B, device)
    with torch.no_grad():
        xbc, tail_new = sd_conv_silu_prefill(xBC, w, b, tail)
        ref_xbc, ref_tail = sd_conv_silu_prefill_ref(xBC, w, b, tail)
    assert xbc.is_contiguous() and xbc.shape == (B, T, C) and xbc.dtype == torch.bfloat16
    assert tail_new.shape == (B, C, 3) and tail_new.dtype == torch.float32
    assert_all_match({"xbc": xbc, "tail": tail_new}, {"xbc": ref_xbc, "tail": ref_tail}, prefix=f"prefill_B{B}_T{T}")


def test_prefill_consistency(device):
    B, T = 2, 512
    xBC = _proj_slice(B, T, device)
    w, b = _params(device)
    with torch.no_grad():
        whole = sd_conv_silu(xBC, w, b)
        zero = torch.zeros((B, C, 3), dtype=torch.float32, device=device)
        one, tail_one = sd_conv_silu_prefill(xBC, w, b, zero)
        first, tail1 = sd_conv_silu_prefill(xBC[:, :200], w, b, zero)
        second, tail2 = sd_conv_silu_prefill(xBC[:, 200:], w, b, tail1)
    assert_match(one, whole, "prefill_vs_whole_row")
    assert_match(torch.cat([first, second], dim=1), one, "chained_prefill")
    assert_match(tail2, tail_one, "chained_tail")
    assert_match(tail_one, xBC[:, -3:].float().transpose(1, 2), "tail_is_last_inputs")


def test_refusals(device):
    xBC = _proj_slice(1, 8, device)
    w, b = _params(device)
    with pytest.raises(RuntimeError):
        sd_conv_silu(xBC.float(), w, b)                                   # fp32 activations
    with pytest.raises(RuntimeError):
        sd_conv_silu(xBC, w.t().contiguous().t(), b)                      # non-contiguous weight
    with pytest.raises(RuntimeError):
        sd_conv_silu_prefill(xBC, w, b, torch.zeros((1, C, 2), device=device))   # wrong tail
