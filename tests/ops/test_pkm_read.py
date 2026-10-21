"""ops.triton.pkm_read: the PKM value read.

* training ``pkm_value_read`` (output and the gradients of the slot weights and of the value table) at B 8 x T 256
  against the float64 reference (``F.embedding`` + einsum); slot ids with hot slots (thousands of occurrences, many
  64-occurrence chunks) and with uniform ids; value widths 768 (D = 1536) and 1024 (D = 2048);
* gradients of one input only (frozen weights / frozen table) against the both-gradients call;
* prefill ``pkm_weighted_sum`` at b in {1, 8} and the prefill lengths against the float64 reference;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.pkm_read import (
    pkm_value_read,
    pkm_value_read_ref,
    pkm_weighted_sum,
    pkm_weighted_sum_ref,
)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

S, D, H, K = 16384, 768, 4, 16
WIDTHS = (768, 1024)                 # value widths of D = 1536 and D = 2048
M = H * K
B_TRAIN = 8
T_REF = 256
PREFILL_B = (1, 8)
PREFILL_T = (2, 3, 63, 64, 256, 1024, 2048)
DISTS = ("hot", "uniform")


def _inputs(T, B, dist, device, D=D):
    """w: per-head softmax weights (T, B, 64) fp32; u: slot ids (T, B, 64) int64 ("hot": S * U^2, so the low slots
    get thousands of occurrences; "uniform"); V: (S, D) fp32 table N(0, 1/D); go: an upstream gradient (T, B, D)."""
    w = torch.softmax(rand_tensor((T, B, H, K), torch.float32, device) * 2.0, dim=-1).reshape(T, B, M).contiguous()
    r = torch.rand((T, B, M), device=device)
    if dist == "hot":
        r = r * r
    u = (r * S).long().clamp_(0, S - 1)
    V = (rand_tensor((S, D), torch.float32, device) * D ** -0.5).contiguous()
    go = rand_tensor((T, B, D), torch.float32, device) * 1e-2
    return {"w": w, "u": u, "V": V, "go": go}


def _run(fn, inp, need_w=True, need_v=True):
    w = inp["w"].clone().requires_grad_(need_w)
    V = inp["V"].clone().requires_grad_(need_v)
    o = fn(w, inp["u"], V)
    o.backward(inp["go"])
    out = {"o": o.detach()}
    if need_w:
        out["g_w"] = w.grad
    if need_v:
        out["g_V"] = V.grad
    return out


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("dist", DISTS)
def test_pkm_value_read_vs_reference(device, dist, width):
    inp = _inputs(T_REF, B_TRAIN, dist, device, width)
    new = _run(pkm_value_read, inp)
    ref = _run(pkm_value_read_ref, inp)
    assert_all_match(new, ref, prefix=f"read_{dist}_{width}")


@pytest.mark.parametrize("need", ("w", "V"))
def test_pkm_value_read_single_gradient(device, need):
    """With one input frozen the other's gradient equals the both-gradients call's."""
    inp = _inputs(T_REF, B_TRAIN, "hot", device)
    both = _run(pkm_value_read, inp)
    one = _run(pkm_value_read, inp, need_w=(need == "w"), need_v=(need == "V"))
    key = "g_w" if need == "w" else "g_V"
    assert_all_match({"o": one["o"], key: one[key]}, {"o": both["o"], key: both[key]}, prefix=f"only_{need}")


def test_pkm_value_read_no_grad(device):
    inp = _inputs(T_REF, B_TRAIN, "uniform", device)
    with torch.no_grad():
        o = pkm_value_read(inp["w"], inp["u"], inp["V"])
        o_ref = pkm_value_read_ref(inp["w"], inp["u"], inp["V"])
    assert_all_match({"o": o}, {"o": o_ref}, prefix="no_grad")


def test_pkm_value_read_refusals(device):
    inp = _inputs(4, 2, "uniform", device)
    with pytest.raises(RuntimeError):
        pkm_value_read(inp["w"], inp["u"].to(torch.int32), inp["V"])         # int32 ids
    with pytest.raises(RuntimeError):
        pkm_value_read(inp["w"], inp["u"], inp["V"][:, :500].contiguous())   # D % 256 != 0
    with pytest.raises(RuntimeError):
        pkm_value_read(inp["w"].to(torch.bfloat16), inp["u"], inp["V"])      # bf16 weights


# ------------------------------------------------------------------------------------------------ prefill
PREFILL_CASES = [(f"b{b}_t{t}", b, t) for b in PREFILL_B for t in PREFILL_T]


@pytest.mark.parametrize("name,b,T", PREFILL_CASES, ids=[c[0] for c in PREFILL_CASES])
def test_pkm_weighted_sum_vs_reference(device, name, b, T):
    inp = _inputs(T, b, "uniform", device)
    with torch.no_grad():
        o = pkm_weighted_sum(inp["w"], inp["u"], inp["V"])
        o_ref = pkm_weighted_sum_ref(inp["w"], inp["u"], inp["V"])
    assert_all_match({"o": o}, {"o": o_ref}, prefix=name)


def test_pkm_weighted_sum_refusals(device):
    inp = _inputs(4, 2, "uniform", device)
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            pkm_weighted_sum(inp["w"], inp["u"][:, :1], inp["V"])                # ids of another shape
        with pytest.raises(RuntimeError):
            pkm_weighted_sum(inp["w"], inp["u"], inp["V"].to(torch.bfloat16))    # bf16 table
