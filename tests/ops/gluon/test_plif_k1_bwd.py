"""ops.gluon.plif_k1_bwd: the Gluon backward kernel of the training PLIF population.

For the three population kinds at B 8 x T 1024 (C 1536 fp32 input in T-major and B-major layout, C 768 bf16 input):

* the backward outputs (gz, gzp, gx, per-lane gth), given the operands saved by the training forward (eta, pre),
  against the pure-torch reference ``plif_k1_bwd_ref`` (the same recurrences, evaluated sequentially);
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import math

import pytest
import torch

from neuronspark.ops.gluon.plif_k1_bwd import plif_k1_bwd, plif_k1_bwd_ref
from neuronspark.ops.triton.plif import _plif_k1x_fwd
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

T_TRAIN, B_TRAIN = 1024, 8
CASES = [
    ("input2_c1536_fp32_tmajor", 1536, torch.float32, "t"),
    ("gate_c1536_fp32_bmajor", 1536, torch.float32, "b"),
    ("ffn_c768_bf16_tmajor", 768, torch.bfloat16, "t"),
]


def _inputs(C, x_dtype, layout, device, T=T_TRAIN, B=B_TRAIN):
    tau = torch.exp(torch.empty(C, device=device).uniform_(math.log(2.0), math.log(100.0)))
    p = (1.0 / tau).clamp(1e-6, 1 - 1e-6)
    b_eta = (torch.log(p / (1 - p)) + 0.3 * rand_tensor((C,), torch.float32, device)).contiguous()
    w_eta = (0.05 * rand_tensor((C,), torch.float32, device)).contiguous()
    v_th = torch.empty(C, device=device).uniform_(0.15, 0.40).contiguous()
    store = (B, T, C) if layout == "b" else (T, B, C)
    x = rand_tensor(store, torch.float32, device)
    if x_dtype == torch.bfloat16:
        x = (0.5 * x).to(torch.bfloat16)
    zp = (0.3 * rand_tensor((T, B, C), torch.float32, device)).to(torch.bfloat16)
    gS = rand_tensor((T, B, C), x_dtype, device)
    return {"x": x, "layout": layout, "w_eta": w_eta, "b_eta": b_eta, "zp": zp, "v_th": v_th, "gS": gS}


def _view(xs, layout):
    return xs.transpose(0, 1) if layout == "b" else xs


def _forward(inp):
    x = _view(inp["x"], inp["layout"])
    _S, pre, eta = _plif_k1x_fwd(inp["w_eta"], inp["b_eta"], inp["zp"], x, inp["v_th"], True, x.dtype)
    return x, eta, pre


@pytest.mark.parametrize("name,C,x_dtype,layout", CASES, ids=[c[0] for c in CASES])
def test_plif_k1_bwd_vs_reference(device, name, C, x_dtype, layout):
    inp = _inputs(C, x_dtype, layout, device)
    x, eta, pre = _forward(inp)
    new = plif_k1_bwd(eta, x, inp["v_th"], pre, inp["gS"], 8.0)
    ref = plif_k1_bwd_ref(eta, x, inp["v_th"], pre, inp["gS"], 8.0)
    assert_all_match(dict(zip(("gz", "gzp", "gx", "gth"), new)), dict(zip(("gz", "gzp", "gx", "gth"), ref)),
                     prefix=name)


def test_plif_k1_bwd_refusals(device):
    inp = _inputs(768, torch.bfloat16, "t", device, T=32, B=2)
    x, eta, pre = _forward(inp)
    gS = inp["gS"]
    with pytest.raises(RuntimeError):                     # no surrogate slope
        plif_k1_bwd(eta, x, inp["v_th"], pre, gS, 0.0)
    with pytest.raises(RuntimeError):                     # slope of the guarded sigmoid form
        plif_k1_bwd(eta, x, inp["v_th"], pre, gS, 96.0)
    with pytest.raises(RuntimeError):                     # non-contiguous upstream gradient
        plif_k1_bwd(eta, x, inp["v_th"], pre, gS.transpose(0, 1).contiguous().transpose(0, 1), 8.0)
    with pytest.raises(RuntimeError):                     # fp16 upstream gradient
        plif_k1_bwd(eta, x, inp["v_th"], pre, gS.to(torch.float16), 8.0)
