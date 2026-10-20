"""ops.triton.plif: the PLIF population operators.

* training ``plif_scan_train`` (spikes S and the gradients of w_eta, b_eta, zp, x, v_th) at B 8 x T 1024 for the three
  population kinds (C 1536 fp32 input in T-major and B-major layout, C 768 bf16 input), against the pure-torch
  reference (eager gate chain and sequential scan forward; the phase-split backward's expressions with the bipolar
  sigmoid surrogate, sequential over t);
* training ``plif_centred_input`` (output and input gradient) against the reference;
* prefill ``plif_gate_prefill`` and ``plif_scan_prefill`` (S and the carried membrane) at b in {1, 8} and the prefill
  lengths, against the references; a split prefill against one call; ``plif_gate_prefill`` reading the bf16 eta
  projection (widened on load) against the same kernel and the reference on the projection widened beforehand;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import math

import pytest
import torch

from neuronspark.ops.triton.plif import (
    plif_centred_input,
    plif_centred_input_ref,
    plif_gate_prefill,
    plif_gate_prefill_ref,
    plif_scan_prefill,
    plif_scan_prefill_ref,
    plif_scan_train,
    plif_scan_train_ref,
)
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

T_TRAIN, B_TRAIN = 1024, 8
# (name, channels, input dtype, layout of x: "t" = T-major contiguous, "b" = (T, B, C) view of B-major storage)
TRAIN_CASES = [
    ("input2_c1536_fp32_tmajor", 1536, torch.float32, "t"),
    ("gate_c1536_fp32_bmajor", 1536, torch.float32, "b"),
    ("ffn_c768_bf16_tmajor", 768, torch.bfloat16, "t"),
]
PREFILL_B = (1, 8)
PREFILL_T = (2, 3, 63, 64, 256, 1024, 2048)
# (name, channels, input dtype): the spike dtype of a population is its input dtype
PREFILL_SITES = [("c1536_fp32", 1536, torch.float32), ("c768_bf16", 768, torch.bfloat16)]


# ------------------------------------------------------------------------------------------------ inputs
def _params(C, device):
    """w_eta, b_eta, v_th as a trained population holds them: b_eta near the launch value logit(1 / tau) with tau
    log-uniform in [2, 100], a small w_eta, thresholds in [0.15, 0.40]."""
    tau = torch.exp(torch.empty(C, device=device).uniform_(math.log(2.0), math.log(100.0)))
    p = (1.0 / tau).clamp(1e-6, 1 - 1e-6)
    b_eta = (torch.log(p / (1 - p)) + 0.3 * rand_tensor((C,), torch.float32, device)).contiguous()
    w_eta = (0.05 * rand_tensor((C,), torch.float32, device)).contiguous()
    v_th = torch.empty(C, device=device).uniform_(0.15, 0.40).contiguous()
    return w_eta, b_eta, v_th


def _train_inputs(C, x_dtype, layout, device, T=T_TRAIN, B=B_TRAIN):
    w_eta, b_eta, v_th = _params(C, device)
    store = (B, T, C) if layout == "b" else (T, B, C)
    x = rand_tensor(store, torch.float32, device)
    if x_dtype == torch.bfloat16:
        x = (0.5 * x).to(torch.bfloat16)
    zp = (0.3 * rand_tensor((T, B, C), torch.float32, device)).to(torch.bfloat16)
    gS = rand_tensor((T, B, C), x_dtype, device)
    return {"x": x, "layout": layout, "w_eta": w_eta, "b_eta": b_eta, "zp": zp, "v_th": v_th, "gS": gS}


def _view(xs, layout):
    return xs.transpose(0, 1) if layout == "b" else xs


def _run_train(fn, inp):
    xs = inp["x"].clone().requires_grad_(True)
    x = _view(xs, inp["layout"])
    w = inp["w_eta"].clone().requires_grad_(True)
    b = inp["b_eta"].clone().requires_grad_(True)
    zp = inp["zp"].clone().requires_grad_(True)
    th = inp["v_th"].clone().requires_grad_(True)
    S = fn(w, b, zp, x, th, alpha=8.0, out_dtype=None)
    S.backward(inp["gS"])
    return {"S": S.detach(), "g_w_eta": w.grad, "g_b_eta": b.grad, "g_zp": zp.grad, "g_x": xs.grad,
            "g_v_th": th.grad}


# ------------------------------------------------------------------------------------------------ training scan
@pytest.mark.sm100
@pytest.mark.parametrize("name,C,x_dtype,layout", TRAIN_CASES, ids=[c[0] for c in TRAIN_CASES])
def test_plif_scan_train_vs_reference(device, name, C, x_dtype, layout):
    inp = _train_inputs(C, x_dtype, layout, device)
    new = _run_train(plif_scan_train, inp)
    ref = _run_train(plif_scan_train_ref, inp)
    assert_all_match(new, ref, prefix=name)


@pytest.mark.parametrize("name,C,x_dtype,layout", TRAIN_CASES, ids=[c[0] for c in TRAIN_CASES])
def test_plif_scan_train_no_grad(device, name, C, x_dtype, layout):
    """Without autograd the forward stores nothing for a backward and gives the same spikes as the reference."""
    inp = _train_inputs(C, x_dtype, layout, device)
    x = _view(inp["x"], layout)
    with torch.no_grad():
        S = plif_scan_train(inp["w_eta"], inp["b_eta"], inp["zp"], x, inp["v_th"])
        S_ref = plif_scan_train_ref(inp["w_eta"], inp["b_eta"], inp["zp"], x, inp["v_th"])
    assert_all_match({"S": S}, {"S": S_ref}, prefix=name)


def test_plif_scan_train_refusals(device):
    inp = _train_inputs(768, torch.bfloat16, "t", device, T=8, B=2)
    w, b, zp, x, th = inp["w_eta"], inp["b_eta"], inp["zp"], inp["x"], inp["v_th"]
    with pytest.raises(RuntimeError):                     # one token: no whole-row scan
        plif_scan_train(w, b, zp[:1].contiguous(), x[:1], th)
    with pytest.raises(RuntimeError):                     # fp32 projection
        plif_scan_train(w, b, zp.float(), x, th)
    with pytest.raises(RuntimeError):                     # channel stride != 1
        plif_scan_train(w[::2].contiguous(), b[::2].contiguous(), zp[..., ::2].contiguous(), x[..., ::2],
                        th[::2].contiguous())
    with pytest.raises(RuntimeError):                     # bf16 threshold
        plif_scan_train(w, b, zp, x, th.to(torch.bfloat16))


# ------------------------------------------------------------------------------------------------ centring
CENTRE_CASES = [("c1536_tmajor", "t"), ("c1536_bmajor", "b")]


def _centre_inputs(layout, device, T=T_TRAIN, B=B_TRAIN, C=1536):
    store = (B, T, C) if layout == "b" else (T, B, C)
    return {"x": rand_tensor(store, torch.float32, device), "layout": layout,
            "g": rand_tensor((T, B, C), torch.float32, device)}


def _run_centre(fn, inp):
    xs = inp["x"].clone().requires_grad_(True)
    out = fn(_view(xs, inp["layout"]))
    out.backward(inp["g"])
    return {"xc": out.detach(), "g_x": xs.grad}


@pytest.mark.parametrize("name,layout", CENTRE_CASES, ids=[c[0] for c in CENTRE_CASES])
def test_plif_centred_input_vs_reference(device, name, layout):
    inp = _centre_inputs(layout, device)
    new = _run_centre(plif_centred_input, inp)
    ref = _run_centre(plif_centred_input_ref, inp)
    assert_all_match(new, ref, prefix=name)


def test_plif_centred_input_refusals(device):
    with pytest.raises(RuntimeError):                     # one token
        plif_centred_input(rand_tensor((1, 2, 1536), torch.float32, device))
    with pytest.raises(RuntimeError):                     # channel stride != 1
        plif_centred_input(rand_tensor((4, 2, 3072), torch.float32, device)[..., ::2])


# ------------------------------------------------------------------------------------------------ prefill
PREFILL_CASES = [(f"{site}_b{b}_t{t}", C, dt, b, t) for (site, C, dt) in PREFILL_SITES for b in PREFILL_B
                 for t in PREFILL_T]


def _prefill_inputs(C, x_dtype, b, T, device):
    w_eta, b_eta, v_th = _params(C, device)
    x = rand_tensor((T, b, C), torch.float32, device)
    if x_dtype == torch.bfloat16:
        x = (0.5 * x).to(torch.bfloat16)
    zp = (0.3 * rand_tensor((T, b, C), torch.float32, device)).to(torch.bfloat16)
    a, drive = plif_gate_prefill_ref(x, w_eta, b_eta, zp)
    v0 = ((torch.rand((b, C), device=device) * 2.0 - 1.0) * v_th).contiguous()       # a carried membrane in [-th, th]
    return {"x": x, "w_eta": w_eta, "b_eta": b_eta, "zp": zp, "v_th": v_th, "a": a, "drive": drive, "v0": v0,
            "out_dtype": "float32" if x_dtype == torch.float32 else "bfloat16"}


def _run_scan(fn, inp):
    v = inp["v0"].clone()
    od = torch.float32 if inp["out_dtype"] == "float32" else torch.bfloat16
    with torch.no_grad():
        S = fn(inp["a"], inp["drive"], inp["v_th"], v, od)
    return {"S": S, "v": v}


@pytest.mark.parametrize("name,C,x_dtype,b,T", PREFILL_CASES, ids=[c[0] for c in PREFILL_CASES])
def test_plif_gate_prefill_vs_reference(device, name, C, x_dtype, b, T):
    inp = _prefill_inputs(C, x_dtype, b, T, device)
    with torch.no_grad():
        a, d = plif_gate_prefill(inp["x"], inp["w_eta"], inp["b_eta"], inp["zp"])
        a_fp32z, d_fp32z = plif_gate_prefill(inp["x"], inp["w_eta"], inp["b_eta"], inp["zp"].float())
    assert_all_match({"a": a, "drive": d}, {"a": inp["a"], "drive": inp["drive"]}, prefix=name)
    # the projection given as its exact fp32 widening gives the same chain
    assert_all_match({"a": a_fp32z, "drive": d_fp32z}, {"a": inp["a"], "drive": inp["drive"]}, prefix=name + "_fp32z")


@pytest.mark.parametrize("name,C,x_dtype,b,T", PREFILL_CASES, ids=[c[0] for c in PREFILL_CASES])
def test_plif_scan_prefill_vs_reference(device, name, C, x_dtype, b, T):
    inp = _prefill_inputs(C, x_dtype, b, T, device)
    new = _run_scan(plif_scan_prefill, inp)
    ref = _run_scan(plif_scan_prefill_ref, inp)
    assert_all_match(new, ref, prefix=name)


@pytest.mark.parametrize("split", (1, 2, 255, 512))
def test_plif_scan_prefill_split_equals_one_call(device, split):
    """A prefill in two calls (the membrane carried in the cache) against one call over the same tokens."""
    inp = _prefill_inputs(1536, torch.float32, 8, 1024, device)
    one = _run_scan(plif_scan_prefill, inp)
    v = inp["v0"].clone()
    with torch.no_grad():
        S1 = plif_scan_prefill(inp["a"][:split], inp["drive"][:split], inp["v_th"], v, torch.float32)
        S2 = plif_scan_prefill(inp["a"][split:], inp["drive"][split:], inp["v_th"], v, torch.float32)
    assert_all_match({"S": torch.cat([S1, S2], dim=0), "v": v}, one, prefix=f"split{split}")


def test_plif_prefill_refusals(device):
    inp = _prefill_inputs(768, torch.bfloat16, 2, 8, device)
    with torch.no_grad():
        with pytest.raises(RuntimeError):                 # bf16 drive
            plif_scan_prefill(inp["a"], inp["drive"].to(torch.bfloat16), inp["v_th"], inp["v0"].clone())
        with pytest.raises(RuntimeError):                 # membrane of the wrong shape
            plif_scan_prefill(inp["a"], inp["drive"], inp["v_th"], inp["v0"][:1].clone())
        with pytest.raises(RuntimeError):                 # projection of another shape
            plif_gate_prefill(inp["x"], inp["w_eta"], inp["b_eta"], inp["zp"][:1])


GATE_BF16_SHAPES = ((2, 1), (16, 8), (128, 8), (1024, 1), (1024, 8), (2048, 8), (896, 8))


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=["x_fp32", "x_bf16"])
@pytest.mark.parametrize("T,B", GATE_BF16_SHAPES, ids=[f"t{t}_b{b}" for t, b in GATE_BF16_SHAPES])
def test_plif_gate_prefill_bf16_projection(device, T, B, x_dtype):
    C = 1536
    g = torch.Generator(device="cpu").manual_seed(T + B)
    x = (torch.randn((T, B, C), generator=g) * 2.0).to(x_dtype).to(device)
    w = (torch.randn(C, generator=g) * 0.3).to(device)
    b = (torch.randn(C, generator=g) * 0.5 - 1.0).to(device)
    z = (torch.randn((T, B, C), generator=g) * 1.5).to(torch.bfloat16).to(device)
    with torch.no_grad(), regime("infer"):
        new = plif_gate_prefill(x, w, b, z)
        widened = plif_gate_prefill(x, w, b, z.to(torch.float32))
        ref = plif_gate_prefill_ref(x, w, b, z.to(torch.float32))
    assert_all_match({"a": new[0], "drive": new[1]}, {"a": widened[0], "drive": widened[1]}, prefix="gate_vs_widened")
    assert_all_match({"a": new[0], "drive": new[1]}, {"a": ref[0], "drive": ref[1]}, prefix="gate_vs_reference")
