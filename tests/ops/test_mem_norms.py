"""ops.triton.mem_norms: memory-layer row norms.

* ``row_rmsnorm`` (weightless and weighted), ``out_norm_gate``, ``nov_norm``: forward and backward at the training
  shape (T 1024, B 8, C 1536) in the trunk's layouts, against the pure-torch references;
* ``rmsnorm_gate_rows``: the zero-centred norm with the output gate, bf16 out, at ``rmsnorm_rows``' row shapes;
* ``rmsnorm_rows``: the prefill / decode row shapes, fp32 and bf16 inputs, with and without the (zero-centred) weight,
  against the torch chain; out-of-domain widths raise.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.mem_norms import (nov_norm, nov_norm_ref, out_norm_gate, out_norm_gate_ref,
                                              rmsnorm_gate_rows, rmsnorm_gate_rows_ref, rmsnorm_rows, rmsnorm_rows_ref,
                                              row_rmsnorm, row_rmsnorm_ref)
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

T, B, D = 1024, 8, 1536
EPS = 1e-6


def _bmajor(device, dtype=torch.float32):
    """a (T, B, D) view of a contiguous (B, T, D) tensor (the trunk's residual-stream layout)."""
    return rand_tensor((B, T, D), dtype, device)


# ------------------------------------------------------------------------------------------------ row_rmsnorm
def _row_run(base, w, g, fn):
    base = base.clone().requires_grad_(True)
    wt = None if w is None else w.clone().requires_grad_(True)
    with regime("train"):
        y = fn(base.transpose(0, 1), wt, EPS)
    y.backward(g)
    out = {"y": y.detach(), "dx": base.grad}
    if wt is not None:
        out["dw"] = wt.grad
    return out


@pytest.mark.parametrize("weighted", [False, True], ids=["frozen", "weighted"])
def test_row_rmsnorm_vs_reference(device, weighted):
    base = _bmajor(device)
    w = (1.0 + 0.1 * rand_tensor((D,), torch.float32, device)) if weighted else None
    g = rand_tensor((T, B, D), torch.float32, device)
    new = _row_run(base, w, g, row_rmsnorm)
    ref = _row_run(base, w, g, row_rmsnorm_ref)
    assert new["y"].stride() == ref["y"].stride()
    assert_all_match(new, ref, prefix="row_rmsnorm")


def test_row_rmsnorm_rejects_bf16(device):
    with pytest.raises(RuntimeError):
        row_rmsnorm(rand_tensor((T, B, D), torch.bfloat16, device), None, EPS)


# ------------------------------------------------------------------------------------------------ out_norm_gate
def _ogate_inputs(device):
    attn = rand_tensor((T, B, D), torch.float32, device)
    w = 0.1 * rand_tensor((D,), torch.float32, device)
    sgl = rand_tensor((T, B, D), torch.bfloat16, device)
    g = rand_tensor((T, B, D), torch.float32, device)
    return attn, w, sgl, g


def _ogate_run(fn, attn, w, sgl, g):
    attn = attn.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    sgl = sgl.clone().requires_grad_(True)
    with regime("train"):
        y = fn(attn, w, sgl, EPS)
    y.backward(g)
    return {"y": y.detach(), "dattn": attn.grad, "dw": w.grad, "dgate": sgl.grad}


def test_out_norm_gate_vs_reference(device):
    attn, w, sgl, g = _ogate_inputs(device)
    assert_all_match(_ogate_run(out_norm_gate, attn, w, sgl, g), _ogate_run(out_norm_gate_ref, attn, w, sgl, g),
                     prefix="out_norm_gate")


# ------------------------------------------------------------------------------------------------ nov_norm
def _nov_run(fn, r, g):
    r = r.clone().requires_grad_(True)
    with regime("train"):
        y = fn(r, EPS)
    y.backward(g)
    return {"y": y.detach(), "dr": r.grad}


def test_nov_norm_vs_reference(device):
    r = rand_tensor((T, B, D), torch.bfloat16, device)
    g = rand_tensor((T, B, D), torch.float32, device)
    assert_all_match(_nov_run(nov_norm, r, g), _nov_run(nov_norm_ref, r, g), prefix="nov_norm")


# ------------------------------------------------------------------------------------------------ rmsnorm_rows
ROW_SHAPES = [(1, 1536), (8, 1536), (2 * 8, 1536), (3 * 1, 1536), (64 * 8, 1536), (1024 * 8, 1536),
              (2048 * 1, 1536), (8, 768)]
WEIGHTS = ["none", "weight", "zero_centered"]


def _rows_case(device, shape, dtype, wmode):
    x = rand_tensor(shape, dtype, device)
    w = None
    if wmode != "none":
        w = 0.1 * rand_tensor((shape[-1],), torch.float32, device)
        if wmode == "weight":
            w = 1.0 + w
    return x, w, wmode == "zero_centered"


@pytest.mark.parametrize("shape", ROW_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in ROW_SHAPES])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("wmode", WEIGHTS)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16], ids=["out_fp32", "out_bf16"])
def test_rmsnorm_rows_vs_reference(device, shape, dtype, wmode, out_dtype):
    x, w, zc = _rows_case(device, shape, dtype, wmode)
    with torch.no_grad(), regime("infer"):
        new = rmsnorm_rows(x, EPS, w, zc, out_dtype)
        ref = rmsnorm_rows_ref(x, EPS, w, zc, out_dtype)
    assert new.dtype == out_dtype and tuple(new.shape) == tuple(shape)
    assert_all_match(new, ref, prefix="rmsnorm_rows")


def test_rmsnorm_rows_three_dim_view(device):
    x = rand_tensor((B, 64, D), torch.float32, device).transpose(0, 1)        # (T, B, D) B-major rows: copied first
    with torch.no_grad():
        new = rmsnorm_rows(x, EPS, None, False, torch.bfloat16)
        ref = rmsnorm_rows_ref(x, EPS, None, False, torch.bfloat16)
    assert_all_match(new, ref, prefix="rmsnorm_rows_view")


@pytest.mark.parametrize("n", [64, 100, 8192])
def test_rmsnorm_rows_rejects_out_of_domain(device, n):
    with pytest.raises(RuntimeError):
        rmsnorm_rows(rand_tensor((4, n), torch.float32, device), EPS)


# ------------------------------------------------------------------------------------------------ rmsnorm_gate_rows
@pytest.mark.parametrize("shape", ROW_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in ROW_SHAPES])
def test_rmsnorm_gate_rows_vs_reference(device, shape):
    x = rand_tensor(shape, torch.float32, device) * 3.0
    w = rand_tensor((int(shape[-1]),), torch.float32, device) * 0.1
    gate = (rand_tensor(shape, torch.float32, device) * 2.0).to(torch.bfloat16)
    with torch.no_grad(), regime("infer"):
        new = rmsnorm_gate_rows(x, EPS, w, gate)
        ref = rmsnorm_gate_rows_ref(x, EPS, w, gate)
    assert new.dtype == torch.bfloat16 and tuple(new.shape) == tuple(shape)
    assert_all_match(new, ref, prefix="rmsnorm_gate_rows")

