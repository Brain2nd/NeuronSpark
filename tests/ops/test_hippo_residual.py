"""ops.triton.hippo_residual: value residual, centring, read gain and PKM tail.

Forward and backward at T 1024, B 8, C 1536 in the trunk's layouts, against the torch references; the read gain and the
PKM tail also without autograd at the inference shapes (prefill T x B in 1024 x 8, 37 x 8, 1024 x 1; decode 1 x 32,
1 x 1) on contiguous T-major rows.  Every comparison is
the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.hippo_residual import (PKM_TAIL_MIN_ROWS, pkm_tail, pkm_tail_ref, read_gain_residual,
                                                 read_gain_residual_ref, value_residual, value_residual_ref, zc_center,
                                                 zc_center_ref)
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

T, B, D, DT = 1024, 8, 1536, 256


# ------------------------------------------------------------------------------------------------ value_residual
def _vres_inputs(device, first):
    v = rand_tensor((T, B, D), torch.bfloat16, device)
    lam = 0.5 + 0.1 * rand_tensor((T, B, D), torch.float32, device)
    carry = None if first else rand_tensor((T, B, D), torch.bfloat16, device)
    gv = 0.1 * rand_tensor((T, B, DT), torch.float32, device)
    g = rand_tensor((T, B, D), torch.bfloat16, device)
    return v, lam, carry, gv, g


def _vres_run(fn, v, lam, carry, gv, g):
    v = v.clone().requires_grad_(True)
    lam = lam.clone().requires_grad_(True)
    gv = gv.clone().requires_grad_(True)
    c = None if carry is None else carry.clone().requires_grad_(True)
    with regime("train"):
        y = fn(v, lam, c, gv)
    y.backward(g)
    out = {"y": y.detach(), "dv": v.grad, "dlam": lam.grad, "dgv": gv.grad}
    if c is not None:
        out["dcarry"] = c.grad
    return out


@pytest.mark.parametrize("first", [True, False], ids=["layer0", "later"])
def test_value_residual_vs_reference(device, first):
    args = _vres_inputs(device, first)
    assert_all_match(_vres_run(value_residual, *args), _vres_run(value_residual_ref, *args), prefix="value_residual")


# ------------------------------------------------------------------------------------------------ zc_center
def _zc_inputs(device):
    zb = rand_tensor((B, T, D), torch.float32, device)            # z: the (T, B, D) view of a B-major tensor
    zcum = torch.cat([torch.zeros_like(zb.transpose(0, 1)[:1]), zb.transpose(0, 1).cumsum(0)[:-1]], dim=0)
    g = rand_tensor((T, B, D), torch.float32, device)
    return zb, zcum, g


def _zc_run(fn, zb, zcum, g):
    zb = zb.clone().requires_grad_(True)
    zcum = zcum.clone().requires_grad_(True)
    with regime("train"):
        y = fn(zb.transpose(0, 1), zcum)
    y.backward(g)
    return {"y": y.detach(), "dz": zb.grad, "dzcum": zcum.grad}


def test_zc_center_vs_reference(device):
    zb, zcum, g = _zc_inputs(device)
    new = _zc_run(zc_center, zb, zcum, g)
    assert new["y"].is_contiguous()
    assert_all_match(new, _zc_run(zc_center_ref, zb, zcum, g), prefix="zc_center")


# ------------------------------------------------------------------------------------------------ read_gain_residual
def _rg_inputs(device):
    hb = rand_tensor((B, T, D), torch.float32, device)
    r = rand_tensor((T, B, D), torch.bfloat16, device)
    f = 4.0 * torch.rand((T, B), device=device)
    g = rand_tensor((T, B, D), torch.float32, device)
    return hb, r, f, g


def _rg_run(fn, hb, r, f, g):
    hb = hb.clone().requires_grad_(True)
    r = r.clone().requires_grad_(True)
    f = f.clone().requires_grad_(True)
    y = fn(hb.transpose(0, 1), r, f)
    y.backward(g)
    return {"y": y.detach(), "dh": hb.grad, "dr": r.grad, "df": f.grad}


def test_read_gain_residual_vs_reference(device):
    args = _rg_inputs(device)
    assert_all_match(_rg_run(read_gain_residual, *args), _rg_run(read_gain_residual_ref, *args),
                     prefix="read_gain_residual")


# ------------------------------------------------------------------------------------------------ pkm_tail
def _pkt_inputs(device):
    hb = rand_tensor((B, T, D), torch.float32, device)
    ffn = rand_tensor((T, B, D), torch.bfloat16, device)
    y = rand_tensor((T, B, D), torch.bfloat16, device)
    rho = 2.0 * torch.sigmoid(-4.0 * torch.rand((T, B), device=device))
    g = rand_tensor((T, B, D), torch.float32, device)
    return hb, ffn, y, rho, g


def _pkt_run(fn, hb, ffn, y, rho, g):
    hb = hb.clone().requires_grad_(True)
    ffn = ffn.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    with regime("train"):
        out = fn(hb.transpose(0, 1), ffn, y, rho)
    out.backward(g)
    return {"out": out.detach(), "dh": hb.grad, "dffn": ffn.grad, "dy": y.grad}


def test_pkm_tail_vs_reference(device):
    args = _pkt_inputs(device)
    assert_all_match(_pkt_run(pkm_tail, *args), _pkt_run(pkm_tail_ref, *args), prefix="pkm_tail")


def test_pkm_tail_rejects_rho_with_grad(device):
    hb, ffn, y, rho, _ = _pkt_inputs(device)
    with pytest.raises(RuntimeError):
        pkm_tail(hb.transpose(0, 1), ffn, y, rho.requires_grad_(True))


# ------------------------------------------------------------------------------------------------ inference shapes
INFER_SHAPES = ((1024, 8), (37, 8), (1024, 1), (1, 32), (1, 1))


@pytest.mark.parametrize("Ti,Bi", INFER_SHAPES, ids=[f"t{t}_b{b}" for t, b in INFER_SHAPES])
def test_read_gain_and_pkm_tail_inference(device, Ti, Bi):
    h = rand_tensor((Ti, Bi, D), torch.float32, device) * 5.0
    r = rand_tensor((Ti, Bi, D), torch.bfloat16, device)
    f = torch.rand((Ti, Bi), device=device)
    ffn = rand_tensor((Ti, Bi, D), torch.bfloat16, device)
    y = rand_tensor((Ti, Bi, D), torch.bfloat16, device)
    rho = 2.0 * torch.sigmoid(-4.0 * torch.rand((Ti, Bi), device=device))
    with torch.no_grad(), regime("infer"):
        h1 = read_gain_residual(h, r, f)
        h1_ref = read_gain_residual_ref(h, r, f)
        assert_all_match({"h1": h1}, {"h1": h1_ref}, prefix="read_gain_infer")
        if Ti * Bi >= PKM_TAIL_MIN_ROWS:
            assert_all_match({"out": pkm_tail(h1, ffn, y, rho)}, {"out": pkm_tail_ref(h1, ffn, y, rho)}, prefix="pkm_tail_infer")
        else:
            with pytest.raises(RuntimeError):           # below the reproduced reduction layout: the caller keeps the torch lines
                pkm_tail(h1, ffn, y, rho)

