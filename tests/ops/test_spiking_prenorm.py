"""ops.triton.spiking_prenorm: the spiking-layer pre-norm (``spiking_norm``) and the residual add + next pre-norm
(``spiking_resid_norm``).

* against the pure-torch references ``spiking_norm_ref`` / ``spiking_resid_norm_ref`` (autograd): outputs and every
  gradient;
* the layout contract (hn in h's layout, h_new with torch's add strides) and the input refusals.

Shapes: the (T, B, D) = (1024, 8, 1536) fp32 stream of a training step and a shorter one, stored B-major (the
(0, 1)-transposed view of a contiguous (B, T, D) tensor).  Every comparison is the per-operator rule of
``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.spiking_prenorm import (spiking_norm, spiking_norm_ref, spiking_resid_norm,
                                                    spiking_resid_norm_ref)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

EPS = 1e-5
SHAPES = [(1024, 8, 1536), (256, 2, 1536)]          # (T, B, D)


def _stream(T, B, D, device, dtype=torch.float32, scale=1.0):
    """(T, B, D) view of a contiguous (B, T, D) tensor (the B-major stream)"""
    return (rand_tensor((B, T, D), torch.float32, device) * scale).to(dtype).transpose(0, 1)


def _gain(D, device):
    return 1.0 + 0.1 * rand_tensor((D,), torch.float32, device)


def _norm_case(T, B, D, device):
    return {"h": _stream(T, B, D, device, scale=3.0), "gain": _gain(D, device),
            "dy": _stream(T, B, D, device, dtype=torch.bfloat16)}


def _resid_case(T, B, D, device):
    return {"h": _stream(T, B, D, device, scale=3.0), "upd": _stream(T, B, D, device, dtype=torch.bfloat16),
            "gain": _gain(D, device), "g_h": _stream(T, B, D, device, scale=0.1),
            "g_y": _stream(T, B, D, device, dtype=torch.bfloat16)}


def _run_norm(fn, c):
    h = c["h"].detach().clone().requires_grad_(True)
    g = c["gain"].detach().clone().requires_grad_(True)
    hn = fn(h, g, EPS)
    hn.backward(c["dy"])
    return {"hn": hn.detach(), "dh": h.grad, "dgain": g.grad}


def _run_resid(fn, c):
    h = c["h"].detach().clone().requires_grad_(True)
    u = c["upd"].detach().clone().requires_grad_(True)
    g = c["gain"].detach().clone().requires_grad_(True)
    h_new, hn = fn(h, u, g, EPS)
    torch.autograd.backward([h_new, hn], [c["g_h"], c["g_y"]])
    return {"h_new": h_new.detach(), "hn": hn.detach(), "dh": h.grad, "dupd": u.grad, "dgain": g.grad}


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_spiking_norm_vs_reference(device, shape):
    c = _norm_case(*shape, device)
    assert_all_match(_run_norm(spiking_norm, c), _run_norm(spiking_norm_ref, c), prefix="spiking_norm_ref")


@pytest.mark.parametrize("shape", SHAPES, ids=[str(s) for s in SHAPES])
def test_spiking_resid_norm_vs_reference(device, shape):
    c = _resid_case(*shape, device)
    assert_all_match(_run_resid(spiking_resid_norm, c), _run_resid(spiking_resid_norm_ref, c),
                     prefix="spiking_resid_norm_ref")


def test_layouts(device):
    T, B, D = 64, 4, 1536
    h = _stream(T, B, D, device)
    upd = _stream(T, B, D, device, dtype=torch.bfloat16)
    g = _gain(D, device)
    with torch.no_grad():
        hn = spiking_norm(h, g, EPS)
        assert hn.dtype == torch.bfloat16 and hn.stride() == h.stride()
        assert hn.transpose(0, 1).is_contiguous()
        h_new, hn2 = spiking_resid_norm(h, upd, g, EPS)
        assert h_new.dtype == torch.float32 and h_new.stride() == (h + upd.float()).stride()
        assert hn2.stride() == h_new.stride()
    # a T-major contiguous stream works as well
    hc = h.contiguous()
    with torch.no_grad():
        assert spiking_norm(hc, g, EPS).is_contiguous()


def test_refusals(device):
    T, B = 16, 2
    g = _gain(1536, device)
    with pytest.raises(RuntimeError):
        spiking_norm(_stream(T, B, 512, device), _gain(512, device), EPS)             # D <= 512
    with pytest.raises(RuntimeError):
        spiking_norm(_stream(T, B, 1536, device, dtype=torch.bfloat16), g, EPS)       # bf16 stream
    with pytest.raises(RuntimeError):
        spiking_norm(_stream(T, B, 1536, device), g.double(), EPS)                    # fp64 gain
    with pytest.raises(RuntimeError):
        spiking_resid_norm(_stream(T, B, 1536, device), _stream(T, B, 1536, device), g, EPS)   # fp32 update
