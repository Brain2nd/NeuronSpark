"""ops.linear: the fp32-weight-gradient linear op.

* against the pure-torch reference ``linear_ref`` (float64 products of the bf16-rounded operands) on the projection
  shapes of one training step: output, input gradient, weight gradient;
* an odd output-feature count (the padded layout) with an in-place update of the returned view, against the reference;
* the inference path with and without the bf16 weight cache, and the stale-cache refusal;
* a tied weight with two uses in one backward (readout + lookup), against the reference.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuronspark.ops.linear import cache_bf16_, drop_bf16_, linear, linear_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match, rand_tensor

pytestmark = pytest.mark.cuda

# (name, input shape, output features, input dtype): the projection shapes of one training step (B 8, T 1024)
SHAPES = [
    ("spiking_in_proj", (8, 1024, 1536), 18832, torch.bfloat16),
    ("spiking_w_out", (8192, 9216), 1536, torch.bfloat16),
    ("mem_qkv_folded", (8192, 1536), 4608, torch.bfloat16),
    ("mem_ffn_gate", (1024, 8, 1536), 768, torch.bfloat16),
    ("plif_eta_v", (1024, 8, 1536), 16, torch.float32),
    ("plif_eta_u", (1024, 8, 16), 1536, torch.bfloat16),
    ("readout", (2048, 1536), 151669, torch.float32),
]
# the reference multiplies in float64: the readout is checked on fewer rows there
REF_ROWS_CAP = {"readout": (512, 1536)}


def _inputs(shape, n, x_dtype, device):
    x = rand_tensor(shape, x_dtype, device)
    w = rand_tensor((n, shape[-1]), torch.float32, device) * 0.02
    dy = rand_tensor(tuple(shape[:-1]) + (n,), torch.bfloat16, device)
    return x, w, dy


def _kernel(x, w, dy):
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    with regime("train"):
        y = linear(x, w)
    y.backward(dy)
    return {"y": y.detach(), "dx": x.grad, "dw": w.grad}


def _ref(x, w, dy):
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    y = linear_ref(x, w)
    y.backward(dy)
    return {"y": y.detach(), "dx": x.grad, "dw": w.grad}



@pytest.mark.parametrize("name,shape,n,x_dtype", SHAPES, ids=[s[0] for s in SHAPES])
def test_linear_vs_reference(device, name, shape, n, x_dtype):
    shape = REF_ROWS_CAP.get(name, shape)
    x, w, dy = _inputs(shape, n, x_dtype, device)
    new = _kernel(x, w, dy)
    ref = _ref(x, w, dy)
    assert_all_match(new, ref, prefix=name)


def test_odd_features_padded_view_inplace(device):
    """N = 4099 (not a multiple of 8): the (R, N) view of the padded output, updated in place (as the readout balance
    updates the logits), then the backward; output, input gradient and weight gradient against the reference."""
    R, K, N = 64, 256, 4099
    x0 = rand_tensor((R, K), torch.float32, device)
    w0 = rand_tensor((N, K), torch.float32, device) * 0.02
    c0 = rand_tensor((R, 1), torch.bfloat16, device)
    dy = rand_tensor((R, N), torch.bfloat16, device)

    def run(fn, train):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        c = c0.clone().requires_grad_(True)
        if train:
            with regime("train"):
                y = fn(x, w)
        else:
            y = fn(x, w)
        y.add_(c)
        y.backward(dy)
        return {"y": y.detach(), "dx": x.grad, "dw": w.grad, "dc": c.grad}

    new = run(linear, True)
    assert tuple(new["y"].shape) == (R, N) and new["y"].dtype == torch.bfloat16 and new["dw"].dtype == torch.float32
    assert_all_match(new, run(linear_ref, False), prefix="odd_n")


def test_linear_frozen_weight_has_no_weight_gradient(device):
    x, w, dy = _inputs((256, 1536), 768, torch.bfloat16, device)
    x = x.requires_grad_(True)
    with regime("train"):
        y = linear(x, w)
    y.backward(dy)
    assert w.grad is None and x.grad is not None


def test_linear_rejects_trained_bf16_weight(device):
    x = rand_tensor((16, 64), torch.bfloat16, device)
    w = rand_tensor((32, 64), torch.bfloat16, device).requires_grad_(True)
    with pytest.raises(RuntimeError):
        linear(x, w)


def test_inference_path_and_cache(device):
    x, w, _ = _inputs((8, 1, 1536), 4608, torch.bfloat16, device)
    with torch.no_grad(), regime("infer"):
        plain = linear(x, w)
        expected = F.linear(x.to(torch.bfloat16), w.to(torch.bfloat16))
        assert_match(plain, expected, "inference_no_cache")
        cache_bf16_(w)
        cached = linear(x, w)
        assert_match(cached, expected, "inference_cached")
        w.add_(1.0)                       # an in-place update makes the cache stale
        with pytest.raises(RuntimeError):
            linear(x, w)
        drop_bf16_(w)
        after = linear(x, w)
        assert_match(after, F.linear(x.to(torch.bfloat16), w.to(torch.bfloat16)), "inference_after_drop")


def test_tied_weight_two_uses(device):
    V, D, R = 4096, 1536, 1024
    E0 = rand_tensor((V, D), torch.float32, device) * 0.02
    ids = torch.randint(0, V, (R,), device=device)
    rows = rand_tensor((R, D), torch.float32, device)
    g_logits = rand_tensor((R, V), torch.float32, device)
    g_emb = rand_tensor((R, D), torch.float32, device)

    def run(fn):
        E = E0.clone().requires_grad_(True)
        logits = fn(rows, E)
        loss = (logits.float() * g_logits).sum() + (F.embedding(ids, E) * g_emb).sum()
        loss.backward()
        return E.grad

    with regime("train"):
        new = run(linear)
    ref = run(linear_ref)
    assert new.dtype == torch.float32
    assert_match(new, ref, "tied_weight_grad")
