"""ops.triton.hippo_heads: per-head pool preparation and value slices (training).

* ``pool_heads``: the six heads' normalised (gained) q, normalised k and biased gate, forward and backward (q, k, gate,
  gain, bias gradients) at T 1024, B 8, against the torch reference;
* ``value_split``: the six bf16 value slices and v's gradient, against the torch reference.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import math

import pytest
import torch

from neuronspark.ops.triton.hippo_heads import pool_heads, pool_heads_ref, value_split, value_split_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

T, B, H, DK = 1024, 8, 6, 256
D = H * DK


def _inputs(device):
    q = rand_tensor((T, B, D), torch.bfloat16, device)
    k = rand_tensor((T, B, D), torch.bfloat16, device)
    gate = rand_tensor((T, B, D), torch.float32, device)
    gain = torch.exp(torch.linspace(0.0, math.log(0.37), H, device=device)).to(torch.bfloat16)
    wbias = torch.linspace(0.0, 6.0, H, device=device)
    grads = [rand_tensor((T, B, DK), torch.float32, device) for _ in range(3 * H)]
    return q, k, gate, gain, wbias, grads


def _run(fn, q, k, gate, gain, wbias, grads):
    q, k, gate = (t.clone().requires_grad_(True) for t in (q, k, gate))
    gain = gain.clone().requires_grad_(True)
    wbias = wbias.clone().requires_grad_(True)
    with regime("train"):
        qn, kn, g = fn(q, k, gate, gain, wbias, H, DK)
    outs = list(qn) + list(kn) + list(g)
    torch.autograd.backward(outs, grads)
    res = {f"out{i}": o.detach() for i, o in enumerate(outs)}
    res.update(dq=q.grad, dk=k.grad, dgate=gate.grad, dgain=gain.grad, dbias=wbias.grad)
    return res


def test_pool_heads_vs_reference(device):
    q, k, gate, gain, wbias, grads = _inputs(device)
    new = _run(pool_heads, q, k, gate, gain, wbias, grads)
    ref = _run(pool_heads_ref, q, k, gate, gain, wbias, grads)
    assert new["dgain"].dtype == torch.bfloat16 and new["dbias"].dtype == torch.float32
    assert_all_match(new, ref, prefix="pool_heads")


def test_pool_heads_partial_gradient_raises(device):
    q, k, gate, gain, wbias, _ = _inputs(device)
    q = q.requires_grad_(True)
    with regime("train"):
        qn, kn, g = pool_heads(q, k, gate, gain, wbias, H, DK)
    with pytest.raises(RuntimeError):
        qn[0].sum().backward()


# ------------------------------------------------------------------------------------------------ value_split
def _vsplit_run(fn, v, grads):
    v = v.clone().requires_grad_(True)
    outs = fn(v, H, DK)
    torch.autograd.backward(list(outs), grads)
    res = {f"v{i}": o.detach() for i, o in enumerate(outs)}
    res["dv"] = v.grad
    return res


def test_value_split_vs_reference(device):
    v = rand_tensor((T, B, D), torch.bfloat16, device)
    grads = [rand_tensor((T, B, DK), torch.bfloat16, device) for _ in range(H)]
    new = _vsplit_run(value_split, v, grads)
    ref = _vsplit_run(value_split_ref, v, grads)
    assert new["dv"].is_contiguous()
    assert_all_match(new, ref, prefix="value_split")
