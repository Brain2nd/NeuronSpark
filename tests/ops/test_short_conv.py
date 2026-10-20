"""ops.triton.short_conv: the qkv short convolution in its three regimes.

* ``short_conv_train`` at T 1024, B 8, C 4608: output, input gradient and the fp32 weight gradient, against the fp32
  tap-chain reference; two uses of the weight in one graph;
* ``short_conv_silu_prefill`` (the conv and its SiLU) at b in {1, 8}, T in {3, 63, 64, 256, 1024, 2048}, chained over
  two calls (the carried taps);
* ``short_conv_step`` at b in {1, 2, 4, 8, 16, 32}, several consecutive steps;

the inference forms against the torch chains (outputs and the rolled history), and the steps (with the SiLU applied to
their outputs) against one prefill.
Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

import torch.nn.functional as F

from neuronspark.ops.triton.short_conv import (short_conv_silu_prefill, short_conv_silu_prefill_ref, short_conv_step,
                                               short_conv_step_ref, short_conv_train, short_conv_train_ref)
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

T, B, C = 1024, 8, 4608


def _weight(device):
    w = 0.5 * rand_tensor((C, 1, 4), torch.float32, device)
    w[:, 0, 3] += 1.0                                              # near the identity launch (tap 3 = 1)
    return w


# ------------------------------------------------------------------------------------------------ training
def _train_run(fn, qkv, w, g):
    qkv = qkv.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    with regime("train"):
        y = fn(qkv, w)
    y.backward(g)
    return {"y": y.detach(), "dx": qkv.grad, "dw": w.grad}


def test_short_conv_train_vs_reference(device):
    qkv = rand_tensor((T, B, C), torch.bfloat16, device)
    w = _weight(device)
    g = rand_tensor((T, B, C), torch.bfloat16, device)
    new = _train_run(short_conv_train, qkv, w, g)
    ref = _train_run(short_conv_train_ref, qkv, w, g)
    assert new["y"].is_contiguous() and new["dw"].dtype == torch.float32
    assert_all_match(new, ref, prefix="short_conv_train")


def test_short_conv_train_gradient_accumulates(device):
    """Two uses of the weight in one graph: autograd sums the fp32 gradients (the per-use values are the kernel's)."""
    qkv = rand_tensor((T, B, C), torch.bfloat16, device)
    w = _weight(device).requires_grad_(True)
    g = rand_tensor((T, B, C), torch.bfloat16, device)
    with regime("train"):
        (short_conv_train(qkv, w).float() * g.float()).sum().backward()
    one = w.grad.clone()
    w.grad = None
    with regime("train"):
        loss = (short_conv_train(qkv, w).float() * g.float()).sum() + (short_conv_train(qkv, w).float() * g.float()).sum()
    loss.backward()
    assert_all_match(w.grad, 2.0 * one, prefix="short_conv_train_two_uses")


# ------------------------------------------------------------------------------------------------ prefill
PREFILL = [(b, t) for b in (1, 8) for t in (3, 63, 64, 256, 1024, 2048)]


def _hist(device, b):
    return rand_tensor((b, C, 3), torch.bfloat16, device).float()   # bf16-exact carried taps


def _prefill_run(fn, x1, x2, hist, w):
    hist = hist.clone()
    with torch.no_grad(), regime("infer"):
        y1 = fn(x1, hist, w)
        h1 = hist.clone()
        y2 = fn(x2, hist, w)
    return {"y1": y1, "h1": h1, "y2": y2, "h2": hist}


@pytest.mark.parametrize("b,t", PREFILL, ids=[f"b{b}_t{t}" for b, t in PREFILL])
def test_short_conv_silu_prefill_vs_reference(device, b, t):
    x1 = rand_tensor((t, b, C), torch.bfloat16, device)
    x2 = rand_tensor((t, b, C), torch.bfloat16, device)
    hist, w = _hist(device, b), _weight(device)
    assert_all_match(_prefill_run(short_conv_silu_prefill, x1, x2, hist, w),
                     _prefill_run(short_conv_silu_prefill_ref, x1, x2, hist, w), prefix="short_conv_silu_prefill")


def test_short_conv_silu_prefill_rejects_short_call(device):
    with pytest.raises(RuntimeError):
        short_conv_silu_prefill(rand_tensor((2, 1, C), torch.bfloat16, device), _hist(device, 1), _weight(device))


# ------------------------------------------------------------------------------------------------ decode step
STEP_B = [1, 2, 4, 8, 16, 32]
N_STEPS = 4


def _step_run(fn, xs, hist, w):
    hist = hist.clone()
    out = {}
    with torch.no_grad(), regime("infer"):
        for i, x in enumerate(xs):
            out[f"y{i}"] = fn(x, hist, w)
    out["hist"] = hist
    return out


@pytest.mark.parametrize("b", STEP_B, ids=[f"b{b}" for b in STEP_B])
def test_short_conv_step_vs_reference(device, b):
    xs = [rand_tensor((b, C), torch.bfloat16, device) for _ in range(N_STEPS)]
    hist, w = _hist(device, b), _weight(device)
    assert_all_match(_step_run(short_conv_step, xs, hist, w), _step_run(short_conv_step_ref, xs, hist, w),
                     prefix="short_conv_step")


def test_short_conv_step_matches_prefill(device):
    """Consecutive steps from a carried history (their outputs through SiLU) match one prefill call over the same tokens
    under the rule."""
    b, t = 8, 16
    x = rand_tensor((t, b, C), torch.bfloat16, device)
    hist, w = _hist(device, b), _weight(device)
    h_pre, h_step = hist.clone(), hist.clone()
    with torch.no_grad():
        y_pre = short_conv_silu_prefill(x, h_pre, w)
        y_step = F.silu(torch.stack([short_conv_step(x[i], h_step, w) for i in range(t)], dim=0))
    assert_all_match({"y": y_step, "hist": h_step}, {"y": y_pre, "hist": h_pre}, prefix="step_vs_prefill")
