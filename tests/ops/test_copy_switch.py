"""ops.triton.copy_switch: the copy-switch auxiliary loss of the six memory layers.

* the aux value and the gradients of every layer's predictor (w, v, b) at B 8 x T 1024, D 1536, against the
  pure-torch reference (boolean (B, T, T) labels, detached predictor logits, pi-weighted binary cross entropy);
* the copy labels against the reference;
* operands in T-major and in B-major layout, with ignored targets; row counts off the kernels' row blocks;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.copy_switch import (
    copy_switch_aux,
    copy_switch_aux_ref,
    copy_switch_labels,
    copy_switch_labels_ref,
)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

B, T, D, L = 8, 1024, 1536, 6
LAM = 0.5
VOCAB = 48                       # a small vocabulary: repeated bigrams, i.e. positive copy labels


def _ids(device, b=B, t=T):
    ids = torch.randint(0, VOCAB, (b, t), device=device, dtype=torch.int64)
    targets = torch.roll(ids, shifts=-1, dims=1)
    targets[:, -1] = -100
    drop = torch.rand((b, t), device=device) < 0.02
    targets = torch.where(drop, torch.full_like(targets, -100), targets)
    return ids, targets


def _layers(device, b=B, t=T, d=D, n=L):
    """Per layer: z (T, B, D) fp32 (layers 0, 2, 4 T-major, 1, 3, 5 views of B-major storage), r (T, B, D) bf16, pi
    (T, B) fp32 in [0, 1) (0 at t < 2), and the predictor's w, v (D,) fp32, b (1,) fp32."""
    feats, weights = [], []
    for li in range(n):
        if li % 2:
            z = rand_tensor((b, t, d), torch.float32, device).transpose(0, 1)
        else:
            z = rand_tensor((t, b, d), torch.float32, device)
        r = (0.3 * rand_tensor((t, b, d), torch.float32, device)).to(torch.bfloat16)
        pi = torch.rand((t, b), device=device)
        pi[:2] = 0.0
        feats.append((z, r, pi))
        w = (0.02 * rand_tensor((d,), torch.float32, device)).contiguous()
        v = (0.02 * rand_tensor((d,), torch.float32, device)).contiguous()
        bias = torch.full((1,), 4.0, device=device) + 0.1 * rand_tensor((1,), torch.float32, device)
        weights.append((w, v, bias))
    return feats, weights


def _run(fn, ids, targets, feats, weights):
    params = [tuple(p.clone().requires_grad_(True) for p in wt) for wt in weights]
    aux = fn(ids, targets, feats, params, LAM)
    aux.backward()
    out = {"aux": aux.detach()}
    for li, (w, v, bias) in enumerate(params):
        out[f"dw{li}"] = w.grad
        out[f"dv{li}"] = v.grad
        out[f"db{li}"] = bias.grad
    return out


def test_copy_switch_aux_vs_reference(device):
    ids, targets = _ids(device)
    feats, weights = _layers(device)
    new = _run(copy_switch_aux, ids, targets, feats, weights)
    ref = _run(copy_switch_aux_ref, ids, targets, feats, weights)
    assert_all_match(new, ref, prefix="copy_switch")


@pytest.mark.parametrize("t", (2, 3, 64, 1000, 1024))
def test_copy_switch_labels_vs_reference(device, t):
    ids, targets = _ids(device, t=t)
    label, valid = copy_switch_labels(ids, targets)
    label_ref, valid_ref = copy_switch_labels_ref(ids, targets)
    assert_all_match({"label": label, "valid": valid.to(torch.float32)},
                     {"label": label_ref, "valid": valid_ref.to(torch.float32)}, prefix=f"labels_t{t}")


def test_copy_switch_aux_small_rows(device):
    """Row counts that are not multiples of the kernels' row blocks (16 forward, 128 backward)."""
    ids, targets = _ids(device, b=3, t=77)
    feats, weights = _layers(device, b=3, t=77, n=2)
    new = _run(copy_switch_aux, ids, targets, feats, weights)
    ref = _run(copy_switch_aux_ref, ids, targets, feats, weights)
    assert_all_match(new, ref, prefix="small")


def test_copy_switch_aux_refusals(device):
    ids, targets = _ids(device, b=2, t=16)
    feats, weights = _layers(device, b=2, t=16, d=256, n=2)
    with pytest.raises(RuntimeError):                            # one weight set short
        copy_switch_aux(ids, targets, feats, weights[:1], LAM)
    bad = [(feats[0][0][:8], feats[0][1], feats[0][2])] + feats[1:]
    with pytest.raises(RuntimeError):                            # z of another length
        copy_switch_aux(ids, targets, bad, weights, LAM)
    badw = [(weights[0][0].to(torch.bfloat16), weights[0][1], weights[0][2])] + weights[1:]
    with pytest.raises(RuntimeError):                            # bf16 predictor weight
        copy_switch_aux(ids, targets, feats, badw, LAM)
