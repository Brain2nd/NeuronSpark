"""ops.triton.pool_attn: the memory layers' pool attention (training).

One pool head (``kshared=False``) and the four extra queries sharing the last head's keys / values / mask
(``kshared=True``), at B 8, T 1024 and a short row: the bf16 output and the gradients of q, k, v and the mask, against
the float64 reference.  The extra queries' shared key / value gradients are compared per query head (the attention's
own gradients, each rounded to bf16 before the expand sums them), and the full backward's shared key / value
gradients against the sum of the per-query ones.  Every comparison is the per-operator rule of
``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from neuronspark.ops.triton.pool_attn import pool_attention, pool_attention_ref
from neuronspark.ops.triton.pool_bias import pool_bias_mask_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

DK = DV = 256
M = 4
SCALE = 16.0
SHAPES = [(8, 1024), (2, 63)]


def _mask(device, b, t):
    cum = (-0.05 * torch.rand((t, b), device=device)).cumsum(dim=0)
    past = torch.ones(t, t, device=device, dtype=torch.bool).tril(-1)
    m = ((torch.rand((b, t, t), device=device) < 0.05) & past).to(torch.float32)
    return pool_bias_mask_ref(cum, 0.5 * m, t, b).contiguous()       # (B, 1, T, T) bf16


def _unit(x):
    return F.normalize(x, dim=-1)


def _head_inputs(device, b, t):
    q = _unit(rand_tensor((t, b, DK), torch.float32, device)) * 0.8
    k = _unit(rand_tensor((t, b, DK), torch.float32, device))
    v = rand_tensor((t, b, DV), torch.bfloat16, device)
    g = rand_tensor((t, b, DV), torch.bfloat16, device)
    return q, k, v, _mask(device, b, t), g


def _xq_inputs(device, b, t):
    Q = _unit(rand_tensor((M, t, b, DK), torch.float32, device)) * torch.exp(torch.rand((M, t, b, 1), device=device))
    k = _unit(rand_tensor((t, b, DK), torch.float32, device))
    v = rand_tensor((t, b, DV), torch.bfloat16, device)
    g = rand_tensor((t, b, M, DV), torch.bfloat16, device)
    return Q, k, v, _mask(device, b, t), g


def _run(fn, q, k, v, mask, g, kshared):
    q = q.clone().requires_grad_(True)
    k = k.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    mask = mask.clone().requires_grad_(True)
    qq = q.permute(1, 2, 0, 3) if kshared else q                    # the extra queries: stack(queries, 0) -> (T, B, M, dk)
    with regime("train"):
        y = fn(qq, k, v, mask, SCALE, kshared)
    y.backward(g)
    return {"y": y.detach(), "dq": q.grad, "dk": k.grad, "dv": v.grad, "dmask": mask.grad}


@pytest.mark.parametrize("b,t", SHAPES, ids=[f"b{b}_t{t}" for b, t in SHAPES])
def test_pool_attention_head_vs_reference(device, b, t):
    args = _head_inputs(device, b, t)
    new = _run(pool_attention, *args, kshared=False)
    ref = _run(pool_attention_ref, *args, kshared=False)
    assert new["y"].dtype == torch.bfloat16 and tuple(new["y"].shape) == (t, b, DV)
    assert_all_match(new, ref, prefix="pool_attention_head")


def _one_query(g, m):
    """The output gradient of query m alone (the other queries' output gradients zero)."""
    gm = torch.zeros_like(g)
    gm[:, :, m] = g[:, :, m]
    return gm


@pytest.mark.parametrize("b,t", SHAPES, ids=[f"b{b}_t{t}" for b, t in SHAPES])
def test_pool_attention_extra_queries_vs_reference(device, b, t):
    """The M queries share the head's key, value and mask.  The attention's key / value gradients are per query head,
    each rounded to bf16 on its own; the expand of the shared key / value then sums the M of them (key: fp32 after the
    widening cast, value: bf16).  Where the M terms cancel, one bf16 rounding step of one term is a large fraction of
    their sum, so the key / value gradients are compared per query head: the backward of query m's output gradient
    alone gives query m's rounded key / value gradient (the other heads contribute exact zeros).  The full backward's
    shared key / value gradients are compared against the sum of those per-query gradients, and its output, query
    gradients and mask gradient (summed over the heads before its one rounding) against the reference."""
    Q, k, v, mask, g = _xq_inputs(device, b, t)
    new = _run(pool_attention, Q, k, v, mask, g, kshared=True)
    ref = _run(pool_attention_ref, Q, k, v, mask, g, kshared=True)
    assert tuple(new["y"].shape) == (t, b, M, DV)
    direct = ("y", "dq", "dmask")
    assert_all_match({n: new[n] for n in direct}, {n: ref[n] for n in direct}, prefix="pool_attention_xq")
    grads = ("dq", "dk", "dv", "dmask")
    per_query = []
    for m in range(M):
        gm = _one_query(g, m)
        new_m = _run(pool_attention, Q, k, v, mask, gm, kshared=True)
        ref_m = _run(pool_attention_ref, Q, k, v, mask, gm, kshared=True)
        assert_all_match({n: new_m[n] for n in grads}, {n: ref_m[n] for n in grads},
                         prefix=f"pool_attention_xq.query{m}")
        per_query.append(new_m)
    dk_sum = torch.stack([r["dk"] for r in per_query], dim=0).sum(dim=0)
    dv_sum = torch.stack([r["dv"] for r in per_query], dim=0).to(torch.float32).sum(dim=0).to(torch.bfloat16)
    assert_all_match({"dk": new["dk"], "dv": new["dv"]}, {"dk": dk_sum, "dv": dv_sum},
                     prefix="pool_attention_xq.shared")


def test_pool_attention_rejects_mismatched_mask(device):
    q, k, v, mask, _ = _head_inputs(device, 2, 63)
    with pytest.raises(RuntimeError):
        pool_attention(q, k, v, mask[:1], SCALE, False)


def test_pool_attention_scale_is_dendritic_gain():
    assert SCALE == math.sqrt(DK)
