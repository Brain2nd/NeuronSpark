"""ops.triton.mem_rope: interleaved RoPE of the memory layer.

* ``silu_rope`` on the q / k slices of the short conv's T-major output (T 1024, B 8, C 1536), forward and backward;
* ``rope_paired`` on an extra query's (T, B, 256) projection with the last head's table slice, forward and backward;
* ``rope_interleaved`` (prefill) at b in {1, 8} and T in {2, 3, 63, 64, 256, 1024, 2048};

each against its torch reference.  Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched
ratio 1.0).
"""
import pytest
import torch

from neuronspark.modules.rope import rope_head_band_tables
from neuronspark.ops.triton.mem_rope import (rope_interleaved, rope_interleaved_ref, rope_paired, rope_paired_ref,
                                             silu_rope, silu_rope_ref)
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

T, B, D, H, DK = 1024, 8, 1536, 6, 256


def _tables(device, t):
    cos, sin = rope_head_band_tables(D, H, 8192, 1e4, device=device)
    return cos[:t].to(torch.bfloat16).unsqueeze(1), sin[:t].to(torch.bfloat16).unsqueeze(1)   # (T, 1, 768) bf16


# ------------------------------------------------------------------------------------------------ silu_rope
def _qkv(device):
    return rand_tensor((T, B, 3 * D), torch.bfloat16, device)


def _srope_run(fn, qkv, cos, sin, g, part):
    qkv = qkv.clone().requires_grad_(True)
    x = qkv[..., part * D:(part + 1) * D]
    with regime("train"):
        y = fn(x, cos, sin)
    y.backward(g)
    return {"y": y.detach(), "dqkv": qkv.grad}


@pytest.mark.parametrize("part", [0, 1], ids=["q", "k"])
def test_silu_rope_vs_reference(device, part):
    qkv = _qkv(device)
    cos, sin = _tables(device, T)
    g = rand_tensor((T, B, D), torch.bfloat16, device)
    new = _srope_run(silu_rope, qkv, cos, sin, g, part)
    ref = _srope_run(silu_rope_ref, qkv, cos, sin, g, part)
    assert_all_match(new, ref, prefix="silu_rope")


def test_silu_rope_two_dim_tables(device):
    qkv = _qkv(device)
    cos, sin = _tables(device, T)
    g = rand_tensor((T, B, D), torch.bfloat16, device)
    a = _srope_run(silu_rope, qkv, cos, sin, g, 0)
    b = _srope_run(silu_rope, qkv, cos[:, 0], sin[:, 0], g, 0)
    assert_all_match(a, b, prefix="silu_rope_2d_tables")


# ------------------------------------------------------------------------------------------------ rope_paired
def _xq_tables(device):
    cos, sin = _tables(device, T)
    lo = (H - 1) * (DK // 2)
    return cos[..., lo:lo + DK // 2], sin[..., lo:lo + DK // 2]


def _rope_run(fn, x, cos, sin, g):
    x = x.clone().requires_grad_(True)
    with regime("train"):
        y = fn(x, cos, sin)
    y.backward(g)
    return {"y": y.detach(), "dx": x.grad}


def test_rope_paired_vs_reference(device):
    x = rand_tensor((T, B, DK), torch.bfloat16, device)
    cos, sin = _xq_tables(device)
    g = rand_tensor((T, B, DK), torch.bfloat16, device)
    assert_all_match(_rope_run(rope_paired, x, cos, sin, g), _rope_run(rope_paired_ref, x, cos, sin, g),
                     prefix="rope_paired")


# ------------------------------------------------------------------------------------------------ rope_interleaved
PREFILL = [(b, t) for b in (1, 8) for t in (2, 3, 63, 64, 256, 1024, 2048)]


@pytest.mark.parametrize("b,t", PREFILL, ids=[f"b{b}_t{t}" for b, t in PREFILL])
def test_rope_interleaved_vs_reference(device, b, t):
    x = rand_tensor((t, b, D), torch.bfloat16, device)
    cos, sin = _tables(device, t)
    with torch.no_grad():
        assert_all_match(rope_interleaved(x, cos, sin), rope_interleaved_ref(x, cos, sin), prefix="rope_interleaved")
