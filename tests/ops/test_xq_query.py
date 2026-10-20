"""ops.triton.xq_query: the extra-query formation of a prefill call.

``xq_queries_prefill`` (one kernel for all M queries: SiLU, interleaved RoPE, normalisation, gain, one rounding to
bf16) against its torch chain ``xq_queries_prefill_ref`` (the memory layer's formation under the bf16 autocast of the
inference regime, fp32 queries) at the 1MIX shapes: M = 4 queries of dk = 256, bf16 projections, psi (T, B, 4) fp32
clamped to [-3, 3], a one-element bf16 head gain; T x B in {2 x 1, 16 x 8, 128 x 8, 1024 x 1, 1024 x 8, 2048 x 8,
896 x 8}.  Two table layouts: (T, 1, 128) fp32 tables of real positions shared by the rows, and the module's layout
(the last head's band of the (T, B, 768) bf16 tables gathered at per-row positions).

Compared by the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0) as a cross-dtype comparison
(bf16 output against the fp32 chain, which the ring read rounds to bf16); the output layout is the chain's.
"""
import pytest
import torch

from neuronspark.modules.rope import rope_head_band_tables
from neuronspark.ops.triton.xq_query import xq_queries_prefill, xq_queries_prefill_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_match

pytestmark = pytest.mark.cuda

M, DK, H = 4, 256, 6
SHAPES = ((2, 1), (16, 8), (128, 8), (1024, 1), (1024, 8), (2048, 8), (896, 8))
SEEDS = (0, 1)


def _inputs(T: int, B: int, seed: int, device):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    qs = [(torch.randn((T, B, DK), generator=g) * 0.8).to(torch.bfloat16).to(device).contiguous() for _ in range(M)]
    psi = (torch.randn((T, B, M), generator=g) * 1.5).clamp(-3.0, 3.0).to(device)
    gain = torch.exp(torch.tensor(0.15)).to(torch.bfloat16).to(device)
    return g, qs, psi, gain


def _shared_tables(T: int, g: torch.Generator, device):
    pos = torch.arange(T, dtype=torch.float32) + float(torch.randint(0, 3000, (), generator=g))
    inv = 1.0 / (10000.0 ** (torch.arange(DK // 2, dtype=torch.float32) / (DK // 2)))
    ang = pos[:, None] * inv[None, :]
    return torch.cos(ang).view(T, 1, DK // 2).to(device), torch.sin(ang).view(T, 1, DK // 2).to(device)


def _module_tables(T: int, B: int, g: torch.Generator, device):
    cos_t, sin_t = rope_head_band_tables(H * DK, H, 8192, 1e4, device=device)
    pos0 = torch.randint(0, 8192 - T, (B,), generator=g).to(device)
    positions = pos0.view(1, B) + torch.arange(T, device=device, dtype=torch.int64).view(T, 1)
    lo = (H - 1) * (DK // 2)
    cos = cos_t[positions].to(torch.bfloat16)[..., lo:lo + DK // 2]
    sin = sin_t[positions].to(torch.bfloat16)[..., lo:lo + DK // 2]
    return cos, sin


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("tables", ["shared_fp32", "module_bf16"])
@pytest.mark.parametrize("T,B", SHAPES, ids=[f"t{t}_b{b}" for t, b in SHAPES])
def test_xq_queries_prefill_vs_reference(device, T, B, tables, seed):
    g, qs, psi, gain = _inputs(T, B, 100 * seed + T + B, device)
    cos, sin = _shared_tables(T, g, device) if tables == "shared_fp32" else _module_tables(T, B, g, device)
    with torch.no_grad(), regime("infer"):
        new = xq_queries_prefill(qs, psi, gain, cos, sin)
        ref = xq_queries_prefill_ref(qs, psi, gain, cos, sin)
    assert tuple(new.shape) == (B, M, T, DK) and new.dtype == torch.bfloat16
    assert tuple(new.stride()) == tuple(ref.stride())
    assert_match(new, ref, "xq_queries", check_dtype=False)
