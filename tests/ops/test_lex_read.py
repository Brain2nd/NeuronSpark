"""ops.triton.lex_read: the lexical read of a prefill call after its projections.

``lex_read_prefill`` (one kernel per call: the 64-wide q / k RMS norms with zero-centred gains, the gate
sigmoid(q . k / sqrt(64) + beta), the fp32 residual h + alpha o) against its torch chain ``lex_read_prefill_ref`` at
the default configuration's shapes: h (T, B, 1536) fp32, q_raw / k_raw (T, B, 64) bf16, o (T, B, 1536) bf16, fp32 gains,
beta in {-2.0 (the configured gate bias), 0.0, 3.0}; T x B in {2 x 1, 16 x 8, 128 x 8, 1024 x 1, 1024 x 8, 2048 x 8, 896
x 8}. A B-major residual stream (the (T, B, D) view of a (B, T, D) tensor) is checked too.

Compared by the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.lex_read import lex_read_prefill, lex_read_prefill_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_match

pytestmark = pytest.mark.cuda

D, K, EPS = 1536, 64, 1e-6
SHAPES = ((2, 1), (16, 8), (128, 8), (1024, 1), (1024, 8), (2048, 8), (896, 8))
BETAS = (-2.0, 0.0, 3.0)
SEEDS = (0, 1)


def _inputs(T: int, B: int, seed: int, beta0: float, device, b_major: bool = False):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if b_major:
        h = (torch.randn((B, T, D), generator=g) * 4.0).to(device).transpose(0, 1)
    else:
        h = (torch.randn((T, B, D), generator=g) * 4.0).to(device)
    q = (torch.randn((T, B, K), generator=g) * 0.7).to(torch.bfloat16).to(device)
    k = (torch.randn((T, B, K), generator=g) * 0.7).to(torch.bfloat16).to(device)
    o = (torch.randn((T, B, D), generator=g) * 0.5).to(torch.bfloat16).to(device)
    gq = (torch.randn(K, generator=g) * 0.1).to(device)
    gk = (torch.randn(K, generator=g) * 0.1).to(device)
    beta = torch.tensor([beta0], dtype=torch.float32, device=device)
    return h, q, k, o, gq, gk, beta


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("beta0", BETAS)
@pytest.mark.parametrize("T,B", SHAPES, ids=[f"t{t}_b{b}" for t, b in SHAPES])
def test_lex_read_prefill_vs_reference(device, T, B, beta0, seed):
    args = _inputs(T, B, 100 * seed + T + B, beta0, device)
    with torch.no_grad(), regime("infer"):
        new = lex_read_prefill(*args, EPS)
        ref = lex_read_prefill_ref(*args, EPS)
    assert tuple(new.shape) == (T, B, D) and new.dtype == torch.float32 and new.is_contiguous()
    assert_match(new, ref, "lex_read")


@pytest.mark.parametrize("T,B", ((128, 8), (1024, 8)), ids=["t128_b8", "t1024_b8"])
def test_lex_read_prefill_b_major_stream(device, T, B):
    args = _inputs(T, B, 7 + T, -2.0, device, b_major=True)
    with torch.no_grad(), regime("infer"):
        new = lex_read_prefill(*args, EPS)
        ref = lex_read_prefill_ref(*args, EPS)
    assert_match(new, ref, "lex_read_b_major")
