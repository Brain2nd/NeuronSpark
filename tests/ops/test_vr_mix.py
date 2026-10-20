"""ops.triton.vr_mix: the value-residual mix and GLU tail of a prefill call.

``vr_mix_prefill`` (one kernel) against its torch chain ``vr_mix_prefill_ref`` for hippocampal layer 0 (``v (1 + lam)``)
and a later layer (``v + lam carry``): v / carry (T, B, 1536) bf16 contiguous and as views of a (B, 3 x 1536, T)
buffer (token axis contiguous), gl (T, B, 6) bf16, lam0 (1536,) fp32, glu_add (T, B, 256) fp32; three value scales.

Shapes T x B in {2 x 1, 37 x 8, 128 x 8, 1024 x 1, 1024 x 8, 2048 x 8}.  Compared by the per-operator rule of
``neuronspark.utils.testing`` (matched ratio 1.0); the refusals outside the contract.
"""
import pytest
import torch

from neuronspark.ops.triton.vr_mix import vr_mix_prefill, vr_mix_prefill_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_match

pytestmark = pytest.mark.cuda

C, H, DV = 1536, 6, 256
SHAPES = ((2, 1), (37, 8), (128, 8), (1024, 1), (1024, 8), (2048, 8))


def _values(g, T, B, scale, layout, device):
    if layout == "conv_view":
        buf = (torch.randn((B, 3 * C, T), generator=g) * scale).to(torch.bfloat16).to(device)
        return buf[:, 2 * C:, :].permute(2, 0, 1)
    return (torch.randn((T, B, C), generator=g) * scale).to(torch.bfloat16).to(device)


@pytest.mark.parametrize("scale", [1.0, 0.01, 30.0])
@pytest.mark.parametrize("layout", ["contiguous", "conv_view"])
@pytest.mark.parametrize("first", [True, False], ids=["layer0", "later"])
@pytest.mark.parametrize("T,B", SHAPES, ids=[f"t{t}_b{b}" for t, b in SHAPES])
def test_vr_mix_prefill_vs_reference(device, T, B, first, layout, scale):
    g = torch.Generator(device="cpu").manual_seed(11 * T + B)
    v = _values(g, T, B, scale, layout, device)
    carry = None if first else _values(g, T, B, scale, layout, device)
    gl = (torch.randn((T, B, H), generator=g) * 1.5).to(torch.bfloat16).to(device)
    lam0 = (torch.randn(C, generator=g) * 0.4).to(device)
    glu_add = (torch.randn((T, B, DV), generator=g) * scale).to(device)
    with torch.no_grad(), regime("infer"):
        new = vr_mix_prefill(v, carry, gl, lam0, glu_add)
        ref = vr_mix_prefill_ref(v, carry, gl, lam0, glu_add)
    assert tuple(new.shape) == (T, B, C) and new.dtype == torch.bfloat16 and new.is_contiguous()
    assert_match(new, ref, "vr_mix")


def test_vr_mix_prefill_refusals(device):
    T, B = 4, 2
    v = torch.zeros((T, B, C), dtype=torch.bfloat16, device=device)
    gl = torch.zeros((T, B, H), dtype=torch.bfloat16, device=device)
    lam0 = torch.zeros(C, device=device)
    ga = torch.zeros((T, B, DV), device=device)
    with pytest.raises(RuntimeError, match="v must be bf16"):
        vr_mix_prefill(v.float(), None, gl, lam0, ga)
    with pytest.raises(RuntimeError, match="carry must be bf16"):
        vr_mix_prefill(v, v.float(), gl, lam0, ga)
    with pytest.raises(RuntimeError, match="channels for"):
        vr_mix_prefill(v, None, torch.zeros((T, B, 7), dtype=torch.bfloat16, device=device), lam0, ga)
    with pytest.raises(RuntimeError, match="lam0 must be"):
        vr_mix_prefill(v, None, gl, lam0.double(), ga)
    with pytest.raises(RuntimeError, match="glu_add must be"):
        vr_mix_prefill(v, None, gl, lam0, ga.to(torch.bfloat16))
