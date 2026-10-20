"""ops.triton.vr_lambda: the value-residual mix of a prefill call; and the prefill PLIF gate on the bf16 projection.

* ``vr_lambda_prefill`` (one kernel: lam + tanh(f32(gl)) repeated over each head's channels) against its torch chain
  ``vr_lambda_prefill_ref``: gl (T, B, 6) bf16, lam (1536,) fp32;
* ``plif_gate_prefill`` reading the bf16 eta projection (widened on load, the prefill form of the neuron) against the
  same kernel and the reference on the projection widened to fp32 beforehand: x (T, B, 1536) fp32 / bf16.

Shapes T x B in {2 x 1, 16 x 8, 128 x 8, 1024 x 1, 1024 x 8, 2048 x 8, 896 x 8}.  Compared by the per-operator rule of
``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.plif import plif_gate_prefill, plif_gate_prefill_ref
from neuronspark.ops.triton.vr_lambda import vr_lambda_prefill, vr_lambda_prefill_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match

pytestmark = pytest.mark.cuda

C, H = 1536, 6
SHAPES = ((2, 1), (16, 8), (128, 8), (1024, 1), (1024, 8), (2048, 8), (896, 8))


@pytest.mark.parametrize("T,B", SHAPES, ids=[f"t{t}_b{b}" for t, b in SHAPES])
def test_vr_lambda_prefill_vs_reference(device, T, B):
    g = torch.Generator(device="cpu").manual_seed(7 + T)
    gl = (torch.randn((T, B, H), generator=g) * 1.2).to(torch.bfloat16).to(device)
    lam = (torch.randn(C, generator=g) * 0.4).to(device)
    with torch.no_grad(), regime("infer"):
        new = vr_lambda_prefill(gl, lam)
        ref = vr_lambda_prefill_ref(gl, lam)
    assert tuple(new.shape) == (T, B, C) and new.dtype == torch.float32 and new.is_contiguous()
    assert_match(new, ref, "vr_lambda")


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=["x_fp32", "x_bf16"])
@pytest.mark.parametrize("T,B", SHAPES, ids=[f"t{t}_b{b}" for t, b in SHAPES])
def test_plif_gate_prefill_bf16_projection(device, T, B, x_dtype):
    g = torch.Generator(device="cpu").manual_seed(T + B)
    x = (torch.randn((T, B, C), generator=g) * 2.0).to(x_dtype).to(device)
    w = (torch.randn(C, generator=g) * 0.3).to(device)
    b = (torch.randn(C, generator=g) * 0.5 - 1.0).to(device)
    z = (torch.randn((T, B, C), generator=g) * 1.5).to(torch.bfloat16).to(device)
    with torch.no_grad(), regime("infer"):
        new = plif_gate_prefill(x, w, b, z)
        widened = plif_gate_prefill(x, w, b, z.to(torch.float32))
        ref = plif_gate_prefill_ref(x, w, b, z.to(torch.float32))
    assert_all_match({"a": new[0], "drive": new[1]}, {"a": widened[0], "drive": widened[1]}, prefix="gate_vs_widened")
    assert_all_match({"a": new[0], "drive": new[1]}, {"a": ref[0], "drive": ref[1]}, prefix="gate_vs_reference")
