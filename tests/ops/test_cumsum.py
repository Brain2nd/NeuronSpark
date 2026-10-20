"""ops.triton.cumsum: time-axis cumulative sums and the carried running means.

* ``cumsum0`` (training): narrow (T, B) pool clocks and wide (T, B, C) running sums in the trunk's layouts, forward
  and backward, against ``torch.cumsum``; a single column takes torch's own scan;
* ``exact_cumsum0`` / ``centred_running`` (prefill): b in {1, 8}, T in {2, 3, 63, 64, 256, 1024, 2048}, against the
  torch chains (the carried sum is compared too).

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.cumsum import (centred_running, centred_running_ref, cumsum0, cumsum0_ref, exact_cumsum0,
                                           exact_cumsum0_ref)
from neuronspark.utils.testing import assert_all_match, assert_match, rand_tensor

pytestmark = pytest.mark.cuda

T, B, D = 1024, 8, 1536


def _cases(device):
    """(name, base tensor requiring grad is built from it, view function) of the training call sites."""
    return [
        ("clock_TB", -0.05 * torch.rand((T, B), device=device), lambda t: t),
        ("wide_bmajor", rand_tensor((B, T, D), torch.float32, device), lambda t: t.transpose(0, 1)),
        ("wide_contig", rand_tensor((T, B, D), torch.float32, device), lambda t: t),
        ("single_column", rand_tensor((T,), torch.float32, device), lambda t: t),
    ]


def _run(fn, base, view, g):
    base = base.clone().requires_grad_(True)
    y = fn(view(base))
    y.backward(g)
    return {"y": y.detach(), "dx": base.grad}


@pytest.mark.parametrize("idx", range(4), ids=["clock_TB", "wide_bmajor", "wide_contig", "single_column"])
def test_cumsum0_vs_reference(device, idx):
    name, base, view = _cases(device)[idx]
    g = rand_tensor(tuple(view(base).shape), torch.float32, device)
    assert_all_match(_run(cumsum0, base, view, g), _run(cumsum0_ref, base, view, g), prefix=f"cumsum0[{name}]")


def test_cumsum0_narrow_is_ordered_scan(device):
    """The narrow width keeps ATen's sequential order: the rule against torch.cumsum on a long, ill-conditioned row."""
    x = torch.randn((4096, 8), device=device) * torch.logspace(-6, 6, 4096, device=device)[:, None]
    with torch.no_grad():
        assert_match(cumsum0(x), torch.cumsum(x, dim=0), "cumsum0_narrow_long")


# ------------------------------------------------------------------------------------------------ prefill
PREFILL = [(b, t) for b in (1, 8) for t in (2, 3, 63, 64, 256, 1024, 2048)]


def _clock_rows(device, b, t):
    """the pool-clock prefill scan: cumsum(cat([carried clock, per-step increments])) over (t + 1, b * heads)."""
    return torch.cat([-3.0 * torch.rand((1, b * 6), device=device),
                      -0.05 * torch.rand((t, b * 6), device=device)], dim=0)


@pytest.mark.parametrize("b,t", PREFILL, ids=[f"b{b}_t{t}" for b, t in PREFILL])
def test_exact_cumsum0_vs_reference(device, b, t):
    for x in (_clock_rows(device, b, t), rand_tensor((t, b, D), torch.float32, device)):
        with torch.no_grad():
            assert_match(exact_cumsum0(x), exact_cumsum0_ref(x), "exact_cumsum0")


def test_exact_cumsum0_rejects_single_column(device):
    with pytest.raises(RuntimeError):
        exact_cumsum0(torch.randn((64, 1), device=device))


def _centred_inputs(device, b, t, x_dtype):
    x = rand_tensor((t, b, D), x_dtype, device)
    run_sum = rand_tensor((b, D), torch.float32, device) * 10.0
    count = torch.randint(0, 2048, (b,), device=device, dtype=torch.int64)
    count[0] = 0                                                   # a row starting fresh
    return x, run_sum, count


@pytest.mark.parametrize("b,t", PREFILL, ids=[f"b{b}_t{t}" for b, t in PREFILL])
@pytest.mark.parametrize("x_dtype,out_dtype", [(torch.float32, torch.float32), (torch.float32, torch.bfloat16),
                                               (torch.bfloat16, torch.bfloat16)], ids=["f32f32", "f32bf16", "bf16bf16"])
def test_centred_running_vs_reference(device, b, t, x_dtype, out_dtype):
    x, run_sum, count = _centred_inputs(device, b, t, x_dtype)
    rs_new, rs_ref = run_sum.clone(), run_sum.clone()
    with torch.no_grad():
        new = centred_running(x, rs_new, count, out_dtype)
        ref = centred_running_ref(x, rs_ref, count, out_dtype)
    assert new.dtype == out_dtype
    assert_all_match({"y": new, "run_sum": rs_new}, {"y": ref, "run_sum": rs_ref}, prefix="centred_running")


def test_centred_running_rejects_single_step(device):
    x, run_sum, count = _centred_inputs(device, 1, 1, torch.float32)
    with pytest.raises(RuntimeError):
        centred_running(x, run_sum, count, torch.float32)
