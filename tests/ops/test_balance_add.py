"""ops.triton.balance_add: the readout balance of the inference and scoring readout in one kernel.

``balance_add`` against its torch chain ``balance_add_ref``: logits (R, 151669) bf16 (dense, and a row stride above V),
contributions (R, 16) and (R, 32) bf16, widths (16, 32) and (16,).  Rows R in {1, 2, 8, 32, 256, 2048}.  Compared by
the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.balance_add import balance_add, balance_add_ref
from neuronspark.utils.testing import assert_match

pytestmark = pytest.mark.cuda

V = 151669
ROWS = (1, 2, 8, 32, 256, 2048)
WIDTHS = ((16, 32), (16,))


def _inputs(R, widths, row_pad, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = (torch.randn((R, V + row_pad), generator=g) * 4.0).to(torch.bfloat16).to(device)
    logits = base[:, :V]
    contribs = [(torch.randn((R, w), generator=g) * 0.7).to(torch.bfloat16).to(device) for w in widths]
    return logits, contribs


@pytest.mark.parametrize("row_pad", [0, 64], ids=["dense", "row_stride"])
@pytest.mark.parametrize("widths", WIDTHS, ids=["w16_32", "w16"])
@pytest.mark.parametrize("R", ROWS, ids=[f"r{r}" for r in ROWS])
def test_balance_add_vs_reference(device, R, widths, row_pad):
    logits, contribs = _inputs(R, widths, row_pad, 11 + R + len(widths), device)
    before = logits.clone()
    with torch.no_grad():
        new = balance_add(logits, contribs, widths)
        ref = balance_add_ref(logits, contribs, widths)
    assert tuple(new.shape) == (R, V) and new.dtype == torch.bfloat16 and new.is_contiguous()
    assert torch.equal(logits, before)
    assert_match(new, ref, "balance_add")
