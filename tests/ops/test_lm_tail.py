"""ops.triton.lm_tail: the fused readout balance + cross entropy of the training loss.

* against the pure-torch reference ``balance_ce_per_token_ref`` (the balance as bf16 torch ops + ``F.cross_entropy``
  on the fp32 copy, differentiated by autograd): per-token loss, logits gradient and both contribution gradients;
* a logits view with a row stride larger than V against the dense layout;
* the input refusals.

Shapes: R = 2048 rows (the torch reference's memory bounds the row count), V = 151669, widths (16, 32).  Every
comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.lm_tail import IGNORE_INDEX, balance_ce_per_token, balance_ce_per_token_ref
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, assert_match, rand_tensor

pytestmark = pytest.mark.cuda

V = 151669
WIDTHS = (16, 32)
R_REF = 2048
# (name, logits scale, fraction of ignored labels, peaked target logit)
VARIANTS = [("plain", 1.0, 0.0, False), ("ignored", 1.0, 0.1, False), ("peaked", 4.0, 0.0, True)]


def _inputs(R, device, seed, scale=1.0, ignore_frac=0.0, peaked=False):
    g = torch.Generator(device=device).manual_seed(int(seed))
    logits = rand_tensor((R, V), torch.float32, device, g) * scale
    labels = torch.randint(0, V, (R,), device=device, generator=g)
    if peaked:
        logits[torch.arange(R, device=device), labels] += 20.0
    if ignore_frac:
        labels[torch.rand((R,), device=device, generator=g) < ignore_frac] = IGNORE_INDEX
    contribs = tuple((rand_tensor((R, w), torch.float32, device, g) * 0.5).to(torch.bfloat16) for w in WIDTHS)
    up = rand_tensor((R,), torch.float32, device, g)
    return logits.to(torch.bfloat16), labels.contiguous(), contribs, up


def _run(fn, logits, labels, contribs, up):
    lg = logits.clone().requires_grad_(True)
    cs = [c.clone().requires_grad_(True) for c in contribs]
    with regime("train"):
        per = fn(lg, labels, tuple(cs), WIDTHS)
    per.backward(up)
    return {"loss": per.detach(), "dlogits": lg.grad, "dc16": cs[0].grad, "dc32": cs[1].grad}


@pytest.mark.parametrize("name,scale,ignore_frac,peaked", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_balance_ce_vs_reference(device, name, scale, ignore_frac, peaked):
    logits, labels, contribs, up = _inputs(R_REF, device, 6100, scale, ignore_frac, peaked)
    new = _run(balance_ce_per_token, logits, labels, contribs, up)
    ref = _run(balance_ce_per_token_ref, logits, labels, contribs, up)
    assert_all_match(new, ref, prefix=f"balance_ce[{name}]")


def test_balance_ce_row_strided_logits(device):
    """A (R, V) view with row stride V + 11 (unit column stride) gives the dense layout's values."""
    logits, labels, contribs, up = _inputs(R_REF, device, 6200)
    buf = torch.zeros((R_REF, V + 11), dtype=torch.bfloat16, device=device)
    buf[:, :V].copy_(logits)
    view = buf[:, :V]
    with torch.no_grad(), regime("train"):
        dense = balance_ce_per_token(logits, labels, contribs, WIDTHS)
        strided = balance_ce_per_token(view, labels, contribs, WIDTHS)
    assert_match(strided, dense, "balance_ce[row stride V + 11]")


def test_balance_ce_forward_without_grad(device):
    logits, labels, contribs, _ = _inputs(R_REF, device, 6300, ignore_frac=0.1)
    with torch.no_grad(), regime("train"):
        new = balance_ce_per_token(logits, labels, contribs, WIDTHS)
        ref = balance_ce_per_token_ref(logits, labels, contribs, WIDTHS)
    assert_match(new, ref, "balance_ce[forward]")


def test_balance_ce_refusals(device):
    logits, labels, contribs, _ = _inputs(64, device, 6400)
    with pytest.raises(RuntimeError):
        balance_ce_per_token(logits.float(), labels, contribs, WIDTHS)                # fp32 logits
    with pytest.raises(RuntimeError):
        balance_ce_per_token(logits, labels.int(), contribs, WIDTHS)                  # int32 labels
    with pytest.raises(RuntimeError):
        balance_ce_per_token(logits, labels, contribs[:1], WIDTHS)                    # widths / contributions differ
    with pytest.raises(RuntimeError):
        balance_ce_per_token(logits.t().contiguous().t(), labels, contribs, WIDTHS)   # column-strided logits
