"""ops.triton.pool_bias: the pool attention's induction term and additive mask (training).

At B 8, T 1024 (and a short row), forward and backward, against the torch references.  Every comparison is the
per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
import pytest
import torch

from neuronspark.ops.triton.pool_bias import (pool_bias_mask, pool_bias_mask_ref, pool_induction_term,
                                              pool_induction_term_ref)
from neuronspark.utils.precision import regime
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

H, ORDERS = 6, 3
SHAPES = [(8, 1024), (2, 63)]


def _matches(device, b, t):
    """three causal bool match matrices (strictly past, sparse), nested as n-gram matches are (m4 <= m3 <= m2)."""
    past = torch.ones(t, t, device=device, dtype=torch.bool).tril(-1)
    m2 = (torch.rand((b, t, t), device=device) < 0.05) & past
    m3 = m2 & (torch.rand((b, t, t), device=device) < 0.5)
    m4 = m3 & (torch.rand((b, t, t), device=device) < 0.5)
    return [m2.contiguous(), m3.contiguous(), m4.contiguous()]


def _beta_full(device):
    return 0.5 * rand_tensor((ORDERS * H,), torch.float32, device)   # pool_ind_beta (18,), viewed (orders, heads)


# ------------------------------------------------------------------------------------------------ induction term
def _ind_run(fn, beta_full, ms, head, g):
    beta_full = beta_full.clone().requires_grad_(True)
    with regime("train"):
        x = fn(beta_full.view(ORDERS, H)[:, head].to(torch.float32), ms)
    x.backward(g)
    return {"x": x.detach(), "dbeta": beta_full.grad}


@pytest.mark.parametrize("b,t", SHAPES, ids=[f"b{b}_t{t}" for b, t in SHAPES])
@pytest.mark.parametrize("head", [0, H - 1])
def test_pool_induction_term_vs_reference(device, b, t, head):
    ms, beta = _matches(device, b, t), _beta_full(device)
    g = rand_tensor((b, t, t), torch.float32, device)
    assert_all_match(_ind_run(pool_induction_term, beta, ms, head, g),
                     _ind_run(pool_induction_term_ref, beta, ms, head, g), prefix="pool_induction_term")


# ------------------------------------------------------------------------------------------------ mask
def _mask_inputs(device, b, t, with_x=True):
    la = -0.05 * torch.rand((t, b), device=device)                    # gamma_h * mean log a: negative increments
    cum = la.cumsum(dim=0)
    x = (pool_induction_term_ref(0.5 * rand_tensor((ORDERS,), torch.float32, device), _matches(device, b, t))
         if with_x else None)
    dm = rand_tensor((b, 1, t, t), torch.bfloat16, device)
    return cum, x, dm


def _finite(m):
    """the mask with its -inf entries (the non-causal support) as -1e4: the rule's inf / nan check then applies to the
    computed entries only, and a misplaced -inf shows as a 1e4 difference."""
    m = m.float()
    return torch.where(torch.isinf(m), torch.sign(m) * 1e4, m)


def _mask_run(fn, cum, x, dm):
    cum = cum.clone().requires_grad_(True)
    xx = None if x is None else x.clone().requires_grad_(True)
    T_, B_ = (int(s) for s in cum.shape)
    with regime("train"):
        m = fn(cum, xx, T_, B_)
    m.backward(dm)
    out = {"mask": _finite(m.detach()), "dcum": cum.grad}
    if xx is not None:
        out["dx"] = xx.grad
    return out


@pytest.mark.parametrize("b,t", SHAPES, ids=[f"b{b}_t{t}" for b, t in SHAPES])
@pytest.mark.parametrize("with_x", [True, False], ids=["with_term", "no_term"])
def test_pool_bias_mask_vs_reference(device, b, t, with_x):
    cum, x, dm = _mask_inputs(device, b, t, with_x)
    with regime("train"):
        raw = pool_bias_mask(cum, x, int(cum.shape[0]), int(cum.shape[1]))
    assert raw.dtype == torch.bfloat16 and raw.is_contiguous() and tuple(raw.shape) == (b, 1, t, t)
    new = _mask_run(pool_bias_mask, cum, x, dm)
    ref = _mask_run(pool_bias_mask_ref, cum, x, dm)
    assert_all_match(new, ref, prefix="pool_bias_mask")
