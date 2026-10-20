"""ops.triton.pkm_topk: the PKM product-key top-k in its three regimes.

* training ``pkm_topk_train`` (sc, u, ii, jj and the gradient of the scores) at B 8 x T 1024 x 4 heads;
* prefill ``pkm_topk_prefill`` (sc, u, ii, jj) at b in {1, 8} and the prefill lengths;
* decode ``pkm_topk_softmax_t1`` (sc, w, u, ii, jj) at b in {1, 2, 4, 8, 16, 32};

each against the pure-torch reference (three torch.topk calls, the candidate add, the gathers).
Score sets: "cosine" (the scaled unit cosines of the read), "ties" (values on a coarse grid: many exact ties, including
signed zeros, which also exercise the full-candidate path of the pruned training kernel).  Every comparison is the
per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0); integer outputs are compared under the same
rule (cast to fp32).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuronspark.ops.triton.pkm_topk import (
    pkm_beta_table,
    pkm_topk_prefill,
    pkm_topk_prefill_ref,
    pkm_topk_softmax_t1,
    pkm_topk_softmax_t1_ref,
    pkm_topk_train,
    pkm_topk_train_ref,
)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

H, N, K, DH = 4, 128, 16, 64
T_TRAIN, B_TRAIN = 1024, 8
PREFILL_B = (1, 8)
PREFILL_T = (2, 3, 63, 64, 256, 1024, 2048)
DECODE_B = (1, 2, 4, 8, 16, 32)
KINDS = ("cosine", "ties")


def _gain(device):
    log_gain = 0.1 * rand_tensor((H * 2,), torch.float32, device)
    return pkm_beta_table(log_gain, H, DH)                     # (H, 2) = 8 exp(lambda)


def _raw_scores(lead, kind, device):
    """(*lead, H, 2, N) fp32 unit sub-key cosines ("cosine") or tie-heavy values with signed zeros ("ties")."""
    if kind == "cosine":
        q = F.normalize(rand_tensor((*lead, H, 2, DH), torch.float32, device), dim=-1)
        k = F.normalize(rand_tensor((H, 2, N, DH), torch.float32, device), dim=-1)
        return torch.einsum("...had,hand->...han", q, k).contiguous()
    v = torch.round(rand_tensor((*lead, H, 2, N), torch.float32, device) * 2.0) / 8.0
    neg_zero = torch.rand(v.shape, device=device) < 0.5
    return torch.where((v == 0) & neg_zero, torch.full_like(v, -0.0), v).contiguous()


def _scaled(lead, kind, device):
    beta = _gain(device)
    view = (1,) * len(lead) + (H, 2, 1)
    return (_raw_scores(lead, kind, device) * beta.view(view)).contiguous()


# ------------------------------------------------------------------------------------------------ training
def _run_train(fn, s, g_sc):
    s = s.clone().requires_grad_(True)
    sc, u, ii, jj = fn(s, K)
    sc.backward(g_sc)
    return {"sc": sc.detach(), "u": u, "ii": ii, "jj": jj, "g_s": s.grad}


@pytest.mark.parametrize("kind", KINDS)
def test_pkm_topk_train_vs_reference(device, kind):
    s = _scaled((T_TRAIN, B_TRAIN), kind, device)
    g_sc = rand_tensor((T_TRAIN, B_TRAIN, H, K), torch.float32, device)
    new = _run_train(pkm_topk_train, s, g_sc)
    ref = _run_train(pkm_topk_train_ref, s, g_sc)
    assert_all_match(new, ref, prefix=f"train_{kind}")


def test_pkm_topk_train_strided_and_no_grad(device):
    """A strided score tensor and a call without autograd give the reference's selection."""
    s = _scaled((64, B_TRAIN), "cosine", device)
    s_strided = torch.cat([s, s], dim=-1)[..., :N]               # inner stride 1, row strides of a wider tensor
    with torch.no_grad():
        new = pkm_topk_train(s_strided, K)
        ref = pkm_topk_train_ref(s_strided, K)
    assert_all_match(dict(zip(("sc", "u", "ii", "jj"), new)), dict(zip(("sc", "u", "ii", "jj"), ref)), prefix="strided")


def test_pkm_topk_train_refusals(device):
    s = _scaled((4, 2), "cosine", device)
    with pytest.raises(RuntimeError):
        pkm_topk_train(s, 8)                                     # k != 16
    with pytest.raises(RuntimeError):
        pkm_topk_train(s[..., :64].contiguous(), K)              # n != 128
    with pytest.raises(RuntimeError):
        pkm_topk_train(s.double(), K)                            # fp64 scores


# ------------------------------------------------------------------------------------------------ prefill
PREFILL_CASES = [(f"{kind}_b{b}_t{t}", kind, b, t) for kind in KINDS for b in PREFILL_B for t in PREFILL_T]


@pytest.mark.parametrize("name,kind,b,T", PREFILL_CASES, ids=[c[0] for c in PREFILL_CASES])
def test_pkm_topk_prefill_vs_reference(device, name, kind, b, T):
    s = _scaled((T, b), kind, device)
    new = pkm_topk_prefill(s[..., 0, :], s[..., 1, :], K)
    ref = pkm_topk_prefill_ref(s[..., 0, :], s[..., 1, :], K)
    assert_all_match(dict(zip(("sc", "u", "ii", "jj"), new)), dict(zip(("sc", "u", "ii", "jj"), ref)), prefix=name)


def test_pkm_topk_prefill_refusals(device):
    s = _scaled((4, 2), "cosine", device)
    with pytest.raises(RuntimeError):
        pkm_topk_prefill(s[..., 0, :], s[..., 1, :], 8)          # k != 16
    with pytest.raises(RuntimeError):
        pkm_topk_prefill(s[..., 0, :96], s[..., 1, :96], K)       # n not a power of two
    with pytest.raises(RuntimeError):
        pkm_topk_prefill(s[..., 0, :], s[:2, ..., 1, :], K)       # halves of different shapes


# ------------------------------------------------------------------------------------------------ decode
DECODE_CASES = [(f"{kind}_b{b}", kind, b) for kind in KINDS for b in DECODE_B]


def _decode_inputs(kind, b, device):
    return _raw_scores((b,), kind, device), _gain(device)


@pytest.mark.parametrize("name,kind,b", DECODE_CASES, ids=[c[0] for c in DECODE_CASES])
def test_pkm_topk_softmax_t1_vs_reference(device, name, kind, b):
    s_raw, beta = _decode_inputs(kind, b, device)
    new = pkm_topk_softmax_t1(s_raw, beta, K)
    ref = pkm_topk_softmax_t1_ref(s_raw, beta, K)
    keys = ("sc", "w", "u", "ii", "jj")
    assert_all_match(dict(zip(keys, new)), dict(zip(keys, ref)), prefix=name)


def test_pkm_topk_softmax_t1_refusals(device):
    s_raw, beta = _decode_inputs("cosine", 2, device)
    with pytest.raises(RuntimeError):
        pkm_topk_softmax_t1(s_raw, beta.t().contiguous(), K)     # gain table (2, H)
    with pytest.raises(RuntimeError):
        pkm_topk_softmax_t1(s_raw, beta, 8)                      # k != 16
    with pytest.raises(RuntimeError):
        pkm_topk_softmax_t1(s_raw.transpose(2, 3).contiguous().transpose(2, 3), beta, K)   # inner stride != 1


# ------------------------------------------------------------------------------------------------ special values
def _special_scores(lead, kind, device):
    """cosines with NaN / +-inf entries ("nonfinite": the inference kernels' row-batched path) or half-integer halves
    ("cand_ties": equal candidate sums)"""
    if kind == "nonfinite":
        v = _raw_scores(lead, "cosine", device)
        v[..., 7] = float("nan")
        v[..., 0, 13] = float("inf")
        v[..., 1, 11] = float("-inf")
        return v.contiguous()
    return (torch.randint(-40, 40, (*lead, H, 2, N), device=device).float() * 0.5).contiguous()


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


@pytest.mark.parametrize("kind", ("nonfinite", "cand_ties"))
def test_pkm_topk_inference_exact_on_special_values(device, kind):
    """the selection outputs are torch.topk's element for element (the scores as bit patterns)"""
    s = _special_scores((256, 8), kind, device)
    new = pkm_topk_prefill(s[..., 0, :], s[..., 1, :], K)
    ref = pkm_topk_prefill_ref(s[..., 0, :], s[..., 1, :], K)
    for name, a, b in zip(("sc", "u", "ii", "jj"), new, ref):
        assert torch.equal(_bits(a.contiguous()), _bits(b.contiguous())), f"prefill {kind} {name}"
    s_raw, beta = _special_scores((32,), kind, device), _gain(device)
    new = pkm_topk_softmax_t1(s_raw, beta, K)
    ref = pkm_topk_softmax_t1_ref(s_raw, beta, K)
    for name, a, b in zip(("sc", "u", "ii", "jj"), new[:1] + new[2:], ref[:1] + ref[2:]):
        assert torch.equal(_bits(a.contiguous()), _bits(b.contiguous())), f"decode {kind} {name}"
