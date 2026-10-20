"""ops.triton.sd_step: the SD decode step and the SD pre-norm rows of the inference paths.

* ``sd_step_core`` (K1 routed by the decode batch, then K2) against its pure-torch reference at every decode batch
  b in {1, 2, 4, 8, 16, 32} (all three K1 routes) and at other batches, for both SSM state dtypes (bf16, the default
  storage, and fp32), over three consecutive steps (the rolled conv tails and the carried state feed the next step):
  the outputs xs, the conv tail and the SSM state after each step; strided state rows, updated in place;
* ``sd_prenorm_rows`` against its reference, at decode row counts and prefill row counts (T * B), and on strided rows;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuronspark.ops.triton.sd_step import (
    MAX_DECODE_BATCH,
    k1_route,
    sd_prenorm_rows,
    sd_prenorm_rows_ref,
    sd_step_core,
    sd_step_core_ref,
)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

D_MODEL = 1536
H, P, N = 144, 64, 128
DI = H * P
CONV = DI + 2 * N
WIDTH = 2 * DI + 2 * N + H
EPS = 1e-5
DECODE_B = [1, 2, 4, 8, 16, 32]           # the decode batch sizes of the inference cache (every K1 route)
OTHER_B = [3, 5, 12, 24]                  # other batch sizes (the general K1 route)
STATE_DTYPES = [torch.bfloat16, torch.float32]
STEPS = 3
PRENORM_ROWS = [1, 2, 8, 32, 1024, 8192]


def params(device):
    """The mixer's step operands: conv weight / bias, A = -exp(A_log), dt_bias, D, theta = 0.02 + softplus(rho) near
    its launch value, gamma near 1 (fp32)."""
    rho = -3.4915204 + 0.5 * rand_tensor((DI,), torch.float32, device)
    return {
        "conv_w": (0.3 * rand_tensor((CONV, 4), torch.float32, device)).contiguous(),
        "conv_b": 0.1 * rand_tensor((CONV,), torch.float32, device),
        "A": -torch.exp(0.5 * rand_tensor((H,), torch.float32, device)),
        "dt_bias": 0.5 * rand_tensor((H,), torch.float32, device),
        "D": rand_tensor((H,), torch.float32, device),
        "theta": (0.02 + F.softplus(rho)).contiguous(),
        "gamma": 1.0 + 0.1 * rand_tensor((DI,), torch.float32, device),
    }


def states(b, state_dtype, device):
    """conv tail (b, 9472, 3) fp32 holding bf16 values; SSM state (b, 144, 64, 128) in ``state_dtype``."""
    conv = rand_tensor((b, CONV, 3), torch.bfloat16, device).float()
    ssm = (0.1 * rand_tensor((b, H, P, N), torch.float32, device)).to(state_dtype)
    return conv, ssm


def projections(b, device, steps=STEPS):
    """``steps`` in-projection outputs (b, 18832) bf16 = [z | xBC | dt]."""
    return [rand_tensor((b, WIDTH), torch.bfloat16, device) for _ in range(steps)]


def _run(fn, prm, conv, ssm, projs):
    conv = conv.clone()
    ssm = ssm.clone()
    out = {}
    with torch.no_grad():
        for s, proj in enumerate(projs):
            xs = fn(proj, conv, ssm, prm["conv_w"], prm["conv_b"], prm["A"], prm["dt_bias"], prm["D"], prm["theta"],
                    prm["gamma"], EPS)
            out[f"xs{s}"] = xs
            out[f"conv{s}"] = conv.clone()
            out[f"ssm{s}"] = ssm.clone()
    return out


@pytest.mark.parametrize("b", DECODE_B + OTHER_B)
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
def test_step_core_vs_reference(device, b, state_dtype):
    prm = params(device)
    conv, ssm = states(b, state_dtype, device)
    projs = projections(b, device)
    new = _run(sd_step_core, prm, conv, ssm, projs)
    ref = _run(sd_step_core_ref, prm, conv, ssm, projs)
    assert_all_match(new, ref, prefix=f"step_ref_b{b}_{k1_route(b)}")


def test_step_core_strided_state_rows(device):
    """The one-kernel routes take any state strides and update the views in place (b = 4: the general route)."""
    prm = params(device)
    conv_c, ssm_c = states(4, torch.bfloat16, device)
    conv = torch.empty((4, 3, CONV), dtype=torch.float32, device=device).transpose(1, 2)   # lag stride CONV
    conv.copy_(conv_c)
    ssm = torch.empty((4, H, P, 2 * N), dtype=torch.bfloat16, device=device)[..., :N]     # row stride 2 N
    ssm.copy_(ssm_c)
    assert not conv.is_contiguous() and not ssm.is_contiguous()
    projs = projections(4, device, steps=1)
    ref = _run(sd_step_core_ref, prm, conv.clone(), ssm.clone(), projs)
    with torch.no_grad():
        xs = sd_step_core(projs[0], conv, ssm, prm["conv_w"], prm["conv_b"], prm["A"], prm["dt_bias"], prm["D"],
                          prm["theta"], prm["gamma"], EPS)
    assert_all_match({"xs0": xs, "conv0": conv, "ssm0": ssm}, ref, prefix="step_strided")


def prenorm_operands(M, device):
    return {"x": 3.0 * rand_tensor((M, D_MODEL), torch.float32, device),
            "gain": 1.0 + 0.1 * rand_tensor((D_MODEL,), torch.float32, device)}


@pytest.mark.parametrize("M", PRENORM_ROWS)
def test_prenorm_rows_vs_reference(device, M):
    o = prenorm_operands(M, device)
    with torch.no_grad():
        assert_all_match({"hn": sd_prenorm_rows(o["x"], o["gain"], EPS)},
                         {"hn": sd_prenorm_rows_ref(o["x"], o["gain"], EPS)}, prefix="prenorm_ref")


def test_prenorm_rows_strided_input(device):
    """Rows with a row stride larger than D (one time step of a B-major (B, T, D) stream)."""
    h = rand_tensor((8, 2, D_MODEL), torch.float32, device)
    gain = 1.0 + 0.1 * rand_tensor((D_MODEL,), torch.float32, device)
    rows = h[:, 0]                                              # (8, D) with row stride 2 D
    assert rows.stride() == (2 * D_MODEL, 1)
    with torch.no_grad():
        assert_all_match({"hn": sd_prenorm_rows(rows, gain, EPS)}, {"hn": sd_prenorm_rows_ref(rows, gain, EPS)},
                         prefix="prenorm_strided")


def test_refusals(device):
    prm = params(device)
    conv, ssm = states(2, torch.bfloat16, device)
    proj = projections(2, device, steps=1)[0]
    args = (prm["conv_w"], prm["conv_b"], prm["A"], prm["dt_bias"], prm["D"], prm["theta"], prm["gamma"], EPS)
    with torch.no_grad():
        big = MAX_DECODE_BATCH + 1
        c_big, s_big = states(big, torch.bfloat16, device)
        with pytest.raises(RuntimeError):
            sd_step_core(projections(big, device, steps=1)[0], c_big, s_big, *args)             # b > 32
        with pytest.raises(RuntimeError):
            sd_step_core(proj.float(), conv, ssm, *args)                                       # fp32 proj
        with pytest.raises(RuntimeError):
            sd_step_core(proj[:, :-1], conv, ssm, *args)                                       # width
        with pytest.raises(RuntimeError):
            sd_step_core(proj, conv, ssm.half(), *args)                                        # fp16 state
        with pytest.raises(RuntimeError):
            sd_step_core(proj, conv[:1], ssm, *args)                                           # state batch
        with pytest.raises(RuntimeError):
            sd_step_core(proj, conv, ssm, prm["conv_w"].t().contiguous().t(), *args[1:])       # non-contiguous conv_w
        with pytest.raises(RuntimeError):
            sd_prenorm_rows(rand_tensor((4, D_MODEL), torch.bfloat16, device), prm["gamma"][:D_MODEL].contiguous(),
                            EPS)                                                               # bf16 x
        with pytest.raises(RuntimeError):
            sd_prenorm_rows(rand_tensor((4, D_MODEL), torch.float32, device), prm["gamma"][:100].contiguous(), EPS)
