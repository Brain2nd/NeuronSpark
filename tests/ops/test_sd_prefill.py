"""ops.sd_prefill: the SD prefill scan from a carried state, and its fused kernels.

* the fused chunk states + state passing kernel (``ops.triton.sd_states.csp_prefill``) against its reference
  ``csp_prefill_ref`` and against the vendored mamba_ssm pair computing the same quantities (``_chunk_state_fwd`` +
  ``_state_passing_fwd`` at the prefill configurations of ``ops.triton.mamba``), from a carried state and from the
  zero state (a first call);
* the chunk-scan forward of the op (the Gluon kernel for whole-chunk calls, the tensor-descriptor kernel otherwise)
  against the reference ``sd_prefill_chunk_scan_ref`` and against the tensor-descriptor kernel on the same stage
  inputs;
* those two at the prefill shapes b x T in {1 x 1024, 8 x 1024, 8 x 896}, three random sets each, with the inputs of
  the layer's call: x / B / C views of one (b, T, 9472) conv output, dt the strided slice of a (b, T, 18832)
  projection, A = -exp(0.5 randn), D randn, dt_bias 0.5 randn, the initial state 0.1 randn (fp32) or zero;
* the op against the pure-torch references stage by stage on smaller shapes (both chunk-scan routes), each stage
  reference evaluated on the op's own inputs to that stage and carrying the kernels' bf16 rounding points: the chunk
  quantities (dt, the chunk cumsum of dt A, CB; fp32), the states entering each chunk (bf16) and final_state (fp32)
  from the op's chunk quantities, and y from the op's chunk quantities and states (why stage by stage:
  ``neuronspark.ops.sd_prefill.sd_prefill_chunk_scan_ref``);
* a prompt split at multiples of the chunk size over several calls with the carried state against the single call;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.sd_prefill import (
    sd_chunk_scan_prefill,
    sd_prefill_chunk_quantities_ref,
    sd_prefill_chunk_scan_ref,
    sd_prefill_stage_tensors,
    sd_prefill_states_ref,
)
from neuronspark.ops.triton.mamba import (
    CHUNK_STATE_FWD_PREFILL,
    STATE_PASSING_FWD_PREFILL,
    _chunk_state_fwd,
    _state_passing_fwd,
)
from neuronspark.ops.triton.sd_scan import chunk_scan_fwd_tma
from neuronspark.ops.triton.sd_states import csp_prefill, csp_prefill_ref
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N = 144, 64, 128
DI = H * P
CONV = DI + 2 * N
WIDTH = 2 * DI + 2 * N + H
CHUNK = 256
CHECK_SHAPES = [(1, 1024), (8, 1024), (8, 896)]
CHECK_SEEDS = [0, 1, 2]
INIT_KINDS = ["carried", "zero"]
REF_SHAPES = [(1, 2), (1, 3), (2, 63), (2, 64), (1, 256), (2, 300), (2, 512)]


def operands(b, T, device, state_kind="bf16"):
    """The scan's inputs in the layer's layout: the conv output ``xbc`` (b, T, 9472) bf16, the raw step sizes ``dt``
    (b, T, 144) bf16, A = -exp(0.5 randn), D, dt_bias, and the initial state (0.1 randn; rounded to bf16 values for
    ``state_kind == "bf16"``)."""
    xbc = rand_tensor((b, T, CONV), torch.bfloat16, device)
    dt = rand_tensor((b, T, H), torch.bfloat16, device)
    A = -torch.exp(0.5 * rand_tensor((H,), torch.float32, device))
    D = rand_tensor((H,), torch.float32, device)
    dt_bias = 0.5 * rand_tensor((H,), torch.float32, device)
    init = 0.1 * rand_tensor((b, H, P, N), torch.float32, device)
    if state_kind == "bf16":
        init = init.to(torch.bfloat16).float()
    return {"xbc": xbc, "dt": dt, "A": A, "D": D, "dt_bias": dt_bias, "init": init}


def check_operands(b, T, seed, init_kind, device):
    """The prefill-shape operand sets: drawn on the CPU from a generator seeded per (seed, b, T) in the order conv
    output, dt, A, D, dt_bias, initial state; ``init_kind == "zero"`` replaces the initial state by zeros."""
    g = torch.Generator(device="cpu").manual_seed(seed * 7919 + b * 131 + T)
    xbc = torch.randn((b, T, CONV), generator=g).to(torch.bfloat16).to(device)
    dt = torch.randn((b, T, H), generator=g).to(torch.bfloat16).to(device)
    A = (-torch.exp(torch.randn((H,), generator=g) * 0.5)).to(device)
    D = torch.randn((H,), generator=g).to(device)
    dt_bias = (torch.randn((H,), generator=g) * 0.5).to(device)
    init = (torch.randn((b, H, P, N), generator=g) * 0.1).to(device)
    if init_kind == "zero":
        init = torch.zeros_like(init)
    return {"xbc": xbc, "dt": dt, "A": A, "D": D, "dt_bias": dt_bias, "init": init}


def views(o):
    """(x, dt, Bm, Cm): row-strided views of the conv output and of a full projection (row stride 18832)."""
    xbc = o["xbc"]
    b, T, _ = xbc.shape
    proj = torch.zeros((b, T, WIDTH), dtype=torch.bfloat16, device=xbc.device)
    proj[..., WIDTH - H:] = o["dt"]
    x = xbc[..., :DI].view(b, T, H, P)
    Bm = xbc[..., DI:DI + N].view(b, T, 1, N)
    Cm = xbc[..., DI + N:].view(b, T, 1, N)
    return x, proj[..., WIDTH - H:], Bm, Cm


def _op(o, impl=sd_chunk_scan_prefill):
    x, dt, Bm, Cm = views(o)
    with torch.no_grad():
        y, final = impl(x, dt, o["A"], Bm, Cm, o["D"], o["dt_bias"], o["init"], chunk_size=CHUNK)
    return {"y": y, "final_state": final}


def _stages(o):
    x, dt, Bm, Cm = views(o)
    with torch.no_grad():
        st = sd_prefill_stage_tensors(x, dt, o["A"], Bm, Cm, o["D"], o["dt_bias"], o["init"], chunk_size=CHUNK)
    return x, dt, Bm, Cm, st


def _mamba_states(Bm, x, dt_out, dA_cumsum, init):
    """The vendored mamba_ssm chunk states (fp32) and state passing (from ``init``, states in bf16) at the prefill
    configurations."""
    b, T, nheads, headdim = x.shape
    dstate = Bm.shape[-1]
    pre = _chunk_state_fwd(Bm, x, dt_out, dA_cumsum, seq_idx=None, states_in_fp32=True, config=CHUNK_STATE_FWD_PREFILL)
    nc = int(pre.shape[1])
    states, final = _state_passing_fwd(pre.reshape(b, nc, nheads, headdim * dstate), dA_cumsum[:, :, :, -1],
                                       initial_states=init.reshape(b, nheads, headdim * dstate), seq_idx=None,
                                       chunk_size=CHUNK, out_dtype=torch.bfloat16, config=STATE_PASSING_FWD_PREFILL)
    return {"states": states.reshape(b, nc, nheads, headdim, dstate),
            "final_state": final.reshape(b, nheads, headdim, dstate)}


# ------------------------------------------------------------------------------------------------ prefill shapes
@pytest.mark.parametrize("b,T", CHECK_SHAPES)
@pytest.mark.parametrize("seed", CHECK_SEEDS)
@pytest.mark.parametrize("init_kind", INIT_KINDS)
def test_csp_prefill_vs_reference_and_mamba(device, b, T, seed, init_kind):
    o = check_operands(b, T, seed, init_kind, device)
    x, dt, Bm, Cm, st = _stages(o)
    with torch.no_grad():
        states, final = csp_prefill(Bm, x, st["dt_out"], st["dA_cumsum"], o["init"], torch.bfloat16)
        mamba = _mamba_states(Bm, x, st["dt_out"], st["dA_cumsum"], o["init"])
    new = {"states": states, "final_state": final}
    assert states.dtype == torch.bfloat16 and final.dtype == torch.float32
    tag = f"b{b}_T{T}_s{seed}_{init_kind}"
    assert_all_match(new, {"states": st["states"], "final_state": st["final_state"]}, prefix=f"csp_op_{tag}")
    assert_all_match(new, mamba, prefix=f"csp_vs_mamba_{tag}")
    states_ref, final_ref = csp_prefill_ref(Bm, x, st["dt_out"], st["dA_cumsum"], o["init"], torch.bfloat16)
    assert_all_match(new, {"states": states_ref, "final_state": final_ref}, prefix=f"csp_vs_ref_{tag}")


@pytest.mark.parametrize("b,T", CHECK_SHAPES)
@pytest.mark.parametrize("seed", CHECK_SEEDS)
@pytest.mark.parametrize("init_kind", INIT_KINDS)
def test_chunk_scan_vs_reference_and_tma(device, b, T, seed, init_kind):
    o = check_operands(b, T, seed, init_kind, device)
    x, dt, Bm, Cm, st = _stages(o)
    with torch.no_grad():
        y_tma = chunk_scan_fwd_tma(st["CB"], x, st["dt_out"], st["dA_cumsum"], Cm, st["states"], o["D"])
    assert st["y"].dtype == torch.bfloat16 and st["y"].is_contiguous()
    tag = f"b{b}_T{T}_s{seed}_{init_kind}"
    assert_all_match({"y": st["y"]}, {"y": y_tma}, prefix=f"scan_vs_tma_{tag}")
    y_ref = sd_prefill_chunk_scan_ref(x, Cm, o["D"], st["dt_out"], st["dA_cumsum"], st["CB"], st["states"])
    assert_all_match({"y": st["y"]}, {"y": y_ref}, prefix=f"scan_vs_ref_{tag}")


# ------------------------------------------------------------------------------------------------ stage references
@pytest.mark.parametrize("b,T", REF_SHAPES)
def test_prefill_scan_vs_reference(device, b, T):
    o = operands(b, T, device, "bf16")
    x, dt, Bm, Cm, st = _stages(o)
    new = _op(o)
    # stages 1 and 3: the chunk quantities
    names = ("dt_out", "dA_cumsum", "CB")
    q_ref = sd_prefill_chunk_quantities_ref(dt, o["A"], Bm, Cm, o["dt_bias"], chunk_size=CHUNK)
    assert_all_match({k: st[k] for k in names}, dict(zip(names, q_ref)), prefix=f"prefill_chunkq_b{b}_T{T}")
    # stage 2 on the op's stage-1 results: the states entering each chunk and the final state
    states_ref, final_ref = sd_prefill_states_ref(x, Bm, o["init"], st["dt_out"], st["dA_cumsum"], out_dtype=Cm.dtype)
    assert_all_match({"states": st["states"]}, {"states": states_ref}, prefix=f"prefill_states_b{b}_T{T}")
    # stage 4 on the op's stage 1 / 2 / 3 results
    y_ref = sd_prefill_chunk_scan_ref(x, Cm, o["D"], st["dt_out"], st["dA_cumsum"], st["CB"], st["states"])
    assert_all_match(new, {"y": y_ref, "final_state": final_ref}, prefix=f"prefill_ref_b{b}_T{T}")


@pytest.mark.parametrize("b,splits", [(1, (256, 256, 512)), (8, (512, 512)), (2, (256, 44))])
def test_prefill_split_at_chunk_multiples(device, b, splits):
    """Calls split at multiples of 256 with the carried state reproduce the single call (the same association)."""
    T = sum(splits)
    o = operands(b, T, device, "bf16")
    whole = _op(o)
    x, dt, Bm, Cm = views(o)
    state = o["init"]
    ys = []
    t0 = 0
    with torch.no_grad():
        for L in splits:
            sl = slice(t0, t0 + L)
            y, state = sd_chunk_scan_prefill(x[:, sl], dt[:, sl], o["A"], Bm[:, sl], Cm[:, sl], o["D"], o["dt_bias"],
                                             state, chunk_size=CHUNK)
            ys.append(y)
            t0 += L
    assert_all_match({"y": torch.cat(ys, dim=1), "final_state": state}, whole, prefix=f"prefill_split_b{b}")


@pytest.mark.parametrize("b,T", [(1, 1024), (8, 1024), (2, 300)], ids=["b1_t1024", "b8_t1024", "b2_t300"])
def test_prefill_scan_bf16_initial_state(device, b, T):
    """A bf16 carried state handed to the scan as is (the state-passing kernel widens it on load) gives the outputs of its
    fp32 upcast."""
    o = operands(b, T, device)
    o16 = dict(o, init=o["init"].to(torch.bfloat16))
    assert_all_match(_op(o16), _op(o), prefix="bf16_initial_state")


def test_inputs_unchanged(device):
    for T in (300, 512):
        o = operands(2, T, device, "fp32")
        before = {k: v.clone() for k, v in o.items()}
        _op(o)
        for k in o:
            assert_all_match({k: o[k]}, {k: before[k]}, prefix=f"unchanged_T{T}")


def test_refusals(device):
    o = operands(1, 64, device)
    x, dt, Bm, Cm = views(o)
    A, D, dtb, init = o["A"], o["D"], o["dt_bias"], o["init"]
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            sd_chunk_scan_prefill(x, dt, A, Bm, Cm, D, dtb, init, chunk_size=128)              # chunk size
        with pytest.raises(RuntimeError):
            sd_chunk_scan_prefill(x, dt, A, Bm, Cm, D, dtb, init.to(torch.float16), chunk_size=CHUNK)    # fp16 state
        with pytest.raises(RuntimeError):
            sd_chunk_scan_prefill(x.float(), dt, A, Bm, Cm, D, dtb, init, chunk_size=CHUNK)    # fp32 x
        with pytest.raises(RuntimeError):
            sd_chunk_scan_prefill(x, dt[..., :H - 1], A, Bm, Cm, D, dtb, init, chunk_size=CHUNK)   # dt width
        with pytest.raises(RuntimeError):
            sd_chunk_scan_prefill(x, dt, A, Bm.expand(1, 64, 2, N), Cm.expand(1, 64, 2, N), D, dtb, init,
                                  chunk_size=CHUNK)                                            # ngroups 2
    A_req = A.clone().requires_grad_(True)
    with pytest.raises(RuntimeError):
        sd_chunk_scan_prefill(x, dt, A_req, Bm, Cm, D, dtb, init, chunk_size=CHUNK)            # autograd requested
    # more than 64 chunks in one call
    o_long = operands(1, 64 * CHUNK + 1, device)
    x, dt, Bm, Cm = views(o_long)
    with torch.no_grad(), pytest.raises(RuntimeError):
        sd_chunk_scan_prefill(x, dt, o_long["A"], Bm, Cm, o_long["D"], o_long["dt_bias"], o_long["init"],
                              chunk_size=CHUNK)
