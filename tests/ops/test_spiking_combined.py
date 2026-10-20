"""ops.spiking_combined: ``SpikingChunkScanSomaFn`` / ``spiking_chunk_scan_soma`` -- the spiking layer's conv, SSD chunk
scan, gated norm and soma read-out as one autograd Function (training regime).

* against the pure-torch references of its operators at B 1, T 512, each fed with the Function's own input to that
  operator (the forward intermediates from the forward op, the backward stage inputs from the same stage launches the
  backward op makes): the conv, the chunk cumsum, CB, the passed states, the chunk scan, the gated norm, the soma
  read-out, the soma and gated-norm backward, the SSD backward, the conv backward.  Every output (xs) and gradient
  (zxbcdt, conv weight / bias, dt_bias, A, D, theta, gamma) of the Function and every intermediate are compared.
  (An end-to-end pure-torch chain recomputes each bf16 intermediate independently; where the fp32 values lie on
  either side of a rounding boundary the intermediate differs by one unit in the last place, and the next operator can
  carry that past the rule's relative bound.)
* autocast independence (T 512, two chunks; the state gradients need at least two);
* the call contract (full chunks, chunk 256, dtypes) and the layout of xs.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from neuronspark.ops.spiking_combined import NORM_EPS, spiking_chunk_scan_soma
from neuronspark.ops.triton.spiking_conv import spiking_conv_silu_ref
from neuronspark.ops.triton.spiking_glue import bwd_store_glue, bwd_store_glue_ref
from neuronspark.ops.triton.spiking_scan import chunk_cumsum_fwd_ref, chunk_scan_fwd_tma_ref
from neuronspark.ops.triton.spiking_soma import (
    gated_norm_soma_fwd_ref,
    gated_rmsnorm_bwd,
    gated_rmsnorm_bwd_ref,
    spiking_soma_readout_ref,
    soma_bwd,
    soma_bwd_ref,
)
from neuronspark.ops.gluon.spiking_chunk_states import chunk_states_train
from neuronspark.ops.triton.spiking_states import csp_fused_ref
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = [pytest.mark.cuda, pytest.mark.sm100]

H, P, N = 144, 64, 128
CHUNK = 256
DI = H * P
C_CONV = DI + 2 * N
WIDTH = 2 * DI + 2 * N + H
ALPHA = 8.0
NAMES = ("zxbcdt", "conv_w", "conv_b", "dt_bias", "A", "D", "theta", "gamma")


def operands(B, T, device):
    """an in-projection output (std ~0.8) and spiking-layer parameters near the launch distribution (dt_bias from the dt
    range
    [1e-3, 0.1], A = -exp(log U(1, 16)), D ~ 1, theta = 0.02 + softplus(rho), gamma ~ 1)"""
    zx = (rand_tensor((B, T, WIDTH), torch.float32, device) * 0.8)
    A = -torch.exp(torch.log(1.0 + 15.0 * torch.rand(H, device=device)))
    u = torch.rand(H, device=device)
    dt = torch.exp(u * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
    p = {
        "zxbcdt": zx.to(torch.bfloat16).contiguous(),
        "conv_w": (rand_tensor((C_CONV, 4), torch.float32, device) * 0.3).contiguous(),
        "conv_b": rand_tensor((C_CONV,), torch.float32, device) * 0.1,
        "dt_bias": (dt + torch.log(-torch.expm1(-dt))).contiguous(),
        "A": A.contiguous(),
        "D": 1.0 + 0.1 * rand_tensor((H,), torch.float32, device),
        "theta": (0.02 + F.softplus(-3.4915204 + 0.5 * rand_tensor((DI,), torch.float32, device))).contiguous(),
        "gamma": 1.0 + 0.1 * rand_tensor((DI,), torch.float32, device),
    }
    p["dxs"] = rand_tensor((B * T, DI), torch.bfloat16, device)
    return p


def _run(fn, p):
    leaves = {k: p[k].detach().clone().requires_grad_(True) for k in NAMES}
    xs = fn(*(leaves[k] for k in NAMES), alpha=ALPHA, chunk_size=256)
    xs.backward(p["dxs"])
    out = {"xs": xs.detach()}
    out.update({"d" + k: leaves[k].grad for k in NAMES})
    return out


def _per_operator(p):
    """(references of the Function's outputs, the Function's intermediates, their references): each operator of the
    Function's call sequence (spiking_combined forward / backward) against its pure-torch reference fed with the
    Function's own input to it.  The forward intermediates come from the forward op (the call the Function makes); the
    backward stage inputs (grad_n, dy, the conv-input gradient) from the stage launches the backward op makes, on the
    same
    tensors."""
    zx = p["zxbcdt"]
    dev = zx.device
    Bn, T, _ = zx.shape
    M = Bn * T
    th, ga, D, A, dt_bias = p["theta"], p["gamma"], p["D"], p["A"], p["dt_bias"]
    ones = torch.ones(DI, dtype=torch.float32, device=dev)
    z_in, xBC_in, dt_raw = torch.split(zx, [DI, C_CONV, H], dim=-1)
    z2 = z_in.view(M, DI)
    with torch.no_grad():
        xs, y, rstd, xbc, dt, dA, CB, states_passed, n = torch.ops.neuronspark.spiking_chunk_scan_soma_fwd(
            zx, p["conv_w"], p["conv_b"], dt_bias, A, D, th, ga, CHUNK)
    x = xbc[..., :DI].unflatten(-1, (H, P))
    Bm = xbc[..., DI:DI + N].unflatten(-1, (1, N))
    Cm = xbc[..., DI + N:].unflatten(-1, (1, N))
    y2 = y.view(M, DI)
    got, want = {}, {}

    def stage(name, new, ref):
        got["stage_" + name] = new
        want["stage_" + name] = ref

    # forward
    stage("conv_out", xbc, spiking_conv_silu_ref(xBC_in, p["conv_w"], p["conv_b"]))
    r_dA, r_dt = chunk_cumsum_fwd_ref(dt_raw, A, dt_bias, CHUNK)
    stage("dA_cumsum", dA, r_dA)
    stage("dt", dt, r_dt)
    nc = T // CHUNK
    Cc = Cm[:, :, 0].double().reshape(Bn, nc, CHUNK, N)
    Bc = Bm[:, :, 0].double().reshape(Bn, nc, CHUNK, N)
    stage("CB", CB, torch.einsum("bcln,bcsn->bcls", Cc, Bc)[:, :, None].float())
    with torch.no_grad():
        _, states = chunk_states_train(Bm, x, dt, dA, torch.bfloat16)  # the passed states the forward op scans with
    _, r_states = csp_fused_ref(Bm, x, dt, dA, torch.bfloat16)
    _, r_passed = csp_fused_ref(Bm, x, dt, dA, torch.float32)
    stage("states_passed", states_passed, r_passed)
    stage("states", states, r_states)
    stage("y", y, chunk_scan_fwd_tma_ref(CB, x, dt, dA, Cm, states, D))
    r_norm = gated_norm_soma_fwd_ref(y2, ones, NORM_EPS, z2, th, ga)
    stage("rstd", rstd, r_norm[1])
    stage("n", n, r_norm[2])
    # backward
    dxs = p["dxs"]
    with torch.no_grad():
        gn, _, _ = soma_bwd(dxs, n, th, ga, ALPHA)
        dy, _ = gated_rmsnorm_bwd(gn, y2, ones, rstd, z2, torch.empty((M, DI), dtype=torch.bfloat16, device=dev))
        dy4 = dy.view(Bn, T, H, P)
        dxBC = torch.empty((Bn, T, C_CONV), dtype=torch.bfloat16, device=dev)
        bwd_store_glue(dy4, x, dt_raw, A, Bm, Cm, CHUNK, D, dt_bias, dxBC[..., :DI].unflatten(-1, (H, P)),
                       torch.empty((Bn, T, H), dtype=torch.bfloat16, device=dev),
                       dxBC[..., DI:DI + N].unflatten(-1, (1, N)), dxBC[..., DI + N:].unflatten(-1, (1, N)),
                       dA, dt, CB, states_passed)
    r_gn, r_dtheta, r_dgamma = soma_bwd_ref(dxs, n, th, ga, ALPHA)
    stage("grad_n", gn, r_gn)
    r_dy, r_dz = gated_rmsnorm_bwd_ref(gn, y2, ones, rstd, z2)
    stage("dy", dy, r_dy)
    r_dx, r_ddt, r_dA, r_dB, r_dC, r_dD, r_ddt_bias = bwd_store_glue_ref(
        dy4, x, dt_raw, A, Bm, Cm, CHUNK, D, dt_bias, None, None, None, None, dA, dt, CB, states_passed)
    stage("dxBC", dxBC, torch.cat([r_dx.flatten(-2), r_dB.flatten(-2), r_dC.flatten(-2)], dim=-1))
    xl = xBC_in.detach().requires_grad_(True)
    wl = p["conv_w"].detach().clone().requires_grad_(True)
    bl = p["conv_b"].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        r_gx, r_gw, r_gb = torch.autograd.grad(spiking_conv_silu_ref(xl, wl, bl), (xl, wl, bl), dxBC)
    ref = {
        "xs": spiking_soma_readout_ref(n, th, ga),
        "dzxbcdt": torch.cat([r_dz.view(Bn, T, DI), r_gx, r_ddt], dim=-1),
        "dconv_w": r_gw,
        "dconv_b": r_gb,
        "ddt_bias": r_ddt_bias,
        "dA": r_dA,
        "dD": r_dD,
        "dtheta": r_dtheta,
        "dgamma": r_dgamma,
    }
    return ref, got, want


def test_vs_reference(device):
    p = operands(1, 512, device)
    new = _run(spiking_chunk_scan_soma, p)
    ref, got, want = _per_operator(p)
    new.update(got)
    ref.update(want)
    assert_all_match(new, ref, prefix="spiking_combined_ref")


def test_autocast_independent(device):
    p = operands(1, 512, device)
    plain = _run(spiking_chunk_scan_soma, p)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        auto = _run(spiking_chunk_scan_soma, p)
    assert_all_match(auto, plain, prefix="autocast")


def test_layout_and_contract(device):
    p = operands(1, 256, device)
    with torch.no_grad():
        xs = spiking_chunk_scan_soma(*(p[k] for k in NAMES), alpha=ALPHA, chunk_size=256)
    assert xs.shape == (256, DI) and xs.dtype == torch.bfloat16 and xs.is_contiguous()
    with pytest.raises(RuntimeError):
        spiking_chunk_scan_soma(p["zxbcdt"][:, :200].contiguous(), *(p[k] for k in NAMES[1:]), alpha=ALPHA,
                           chunk_size=256)                                    # partial chunk
    with pytest.raises(RuntimeError):
        spiking_chunk_scan_soma(*(p[k] for k in NAMES), alpha=ALPHA, chunk_size=128)   # chunk size
    with pytest.raises(RuntimeError):
        spiking_chunk_scan_soma(p["zxbcdt"].float(), *(p[k] for k in NAMES[1:]), alpha=ALPHA, chunk_size=256)
    with pytest.raises(RuntimeError):
        spiking_chunk_scan_soma(*(p[k] for k in NAMES), alpha=0.0, chunk_size=256)
