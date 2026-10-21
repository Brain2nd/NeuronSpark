"""ops.triton.dendritic_chunk: the write-strength transform with its in-chunk decay sums and the chunk-local overlap of
the read and write directions, forward and backward, against their torch references.

* ``chunk_dt_cumsum`` / ``chunk_dt_cumsum_bwd``: bf16 and fp32 dt (the bf16 one a strided view of a wider row, as the
  layer's projection is split), with and without dt_bias, the default limit (0, inf) and a finite one (lo 1e-3, hi 0.1:
  both clamp sides active), sequence lengths 1024 / 300 / 37 (whole and partial last chunks), 20 heads (a partial head
  tile), large positive inputs (the linear branch of the softplus above 20);
* ``chunk_cb`` / ``chunk_cb_bwd``: dstate 128 and 64, the same sequence lengths, ngroups 1 and 2, dCB read directly and
  through its transposed view (the dC call), with and without a residual, bf16 and fp32 outputs.

Every comparison is the per-operator rule of ``neuronspark.utils.testing`` (matched ratio 1.0).
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.dendritic_chunk import (chunk_cb, chunk_cb_bwd, chunk_cb_bwd_ref, chunk_cb_ref,
                                                   chunk_dt_cumsum, chunk_dt_cumsum_bwd, chunk_dt_cumsum_bwd_ref,
                                                   chunk_dt_cumsum_ref)
from neuronspark.utils.testing import assert_match, rand_tensor

pytestmark = pytest.mark.cuda

CS = 256
LIMITS = [(0.0, float("inf")), (1e-3, 0.1)]


def _dt_inputs(device, batch, seqlen, nheads, dtype, strided, big=False):
    g = torch.Generator(device="cpu").manual_seed(seqlen * 7 + nheads)
    raw = torch.randn(batch, seqlen, nheads + (5 if strided else 0), generator=g) * (12.0 if big else 2.0) - 1.0
    raw = raw.to(dtype).to(device)
    dt = raw[..., :nheads] if strided else raw
    A = -torch.exp(torch.randn(nheads, generator=g)).to(device)
    bias = (torch.randn(nheads, generator=g) * 0.5).to(device)
    return dt, A, bias


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("dtype,strided", [(torch.bfloat16, True), (torch.float32, False)])
@pytest.mark.parametrize("seqlen", [1024, 300, 37])
def test_dt_cumsum_forward(device, seqlen, dtype, strided, with_bias, limit):
    dt, A, bias = _dt_inputs(device, 2, seqlen, 20, dtype, strided)
    b = bias if with_bias else None
    da, d = chunk_dt_cumsum(dt, A, CS, dt_bias=b, dt_limit=limit)
    da_r, d_r = chunk_dt_cumsum_ref(dt, A, CS, dt_bias=b, dt_limit=limit)
    assert da.shape == d.shape == (2, 20, -(-seqlen // CS), CS)
    assert_match(d, d_r, f"dt_out_T{seqlen}")
    assert_match(da, da_r, f"dA_cs_T{seqlen}")


def test_dt_cumsum_softplus_linear_branch(device):
    dt, A, bias = _dt_inputs(device, 2, 300, 20, torch.float32, False, big=True)
    da, d = chunk_dt_cumsum(dt, A, CS, dt_bias=bias)
    da_r, d_r = chunk_dt_cumsum_ref(dt, A, CS, dt_bias=bias)
    assert_match(d, d_r, "dt_out_big")
    assert_match(da, da_r, "dA_cs_big")


@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("dtype,strided", [(torch.bfloat16, True), (torch.float32, False)])
@pytest.mark.parametrize("seqlen", [1024, 300, 37])
def test_dt_cumsum_backward(device, seqlen, dtype, strided, with_bias, limit):
    batch, nheads = 2, 20
    dt, A, bias = _dt_inputs(device, batch, seqlen, nheads, dtype, strided)
    b = bias if with_bias else None
    nc = -(-seqlen // CS)
    dda = rand_tensor((batch, nheads, nc, CS), torch.float32, device)
    ddt_out = rand_tensor((batch, nheads, nc, CS), torch.float32, device)
    ddt = torch.empty_like(dt)
    ws_a = torch.empty(batch, nc, nheads, device=device)
    ws_b = torch.empty(batch, nc, nheads, device=device)
    chunk_dt_cumsum_bwd(dda, ddt_out, dt, A, b, ddt, ws_a, ws_b, dt_limit=limit)
    ddt_r, wa_r, wb_r = chunk_dt_cumsum_bwd_ref(dda, ddt_out, dt, A, b, dt_limit=limit)
    assert_match(ddt, ddt_r.to(dt.dtype), f"ddt_T{seqlen}")
    assert_match(ws_a, wa_r, f"ws_dA_T{seqlen}")
    assert_match(ws_b, wb_r, f"ws_ddt_bias_T{seqlen}")


def _rows(device, batch, seqlen, ngroups, k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    wide = torch.randn(batch, seqlen, ngroups, 2 * k + 8, generator=g).to(torch.bfloat16).to(device)
    return wide[..., :k], wide[..., k + 8:2 * k + 8]                    # strided views with a unit last stride


@pytest.mark.parametrize("ngroups", [1, 2])
@pytest.mark.parametrize("k", [128, 64])
@pytest.mark.parametrize("seqlen", [1024, 300, 37])
def test_cb_forward(device, seqlen, k, ngroups):
    C, B = _rows(device, 2, seqlen, ngroups, k, seqlen + k)
    cb = chunk_cb(C, B, CS)
    assert cb.shape == (2, -(-seqlen // CS), ngroups, CS, CS) and cb.dtype == torch.float32
    assert_match(cb, chunk_cb_ref(C, B, CS), f"CB_T{seqlen}_k{k}_g{ngroups}")


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("residual", [None, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ngroups", [1, 2])
@pytest.mark.parametrize("seqlen", [1024, 300, 37])
def test_cb_backward(device, seqlen, ngroups, residual, transposed):
    batch, k = 2, 128
    C, B = _rows(device, batch, seqlen, ngroups, k, seqlen + 3)
    nc = -(-seqlen // CS)
    dcb = rand_tensor((batch, nc, ngroups, CS, CS), torch.float32, device)
    dout = dcb.transpose(-1, -2) if transposed else dcb
    res = rand_tensor(C.shape, residual, device) if residual is not None else None
    out = chunk_cb_bwd(C, dout, residual=res)
    assert out.shape == C.shape and out.dtype == C.dtype
    assert_match(out, chunk_cb_bwd_ref(C, dout, res), f"CBbwd_T{seqlen}_g{ngroups}_res{residual}_t{transposed}")


def test_cb_backward_into_fp32_buffer(device):
    C, B = _rows(device, 2, 300, 1, 128, 11)
    dcb = rand_tensor((2, 2, 1, CS, CS), torch.float32, device)
    out = torch.empty(C.shape, dtype=torch.float32, device=device)
    chunk_cb_bwd(B, dcb.transpose(-1, -2), residual=None, out=out)
    assert_match(out, chunk_cb_bwd_ref(B, dcb.transpose(-1, -2), None, torch.float32), "CBbwd_fp32_out")


def test_operand_form_refused(device):
    C, B = _rows(device, 1, 64, 1, 128, 5)
    with pytest.raises(RuntimeError):
        chunk_cb(C.float(), B.float(), CS)
    with pytest.raises(RuntimeError):
        chunk_cb_bwd(C, torch.zeros(1, 2, 1, CS, CS, device=device))                      # 2 chunks for seqlen 64
