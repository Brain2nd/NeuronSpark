"""ops.triton.sd_soma: the gated RMSNorm + soma read-out of the SD layer.

* ``gated_norm_soma_fwd`` / ``gated_norm_soma_bwd`` (training: the fused forward storing n, the two-stage backward)
  on the rows of a training step (8192, 9216) and fewer, with z the strided slice of a projection (row stride 18832);
* ``sd_soma_readout`` (prefill registry op) at prefill row counts;
* each against its pure-torch reference, operator by operator on the same inputs: the gated norm (rstd, n); the
  soma read-out x_soma given the stored n (the read-out is defined on n, which is compared on its own); the soma
  backward (grad_n, dtheta, dgamma) given the stored n; the gated-norm backward (dy, dz) given the first stage's
  grad_n and the stored rstd.  A bf16 intermediate that is recomputed independently can differ by one unit in the
  last place where the fp32 values lie on either side of a rounding boundary, and the next operator can carry that
  difference past the rule's relative bound; each reference therefore takes the operator's own inputs.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuronspark.ops.triton.sd_soma import (
    gated_norm_soma_bwd,
    gated_norm_soma_fwd,
    gated_norm_soma_fwd_ref,
    gated_rmsnorm_bwd_ref,
    sd_soma_readout,
    sd_soma_readout_ref,
    soma_bwd,
    soma_bwd_ref,
)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

DI = 9216
WIDTH = 18832
ALPHA = 8.0
EPS = 1e-5
ROWS = [8192, 1024, 77]
READOUT_ROWS = [1, 2, 63, 1024, 8192]


def operands(M, device):
    """x (M, 9216) bf16 (an SSD output), z the gate slice of a (M, 18832) projection, the unit weight, theta =
    0.02 + softplus(rho) near its launch value, gamma near 1, an incoming gradient dxs and the dz buffer view"""
    x = rand_tensor((M, DI), torch.bfloat16, device)
    proj = rand_tensor((M, WIDTH), torch.bfloat16, device)
    rho = -3.4915204 + 0.5 * rand_tensor((DI,), torch.float32, device)
    theta = (0.02 + F.softplus(rho)).contiguous()
    gamma = 1.0 + 0.1 * rand_tensor((DI,), torch.float32, device)
    dxs = rand_tensor((M, DI), torch.bfloat16, device)
    return {"x": x, "proj": proj, "w": torch.ones(DI, dtype=torch.float32, device=device), "theta": theta,
            "gamma": gamma, "dxs": dxs}


def _fwd(o):
    return gated_norm_soma_fwd(o["x"], o["w"], EPS, o["proj"][:, :DI], o["theta"], o["gamma"])


def _bwd(o, xo_rstd_n):
    _, rstd, n = xo_rstd_n
    dproj = torch.empty_like(o["proj"])
    dz_buf = dproj[:, :DI]
    dy, dz, gth, gga = gated_norm_soma_bwd(o["dxs"], n, o["x"], o["w"], rstd, o["proj"][:, :DI], dz_buf, o["theta"],
                                           o["gamma"], ALPHA)
    assert dz.data_ptr() == dz_buf.data_ptr()
    return {"dy": dy, "dz": dz.contiguous(), "dtheta": gth, "dgamma": gga}


@pytest.mark.parametrize("M", ROWS)
def test_gated_norm_soma_vs_reference(device, M):
    o = operands(M, device)
    z = o["proj"][:, :DI]
    f = _fwd(o)
    xo, rstd, n = f
    rf = gated_norm_soma_fwd_ref(o["x"], o["w"], EPS, z, o["theta"], o["gamma"])
    assert_all_match({"rstd": rstd, "n": n}, {"rstd": rf[1], "n": rf[2]}, prefix="soma_fwd_ref")
    assert_all_match({"xo": xo}, {"xo": sd_soma_readout_ref(n, o["theta"], o["gamma"])}, prefix="soma_fwd_ref")
    new = _bwd(o, f)
    # the composite backward's first stage (the same launch on the same inputs) supplies grad_n to the second
    gn, gth, gga = soma_bwd(o["dxs"], n, o["theta"], o["gamma"], ALPHA)
    rgn, rth, rga = soma_bwd_ref(o["dxs"], n, o["theta"], o["gamma"], ALPHA)
    assert_all_match({"grad_n": gn, "dtheta": gth, "dgamma": gga}, {"grad_n": rgn, "dtheta": rth, "dgamma": rga},
                     prefix="soma_bwd_stage1_ref")
    rdy, rdz = gated_rmsnorm_bwd_ref(gn, o["x"], o["w"], rstd, z)
    assert_all_match(new, {"dy": rdy, "dz": rdz, "dtheta": rth, "dgamma": rga}, prefix="soma_bwd_ref")


@pytest.mark.parametrize("M", READOUT_ROWS)
def test_readout_vs_reference(device, M):
    o = operands(M, device)
    n = rand_tensor((M, DI), torch.bfloat16, device) * 0.2
    with torch.no_grad():
        assert_all_match({"x": sd_soma_readout(n, o["theta"], o["gamma"])},
                         {"x": sd_soma_readout_ref(n, o["theta"], o["gamma"])}, prefix="readout_ref")


def test_refusals(device):
    o = operands(4, device)
    with pytest.raises(RuntimeError):
        gated_norm_soma_fwd(o["x"][:, :4096], o["w"][:4096], EPS, o["proj"][:, :4096], o["theta"][:4096],
                            o["gamma"][:4096])                                   # not 9 x 1024 columns
    with pytest.raises(RuntimeError):
        sd_soma_readout(o["x"].float(), o["theta"], o["gamma"])                  # fp32 n
    with pytest.raises(RuntimeError):
        sd_soma_readout(o["x"], o["theta"].double(), o["gamma"])                 # fp64 theta
