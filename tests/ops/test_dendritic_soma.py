"""ops.triton.dendritic_soma: the gated RMSNorm + soma read-out of the dendritic neuron layer.

* ``gated_norm_soma_fwd`` / ``gated_norm_soma_bwd`` (training: the fused forward storing n, the two-stage backward)
  on the rows of a training step (8192, di) and fewer, with z the strided slice of a projection, at the widths of the
  default configuration (di 9216, projection 18832) and of the D = 2048 configuration (di 12288, projection 25024);
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

from neuronspark.ops.triton.dendritic_soma import (
    gated_norm_soma_bwd,
    gated_norm_soma_fwd,
    gated_norm_soma_fwd_ref,
    gated_rmsnorm_bwd_ref,
    dendritic_soma_readout_ref,
    soma_bwd,
    soma_bwd_ref,
)
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

WIDTHS = [(9216, 18832), (12288, 25024)]      # (di, projection width) of D = 1536 and D = 2048
ALPHA = 8.0
EPS = 1e-5
ROWS = [8192, 1024, 77]


def operands(M, device, di=9216, width=18832):
    """x (M, di) bf16 (an SSD output), z the gate slice of a (M, width) projection, the unit weight, theta =
    0.02 + softplus(rho) near its launch value, gamma near 1, an incoming gradient dxs and the dz buffer view"""
    x = rand_tensor((M, di), torch.bfloat16, device)
    proj = rand_tensor((M, width), torch.bfloat16, device)
    rho = -3.4915204 + 0.5 * rand_tensor((di,), torch.float32, device)
    theta = (0.02 + F.softplus(rho)).contiguous()
    gamma = 1.0 + 0.1 * rand_tensor((di,), torch.float32, device)
    dxs = rand_tensor((M, di), torch.bfloat16, device)
    return {"di": di, "x": x, "proj": proj, "w": torch.ones(di, dtype=torch.float32, device=device), "theta": theta,
            "gamma": gamma, "dxs": dxs}


def _fwd(o):
    return gated_norm_soma_fwd(o["x"], o["w"], EPS, o["proj"][:, :o["di"]], o["theta"], o["gamma"])


def _bwd(o, xo_rstd_n):
    _, rstd, n = xo_rstd_n
    di = o["di"]
    dproj = torch.empty_like(o["proj"])
    dz_buf = dproj[:, :di]
    dy, dz, gth, gga = gated_norm_soma_bwd(o["dxs"], n, o["x"], o["w"], rstd, o["proj"][:, :di], dz_buf, o["theta"],
                                           o["gamma"], ALPHA)
    assert dz.data_ptr() == dz_buf.data_ptr()
    return {"dy": dy, "dz": dz.contiguous(), "dtheta": gth, "dgamma": gga}


@pytest.mark.parametrize("di,width", WIDTHS)
@pytest.mark.parametrize("M", ROWS)
def test_gated_norm_soma_vs_reference(device, M, di, width):
    o = operands(M, device, di, width)
    z = o["proj"][:, :di]
    f = _fwd(o)
    xo, rstd, n = f
    rf = gated_norm_soma_fwd_ref(o["x"], o["w"], EPS, z, o["theta"], o["gamma"])
    assert_all_match({"rstd": rstd, "n": n}, {"rstd": rf[1], "n": rf[2]}, prefix="soma_fwd_ref")
    assert_all_match({"xo": xo}, {"xo": dendritic_soma_readout_ref(n, o["theta"], o["gamma"])}, prefix="soma_fwd_ref")
    new = _bwd(o, f)
    # the composite backward's first stage (the same launch on the same inputs) supplies grad_n to the second
    gn, gth, gga = soma_bwd(o["dxs"], n, o["theta"], o["gamma"], ALPHA)
    rgn, rth, rga = soma_bwd_ref(o["dxs"], n, o["theta"], o["gamma"], ALPHA)
    assert_all_match({"grad_n": gn, "dtheta": gth, "dgamma": gga}, {"grad_n": rgn, "dtheta": rth, "dgamma": rga},
                     prefix="soma_bwd_stage1_ref")
    rdy, rdz = gated_rmsnorm_bwd_ref(gn, o["x"], o["w"], rstd, z)
    assert_all_match(new, {"dy": rdy, "dz": rdz, "dtheta": rth, "dgamma": rga}, prefix="soma_bwd_ref")


def test_refusals(device):
    o = operands(4, device)
    with pytest.raises(RuntimeError):
        gated_norm_soma_fwd(o["x"][:, :4000], o["w"][:4000], EPS, o["proj"][:, :4000], o["theta"][:4000],
                            o["gamma"][:4000])                                   # not a multiple of 1024 columns
