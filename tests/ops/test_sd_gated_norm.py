"""ops.triton.sd_gated_norm: the SD gated RMSNorm + soma read-out of the prefill path.

* ``sd_gated_norm_soma`` on prefill row counts, y (M, 9216) bf16 rows and z the strided gate slice of a (M, 18832)
  projection, theta / gamma (9216,) fp32, over several input scales and eps values, stage by stage: the kernel's norm n
  (stored by its test hook) against the gated-norm reference, and the read-out x against the soma reference of that same
  n (a bf16 intermediate recomputed independently can differ by one unit in the last place at a rounding boundary, and
  the soma can carry that past the rule's relative bound; each reference therefore takes the operator's own input); x
  is also checked to be the public op's output;
* against the reference across the kernel's width domain, up to its edge N = 16384;
* inputs unchanged, and the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

import torch.nn.functional as F

from neuronspark.ops.triton.sd_gated_norm import _gated_norm_soma_fwd, sd_gated_norm_soma, sd_gated_rmsnorm_ref
from neuronspark.ops.triton.sd_soma import sd_soma_readout_ref
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

DI = 9216
WIDTH = 18832
ROWS = [1, 2, 37, 1024, 8192]
# (y scale, z scale, eps)
CASES = [(1.0, 3.0, 1e-5), (4.0, 4.0, 1e-6), (0.05, 0.05, 1e-5), (30.0, 30.0, 1e-6)]


def operands(M, scale_y, scale_z, device, N=DI, width=WIDTH):
    """y (M, N) bf16, z = the first N columns of a (M, width) bf16 projection (row stride ``width``), and the soma's
    theta (a softplus-positive threshold) and gamma (N,) fp32"""
    y = (rand_tensor((M, N), torch.float32, device) * scale_y).to(torch.bfloat16)
    proj = (rand_tensor((M, width), torch.float32, device) * scale_z).to(torch.bfloat16)
    theta = (0.02 + F.softplus(rand_tensor((N,), torch.float32, device) * 0.5 - 2.5)).contiguous()
    gamma = (rand_tensor((N,), torch.float32, device) * 0.3 + 1.0).contiguous()
    return {"y": y, "proj": proj, "theta": theta, "gamma": gamma}


def _run(o, eps):
    z = o["proj"][:, :int(o["y"].shape[1])]
    assert z.stride() == (int(o["proj"].shape[1]), 1)
    with torch.no_grad():
        x = sd_gated_norm_soma(o["y"], z, o["theta"], o["gamma"], eps)
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and tuple(x.shape) == tuple(o["y"].shape)
    return x


def _check_stages(o, eps, prefix):
    """the kernel's n vs the gated-norm reference; x vs the soma reference of the kernel's own n; x == the public op's"""
    N = int(o["y"].shape[1])
    z = o["proj"][:, :N]
    w = torch.ones((N,), dtype=torch.float32, device=o["y"].device)
    with torch.no_grad():
        x, n = _gated_norm_soma_fwd(o["y"], w, z, o["theta"], o["gamma"], eps, store_n=True)
    assert torch.equal(x, _run(o, eps))
    assert_all_match({"n": n}, {"n": sd_gated_rmsnorm_ref(o["y"], z, eps)}, prefix=prefix + "_norm")
    assert_all_match({"x": x}, {"x": sd_soma_readout_ref(n, o["theta"], o["gamma"])}, prefix=prefix + "_readout")


@pytest.mark.parametrize("M", ROWS)
@pytest.mark.parametrize("case", range(len(CASES)))
def test_gated_norm_soma_vs_reference(device, M, case):
    sy, sz, eps = CASES[case]
    o = operands(M, sy, sz, device)
    _check_stages(o, eps, "gated_norm_soma")


@pytest.mark.parametrize("N", [8200, 12288, 16384])
def test_gated_norm_soma_domain_vs_reference(device, N):
    o = operands(64, 1.0, 3.0, device, N=N, width=N + 512)
    _check_stages(o, 1e-5, f"gated_norm_soma_N{N}")


def test_inputs_unchanged(device):
    o = operands(16, 1.0, 3.0, device)
    before = {k: v.clone() for k, v in o.items()}
    _run(o, 1e-5)
    for k in o:
        assert_all_match({k: o[k]}, {k: before[k]}, prefix="unchanged")


def test_refusals(device):
    o = operands(4, 1.0, 1.0, device)
    z = o["proj"][:, :DI]
    th, ga = o["theta"], o["gamma"]
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            sd_gated_norm_soma(o["y"][:, :4096], z[:, :4096], th[:4096], ga[:4096], 1e-5)   # N outside (8192, 16384]
        with pytest.raises(RuntimeError):
            sd_gated_norm_soma(o["y"].float(), z, th, ga, 1e-5)                      # fp32 y
        with pytest.raises(RuntimeError):
            sd_gated_norm_soma(o["y"], o["proj"][:, :2 * DI:2], th, ga, 1e-5)        # non-unit last stride of z
        with pytest.raises(RuntimeError):
            sd_gated_norm_soma(o["y"], z[:2], th, ga, 1e-5)                          # shape mismatch
        with pytest.raises(RuntimeError):
            sd_gated_norm_soma(o["y"], z, th.double(), ga, 1e-5)                     # fp64 theta
        with pytest.raises(RuntimeError):
            sd_gated_norm_soma(o["y"].cpu(), z.cpu(), th.cpu(), ga.cpu(), 1e-5)      # not CUDA
