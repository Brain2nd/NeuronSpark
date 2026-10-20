"""ops.triton.sd_gated_norm: the SD gated RMSNorm of the prefill path.

* ``sd_gated_rmsnorm`` against its pure-torch reference on prefill row counts, y (M, 9216) bf16 rows and z the
  strided gate slice of a (M, 18832) projection, over several input scales and eps values;
* against the reference across the kernel's width domain, up to its edge N = 16384;
* inputs unchanged, and the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch

from neuronspark.ops.triton.sd_gated_norm import sd_gated_rmsnorm, sd_gated_rmsnorm_ref
from neuronspark.utils.testing import assert_all_match, rand_tensor

pytestmark = pytest.mark.cuda

DI = 9216
WIDTH = 18832
ROWS = [1, 2, 37, 1024, 8192]
# (y scale, z scale, eps)
CASES = [(1.0, 3.0, 1e-5), (4.0, 4.0, 1e-6), (0.05, 0.05, 1e-5), (30.0, 30.0, 1e-6)]


def operands(M, scale_y, scale_z, device, N=DI, width=WIDTH):
    """y (M, N) bf16 and z = the first N columns of a (M, width) bf16 projection (row stride ``width``)"""
    y = (rand_tensor((M, N), torch.float32, device) * scale_y).to(torch.bfloat16)
    proj = (rand_tensor((M, width), torch.float32, device) * scale_z).to(torch.bfloat16)
    return {"y": y, "proj": proj}


def _run(o, eps):
    z = o["proj"][:, :int(o["y"].shape[1])]
    assert z.stride() == (int(o["proj"].shape[1]), 1)
    with torch.no_grad():
        n = sd_gated_rmsnorm(o["y"], z, eps)
    assert n.dtype == torch.bfloat16 and n.is_contiguous() and tuple(n.shape) == tuple(o["y"].shape)
    return n


@pytest.mark.parametrize("M", ROWS)
@pytest.mark.parametrize("case", range(len(CASES)))
def test_gated_rmsnorm_vs_reference(device, M, case):
    sy, sz, eps = CASES[case]
    o = operands(M, sy, sz, device)
    z = o["proj"][:, :DI]
    assert_all_match({"n": _run(o, eps)}, {"n": sd_gated_rmsnorm_ref(o["y"], z, eps)}, prefix="gated_rmsnorm_ref")


@pytest.mark.parametrize("N", [8200, 12288, 16384])
def test_gated_rmsnorm_domain_vs_reference(device, N):
    o = operands(64, 1.0, 3.0, device, N=N, width=N + 512)
    z = o["proj"][:, :N]
    assert_all_match({"n": _run(o, 1e-5)}, {"n": sd_gated_rmsnorm_ref(o["y"], z, 1e-5)}, prefix=f"gated_rmsnorm_N{N}")


def test_inputs_unchanged(device):
    o = operands(16, 1.0, 3.0, device)
    before = {k: v.clone() for k, v in o.items()}
    _run(o, 1e-5)
    for k in o:
        assert_all_match({k: o[k]}, {k: before[k]}, prefix="unchanged")


def test_refusals(device):
    o = operands(4, 1.0, 1.0, device)
    z = o["proj"][:, :DI]
    with torch.no_grad():
        with pytest.raises(RuntimeError):
            sd_gated_rmsnorm(o["y"][:, :4096], z[:, :4096], 1e-5)          # N outside (8192, 16384]
        with pytest.raises(RuntimeError):
            sd_gated_rmsnorm(o["y"].float(), z, 1e-5)                      # fp32 y
        with pytest.raises(RuntimeError):
            sd_gated_rmsnorm(o["y"], o["proj"][:, :2 * DI:2], 1e-5)        # non-unit last stride of z
        with pytest.raises(RuntimeError):
            sd_gated_rmsnorm(o["y"], z[:2], 1e-5)                          # shape mismatch
        with pytest.raises(RuntimeError):
            sd_gated_rmsnorm(o["y"].cpu(), z.cpu(), 1e-5)                  # not CUDA
