"""ops.triton.dendritic_step: the dendritic-layer decode step and the dendritic-layer pre-norm rows of the inference
paths.

* ``dendritic_step_core`` (K1 routed by the decode batch, then K2) at every decode batch b in {1, 2, 4, 8, 16, 32}
  (all three K1 routes) and at other batches, for both SSM state dtypes (bf16, the default storage, and fp32), at the
  widths of D = 1536 (H 144) and D = 2048 (H 192), over three consecutive steps (the rolled conv tails and the carried
  state feed the next step).  Each stage against its reference on the operator's own inputs: K1 (the conv tail and
  the SSM state after the step, the K1 output) against :func:`dendritic_step_conv_ref` +
  :func:`dendritic_step_state_ref` on the step's input states; K2 (xs) against :func:`dendritic_step_k2_ref` on the
  kernel's K1 output.  The public call equals the two launches.  K1 rounds its conv output to bf16 after the kernel's
  approximate SiLU: a row past the rule passes only if flipping some of the conv outputs its failing elements read
  (their x channels, the row's B and C) that lie within 1e-6 (relative; the kernel's fp32 conv output differs from the
  reference's by a few units: FMA chain, approximate exp2 and reciprocal) of a bf16 rounding boundary in the
  reference makes the whole row pass.
  ``n = bf16(g * rstd)`` is rounded inside K2: where the fp32 ``g * rstd`` of the kernel and of the reference (another
  summation order of the row norm) lie on either side of a bf16 rounding boundary, n differs by one unit in the last
  place and the read-out can carry that past the rule's relative bound (gamma > 1 and the final bf16 rounding); such
  an element passes only if the reference's fp32 value lies within 1e-5 (relative) of that boundary and the kernel's
  xs is exactly the read-out of the neighbouring bf16 value;
* strided state rows, updated in place: the same values as the call on contiguous states;
* ``dendritic_prenorm_rows`` against its reference, at decode row counts and prefill row counts (T * B), at D = 1536
  and 2048, and on strided rows;
* the refusals outside the contract.

Every comparison is the per-operator rule of ``neuronspark.utils.testing``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuronspark.ops.triton import dendritic_step as ds
from neuronspark.ops.triton.dendritic_soma import dendritic_soma_readout_ref
from neuronspark.ops.triton.dendritic_step import (
    MAX_DECODE_BATCH,
    k1_route,
    k1_tile,
    dendritic_prenorm_rows,
    dendritic_prenorm_rows_ref,
    dendritic_step_conv_ref,
    dendritic_step_core,
    dendritic_step_k2_ref,
    dendritic_step_state_ref,
)
from neuronspark.utils.testing import assert_all_match, match_stats, rand_tensor

pytestmark = pytest.mark.cuda

D_MODEL = 1536
D_MODELS = (1536, 2048)
H, P, N = 144, 64, 128
HS = (144, 192)                           # dendritic heads of D = 1536 and D = 2048 (expand 6, head width 64)
DI = H * P
CONV = DI + 2 * N
EPS = 1e-5
DECODE_B = [1, 2, 4, 8, 16, 32]           # the decode batch sizes of the inference cache (every K1 route)
OTHER_B = [3, 5, 12, 24]                  # other batch sizes (the general K1 route)
STATE_DTYPES = [torch.bfloat16, torch.float32]
STEPS = 3
PRENORM_ROWS = [1, 2, 8, 32, 1024, 8192]


def params(device, h=H):
    """The mixer's step operands: conv weight / bias, A = -exp(A_log), dt_bias, D, theta = 0.02 + softplus(rho) near
    its launch value, gamma near 1 (fp32); h heads of width P."""
    di, conv = h * P, h * P + 2 * N
    rho = -3.4915204 + 0.5 * rand_tensor((di,), torch.float32, device)
    return {
        "conv_w": (0.3 * rand_tensor((conv, 4), torch.float32, device)).contiguous(),
        "conv_b": 0.1 * rand_tensor((conv,), torch.float32, device),
        "A": -torch.exp(0.5 * rand_tensor((h,), torch.float32, device)),
        "dt_bias": 0.5 * rand_tensor((h,), torch.float32, device),
        "D": rand_tensor((h,), torch.float32, device),
        "theta": (0.02 + F.softplus(rho)).contiguous(),
        "gamma": 1.0 + 0.1 * rand_tensor((di,), torch.float32, device),
    }


def states(b, state_dtype, device, h=H):
    """conv tail (b, h P + 2 N, 3) fp32 holding bf16 values; SSM state (b, h, P, N) in ``state_dtype``."""
    conv = rand_tensor((b, h * P + 2 * N, 3), torch.bfloat16, device).float()
    ssm = (0.1 * rand_tensor((b, h, P, N), torch.float32, device)).to(state_dtype)
    return conv, ssm


def projections(b, device, steps=STEPS, h=H):
    """``steps`` in-projection outputs (b, 2 h P + 2 N + h) bf16 = [z | xBC | dt]."""
    return [rand_tensor((b, 2 * h * P + 2 * N + h), torch.bfloat16, device) for _ in range(steps)]


def _stages(route, proj, conv, ssm, prm):
    """The kernel's two launches of one step (states advanced in place): (K1 output, xs)."""
    di, _, H, N, P = ds._widths(proj, prm["conv_w"], prm["A"], prm["theta"])
    b = int(proj.shape[0])
    args = (proj, conv, ssm, prm["conv_w"], prm["conv_b"], prm["A"], prm["dt_bias"], prm["D"], di, H, N, P)
    if route == "plain":
        y = ds._launch_k1_plain(*args, *k1_tile(b))
        return y, ds._launch_k2_select(y, proj, conv, prm["theta"], prm["gamma"], di, N, EPS)
    y = ds._launch_k1_gated(*args, *k1_tile(b)) if route == "gated" else ds._launch_k1_split(*args)
    return y, ds._launch_k2_from_gated(y, proj, conv, prm["theta"], prm["gamma"], di, N, EPS)


def _neighbour(n, sign):
    """The bf16 value one unit in the last place from ``n`` (away from zero for sign +1, towards it for -1)."""
    bits = n.view(torch.int16).to(torch.int32) + sign
    return bits.to(torch.int16).view(torch.bfloat16)


def _other_side(v32):
    """For fp32 values v: (bf16(v), the bf16 neighbour on v's side of it, |v - their midpoint| / |v|)."""
    r = v32.to(torch.bfloat16)
    away = (v32.abs() > r.float().abs())
    o = torch.where(away, _neighbour(r, 1), _neighbour(r, -1))
    return r, o, (v32 - (r.float() + o.float()) / 2).abs() / v32.abs().clamp_min(1e-30)


def _k1_form(route, y, z):
    """The K1 output from the fp32 y: bf16 y (plain) or the fp32 gated bf16(y) silu(z)."""
    return y.to(torch.bfloat16) if route == "plain" else y.to(torch.bfloat16).float() * (z * torch.sigmoid(z))


def assert_k1_match(route, k1, ssm_k, proj, conv_in, ssm_in, prm, name):
    """The K1 output and the SSM state against the reference on the step's input states by the per-operator rule.  The
    conv output is rounded to bf16 inside K1 (after the approximate SiLU of the kernel); a row past the rule passes
    only if flipping some of the conv outputs its failing elements read whose reference value lies within 1e-6
    (relative) of a bf16 rounding boundary to the other side makes the whole row (K1 output and state) pass."""
    di = int(prm["theta"].shape[0])
    z = proj[:, :di].float()
    xbc, s32 = dendritic_step_conv_ref(proj, conv_in, prm["conv_w"], prm["conv_b"], di)

    def ref(rows, x):
        st = ssm_in[rows].clone()
        y = dendritic_step_state_ref(x, proj[rows], st, prm["A"], prm["dt_bias"], prm["D"], di)
        return _k1_form(route, y, z[rows]), st

    k1_r, ssm_r = ref(slice(None), xbc)
    if match_stats(k1, k1_r, name).passed and match_stats(ssm_k, ssm_r, name).passed:
        return
    _, other, dist = _other_side(s32)
    N = int(ssm_k.shape[-1])

    def past(a, b):
        d = (a.float() - b.float()).abs()
        return (d > 1e-2) & (d > 1e-2 * b.float().abs())

    for r in range(int(proj.shape[0])):
        if match_stats(k1[r], k1_r[r], name).passed and match_stats(ssm_k[r], ssm_r[r], name).passed:
            continue
        # the conv outputs the failing elements read: their own x channel and the row's B / C
        xs = set(past(k1[r], k1_r[r]).nonzero().flatten().tolist())
        xs |= set((past(ssm_k[r], ssm_r[r]).any(dim=-1).flatten()).nonzero().flatten().tolist())
        reads = torch.zeros_like(dist[r], dtype=torch.bool)
        reads[sorted(xs)] = True
        reads[di:di + 2 * N] = True
        cand = ((dist[r] <= 1e-6) & reads).nonzero().flatten().tolist()
        assert 0 < len(cand) <= 12, f"{name}: row {r} past the rule with {len(cand)} near-boundary conv outputs"
        ok = False
        for mask in range(1, 1 << len(cand)):
            x = xbc[r:r + 1].clone()
            for i, c in enumerate(cand):
                if mask >> i & 1:
                    x[0, c] = other[r, c].float()
            k1_f, ssm_f = ref(slice(r, r + 1), x)
            if match_stats(k1[r:r + 1], k1_f, name).passed and match_stats(ssm_k[r:r + 1], ssm_f, name).passed:
                ok = True
                break
        assert ok, f"{name}: row {r} past the rule and not explained by rounding-boundary flips of the conv output"


def assert_k2_match(xs, ref_xs, n32, theta, gamma, name):
    """xs against the reference by the per-operator rule; an element past the rule passes only as a rounding-boundary
    flip of n (see the module docstring)."""
    res = match_stats(xs, ref_xs, name)
    if res.passed:
        return
    d = (xs.float() - ref_xs.float()).abs()
    bad = (d > 1e-2) & (d > 1e-2 * ref_xs.float().abs())
    n = n32.to(torch.bfloat16)
    for sign in (1, -1):
        nb = _neighbour(n, sign)
        mid = (n.float() + nb.float()) / 2
        near = (n32 - mid).abs() <= 1e-5 * n32.abs()
        same = dendritic_soma_readout_ref(nb, theta, gamma) == xs
        bad &= ~(near & same)
    assert not bool(bad.any()), f"{res}; {int(bad.sum())} elements are not rounding-boundary flips of n"


@pytest.mark.parametrize("h", HS)
@pytest.mark.parametrize("b", DECODE_B + OTHER_B)
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
def test_step_core_vs_reference(device, b, state_dtype, h):
    prm = params(device, h)
    conv, ssm = states(b, state_dtype, device, h)
    route = k1_route(b)
    di = h * P
    with torch.no_grad():
        for s, proj in enumerate(projections(b, device, h=h)):
            tag = f"step{s}_h{h}_b{b}_{route}"
            pub_conv, pub_ssm = conv.clone(), ssm.clone()
            xs = dendritic_step_core(proj, pub_conv, pub_ssm, prm["conv_w"], prm["conv_b"], prm["A"], prm["dt_bias"],
                                     prm["D"], prm["theta"], prm["gamma"], EPS)
            st_conv, st_ssm = conv.clone(), ssm.clone()
            y, xs_st = _stages(route, proj, st_conv, st_ssm, prm)
            assert torch.equal(xs, xs_st) and torch.equal(pub_conv, st_conv) and torch.equal(pub_ssm, st_ssm), tag
            conv_dim = int(prm["conv_w"].shape[0])
            ref_conv = torch.cat([conv, proj[:, di:di + conv_dim].float()[:, :, None]], dim=-1)[:, :, 1:]
            assert_all_match({"conv": pub_conv}, {"conv": ref_conv}, prefix=tag)
            assert_k1_match(route, y, pub_ssm, proj, conv, ssm, prm, f"{tag}.k1")
            z = proj[:, :di].float()
            g = y.float() * (z * torch.sigmoid(z)) if route == "plain" else y
            ref_xs, n32 = dendritic_step_k2_ref(g, prm["theta"], prm["gamma"], EPS)
            assert_k2_match(xs, ref_xs, n32, prm["theta"], prm["gamma"], f"{tag}.xs")
            conv, ssm = pub_conv, pub_ssm                            # the next step starts from the kernel's states


def test_step_core_strided_state_rows(device):
    """The one-kernel routes take any state strides and update the views in place (b = 4: the general route)."""
    prm = params(device)
    conv_c, ssm_c = states(4, torch.bfloat16, device)
    conv = torch.empty((4, 3, CONV), dtype=torch.float32, device=device).transpose(1, 2)   # lag stride CONV
    conv.copy_(conv_c)
    ssm = torch.empty((4, H, P, 2 * N), dtype=torch.bfloat16, device=device)[..., :N]     # row stride 2 N
    ssm.copy_(ssm_c)
    assert not conv.is_contiguous() and not ssm.is_contiguous()
    proj = projections(4, device, steps=1)[0]
    with torch.no_grad():
        ops = (prm["conv_w"], prm["conv_b"], prm["A"], prm["dt_bias"], prm["D"], prm["theta"], prm["gamma"], EPS)
        xs = dendritic_step_core(proj, conv, ssm, *ops)
        xs_c = dendritic_step_core(proj, conv_c, ssm_c, *ops)
    assert torch.equal(xs, xs_c) and torch.equal(conv, conv_c) and torch.equal(ssm, ssm_c)


def prenorm_operands(M, device, d=D_MODEL):
    return {"x": 3.0 * rand_tensor((M, d), torch.float32, device),
            "gain": 1.0 + 0.1 * rand_tensor((d,), torch.float32, device)}


@pytest.mark.parametrize("d", D_MODELS)
@pytest.mark.parametrize("M", PRENORM_ROWS)
def test_prenorm_rows_vs_reference(device, M, d):
    o = prenorm_operands(M, device, d)
    with torch.no_grad():
        assert_all_match({"hn": dendritic_prenorm_rows(o["x"], o["gain"], EPS)},
                         {"hn": dendritic_prenorm_rows_ref(o["x"], o["gain"], EPS)}, prefix="prenorm_ref")


def test_prenorm_rows_strided_input(device):
    """Rows with a row stride larger than D (one time step of a B-major (B, T, D) stream)."""
    h = rand_tensor((8, 2, D_MODEL), torch.float32, device)
    gain = 1.0 + 0.1 * rand_tensor((D_MODEL,), torch.float32, device)
    rows = h[:, 0]                                              # (8, D) with row stride 2 D
    assert rows.stride() == (2 * D_MODEL, 1)
    with torch.no_grad():
        assert_all_match({"hn": dendritic_prenorm_rows(rows, gain, EPS)}, {"hn": dendritic_prenorm_rows_ref(rows, gain,
                                                                                                        EPS)},
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
            dendritic_step_core(projections(big, device, steps=1)[0], c_big, s_big, *args)             # b > 32
        with pytest.raises(RuntimeError):
            dendritic_step_core(proj.float(), conv, ssm, *args)                                       # fp32 proj
        with pytest.raises(RuntimeError):
            dendritic_step_core(proj[:, :-1], conv, ssm, *args)                                       # width
        with pytest.raises(RuntimeError):
            dendritic_step_core(proj, conv, ssm.half(), *args)                                        # fp16 state
        with pytest.raises(RuntimeError):
            dendritic_step_core(proj, conv[:1], ssm, *args)                                           # state batch
        with pytest.raises(RuntimeError):
            dendritic_step_core(proj, conv, ssm, prm["conv_w"].t().contiguous().t(),
                              *args[1:])       # non-contiguous conv_w
        with pytest.raises(RuntimeError):
            dendritic_prenorm_rows(rand_tensor((4, D_MODEL), torch.bfloat16, device),
                                   prm["gamma"][:D_MODEL].contiguous(),
                            EPS)                                                               # bf16 x
        with pytest.raises(RuntimeError):
            dendritic_prenorm_rows(rand_tensor((4, D_MODEL), torch.float32, device), prm["gamma"][:100].contiguous(),
                                   EPS)
