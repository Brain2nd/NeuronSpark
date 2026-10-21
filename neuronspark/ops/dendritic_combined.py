"""The dendritic neuron layer's chunk scan with its soma read-out as one autograd Function (training regime): the
synaptic filter, the chunked scan, the gated norm and the soma in one Function, store-not-recompute.

Registry op ``dendritic_chunk_scan_soma(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, *, alpha=8.0,
chunk_size=256) -> xs``:

* ``zxbcdt`` (B, T, 2 d_inner + 2 N + H) bf16 contiguous: the in-projection output, columns z | x B C | dt
  (default configuration: 9216 | 9472 | 144 = 18832); T % chunk_size == 0 (full chunks); the backward needs at least
  2 chunks (the state gradients ``gluon.dendritic_dstates.dstates_train``; it raises otherwise);
* ``conv_w`` (d_inner + 2 N, 4) fp32 contiguous (column 3 = lag 0), ``conv_b`` (d_inner + 2 N,) fp32;
* ``dt_bias``, ``A`` (= -exp(A_log)), ``D`` (H,) fp32; ``theta`` (= floor + softplus(rho)), ``gamma`` (d_inner,) fp32;
* ``alpha`` the soma surrogate slope (> 0; 8.0 in the default configuration), ``chunk_size`` 256;
* ``xs`` (B T, d_inner) bf16 contiguous = W_out's input; gradients for zxbcdt (bf16), conv_w / conv_b / dt_bias / A /
  D / theta / gamma (fp32).

Forward:
  1. causal conv k4 + SiLU on x|B|C (:mod:`neuronspark.ops.triton.dendritic_conv`), output (B, T, C) bf16 kept;
  2. dt = softplus(dt + dt_bias) and its per-chunk cumsum of dt A (``dendritic_chunk.chunk_dt_cumsum``);
  3. chunk states + state passing (Gluon ``chunk_states_train``, Blackwell): the passed states (the state entering each
     chunk) kept in fp32 for the backward's state gradients, and in bf16 for the chunk scan;
  4. CB = C B^T per chunk (``dendritic_chunk.chunk_cb``, fp32), kept;
  5. the chunk scan (Gluon ``chunk_scan_fwd_gluon``, Blackwell), y bf16 kept;
  6. gated RMSNorm (unit weight, eps 1e-5) + bipolar soma + gamma (``dendritic_soma.gated_norm_soma_fwd``): xs, with the
  norm's
     rstd and its bf16 output n kept.
Backward: the soma + gated-norm backward on the kept n (``dendritic_soma.gated_norm_soma_bwd``, dz written into the
projection gradient), the SSD backward on the kept tensors (``dendritic_glue.bwd_store_glue``, dx / dB / dC into the
conv-input gradient, ddt into the projection gradient), the conv backward (dx into the projection gradient; weight /
bias gradients from deterministic partial sums).

Numerics: the composition of the kernels above (each kernel's rounding points are stated in its module docstring); the
Function as a whole is checked against :func:`dendritic_chunk_scan_soma_ref` with the per-operator tolerance rule of
:mod:`neuronspark.utils.testing`.  The forward and backward run as ``torch.library`` custom ops
(``neuronspark::dendritic_chunk_scan_soma_fwd`` / ``_bwd``), opaque to ``torch.compile``.  Requirements: CUDA, Blackwell
(the Gluon chunk scan), the first call eager (the Gluon work split and the dda tile table are built on the first call; a
capture raises), autocast-agnostic (every dtype is explicit; the training regime runs it under bf16 autocast).

Memory kept from forward to backward per call (default configuration, B 8, T 1024): zxbcdt 309 MB (the input), the conv
output 155 MB, y and n 151 MB each, the fp32 passed states 151 MB, CB 8.4 MB, dt / dA_cumsum 4.7 MB each.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from neuronspark.ops.gluon.dendritic_chunk_scan_fwd import chunk_scan_fwd_gluon
from neuronspark.ops.triton import require_cuda, require_sm100
from neuronspark.ops.triton.dendritic_chunk import chunk_cb, chunk_dt_cumsum
from neuronspark.ops.triton.dendritic_conv import (_conv_bwd, _conv_fwd, _ensure_stride, dendritic_conv_silu_ref,
                                                   split_dwb)
from neuronspark.ops.gluon.dendritic_chunk_states import chunk_states_train
from neuronspark.ops.triton.dendritic_glue import bwd_store_glue
from neuronspark.ops.triton.dendritic_scan import ssd_forward_ref
from neuronspark.ops.triton.dendritic_soma import (_gated_norm_ref, bipolar_release, gated_norm_soma_bwd,
                                                 gated_norm_soma_fwd)

__all__ = ["NORM_EPS", "CHUNK_SIZE", "DendriticChunkScanSomaFn", "dendritic_chunk_scan_soma",
           "dendritic_chunk_scan_soma_ref"]

NORM_EPS = 1e-5          # the gated RMSNorm eps of the dendritic neuron population (config dendritic_norm_eps)
CHUNK_SIZE = 256         # the only chunk size the backward kernels cover (32 x 8 dda tiles)
_DT_LIMIT = (0.0, float("inf"))

# (width, device index) -> fp32 ones: the gated norm's unit weight (a documented first-call cache)
_ONES: Dict[Tuple[int, int], torch.Tensor] = {}


def _ones(n: int, device: torch.device) -> torch.Tensor:
    key = (int(n), device.index)
    w = _ONES.get(key)
    if w is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("neuronspark.dendritic_chunk_scan_soma: the first call must be eager (not inside a "
                               "CUDA-graph capture)")
        w = torch.ones(int(n), dtype=torch.float32, device=device)
        _ONES[key] = w
    return w


def _geometry(zxbcdt, conv_w, D, theta):
    nheads = int(D.shape[0])
    dim = int(theta.shape[0])
    headdim = dim // nheads
    dstate = (int(conv_w.shape[0]) - dim) // 2
    return nheads, dim, headdim, dstate


# ------------------------------------------------------------------------------------------------ custom ops
@torch.library.custom_op(
    "neuronspark::dendritic_chunk_scan_soma_fwd", mutates_args=(),
    schema="(Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor A, Tensor D, Tensor theta, "
           "Tensor gamma, int chunk_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")
def _dendritic_chunk_scan_soma_fwd_op(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, chunk_size):
    """(xs (B T, d_inner) bf16, y (B, T, H, P) bf16, rstd (B T,) fp32, conv output (B, T, C) bf16, dt (B, H, nc, cs)
    fp32, dA_cumsum (B, H, nc, cs) fp32, CB (B, nc, 1, cs, cs) fp32, passed states (B, nc, H, P, N) fp32, n (B T,
    d_inner) bf16)"""
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    batch, seqlen, _ = zxbcdt.shape
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
    xc = _conv_fwd(_ensure_stride(xBC).transpose(1, 2), conv_w, conv_b)
    xBC_conv = xc.transpose(1, 2)                                                   # (B, T, C) contiguous
    x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
    x = x.unflatten(-1, (nheads, headdim))
    B = B.unflatten(-1, (1, dstate))
    C = C.unflatten(-1, (1, dstate))
    z = z.unflatten(-1, (nheads, headdim))
    # the combined forward's input normalisation is a no-op for the dendritic-layer layout; refused otherwise so that
    # the
    # kept
    # tensors are the ones the kernels read
    if B.stride(-1) != 1 or C.stride(-1) != 1 or (x.stride(-1) != 1 and x.stride(1) != 1) or D.stride(-1) != 1:
        raise RuntimeError("neuronspark.dendritic_chunk_scan_soma: x / B / C / D are not in the dendritic-layer layout")
    dA_cumsum, dt_sp = chunk_dt_cumsum(dt, A, chunk_size, dt_bias=dt_bias, softplus=True, dt_limit=_DT_LIMIT)
    states_passed, states = chunk_states_train(B, x, dt_sp, dA_cumsum, C.dtype, passed_fp32=True)
    CB = chunk_cb(C, B, chunk_size)
    out_x = chunk_scan_fwd_gluon(CB, x, dt_sp, dA_cumsum, C, states, D)
    x_rms = out_x.view(batch * seqlen, dim)
    z_rms = z.view(batch * seqlen, dim)
    xs, rstd, n = gated_norm_soma_fwd(x_rms, _ones(dim, zxbcdt.device), NORM_EPS, z_rms, theta, gamma)
    return xs, out_x, rstd, xBC_conv, dt_sp, dA_cumsum, CB, states_passed, n


@_dendritic_chunk_scan_soma_fwd_op.register_fake
def _dendritic_chunk_scan_soma_fwd_fake(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, chunk_size):
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    batch, seqlen, _ = zxbcdt.shape
    nchunks = -(-int(seqlen) // int(chunk_size))
    dev = zxbcdt.device
    e = torch.empty
    return (e((batch * seqlen, dim), dtype=torch.bfloat16, device=dev),
            e((batch, seqlen, nheads, headdim), dtype=torch.bfloat16, device=dev),
            e((batch * seqlen,), dtype=torch.float32, device=dev),
            e((batch, seqlen, dim + 2 * dstate), dtype=torch.bfloat16, device=dev),
            e((batch, nheads, nchunks, chunk_size), dtype=torch.float32, device=dev),
            e((batch, nheads, nchunks, chunk_size), dtype=torch.float32, device=dev),
            e((batch, nchunks, 1, chunk_size, chunk_size), dtype=torch.float32, device=dev),
            e((batch, nchunks, nheads, headdim, dstate), dtype=torch.float32, device=dev),
            e((batch * seqlen, dim), dtype=torch.bfloat16, device=dev))


@torch.library.custom_op(
    "neuronspark::dendritic_chunk_scan_soma_bwd", mutates_args=(),
    schema="(Tensor dout, Tensor zxbcdt, Tensor conv_w, Tensor conv_b, Tensor dt_bias, Tensor A, Tensor D, "
           "Tensor theta, Tensor gamma, Tensor out_x, Tensor rstd, Tensor xBC_conv, Tensor dt_sp, Tensor dA_cumsum, "
           "Tensor CB, Tensor states_passed, Tensor n, float alpha, int chunk_size) "
           "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)")
def _dendritic_chunk_scan_soma_bwd_op(dout, zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, out_x, rstd, xBC_conv,
                               dt_sp, dA_cumsum, CB, states_passed, n, alpha, chunk_size):
    """(dzxbcdt bf16 (zxbcdt's shape), dconv (5 C,) fp32 = conv weight gradient flattened then conv bias gradient,
    ddt_bias, dA, dD (H,) fp32, dtheta, dgamma (d_inner,) fp32)"""
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    batch, seqlen, _ = zxbcdt.shape
    widths = [dim, dim + 2 * dstate, nheads]
    z, xBC, dt = torch.split(zxbcdt, widths, dim=-1)
    x, B, C = torch.split(xBC_conv, [dim, dstate, dstate], dim=-1)
    x = x.unflatten(-1, (nheads, headdim))
    B = B.unflatten(-1, (1, dstate))
    C = C.unflatten(-1, (1, dstate))
    dzxbcdt = torch.empty_like(zxbcdt)
    dz, dxBC_given, ddt_given = torch.split(dzxbcdt, widths, dim=-1)
    dxBC = torch.empty_like(xBC)
    dx, dB, dC = torch.split(dxBC, [dim, dstate, dstate], dim=-1)
    dx = dx.unflatten(-1, (nheads, headdim))
    dB = dB.unflatten(-1, (1, dstate))
    dC = dC.unflatten(-1, (1, dstate))
    M = batch * seqlen
    dz = dz.view(M, dim)
    x_rms = out_x.view(M, dim)
    z_rms = z.view(M, dim)
    dy, dz, dtheta, dgamma = gated_norm_soma_bwd(dout.contiguous(), n, x_rms, _ones(dim, zxbcdt.device), rstd, z_rms,
                                                 dz, theta, gamma, alpha)
    dy = dy.view(batch, seqlen, nheads, headdim)
    dx, ddt, dA, dB, dC, dD, ddt_bias = bwd_store_glue(
        dy, x, dt, A, B, C, chunk_size, D, dt_bias, dx=dx, ddt=ddt_given, dB=dB, dC=dC,
        dA_cumsum=dA_cumsum, dt_sp=dt_sp, CB=CB, states_passed=states_passed)
    dxBC_given_update, dconv = _conv_bwd(_ensure_stride(xBC).transpose(1, 2), conv_w, conv_b,
                                         _ensure_stride(dxBC).transpose(1, 2),
                                         _ensure_stride(dxBC_given).transpose(1, 2))
    dxBC_given_update = dxBC_given_update.transpose(1, 2)
    if dxBC_given.stride() != dxBC_given_update.stride():
        dxBC_given.copy_(dxBC_given_update)
    return dzxbcdt, dconv, ddt_bias, dA, dD, dtheta, dgamma


@_dendritic_chunk_scan_soma_bwd_op.register_fake
def _dendritic_chunk_scan_soma_bwd_fake(dout, zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, out_x, rstd,
                                        xBC_conv,
                                 dt_sp, dA_cumsum, CB, states_passed, n, alpha, chunk_size):
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    dev = zxbcdt.device
    e = torch.empty
    return (torch.empty_like(zxbcdt),
            e((5 * int(conv_w.shape[0]),), dtype=torch.float32, device=dev),
            e((nheads,), dtype=torch.float32, device=dev),
            e((nheads,), dtype=torch.float32, device=dev),
            e((nheads,), dtype=torch.float32, device=dev),
            e((dim,), dtype=torch.float32, device=dev),
            e((dim,), dtype=torch.float32, device=dev))


# ------------------------------------------------------------------------------------------------ autograd Function
class DendriticChunkScanSomaFn(torch.autograd.Function):
    """xs = soma(gated_rmsnorm(ssd_scan(conv_silu(x B C), dt, A, D), z), theta, gamma) for the dendritic-layer call,
    store-not-recompute (see the module docstring).  Runs in the ambient autocast state (forward) and re-enters it in
    the backward (``torch.amp.custom_fwd`` / ``custom_bwd`` without input casts)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, alpha, chunk_size):
        (xs, out_x, rstd, xBC_conv, dt_sp, dA_cumsum, CB, states_passed,
         n) = torch.ops.neuronspark.dendritic_chunk_scan_soma_fwd(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma,
                                                            int(chunk_size))
        ctx.save_for_backward(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, out_x, rstd, xBC_conv, dt_sp,
                              dA_cumsum, CB, states_passed, n)
        ctx.alpha = float(alpha)
        ctx.chunk_size = int(chunk_size)
        return xs

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        (zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, out_x, rstd, xBC_conv, dt_sp, dA_cumsum, CB, states_passed,
         n) = ctx.saved_tensors
        if dout.dtype != torch.bfloat16 or tuple(dout.shape) != (int(zxbcdt.shape[0]) * int(zxbcdt.shape[1]),
                                                                  int(theta.shape[0])):
            raise RuntimeError("neuronspark.dendritic_chunk_scan_soma: the incoming gradient must be bf16 "
                               f"({int(zxbcdt.shape[0]) * int(zxbcdt.shape[1])}, {int(theta.shape[0])}), got "
                               f"{dout.dtype} {tuple(dout.shape)}")
        dzxbcdt, dconv, ddt_bias, dA, dD, dtheta, dgamma = torch.ops.neuronspark.dendritic_chunk_scan_soma_bwd(
            dout, zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, out_x, rstd, xBC_conv, dt_sp, dA_cumsum, CB,
            states_passed, n, ctx.alpha, ctx.chunk_size)
        dweight, dbias = split_dwb(dconv, int(conv_w.shape[0]))
        need = ctx.needs_input_grad
        return (dzxbcdt, dweight, dbias, ddt_bias, dA, dD, dtheta if need[6] else None, dgamma if need[7] else None,
                None, None)


# ------------------------------------------------------------------------------------------------ public
def _check_call(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, alpha, chunk_size):
    bad = []
    tensors = (("zxbcdt", zxbcdt), ("conv_w", conv_w), ("conv_b", conv_b), ("dt_bias", dt_bias), ("A", A), ("D", D),
               ("theta", theta), ("gamma", gamma))
    for name, t in tensors:
        if not isinstance(t, torch.Tensor) or not t.is_cuda or t.device != zxbcdt.device:
            raise RuntimeError(f"neuronspark.dendritic_chunk_scan_soma: {name} must be a CUDA tensor on "
                               f"{zxbcdt.device}")
    if zxbcdt.dim() != 3 or zxbcdt.dtype != torch.bfloat16 or not zxbcdt.is_contiguous():
        bad.append(f"zxbcdt must be a contiguous bf16 (B, T, width) tensor, got {zxbcdt.dtype} {tuple(zxbcdt.shape)}")
    if D.dim() != 1 or not D.is_contiguous():
        bad.append("D must be a contiguous (nheads,) tensor")
    else:
        nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
        if dim % nheads or headdim not in (16, 32, 64, 128) or dstate <= 0:
            bad.append(f"d_inner {dim} / nheads {nheads} / dstate {dstate}")
        if zxbcdt.dim() == 3 and int(zxbcdt.shape[-1]) != 2 * dim + 2 * dstate + nheads:
            bad.append(f"zxbcdt width {int(zxbcdt.shape[-1])} != 2 d_inner + 2 N + H = {2 * dim + 2 * dstate + nheads}")
        for name, t, n in (("dt_bias", dt_bias, nheads), ("A", A, nheads), ("theta", theta, dim),
                           ("gamma", gamma, dim), ("conv_b", conv_b, dim + 2 * dstate)):
            if t.dtype != torch.float32 or tuple(t.shape) != (n,) or not t.is_contiguous():
                bad.append(f"{name} must be a contiguous fp32 ({n},) tensor, got {t.dtype} {tuple(t.shape)}")
        if (conv_w.dtype != torch.float32 or tuple(conv_w.shape) != (dim + 2 * dstate, 4)
                or not conv_w.is_contiguous()):
            bad.append(f"conv_w must be a contiguous fp32 ({dim + 2 * dstate}, 4) tensor")
        if D.dtype != torch.float32:
            bad.append("D must be fp32")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size != CHUNK_SIZE:
        bad.append(f"chunk_size must be {CHUNK_SIZE}, got {chunk_size!r}")
    elif zxbcdt.dim() == 3 and int(zxbcdt.shape[1]) % chunk_size:
        bad.append(f"T = {int(zxbcdt.shape[1])} must be a multiple of the chunk size {chunk_size} (full chunks)")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not float(alpha) > 0.0:
        bad.append(f"alpha must be a number > 0, got {alpha!r}")
    if bad:
        raise RuntimeError("neuronspark.dendritic_chunk_scan_soma: the operator covers the dendritic neuron layer's "
                           "call only; "
                           + "; ".join(bad))


def dendritic_chunk_scan_soma(zxbcdt: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor, dt_bias: torch.Tensor,
                       A: torch.Tensor, D: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor, *,
                       alpha: float = 8.0, chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """xs (B T, d_inner) bf16: the dendritic neuron layer's conv, SSD chunk scan, gated norm and soma read-out (see the
    module
    docstring).  Differentiable in every tensor argument."""
    require_cuda(zxbcdt, "dendritic_chunk_scan_soma")
    require_sm100(zxbcdt.device, "dendritic_chunk_scan_soma")
    _check_call(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, alpha, chunk_size)
    return DendriticChunkScanSomaFn.apply(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, float(alpha),
                                        int(chunk_size))


def dendritic_chunk_scan_soma_ref(zxbcdt: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor,
                                  dt_bias: torch.Tensor,
                           A: torch.Tensor, D: torch.Tensor, theta: torch.Tensor, gamma: torch.Tensor, *,
                           alpha: float = 8.0, chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """Pure torch (differentiable): the conv tap chain (``dendritic_conv_silu_ref``), the float32 chunked SSD forward
    with the kernels' softplus form (``ssd_forward_ref``), the fp32 gated RMSNorm (unit weight, eps 1e-5) rounded to
    bf16, the bipolar release with the sigmoid surrogate (``bipolar_release``) times gamma, rounded to bf16. Storage
    roundings kept at the kernels' tensor boundaries (conv output, y, n, xs: bf16); no rounding inside the scan. The
    scan is the
    chunked form of the per-token recurrence."""
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    batch, seqlen, _ = zxbcdt.shape
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
    xbc = dendritic_conv_silu_ref(xBC, conv_w, conv_b)
    x, B, C = torch.split(xbc, [dim, dstate, dstate], dim=-1)
    y = ssd_forward_ref(x.unflatten(-1, (nheads, headdim)), dt, A, B.unflatten(-1, (1, dstate)),
                        C.unflatten(-1, (1, dstate)), chunk_size, D, dt_bias)
    M = batch * seqlen
    yn, _ = _gated_norm_ref(y.reshape(M, dim), torch.ones(dim, dtype=torch.float32, device=zxbcdt.device), NORM_EPS,
                            z.reshape(M, dim))
    n = yn.to(torch.bfloat16)
    s = bipolar_release(n.float(), theta, alpha)
    return (s * gamma).to(torch.bfloat16)
