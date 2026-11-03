"""The dendritic neuron layer's chunk scan with predictive plasticity and its soma read-out (training regime).

Predictive plasticity.  In the Hebbian form (``dendritic_combined``) a compartment integrates the token's synaptic
input whatever it already holds: ``S_t = a_t S_{t-1} + dt_t x_t B_t^T``.  Under predictive plasticity (Urbanczik &
Senn, 2014: a dendritic synapse changes by the mismatch between the somatic activity and the dendrite's own
prediction of it, times the presynaptic activity) the compartments first read out what they predict for the
presynaptic pattern, and only the prediction error is written::

    b_t   = B_t / |B_t|                                   the unit presynaptic pattern (shared by the heads)
    eta_t = 1 - exp(-dt_t)                                the plasticity rate, from the same step dt_t
    S_t   = a_t S_{t-1} + eta_t (x_t - a_t S_{t-1} b_t) b_t^T      a_t = exp(dt_t A): the leak, as before
    y_t   = S_t C_t + D x_t

A pattern seen before is overwritten, not summed with; a pattern never seen is written at the rate eta_t.  No new
parameter: the rate comes from dt, the pattern is the normalised B.  The soma read-out (gated norm, bipolar release,
gamma) is the Hebbian layer's.

Registry op ``dendritic_predictive_scan_soma(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, *, alpha=8.0,
chunk_size=256) -> xs``: the signature of ``dendritic_chunk_scan_soma`` (see its module for the operands).  The
kernel path composes the layer's own kernels for the synaptic filter (``dendritic_conv``) and the soma
(``dendritic_soma``) with the chunked scan of the predictive rule, ``flash-linear-attention``'s
``chunk_gated_delta_rule`` (Triton; the rule above is the gated delta rule with the key normalised, ``beta`` =
eta_t, ``g`` = log a_t, scale 1, one key head for the shared pattern and one value head per dendritic head), each
piece with its own backward (store-not-recompute per piece); the op runs with autocast disabled (the pieces fix their
dtypes: bf16 at the tensor boundaries, fp32 for dt, the leak and the rate).  ``dendritic_predictive_scan_soma_ref`` is
the pure-torch definition: the per-token recurrence above in fp32 after the conv reference, then the fp32 gated norm
and the soma as in ``dendritic_chunk_scan_soma_ref``.

Numerics of the kernel path, per operator against the pure-torch references (``neuronspark.utils.testing`` rule, abs
and rel 1e-2): the conv and the soma pieces are the layer's kernels; the scan's forward (y) passes the rule; the scan's
backward keeps the WY representation of the chunked delta rule in bf16 inside the third-party kernel, so its gradients
(x, the unit pattern, the read pattern, log a, eta) carry unbiased rounding noise of relative RMS 3-4e-3 (one bf16 ulp;
no bias, no structure over chunk positions) and a few per mille of the entries fall outside the per-element rule.  This
is below the standard of the package's own kernels and is the known cost of the experiment dependency.  Requirements
of the kernel path: CUDA and ``flash-linear-attention`` (an experiment dependency, not of the package), T a multiple of
``chunk_size``.  The inference regimes (prefill, decode) of predictive plasticity are not implemented: a layer with
this rule refuses them.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from neuronspark.ops.dendritic_combined import CHUNK_SIZE, NORM_EPS, _check_call, _geometry, _ones
from neuronspark.ops.triton import require_cuda
from neuronspark.ops.triton.dendritic_conv import (_conv_bwd, _conv_fwd, _ensure_stride, dendritic_conv_silu_ref,
                                                   split_dwb)
from neuronspark.ops.triton.dendritic_soma import _gated_norm_ref, bipolar_release, gated_norm_soma_bwd, gated_norm_soma_fwd

__all__ = ["dendritic_predictive_scan_soma", "dendritic_predictive_scan_soma_ref", "predictive_scan_ref",
           "predictive_recurrence_ref"]


# ------------------------------------------------------------------------------------------------ pieces (kernel path)
class _ConvSiLU(torch.autograd.Function):
    """SiLU(causal depthwise conv) on the (B, T, C) projection slice with the layer's conv kernels."""

    @staticmethod
    def forward(ctx, xBC, conv_w, conv_b):
        xc = _conv_fwd(_ensure_stride(xBC).transpose(1, 2), conv_w, conv_b)
        ctx.save_for_backward(xBC, conv_w, conv_b)
        return xc.transpose(1, 2)                                     # (B, T, C) contiguous bf16

    @staticmethod
    def backward(ctx, dout):
        xBC, conv_w, conv_b = ctx.saved_tensors
        dx = torch.empty_like(_ensure_stride(xBC))
        dx_update, dconv = _conv_bwd(_ensure_stride(xBC).transpose(1, 2), conv_w, conv_b,
                                     _ensure_stride(dout.contiguous()).transpose(1, 2), dx.transpose(1, 2))
        dx_update = dx_update.transpose(1, 2)
        if dx.stride() != dx_update.stride():
            dx.copy_(dx_update)
        dweight, dbias = split_dwb(dconv, int(conv_w.shape[0]))
        return dx, dweight, dbias


class _GatedNormSoma(torch.autograd.Function):
    """The gated RMSNorm (unit weight) and the bipolar soma with the layer's soma kernels: x (M, N) bf16, z (M, N)."""

    @staticmethod
    def forward(ctx, x, weight, eps, z, theta, gamma, alpha):
        xs, rstd, n = gated_norm_soma_fwd(x, weight, float(eps), z, theta, gamma)
        ctx.save_for_backward(x, weight, z, theta, gamma, rstd, n)
        ctx.alpha = float(alpha)
        return xs

    @staticmethod
    def backward(ctx, dxs):
        x, weight, z, theta, gamma, rstd, n = ctx.saved_tensors
        dz = torch.empty_like(z)
        dy, dz, dtheta, dgamma = gated_norm_soma_bwd(dxs.contiguous(), n, x, weight, rstd, z, dz, theta, gamma,
                                                     ctx.alpha)
        need = ctx.needs_input_grad
        return dy, None, None, dz, dtheta if need[4] else None, dgamma if need[5] else None, None


def _check(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, alpha, chunk_size):
    """The Hebbian op's call contract (same operands), reported under this op's name."""
    try:
        _check_call(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, alpha, chunk_size)
    except RuntimeError as e:
        raise RuntimeError(str(e).replace("dendritic_chunk_scan_soma", "dendritic_predictive_scan_soma")) from None


def _scan_operands(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta):
    """What the scan reads, from the projection (differentiable): z (B, T, d_inner) bf16 view of zxbcdt, x (B, T, H, P)
    bf16 (the somatic signal to express), unit (B, T, 1, N) bf16 (the unit presynaptic pattern), read (B, T, 1, N) bf16
    (the read-out pattern), log_a (B, T, H) fp32 (log of the leak), eta (B, T, H) fp32 (the plasticity rate)."""
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    batch, seqlen, _ = zxbcdt.shape
    z, xBC, dt_raw = torch.split(zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
    xbc = _ConvSiLU.apply(xBC, conv_w, conv_b)                        # (B, T, C) bf16
    x = xbc[..., :dim].contiguous().view(batch, seqlen, nheads, headdim)
    pattern = xbc[..., dim:dim + dstate].contiguous().view(batch, seqlen, 1, dstate)
    read = xbc[..., dim + dstate:].contiguous().view(batch, seqlen, 1, dstate)
    unit = F.normalize(pattern.float(), dim=-1).to(pattern.dtype)
    dt = F.softplus(dt_raw.float() + dt_bias)                         # (B, T, H) fp32: the step
    return z, x, unit, read, dt * A, -torch.expm1(-dt)


# ------------------------------------------------------------------------------------------------ public
def dendritic_predictive_scan_soma(zxbcdt: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor,
                                   dt_bias: torch.Tensor, A: torch.Tensor, D: torch.Tensor, theta: torch.Tensor,
                                   gamma: torch.Tensor, *, alpha: float = 8.0,
                                   chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """xs (B T, d_inner) bf16: the conv, the predictive-plasticity scan, the gated norm and the soma read-out (see the
    module docstring).  Differentiable in every tensor argument."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    require_cuda(zxbcdt, "dendritic_predictive_scan_soma")
    _check(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta, gamma, alpha, chunk_size)
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    batch, seqlen, _ = zxbcdt.shape
    with torch.autocast("cuda", enabled=False):                      # the op fixes its own dtypes (see the docstring)
        z, x, unit, read, log_a, eta = _scan_operands(zxbcdt, conv_w, conv_b, dt_bias, A, D, theta)
        y, _ = chunk_gated_delta_rule(read, unit, x, log_a, eta, scale=1.0)   # (B, T, H, P) in x's dtype
        y = (y.float() + D.view(1, 1, nheads, 1) * x.float()).to(torch.bfloat16)
        M = batch * seqlen
        return _GatedNormSoma.apply(y.reshape(M, dim), _ones(dim, zxbcdt.device), NORM_EPS, z.reshape(M, dim), theta,
                                    gamma, alpha)


def predictive_recurrence_ref(x: torch.Tensor, unit: torch.Tensor, read: torch.Tensor, log_a: torch.Tensor,
                              eta: torch.Tensor) -> torch.Tensor:
    """The per-token recurrence of predictive plasticity in fp32 (differentiable), on the scan's operands: x (b, l, h, P),
    unit / read (b, l, 1, N), log_a / eta (b, l, h) -> S_t C_t (b, l, h, P) fp32 (no D x, no rounding)."""
    b, l, h, P = (int(s) for s in x.shape)
    N = int(unit.shape[-1])
    xf, a = x.float(), torch.exp(log_a.float())
    uf, rf, ef = unit.float().reshape(b, l, N), read.float().reshape(b, l, N), eta.float()
    S = xf.new_zeros(b, h, P, N)
    ys = []
    for t in range(l):
        S = a[:, t].reshape(b, h, 1, 1) * S
        prediction = torch.einsum("bhpn,bn->bhp", S, uf[:, t])
        error = xf[:, t] - prediction
        S = S + ef[:, t].reshape(b, h, 1, 1) * error.unsqueeze(-1) * uf[:, t].reshape(b, 1, 1, N)
        ys.append(torch.einsum("bhpn,bn->bhp", S, rf[:, t]))
    return torch.stack(ys, dim=1)


def predictive_scan_ref(x: torch.Tensor, dt_raw: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                        D: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """The predictive-plasticity scan in fp32 (differentiable), with the Hebbian scan reference's operands: x (b, l, h,
    P), dt_raw (b, l, h), A / D / dt_bias (h,), B / C (b, l, 1, N) -> y = S_t C_t + D x_t (b, l, h, P) in x's dtype."""
    h = int(x.shape[2])
    dt = F.softplus(dt_raw.float() + dt_bias.float())                 # (b, l, h)
    y = predictive_recurrence_ref(x, F.normalize(B.float(), dim=-1), C, dt * A.float(), -torch.expm1(-dt))
    return (y + D.float().reshape(1, 1, h, 1) * x.float()).to(x.dtype)


def dendritic_predictive_scan_soma_ref(zxbcdt: torch.Tensor, conv_w: torch.Tensor, conv_b: torch.Tensor,
                                       dt_bias: torch.Tensor, A: torch.Tensor, D: torch.Tensor, theta: torch.Tensor,
                                       gamma: torch.Tensor, *, alpha: float = 8.0,
                                       chunk_size: int = CHUNK_SIZE) -> torch.Tensor:
    """Pure torch (differentiable): the conv tap chain (``dendritic_conv_silu_ref``), the fp32 per-token predictive
    recurrence (``predictive_scan_ref``), the fp32 gated RMSNorm (unit weight, eps 1e-5) rounded to bf16, the bipolar
    release with the sigmoid surrogate times gamma, rounded to bf16 (the roundings of ``dendritic_chunk_scan_soma_ref``)."""
    nheads, dim, headdim, dstate = _geometry(zxbcdt, conv_w, D, theta)
    batch, seqlen, _ = zxbcdt.shape
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * dstate, nheads], dim=-1)
    xbc = dendritic_conv_silu_ref(xBC, conv_w, conv_b)
    x, B, C = torch.split(xbc, [dim, dstate, dstate], dim=-1)
    y = predictive_scan_ref(x.unflatten(-1, (nheads, headdim)), dt, A, B.unflatten(-1, (1, dstate)),
                            C.unflatten(-1, (1, dstate)), D, dt_bias)
    M = batch * seqlen
    yn, _ = _gated_norm_ref(y.reshape(M, dim), torch.ones(dim, dtype=torch.float32, device=zxbcdt.device), NORM_EPS,
                            z.reshape(M, dim))
    n = yn.to(torch.bfloat16)
    s = bipolar_release(n.float(), theta, alpha)
    return (s * gamma).to(torch.bfloat16)
