"""Pure-torch dendritic-layer recurrence: the module-level oracle of
:class:`neuronspark.modules.dendritic_neuron_layer.DendriticNeurons`.

The mixer is evaluated token by token, in a chosen dtype (fp32, the default, or fp64), with no chunking, no kernel and
no fused operator::

    hn   = h * rsqrt(mean(h^2) + eps) * pre_norm_gain
    zxbcdt = W_in hn;   z, xBC, d = split(zxbcdt)
    xbc  = SiLU(conv_bias + sum_j conv_weight[:, j] xBC_{t-3+j})          (the carried tail before the first token)
    dt   = softplus_k(d + dt_bias),  a = exp(-dt exp(A_log))
    S    = a S + (dt x) B^T;   y = S C + D x                               (the carried state, else zero)
    n    = g rsqrt(mean(g^2) + eps),  g = y SiLU(z)
    s    = max(n - theta, 0) - max(-n - theta, 0),  theta = theta_floor + softplus(rho)
    upd  = W_out (gamma s)

``softplus_k`` is the selective-scan kernels' form ``where(v <= 20, log(exp(v) + 1), v)`` (Triton >= 3 ``softplus``),
not ``F.softplus``.  The soma is the bipolar release with the model's sigmoid surrogate backward
(:class:`BipolarReleaseSurrogateFn`), so the oracle is differentiable with the gradients the model trains with.

``round_like_kernels=True`` rounds to bf16 at the tensor boundaries where the model's kernels round (the pre-norm
output, the GEMM operands and outputs, the conv output, y, n and the soma read-out) and keeps every other step in
``dtype``; the bf16 rounding of the inter-chunk states inside the whole-row chunked scan is not modelled (the oracle
has no chunks).  The default (False) is the plain recurrence.

Entry points (``params``: an :class:`DendriticNeurons` / :class:`DendriticNeuronLayer`, or a mapping of the ten
parameter tensors by their state-dict names; ``state``: an :class:`~neuronspark.utils.cache.DendriticState` or a mapping
with ``"conv"`` / ``"ssm"``):

* :func:`dendritic_neuron_layer_ref` ``(params, h, state=None)``: ``h`` (T, B, D) -> the update (T, B, D); with
  ``state`` the row continues from the carried conv tail and compartments, which are overwritten in place with the final
  ones (in their own dtypes).
* :func:`dendritic_step_ref` ``(params, h_t, state)``: one token ``h_t`` (B, D) -> the update (B, D); ``state``
  advanced.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BipolarReleaseSurrogateFn",
    "bipolar_release_ref",
    "softplus_kernel_form",
    "dendritic_params",
    "dendritic_neuron_layer_ref",
    "dendritic_step_ref",
    "DENDRITIC_PARAM_NAMES",
]

DENDRITIC_PARAM_NAMES = ("pre_norm_gain", "in_proj.weight", "conv_weight", "conv_bias", "dt_bias", "A_log", "D",
                         "gamma",
                  "W_out", "rho")


class BipolarReleaseSurrogateFn(torch.autograd.Function):
    """``S = max(v - theta, 0) - max(-v - theta, 0)``; backward ``dS/dv = sig(alpha (v - theta)) + sig(alpha (-v -
    theta))``, ``dS/dtheta = -(sig(alpha (v - theta)) - sig(alpha (-v - theta)))``."""

    @staticmethod
    def forward(ctx, v_pre: torch.Tensor, theta: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(v_pre, theta)
        ctx.alpha = float(alpha)
        s_pos = torch.clamp_min(v_pre - theta, 0.0)
        s_neg = torch.clamp_min(-v_pre - theta, 0.0)
        return s_pos - s_neg

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v_pre, theta = ctx.saved_tensors
        alpha = ctx.alpha
        gate_pos = torch.sigmoid((v_pre - theta) * alpha)
        gate_neg = torch.sigmoid((-v_pre - theta) * alpha)
        return grad_output * (gate_pos + gate_neg), -grad_output * (gate_pos - gate_neg), None


def bipolar_release_ref(v_pre: torch.Tensor, theta: torch.Tensor, alpha: float = 8.0) -> torch.Tensor:
    """The dendritic-layer soma with its surrogate backward (``alpha`` > 0)."""
    if float(alpha) <= 0.0:
        raise ValueError(f"bipolar_release_ref: alpha must be positive, got {alpha}")
    if theta.dtype != v_pre.dtype:
        theta = theta.to(v_pre.dtype)
    return BipolarReleaseSurrogateFn.apply(v_pre, theta, float(alpha))


def softplus_kernel_form(v: torch.Tensor) -> torch.Tensor:
    """``where(v <= 20, log(exp(v) + 1), v)``: the dt softplus of the selective-scan kernels."""
    return torch.where(v <= 20.0, torch.log(torch.exp(v) + 1.0), v)


def dendritic_params(params: Any) -> Dict[str, torch.Tensor]:
    """The ten mixer parameters by state-dict name, from a DendriticNeurons, a DendriticNeuronLayer or a mapping."""
    if isinstance(params, nn.Module):
        module = getattr(params, "mixer_a", params)
        own = dict(module.named_parameters())
    elif isinstance(params, Mapping):
        own = dict(params)
    else:
        raise TypeError("dendritic_params: expected a DendriticNeurons / DendriticNeuronLayer or a mapping, got "
                        f"{type(params).__name__}")
    missing = [n for n in DENDRITIC_PARAM_NAMES if n not in own]
    if missing:
        raise KeyError(f"dendritic_params: missing parameters {missing}")
    return {n: own[n] for n in DENDRITIC_PARAM_NAMES}


def _state_tensors(state: Any):
    if isinstance(state, Mapping):
        return state["conv"], state["ssm"]
    return state.conv, state.ssm


def _shapes(p: Dict[str, torch.Tensor]):
    d_inner = int(p["gamma"].shape[0])
    nheads = int(p["A_log"].shape[0])
    conv_dim = int(p["conv_weight"].shape[0])
    state_size = (conv_dim - d_inner) // 2
    head_dim = d_inner // nheads
    width = int(p["conv_weight"].shape[1])
    return d_inner, nheads, head_dim, state_size, conv_dim, width


def _rounder(round_like_kernels: bool, cd: torch.dtype):
    if not round_like_kernels:
        return lambda t: t
    return lambda t: t.to(torch.bfloat16).to(cd)


def _pre_norm(h: torch.Tensor, gain: torch.Tensor, eps: float, cd: torch.dtype) -> torch.Tensor:
    x = h.to(cd)
    rstd = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return x * rstd * gain.to(cd)


def _readout(p: Dict[str, torch.Tensor], y: torch.Tensor, z: torch.Tensor, theta_floor: float, norm_eps: float,
             alpha: float, cd: torch.dtype, bf) -> torch.Tensor:
    g = y * F.silu(z)
    n = bf(g * torch.rsqrt(g.square().mean(dim=-1, keepdim=True) + norm_eps))
    theta = theta_floor + F.softplus(p["rho"].to(cd))
    s = bipolar_release_ref(n, theta, alpha)
    xs = bf(s * p["gamma"].to(cd))
    return bf(F.linear(xs, bf(p["W_out"].to(cd))))


def dendritic_neuron_layer_ref(params: Any, h: torch.Tensor, state: Any = None, *, theta_floor: float = 0.02,
                 norm_eps: float = 1e-5, alpha: float = 8.0, dtype: torch.dtype = torch.float32,
                 round_like_kernels: bool = False) -> torch.Tensor:
    """The mixer's update for the stream ``h`` (T, B, D) as a per-token recurrence in ``dtype`` (fp32 / fp64);
    returns (T, B, D) in ``dtype``.  ``state`` (conv (B, conv_dim, 3), ssm (B, H, P, N)): start from it and write the
    final tail and compartments back in place; None: the row starts from zero."""
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"dendritic_neuron_layer_ref: dtype must be float32 or float64, got {dtype}")
    if h.dim() != 3:
        raise ValueError(f"dendritic_neuron_layer_ref: expected h (T, B, D), got {tuple(h.shape)}")
    p = dendritic_params(params)
    cd = dtype
    bf = _rounder(round_like_kernels, cd)
    di, H, P, N, C, K = _shapes(p)
    T, b, _ = (int(s) for s in h.shape)
    hn = bf(_pre_norm(h, p["pre_norm_gain"], norm_eps, cd)).transpose(0, 1)           # (B, T, D)
    zxbcdt = bf(F.linear(hn, bf(p["in_proj.weight"].to(cd))))
    z, xBC, dtr = zxbcdt.split([di, C, H], dim=-1)
    # delay compartments xBC_{t-1..t-3}: the carried tail, else zero; tap j multiplies xBC_{t-3+j}
    if state is None:
        padded = F.pad(xBC.transpose(1, 2), (K - 1, 0))                                   # (B, C, T + 3)
    else:
        conv_state, _ = _state_tensors(state)
        padded = torch.cat([conv_state.to(cd), xBC.transpose(1, 2)], dim=-1)
    pre = p["conv_bias"].to(cd)[None, :, None].expand(b, C, T)
    for j in range(K):
        pre = pre + p["conv_weight"].to(cd)[None, :, j, None] * padded[:, :, j:j + T]
    if state is not None:
        conv_state.copy_(padded[..., -(K - 1):].to(conv_state.dtype))
    xbc = bf(F.silu(pre).transpose(1, 2))                                                  # (B, T, C)
    x, Bm, Cm = xbc.split([di, N, N], dim=-1)
    x = x.reshape(b, T, H, P)
    dt = softplus_kernel_form(dtr + p["dt_bias"].to(cd))                                  # (B, T, H)
    a = torch.exp(-dt * torch.exp(p["A_log"].to(cd)))
    Dv = p["D"].to(cd)
    if state is None:
        Hs = torch.zeros(b, H, P, N, dtype=cd, device=h.device)
    else:
        _, ssm_state = _state_tensors(state)
        Hs = ssm_state.to(cd)
    ys = []
    for t in range(T):
        Hs = a[:, t, :, None, None] * Hs + (dt[:, t, :, None] * x[:, t])[..., None] * Bm[:, t, None, None, :]
        ys.append(torch.einsum("bhpn,bn->bhp", Hs, Cm[:, t]) + Dv[None, :, None] * x[:, t])
    if state is not None:
        ssm_state.copy_(Hs.to(ssm_state.dtype))
    y = bf(torch.stack(ys, dim=1).reshape(b, T, di))
    out = _readout(p, y, z, theta_floor, norm_eps, alpha, cd, bf)                        # (B, T, D)
    return out.transpose(0, 1)


def dendritic_step_ref(params: Any, h_t: torch.Tensor, state: Any, *, theta_floor: float = 0.02, norm_eps: float = 1e-5,
                alpha: float = 8.0, dtype: torch.dtype = torch.float32,
                round_like_kernels: bool = False) -> torch.Tensor:
    """One token: ``h_t`` (B, D) -> the mixer's update (B, D) in ``dtype``; ``state`` (conv, ssm) advanced in place
    (the compartments stored in their own dtype, i.e. rounded once per step for a bf16 state)."""
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"dendritic_step_ref: dtype must be float32 or float64, got {dtype}")
    if h_t.dim() != 2:
        raise ValueError(f"dendritic_step_ref: expected h_t (B, D), got {tuple(h_t.shape)}")
    p = dendritic_params(params)
    cd = dtype
    bf = _rounder(round_like_kernels, cd)
    di, H, P, N, C, K = _shapes(p)
    b = int(h_t.shape[0])
    conv_state, ssm_state = _state_tensors(state)
    hn = bf(_pre_norm(h_t, p["pre_norm_gain"], norm_eps, cd))
    zxbcdt = bf(F.linear(hn, bf(p["in_proj.weight"].to(cd))))
    z, xBC, dtr = zxbcdt.split([di, C, H], dim=-1)
    window = torch.cat([conv_state.to(cd), xBC[:, :, None]], dim=-1)                      # (B, C, 4): lags 3, 2, 1, 0
    pre = (window * p["conv_weight"].to(cd)[None]).sum(-1) + p["conv_bias"].to(cd)
    conv_state.copy_(window[..., 1:].to(conv_state.dtype))
    xbc = bf(F.silu(pre))
    x, Bm, Cm = xbc.split([di, N, N], dim=-1)
    x = x.reshape(b, H, P)
    dt = softplus_kernel_form(dtr + p["dt_bias"].to(cd))
    a = torch.exp(-dt * torch.exp(p["A_log"].to(cd)))
    Hs = a[:, :, None, None] * ssm_state.to(cd) + (dt[:, :, None] * x)[..., None] * Bm[:, None, None, :]
    ssm_state.copy_(Hs.to(ssm_state.dtype))
    y = bf((torch.einsum("bhpn,bn->bhp", Hs, Cm) + p["D"].to(cd)[None, :, None] * x).reshape(b, di))
    return _readout(p, y, z, theta_floor, norm_eps, alpha, cd, bf)
