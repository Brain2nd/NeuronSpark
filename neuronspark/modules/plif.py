"""PLIF neuron population of the hippocampal layers: TTT-eta projected-gate leaky integration with the bipolar
soft-reset release, plus the small helpers shared by the hippocampal-layer modules.

Per channel c and row b, input x (T, B, C) (fp32 for the attention-side ``gate_neuron`` and the centred
``input_neuron2``, bf16 for the spiking FFN's gate / up neurons):

    xc   = x                                                  (plain neurons)
         = (x_t - S_t / max(t, 1)) [t > 0],  S_t = sum_{j<t} x_j   (centred neuron, fp32, cast back to x.dtype)
    zp   = bf16( bf16(xc @ eta_V^T) @ eta_U^T )               two bf16 GEMMs (rank 16)
    z    = (w_eta * f32(x) + b_eta) + f32(zp)                  fp32, one RN rounding per op
    eta  = sigmoid(z);  a = 1 - eta;  drive = eta * f32(x)
    pre  = a_t * v_{t-1} + drive_t                             v_{-1} = carried membrane (0 at row start)
    S_t  = max(pre - v_th, 0) - max(-pre - v_th, 0)            bipolar release, v_th (C,) per channel
    v_t  = pre - S_t                                           soft reset: the membrane stays in [-v_th, v_th]

S is returned in x.dtype.  Backward (training form only): the sigmoid surrogate of slope ``alpha`` (8.0) on both
release sides, dS/dpre = sigmoid(alpha (pre - v_th)) + sigmoid(alpha (-pre - v_th)) and
dS/dv_th = -(sigmoid(alpha (pre - v_th)) - sigmoid(alpha (-pre - v_th))).

Forms (one per regime; the module picks the form from its arguments):

* whole row (``state is None``): ``ops.plif_centred_input`` (centred neuron) -> the two eta linears ->
  ``ops.plif_scan_train`` (forward kernel + backward kernel), from a zero membrane;
* prefill (``state`` given, T >= 2, no autograd): ``ops.centred_running`` over the carried input sum (centred
  neuron; written in bf16, the operand dtype of the first eta linear, so the linear's input cast is the kernel's
  store rounding) -> the two eta linears -> ``ops.plif_gate_prefill`` (a, drive; it reads the bf16 projection and
  widens it on load, exactly as an fp32 cast would) -> ``ops.plif_scan_prefill``, which carries the membrane
  ``state.v`` in place;
* step (``state`` given, T == 1, no autograd): the same expressions in torch, one token, membrane updated in place.

Parameters (state-dict keys): ``w`` (C,) frozen launch seed of the retention, ``v_th`` (C,), ``w_eta`` (C,), ``b_eta``
(C,), ``eta_V`` (rank, C), ``eta_U`` (C, rank).  ``b_eta`` is computed from ``w`` in the constructor
(``b_eta = logit(1 - sigmoid(clamp(w, max=logit(0.99))))``, i.e. the launch gate eta = 1 / tau); nothing in this
module writes a parameter after construction, so a loaded checkpoint keeps its trained ``b_eta``.

Launch draws (``init="launch"``, current RNG stream, in this order): ``w`` ~ N(-log(tau - 1), 0.5) for a scalar
``init_tau``, or log(tau_c - 1) + N(0, 0.5) with tau_c log-uniform over the channels for a (tau_min, tau_max) spectrum;
``v_th`` ~ U(0.15, 0.40); ``eta_V`` Xavier-uniform; ``w_eta`` = ``eta_U`` = 0.  ``init="empty"`` allocates every
parameter uninitialised (no draw), for a build followed by a checkpoint load.
"""
from __future__ import annotations

import contextlib
import math
from typing import Iterator, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from neuronspark.ops import namespace
from neuronspark.utils.cache import PLIFState

__all__ = [
    "PLIFNeuron",
    "W_CLAMP_MAX",
    "V_TH_INIT",
    "W_INIT_STD",
    "INITS",
    "plif_b_eta_launch",
    "centred_running_step",
    "rows_rmsnorm",
    "forked_rng",
    "check_init",
    "linear_init",
]

# beta = sigmoid(clamp(w, max=W_CLAMP_MAX)) <= 0.99 (logit(0.99))
W_CLAMP_MAX = 4.595119850134589
# launch threshold spread: v_th ~ U(0.15, 0.40) per channel
V_TH_INIT = (0.15, 0.40)
# launch jitter of the frozen retention seed w
W_INIT_STD = 0.5
# clamp of the launch retained fraction inside the b_eta logit
_ONE_M_BETA_EPS = 1e-6
# construction modes of every hippocampal-layer module
INITS = ("launch", "empty")


# ------------------------------------------------------------------------------------------------ shared helpers
def check_init(init: str) -> str:
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    return init


def linear_init(init: str) -> str:
    """The ``neuronspark.modules.linear.Linear`` init of a projection: PyTorch's default draw at launch (overwritten by
    the module's own launch distribution where it has one), no draw for an empty build."""
    return "default" if check_init(init) == "launch" else "empty"


@contextlib.contextmanager
def forked_rng(device: Optional[Union[str, torch.device]], base: int, layer_idx: int) -> Iterator[None]:
    """Draw on a forked RNG stream seeded ``mechanism_init_seed(base, layer_idx)`` (the CUDA stream of ``device`` is
    forked too when it is a CUDA device); the global streams are restored on exit.  Also disables autograd."""
    from neuronspark.modules import mechanism_init_seed

    dev = torch.device("cpu") if device is None else torch.device(device)
    devices = [dev] if dev.type == "cuda" else []
    seed = mechanism_init_seed(base, layer_idx)
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        yield


def rows_rmsnorm(ops, x: torch.Tensor, eps: float, weight: Optional[torch.Tensor] = None, zero_centered: bool = False,
                 out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``ops.rmsnorm_rows`` on the (rows, N) view of ``x`` (..., N), returned in ``x``'s leading shape (prefill and
    decode norms of the hippocampal layer)."""
    shape = tuple(x.shape)
    n = int(shape[-1])
    y = ops.rmsnorm_rows(x.reshape(-1, n), eps, weight=weight, zero_centered=zero_centered, out_dtype=out_dtype)
    return y.reshape(shape)


def centred_running_step(x: torch.Tensor, run_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """One-token strictly-past centring over a carried sum (decode form, torch): x (1, B, C), run_sum (B, C) fp32 the
    sum of the row's earlier inputs, count (B,) int64 the row position of this token.  Returns the fp32 (1, B, C)
    c = (x - S / max(n, 1)) [n > 0] and advances ``run_sum`` in place by x (the whole row's cumulative sum's next
    addition, in its order)."""
    if x.dim() != 3 or int(x.shape[0]) != 1 or tuple(run_sum.shape) != tuple(x.shape[1:]) \
            or run_sum.dtype != torch.float32:
        raise RuntimeError(f"running mean (step): input {tuple(x.shape)} for a carried sum {tuple(run_sum.shape)} "
                           f"{run_sum.dtype} (expected x (1, B, C), fp32 (B, C))")
    B = int(x.shape[1])
    if not isinstance(count, torch.Tensor) or tuple(count.shape) != (B,) or count.dtype != torch.int64:
        raise RuntimeError(f"running mean (step): the count must be a ({B},) int64 tensor")
    xf = x.float()
    xcum = run_sum.unsqueeze(0)                                           # S_t (a view; read before the add)
    pos = count.view(1, B, 1) + torch.arange(1, device=x.device, dtype=torch.int64).view(1, 1, 1)
    xn = pos.to(torch.float32).clamp_min(1.0)
    has_past = (pos > 0).to(torch.float32)
    out = (xf - xcum / xn) * has_past
    run_sum.add_(xf[0])
    return out


def plif_b_eta_launch(w: torch.Tensor) -> torch.Tensor:
    """(C,) fp32 launch bias of the eta gate from the frozen seed ``w``: logit(1 - beta), beta = sigmoid(clamp(w,
    max=W_CLAMP_MAX)), the retained fraction clamped to [1e-6, 1 - 1e-6]."""
    beta = torch.sigmoid(torch.clamp(w.to(torch.float32), max=W_CLAMP_MAX))
    one_m_beta = (1.0 - beta).clamp(_ONE_M_BETA_EPS, 1 - _ONE_M_BETA_EPS)
    return torch.log(one_m_beta / (1 - one_m_beta))


# ------------------------------------------------------------------------------------------------ the neuron
class PLIFNeuron(nn.Module):
    """``PLIFNeuron(dim, init_tau, *, centred, alpha, eta_rank, device, dtype, impl, init)``.

    ``init_tau``: a scalar launch time constant (> 1) or a ``(tau_min, tau_max)`` spectrum laid out log-uniformly over
    the channels.  ``centred``: the eta projection reads the strictly-past-centred input (the FFN input neuron of the
    hippocampal layer).  ``alpha``: surrogate slope of the backward.  ``impl``: ``"kernel"`` or ``"ref"`` (the operator
    namespace, :mod:`neuronspark.ops`).

    ``forward(x, state=None, pos=None)``: ``x`` (T, B, C); ``state`` the population's :class:`PLIFState` of an inference
    call (None: whole-row form); ``pos`` (B,) int64 the row position of the call's first token (inference calls of the
    centred neuron).  Returns the spikes S (T, B, C) in x.dtype.
    """

    def __init__(self, dim: int, init_tau: Union[float, Sequence[float]] = 2.0, *, centred: bool = False,
                 alpha: float = 8.0, eta_rank: int = 16, device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None, impl: str = "kernel", init: str = "launch") -> None:
        super().__init__()
        check_init(init)
        if int(dim) <= 0 or int(eta_rank) <= 0:
            raise ValueError(f"PLIFNeuron: dim and eta_rank must be positive, got {dim}, {eta_rank}")
        if not float(alpha) > 0.0:
            raise ValueError(f"PLIFNeuron: alpha must be positive, got {alpha}")
        self.dim = int(dim)
        self.centred = bool(centred)
        self.alpha = float(alpha)
        self.eta_rank = int(eta_rank)
        self.impl = impl
        self._ops = namespace(impl)
        f32 = torch.float32 if dtype is None else dtype
        C, R = self.dim, self.eta_rank
        spectrum = isinstance(init_tau, (list, tuple))
        if spectrum:
            lo, hi = float(init_tau[0]), float(init_tau[1])
            if not 1.0 < lo <= hi:
                raise ValueError(f"PLIFNeuron: an init_tau spectrum needs 1 < tau_min <= tau_max, got {init_tau!r}")
            self.init_tau: Union[float, Tuple[float, float]] = (lo, hi)
        else:
            if not float(init_tau) > 1.0:
                raise ValueError(f"PLIFNeuron: init_tau must exceed 1, got {init_tau!r}")
            self.init_tau = float(init_tau)
        if init == "launch":
            # draw order: w, v_th, eta_V
            if spectrum:
                taus = torch.exp(torch.linspace(math.log(lo), math.log(hi), C, device=device))
                w0 = torch.log(taus - 1.0).to(f32) + torch.empty(C, device=device, dtype=f32).normal_(0.0, W_INIT_STD)
            else:
                w0 = torch.empty(C, device=device, dtype=f32).normal_(-math.log(float(init_tau) - 1.0), W_INIT_STD)
            v_th = torch.empty(C, device=device, dtype=f32).uniform_(*V_TH_INIT)
            w_eta = torch.zeros(C, device=device, dtype=f32)
            eta_V = torch.empty((R, C), device=device, dtype=f32)
            nn.init.xavier_uniform_(eta_V)
            eta_U = torch.zeros((C, R), device=device, dtype=f32)
            b_eta = plif_b_eta_launch(w0).to(f32)
        else:
            w0 = torch.empty(C, device=device, dtype=f32)
            v_th = torch.empty(C, device=device, dtype=f32)
            w_eta = torch.empty(C, device=device, dtype=f32)
            b_eta = torch.empty(C, device=device, dtype=f32)
            eta_V = torch.empty((R, C), device=device, dtype=f32)
            eta_U = torch.empty((C, R), device=device, dtype=f32)
        self.w = nn.Parameter(w0, requires_grad=False)
        self.v_th = nn.Parameter(v_th)
        self.w_eta = nn.Parameter(w_eta)
        self.b_eta = nn.Parameter(b_eta)
        self.eta_V = nn.Parameter(eta_V)
        self.eta_U = nn.Parameter(eta_U)

    # ------------------------------------------------------------------------------------------------ properties
    @property
    def beta(self) -> torch.Tensor:
        """(C,) launch retention sigmoid(clamp(w, max=W_CLAMP_MAX)) from the frozen seed (calibration)."""
        return torch.sigmoid(torch.clamp(self.w, max=W_CLAMP_MAX))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, init_tau={self.init_tau}, centred={self.centred}, alpha={self.alpha}, "
                f"eta_rank={self.eta_rank}, impl={self.impl!r}")

    def allocate_inference_cache(self, batch_size: int, device: Optional[torch.device] = None) -> PLIFState:
        """Zero membrane (and zero input sum for the centred neuron) of ``batch_size`` rows."""
        return PLIFState.allocate(int(batch_size), self.dim, self.centred,
                                  self.v_th.device if device is None else torch.device(device))

    # ------------------------------------------------------------------------------------------------ forward
    def forward(self, x: torch.Tensor, state: Optional[PLIFState] = None,
                pos: Optional[torch.Tensor] = None, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """S (T, B, C) in ``x.dtype``; ``out_dtype`` (the inference forms) stores the spikes in that dtype instead -- a
        caller whose only consumers are bf16 projections passes ``torch.bfloat16`` (the rounding of their input cast)."""
        if x.dim() != 3 or int(x.shape[-1]) != self.dim:
            raise ValueError(f"PLIFNeuron: expected x (T, B, {self.dim}), got {tuple(x.shape)}")
        if x.dtype not in (torch.float32, torch.bfloat16):
            raise RuntimeError(f"PLIFNeuron: x must be fp32 or bf16, got {x.dtype}")
        if state is None:
            if out_dtype is not None:
                raise RuntimeError("PLIFNeuron: out_dtype applies to the inference forms (a carried state)")
            return self._forward_train(x)
        if torch.is_grad_enabled():
            raise RuntimeError("PLIFNeuron: the inference forms (a carried state) run without autograd "
                               "(torch.no_grad())")
        B = int(x.shape[1])
        if tuple(state.v.shape) != (B, self.dim) or state.v.dtype != torch.float32:
            raise RuntimeError(f"PLIFNeuron: membrane state {tuple(state.v.shape)} {state.v.dtype} for x "
                               f"{tuple(x.shape)} (expected fp32 ({B}, {self.dim}))")
        if self.centred:
            if state.xsum is None:
                raise RuntimeError("PLIFNeuron: the centred neuron's state carries no input sum (xsum)")
            if pos is None or tuple(pos.shape) != (B,) or pos.dtype != torch.int64:
                raise RuntimeError(f"PLIFNeuron: the centred neuron needs pos ({B},) int64, the row position of the "
                                   "call's first token")
        od = x.dtype if out_dtype is None else out_dtype
        if int(x.shape[0]) == 1:
            return self._forward_step(x, state, pos).to(od)
        return self._forward_prefill(x, state, pos, od)

    def _eta_projection(self, xc: torch.Tensor, contiguous: bool) -> torch.Tensor:
        """bf16 (T, B, C) token term of the gate logit: bf16(bf16(xc @ eta_V^T) @ eta_U^T)."""
        lin = self._ops.linear
        if contiguous:
            return lin(lin(xc.contiguous(), self.eta_V).contiguous(), self.eta_U)
        return lin(lin(xc, self.eta_V), self.eta_U)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Whole-row form (from a zero membrane): S (T, B, C) in x.dtype; differentiable through the scan op."""
        if int(x.shape[0]) < 2:
            raise RuntimeError("PLIFNeuron: the whole-row form needs T >= 2 (single tokens run through an inference "
                               "state)")
        xc = self._ops.plif_centred_input(x) if self.centred else x
        zp = self._eta_projection(xc, contiguous=True)
        S = self._ops.plif_scan_train(self.w_eta, self.b_eta, zp, x, self.v_th, alpha=self.alpha, out_dtype=x.dtype)
        return S.to(x.dtype)

    def _forward_prefill(self, x: torch.Tensor, state: PLIFState, pos: Optional[torch.Tensor],
                         out_dtype: torch.dtype) -> torch.Tensor:
        """Prefill form (T >= 2): the membrane ``state.v`` and the input sum ``state.xsum`` (centred neuron) are carried
        in place."""
        if self.centred:
            # written in bf16: its only consumer is the bf16 eta projection, whose input cast is the same rounding
            xc = self._ops.centred_running(x, state.xsum, pos, torch.bfloat16)
        else:
            xc = x
        zp = self._eta_projection(xc, contiguous=False)                    # bf16, widened in the gate kernel
        a, drive = self._ops.plif_gate_prefill(x, self.w_eta, self.b_eta, zp)
        S = self._ops.plif_scan_prefill(a, drive, self.v_th, state.v, out_dtype)
        return S.to(out_dtype)

    def _forward_step(self, x: torch.Tensor, state: PLIFState, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """Single-token form (torch): the gate chain, one leaky step, the bipolar release, the soft reset; the membrane
        (and the input sum) updated in place."""
        if self.centred:
            xc = centred_running_step(x, state.xsum, pos).to(x.dtype)
        else:
            xc = x
        zp = self._eta_projection(xc, contiguous=False).to(torch.float32)
        z_eta = self.w_eta.to(torch.float32) * x.to(torch.float32) + self.b_eta.to(torch.float32)
        z_eta = z_eta + zp
        eta = torch.sigmoid(z_eta)
        a_eta = 1.0 - eta
        drive = eta * x.to(torch.float32)
        v = state.v.to(torch.float32)
        th = self.v_th.to(torch.float32)
        pre = a_eta[0] * v + drive[0]
        s = torch.clamp_min(pre - th, 0.0)
        s = s - torch.clamp_min(-pre - th, 0.0)
        state.v.copy_(pre - s)
        return s.unsqueeze(0).to(x.dtype)
