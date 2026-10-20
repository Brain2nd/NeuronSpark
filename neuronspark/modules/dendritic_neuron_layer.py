"""The dendritic neuron layer: one population of dendritic neurons (``mixer_a``) with an fp32 residual addition.

Every soma ``(h, p)`` owns ``N`` dendritic compartments.  Their membrane potentials ``S[h, p, :]`` leak with an
input-dependent time constant (``a_t = exp(dt_t A)``) and integrate the token's synaptic input (``dt_t x_t B_t^T``); the
soma reads its dendrites (``S_t C_t``), and the somatic potential is released through a bipolar soft threshold.  The
membrane update has the selective state-space form, which the operators evaluate as a chunked scan in training and as
one step per token in decoding.  Per token ``t`` (1.27B: d_model 1536, d_inner 9216 = 6 * 1536, H = 144 heads of
P = 64 somata, N = 128 compartments, conv width 4)::

    hn_t            = RMSNorm(h_t) * pre_norm_gain                       eps 1e-5
    [z | xBC | d]_t = W_in hn_t                                         in_proj.weight (18832, 1536): rows z, x, B, C, dt
    [x | B | C]_t   = SiLU(conv_bias + sum_j conv_weight[:, j] xBC_{t-3+j})   column 3 = lag 0
    dt_t            = softplus(d_t + dt_bias)                           a_t = exp(dt_t * A),  A = -exp(A_log)
    S_t             = a_t S_{t-1} + dt_t x_t B_t^T                      (B, C shared by the heads)
    y_t             = S_t C_t + D x_t
    n_t             = RMSNorm(y_t * SiLU(z_t))                          unit weight, eps 1e-5
    s_t             = max(n_t - theta, 0) - max(-n_t - theta, 0)        theta = theta_floor + softplus(rho)
    h_{out}         = h_t + W_out (gamma * s_t)                         fp32 residual addition

The soma has no retention and never resets the compartments.  The backward of the soma is the sigmoid surrogate
``ds/dn = sig(alpha (n - theta)) + sig(alpha (-n - theta))``, ``ds/dtheta = -(sig(alpha (n - theta)) -
sig(alpha (-n - theta)))`` with ``alpha = surrogate_alpha = 8.0`` (inside the training operator).

Regimes (the caller picks one per call; :class:`DendriticNeuronLayer` dispatches on ``inference_params`` and T):

* whole row (training, and teacher-forced scoring), ``h`` (T, B, D) fp32 with T a multiple of 256 on the kernel path:
  ``dendritic_norm`` (or the bf16 ``hn`` handed over by the previous dendritic neuron layer) -> ``linear`` with W_in on
  the (B * T, D) rows (one GEMM; a B-major stream makes the transposed view contiguous) -> ``dendritic_chunk_scan_soma``
  (conv, chunked scan, gated norm and soma in one autograd Function) -> ``linear`` with W_out -> the residual addition,
  or, when the next layer is a dendritic neuron layer, ``dendritic_resid_norm`` (the addition and the next layer's
  pre-norm in one operator).
* prefill (inference, T >= 2, carried state): ``dendritic_prenorm_rows`` -> bf16 GEMM with W_in on the (B, T, D) view of
  the T-major stream -> ``dendritic_conv_silu_prefill`` from the carried conv tail -> ``dendritic_chunk_scan_prefill``
  from the carried state (the final state is written back in the cache's storage dtype) -> ``dendritic_gated_norm_soma``
  (the gated norm on the z view of the projection and the soma read-out in one kernel) -> bf16 GEMM with W_out ->
  ``dendritic_res_add``, the residual addition of the transposed GEMM output (the bf16 operand converted exactly to
  fp32, one row-coalesced kernel).
* decode (inference, T == 1): ``dendritic_prenorm_rows`` -> bf16 GEMM -> ``dendritic_step_core`` (conv tail roll, state
  update in the state's storage dtype, gated norm, soma) -> bf16 GEMM -> the residual addition.

Every operator is taken from the namespace chosen at construction (``impl="kernel"`` the kernels, ``impl="ref"`` the
pure-torch references with the same signatures).  The inference GEMMs go through :func:`neuronspark.ops.linear.linear`
without autograd, i.e. ``F.linear(x.bf16(), w)`` with the bf16 weight copy made by ``prepare_inference()`` when it is
current (else ``weight.to(bf16)``, the same values).

Inference state (:class:`neuronspark.utils.cache.DendriticState`): ``conv`` (B, 9472, 3) fp32, the last three conv
inputs; ``ssm`` (B, 144, 64, 128) in the cache's dendritic-layer state dtype. The storage dtype is the model's one
numerics option (``NeuronSparkConfig.inference_dendritic_state_dtype``): ``"bfloat16"`` (default: the state is rounded
to bf16 once per decode step and once per prefill call; all arithmetic stays fp32) or ``"float32"`` (the state is
carried in fp32). Training never stores a state.

Launch (``init="launch"``): every value is drawn on the CPU in fp32 from a private generator seeded with
``mechanism_init_seed(998000, layer_idx)`` (or ``launch_seed``), in the order in_proj.weight N(0, 0.02),
conv_weight N(0, 0.02), A ~ U(1, 16), dt ~ logU(1e-3, 1e-1) clamped at 1e-4, W_out U(+-1/sqrt(d_inner)) /
sqrt(n_residual_branches); conv_bias 0, D 1, gamma 1, pre_norm_gain 1, rho = softplus^-1(theta0 - theta_floor), dt_bias =
dt + log(-expm1(-dt)).  The global random streams are not read.  ``init="empty"`` allocates without drawing (for a
build followed by a checkpoint load).

State-dict keys (per layer ``layers.{i}.mixer_a.``): ``pre_norm_gain``, ``in_proj.weight``, ``conv_weight``,
``conv_bias``, ``dt_bias``, ``A_log``, ``D``, ``gamma``, ``W_out``, ``rho`` (all fp32), in this registration order;
the non-persistent buffer ``norm_ones`` (the unit gated-norm weight) is not saved.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronspark import ops as _ops
from neuronspark.modules import mechanism_init_seed
from neuronspark.modules.linear import Linear
from neuronspark.utils.cache import DendriticState, resolve_dendritic_state_dtype

__all__ = [
    "DendriticNeurons",
    "DendriticNeuronLayer",
    "softplus_inverse",
    "DENDRITIC_LAUNCH_SEED_BASE",
    "DENDRITIC_INIT_STD",
    "DENDRITIC_A_INIT_RANGE",
    "DENDRITIC_DT_RANGE",
    "DENDRITIC_DT_INIT_FLOOR",
    "DENDRITIC_CONV_WIDTH",
    "DENDRITIC_IMPLS",
    "DENDRITIC_INITS",
]

# launch constants (A and dt ranges of the selective state-space initialisation)
DENDRITIC_LAUNCH_SEED_BASE = 998_000          # private launch stream: mechanism_init_seed(998000, layer_idx)
DENDRITIC_INIT_STD = 0.02                     # in_proj.weight and conv_weight ~ N(0, 0.02)
DENDRITIC_A_INIT_RANGE = (1.0, 16.0)          # A ~ U(1, 16), A_log = log A
DENDRITIC_DT_RANGE = (1e-3, 1e-1)             # dt ~ logU(dt_min, dt_max)
DENDRITIC_DT_INIT_FLOOR = 1e-4                # dt clamped below at this value before the softplus inversion
DENDRITIC_CONV_WIDTH = 4                      # causal depthwise conv width (the operators are built for 4)

DENDRITIC_IMPLS = ("kernel", "ref")
DENDRITIC_INITS = ("launch", "empty")

_StateDtype = Union[str, torch.dtype]


def softplus_inverse(v: float) -> float:
    """``x`` with ``softplus(x) = v`` (v > 0): ``v + log(-expm1(-v))``."""
    return float(v + math.log(-math.expm1(-v)))


class DendriticNeurons(nn.Module):
    """One population of dendritic neurons: pre-norm, synaptic projection W_in, causal synaptic filter (conv),
    dendritic membrane state, gated RMSNorm, bipolar soma, gamma, W_out, and the fp32 residual addition (see the module
    docstring).

    ``forward(h, state=None, *, hn=None, next_norm_gain=None) -> (h_new, hn_next)``: ``state`` None runs the whole-row
    regime (``hn_next`` is the bf16 pre-norm of ``h_new`` for the next dendritic neuron layer when ``next_norm_gain`` is
    given, else None); with an :class:`DendriticState` it runs prefill (T >= 2) or the decode step (T == 1) and
    ``hn_next`` is None.
    """

    def __init__(
        self,
        d_model: int = 1536,
        expand: int = 6,
        head_dim: int = 64,
        state_size: int = 128,
        chunk_size: int = 256,
        conv_width: int = DENDRITIC_CONV_WIDTH,
        theta0: float = 0.05,
        theta_floor: float = 0.02,
        surrogate_alpha: float = 8.0,
        n_residual_branches: int = 30,
        norm_eps: float = 1e-5,
        layer_idx: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        impl: str = "kernel",
        init: str = "launch",
        launch_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if impl not in DENDRITIC_IMPLS:
            raise ValueError(f"DendriticNeurons: impl must be one of {DENDRITIC_IMPLS}, got {impl!r}")
        if init not in DENDRITIC_INITS:
            raise ValueError(f"DendriticNeurons: init must be one of {DENDRITIC_INITS}, got {init!r}")
        if dtype != torch.float32:
            raise TypeError("DendriticNeurons: the parameters are fp32 masters (the operators cast their operands), "
                            f"got {dtype}")
        if int(conv_width) != DENDRITIC_CONV_WIDTH:
            raise ValueError(f"DendriticNeurons: the conv operators are built for width {DENDRITIC_CONV_WIDTH}, got "
                             f"{conv_width}")
        d_inner = int(expand) * int(d_model)
        if d_inner % int(head_dim) != 0:
            raise ValueError(f"DendriticNeurons: expand * d_model = {d_inner} must be a multiple of head_dim "
                             f"{head_dim}")
        if not float(theta0) > float(theta_floor) >= 0.0:
            raise ValueError(f"DendriticNeurons: theta0 {theta0} must exceed theta_floor {theta_floor} >= 0")
        if isinstance(n_residual_branches, bool) or int(n_residual_branches) < 1:
            raise ValueError("DendriticNeurons: n_residual_branches must be a positive integer, got "
                             f"{n_residual_branches!r}")
        if float(surrogate_alpha) <= 0.0:
            raise ValueError(f"DendriticNeurons: surrogate_alpha must be positive, got {surrogate_alpha}")
        if launch_seed is not None and (isinstance(launch_seed, bool) or int(launch_seed) < 0):
            raise ValueError("DendriticNeurons: launch_seed must be a non-negative integer or None, got "
                             f"{launch_seed!r}")

        self.d_model = int(d_model)
        self.expand = int(expand)
        self.head_dim = int(head_dim)
        self.state_size = int(state_size)
        self.chunk_size = int(chunk_size)
        self.conv_width = int(conv_width)
        self.d_inner = d_inner
        self.nheads = d_inner // self.head_dim
        self.conv_dim = d_inner + 2 * self.state_size
        self.proj_dim = 2 * d_inner + 2 * self.state_size + self.nheads
        self.theta0 = float(theta0)
        self.theta_floor = float(theta_floor)
        self.surrogate_alpha = float(surrogate_alpha)
        self.n_residual_branches = int(n_residual_branches)
        self.norm_eps = float(norm_eps)
        self.layer_idx = int(layer_idx)
        self.impl = impl
        self.launch_seed = (mechanism_init_seed(DENDRITIC_LAUNCH_SEED_BASE, self.layer_idx) if launch_seed is None
                            else int(launch_seed))

        f32 = dict(device=device, dtype=torch.float32)
        # registration order = state-dict order
        self.pre_norm_gain = nn.Parameter(torch.empty(self.d_model, **f32))
        self.in_proj = Linear(self.d_model, self.proj_dim, bias=False, device=device, dtype=torch.float32, init="empty")
        self.conv_weight = nn.Parameter(torch.empty(self.conv_dim, self.conv_width, **f32))     # [:, 3] = lag 0
        self.conv_bias = nn.Parameter(torch.empty(self.conv_dim, **f32))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, **f32))
        self.A_log = nn.Parameter(torch.empty(self.nheads, **f32))
        self.D = nn.Parameter(torch.empty(self.nheads, **f32))
        self.gamma = nn.Parameter(torch.empty(self.d_inner, **f32))
        self.W_out = nn.Parameter(torch.empty(self.d_model, self.d_inner, **f32))
        self.register_buffer("norm_ones", torch.ones(self.d_inner, **f32), persistent=False)
        self.rho = nn.Parameter(torch.empty(self.d_inner, **f32))
        if init == "launch":
            self.reset_parameters()

    # ------------------------------------------------------------------------------------------------ launch
    @torch.no_grad()
    def launch_state(self, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """The launch values as CPU fp32 tensors keyed by parameter name (see the module docstring).  ``generator``: a
        CPU generator; None = a private generator seeded with ``launch_seed`` (the same tensors at every call)."""
        g = generator
        if g is None:
            g = torch.Generator(device="cpu").manual_seed(self.launch_seed)
        if g.device.type != "cpu":
            raise ValueError("DendriticNeurons.launch_state: the launch draws on the CPU; pass a CPU generator")
        f32 = dict(dtype=torch.float32, device=torch.device("cpu"))
        state: Dict[str, torch.Tensor] = {"pre_norm_gain": torch.ones(self.d_model, **f32)}
        state["in_proj.weight"] = torch.empty(self.proj_dim, self.d_model, **f32).normal_(
            mean=0.0, std=DENDRITIC_INIT_STD, generator=g)
        state["conv_weight"] = torch.empty(self.conv_dim, self.conv_width, **f32).normal_(
            mean=0.0, std=DENDRITIC_INIT_STD, generator=g)
        state["conv_bias"] = torch.zeros(self.conv_dim, **f32)
        A = torch.empty(self.nheads, **f32).uniform_(DENDRITIC_A_INIT_RANGE[0], DENDRITIC_A_INIT_RANGE[1], generator=g)
        state["A_log"] = torch.log(A)
        dt_min, dt_max = DENDRITIC_DT_RANGE
        dt = torch.exp(torch.rand(self.nheads, generator=g, **f32) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=DENDRITIC_DT_INIT_FLOOR)
        state["dt_bias"] = dt + torch.log(-torch.expm1(-dt))
        state["D"] = torch.ones(self.nheads, **f32)
        state["gamma"] = torch.ones(self.d_inner, **f32)
        # kaiming_uniform_(a = sqrt(5)) bound = 1 / sqrt(fan_in), then the residual-branch scale
        bound = 1.0 / math.sqrt(self.d_inner)
        w = torch.empty(self.d_model, self.d_inner, **f32).uniform_(-bound, bound, generator=g)
        state["W_out"] = w / math.sqrt(self.n_residual_branches)
        state["rho"] = torch.full((self.d_inner,), softplus_inverse(self.theta0 - self.theta_floor), **f32)
        return state

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """Copy :meth:`launch_state` into the parameters (on their device)."""
        state = self.launch_state(generator)
        own = dict(self.named_parameters())
        if set(state) != set(own):
            raise RuntimeError(f"DendriticNeurons launch: parameters {sorted(own)} and launch tensors {sorted(state)} "
                               "differ")
        for name, value in state.items():
            if tuple(value.shape) != tuple(own[name].shape):
                raise RuntimeError(f"DendriticNeurons launch: {name} is {tuple(own[name].shape)}, launch "
                                   f"{tuple(value.shape)}")
            own[name].copy_(value)

    # ------------------------------------------------------------------------------------------------ derived values
    def theta(self) -> torch.Tensor:
        """Soma threshold ``theta_floor + softplus(rho)``, fp32 (d_inner,)."""
        return self.theta_floor + F.softplus(self.rho)

    def A(self) -> torch.Tensor:
        """Per-head decay rate ``-exp(A_log)``, fp32 (nheads,)."""
        return -torch.exp(self.A_log.float())

    def linear_weights(self) -> List[torch.Tensor]:
        """The weights of the mixer's GEMMs (W_in, W_out): the tensors the inference bf16 weight cache covers."""
        return [self.in_proj.weight, self.W_out]

    @property
    def _o(self):
        return _ops.namespace(self.impl)

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, d_inner={self.d_inner}, heads={self.nheads}x{self.head_dim}, "
                f"state={self.state_size}, chunk={self.chunk_size}, impl={self.impl!r}")

    # ------------------------------------------------------------------------------------------------ checks
    def _check_params(self) -> None:
        bad = [n for n, p in self.named_parameters() if p.dtype != torch.float32]
        if bad:
            raise TypeError(f"DendriticNeurons (layer {self.layer_idx}): parameters must be fp32 masters, got non-fp32 "
                            f"{bad}")

    def _check_stream(self, h: torch.Tensor) -> None:
        if h.dim() != 3 or int(h.shape[-1]) != self.d_model:
            raise ValueError(f"DendriticNeurons (layer {self.layer_idx}): expected the (T, B, {self.d_model}) residual "
                             f"stream, got {tuple(h.shape)}")
        if h.dtype != torch.float32:
            raise TypeError(f"DendriticNeurons (layer {self.layer_idx}): the residual stream is fp32, got {h.dtype}")

    def _check_state(self, state: DendriticState, batch: int) -> None:
        if not isinstance(state, DendriticState):
            raise TypeError(f"DendriticNeurons (layer {self.layer_idx}): expected a DendriticState, got "
                            f"{type(state).__name__}")
        want_conv = (batch, self.conv_dim, self.conv_width - 1)
        want_ssm = (batch, self.nheads, self.head_dim, self.state_size)
        if tuple(state.conv.shape) != want_conv or state.conv.dtype != torch.float32:
            raise RuntimeError(f"DendriticNeurons (layer {self.layer_idx}): conv state must be {want_conv} fp32, got "
                               f"{tuple(state.conv.shape)} {state.conv.dtype}")
        if tuple(state.ssm.shape) != want_ssm or state.ssm.dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(f"DendriticNeurons (layer {self.layer_idx}): ssm state must be {want_ssm} bf16 or fp32, "
                               f"got {tuple(state.ssm.shape)} {state.ssm.dtype}")

    @staticmethod
    def _check_no_grad(where: str) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError(f"DendriticNeurons {where}: inference calls run under torch.no_grad() (the cached state "
                               "is updated in place)")

    # ------------------------------------------------------------------------------------------------ regimes
    def forward(self, h: torch.Tensor, state: Optional[DendriticState] = None, *, hn: Optional[torch.Tensor] = None,
                next_norm_gain: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if state is None:
            return self.forward_whole_row(h, hn=hn, next_norm_gain=next_norm_gain)
        if hn is not None or next_norm_gain is not None:
            raise ValueError(f"DendriticNeurons (layer {self.layer_idx}): hn / next_norm_gain belong to the whole-row "
                             "regime")
        self._check_stream(h)
        if int(h.shape[0]) == 1:
            return self.step(h, state), None
        return self.forward_prefill(h, state), None

    def forward_whole_row(self, h: torch.Tensor, *, hn: Optional[torch.Tensor] = None,
                          next_norm_gain: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Whole-row regime: ``h`` (T, B, D) fp32 (unit last stride; B-major in training) -> ``(h_new, hn_next)``.

        ``hn``: this mixer's bf16 pre-norm of ``h`` (T, B, D) as returned by the previous dendritic neuron layer's join,
        else it is formed here.  ``next_norm_gain``: the pre-norm gain of the next layer's mixer when that layer is a
        dendritic layer (the join ``dendritic_resid_norm`` then returns ``hn_next``); None ends with ``h +
        upd.float()``.  Differentiable.
        """
        self._check_stream(h)
        self._check_params()
        T, B, Dm = (int(s) for s in h.shape)
        if self.impl == "kernel" and T % self.chunk_size != 0:
            raise RuntimeError(f"DendriticNeurons (layer {self.layer_idx}): the whole-row scan runs full chunks only: "
                               f"T = {T} is not a multiple of {self.chunk_size}")
        o = self._o
        if hn is None:
            hn = o.dendritic_norm(h, self.pre_norm_gain,
                                self.norm_eps)                          # (T, B, D) bf16, h's layout
        elif tuple(hn.shape) != (T, B, Dm) or hn.dtype != torch.bfloat16:
            raise ValueError(f"DendriticNeurons (layer {self.layer_idx}): hn must be the bf16 ({T}, {B}, {Dm}) "
                             f"pre-norm, got {hn.dtype} {tuple(hn.shape)}")
        # W_in on the (B * T, D) rows: for the B-major stream the transposed view is contiguous (one GEMM, no copy)
        zxbcdt = o.linear(hn.transpose(0, 1).reshape(B * T, Dm), self.in_proj.weight).view(B, T, self.proj_dim)
        xs = o.dendritic_chunk_scan_soma(zxbcdt, self.conv_weight, self.conv_bias, self.dt_bias, self.A(), self.D,
                                  self.theta(), self.gamma, alpha=self.surrogate_alpha, chunk_size=self.chunk_size)
        upd = o.linear(xs, self.W_out).view(B, T, Dm).transpose(0, 1)                   # (T, B, D) bf16
        if next_norm_gain is not None:
            h_new, hn_next = o.dendritic_resid_norm(h, upd, next_norm_gain, self.norm_eps)
            return h_new, hn_next
        return h + upd.float(), None

    def forward_prefill(self, h: torch.Tensor, state: DendriticState) -> torch.Tensor:
        """Prefill regime (no grad): ``h`` (T, B, D) fp32 with T >= 2 (T-major contiguous as the model passes it), the
        layer's ``state`` read and advanced in place -> ``h + upd`` (T, B, D) fp32."""
        self._check_no_grad("prefill")
        self._check_stream(h)
        self._check_params()
        T, B, Dm = (int(s) for s in h.shape)
        if T < 2:
            raise ValueError(f"DendriticNeurons (layer {self.layer_idx}): prefill takes T >= 2 (T == 1 is the decode "
                             "step)")
        self._check_state(state, B)
        o = self._o
        di, cd, H, P, N = self.d_inner, self.conv_dim, self.nheads, self.head_dim, self.state_size
        hn = o.dendritic_prenorm_rows(h.reshape(T * B, Dm), self.pre_norm_gain, self.norm_eps).view(T, B, Dm)
        proj = o.linear(hn.transpose(0, 1), self.in_proj.weight)                         # (B, T, 18832) bf16
        z = proj[..., :di]
        xBC = proj[..., di:di + cd]
        dt = proj[..., di + cd:]
        xbc, tail = o.dendritic_conv_silu_prefill(xBC, self.conv_weight, self.conv_bias, state.conv)
        state.conv.copy_(tail)
        x = xbc[..., :di].view(B, T, H, P)
        Bm = xbc[..., di:di + N].view(B, T, 1, N)
        Cm = xbc[..., di + N:].view(B, T, 1, N)
        # the scan starts from the state in fp32; the final state is stored back in the cache's dtype (one rounding
        # per call for a bf16 state)
        y, final_state = o.dendritic_chunk_scan_prefill(x, dt, self.A(), Bm, Cm, self.D, self.dt_bias, state.ssm,
                                                 chunk_size=self.chunk_size)
        state.ssm.copy_(final_state)
        xo = o.dendritic_gated_norm_soma(y.reshape(B * T, di), z.reshape(B * T, di), self.theta(), self.gamma,
                                       self.norm_eps)
        upd = o.linear(xo, self.W_out).view(B, T, Dm).transpose(0, 1)                   # (T, B, D) bf16
        return o.dendritic_res_add(h, upd)                        # h + upd: the bf16 operand converted exactly, fp32

    def step(self, h: torch.Tensor, state: DendriticState) -> torch.Tensor:
        """Decode regime (no grad): ``h`` (1, B, D) fp32, ``state`` advanced in place -> (1, B, D) fp32."""
        self._check_no_grad("decode")
        self._check_stream(h)
        self._check_params()
        if int(h.shape[0]) != 1:
            raise ValueError(f"DendriticNeurons (layer {self.layer_idx}): the decode step takes T == 1, got "
                             f"{int(h.shape[0])}")
        B = int(h.shape[1])
        self._check_state(state, B)
        o = self._o
        hc = h[0]
        if not hc.is_contiguous():
            hc = hc.contiguous()
        hn = o.dendritic_prenorm_rows(hc, self.pre_norm_gain, self.norm_eps)                    # (B, D) bf16
        proj = o.linear(hn, self.in_proj.weight)                                         # (B, 18832) bf16
        xs = o.dendritic_step_core(proj, state.conv, state.ssm, self.conv_weight, self.conv_bias, self.A(),
                                   self.dt_bias,
                            self.D, self.theta(), self.gamma, self.norm_eps)             # (B, 9216) bf16
        out = hc + o.linear(xs, self.W_out).float()
        return out[None]

    # ------------------------------------------------------------------------------------------------ inference cache
    def allocate_inference_cache(self, batch_size: int, max_seqlen: Optional[int] = None,
                                 dendritic_state_dtype: _StateDtype = "bfloat16",
                                 device: Optional[torch.device] = None) -> DendriticState:
        """Zero ``DendriticState``: conv (B, conv_dim, 3) fp32 and ssm (B, nheads, head_dim, state_size) in
        ``dendritic_state_dtype``.  ``max_seqlen`` is not used by the dendritic-layer state (kept for the cache
        signature)."""
        dev = self.W_out.device if device is None else torch.device(device)
        return DendriticState.allocate(int(batch_size), self.conv_dim, self.conv_width, self.nheads, self.head_dim,
                                self.state_size, resolve_dendritic_state_dtype(dendritic_state_dtype), dev)


class DendriticNeuronLayer(nn.Module):
    """A dendritic neuron layer of the stack (1.27B: 18 instances, ``layers.{i}`` for i not in (3, 7, ..., 23)):
    ``mixer_a`` and the fp32 residual addition.

    ``forward(h, inference_params=None, *, hn=None, next_norm_gain=None) -> (h, hn_next | None)``:

    * ``inference_params`` None: the whole-row regime of :meth:`DendriticNeurons.forward_whole_row` (``hn`` /
      ``next_norm_gain`` thread the join between consecutive dendritic neuron layers);
    * ``inference_params`` given: the layer's state ``inference_params.layers[layer_idx]`` (an :class:`DendriticState`)
      is read and advanced in place; T == 1 runs the decode step, T >= 2 the prefill; ``hn_next`` is None.
    """

    def __init__(self, config, layer_idx: int, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32, impl: str = "kernel", init: str = "launch",
                 launch_seed: Optional[int] = None) -> None:
        super().__init__()
        if config.is_hippocampal_layer(layer_idx):
            raise ValueError(f"DendriticNeuronLayer: layer {layer_idx} is a hippocampal layer of the configured layout")
        self.layer_idx = int(layer_idx)
        self.inference_dendritic_state_dtype = str(config.inference_dendritic_state_dtype)
        self.mixer_a = DendriticNeurons(
            d_model=int(config.D),
            expand=int(config.dendritic_expand),
            head_dim=int(config.dendritic_head_dim),
            state_size=int(config.dendritic_state_size),
            chunk_size=int(config.dendritic_chunk_size),
            conv_width=int(config.dendritic_conv_width),
            theta0=float(config.dendritic_theta0),
            theta_floor=float(config.dendritic_theta_floor),
            surrogate_alpha=float(config.dendritic_surrogate_alpha),
            n_residual_branches=int(config.dendritic_residual_branches),
            norm_eps=float(config.dendritic_norm_eps),
            layer_idx=self.layer_idx,
            device=device,
            dtype=dtype,
            impl=impl,
            init=init,
            launch_seed=launch_seed,
        )

    @property
    def impl(self) -> str:
        return self.mixer_a.impl

    def linear_weights(self) -> List[torch.Tensor]:
        return self.mixer_a.linear_weights()

    def forward(self, h: torch.Tensor, inference_params=None, *, hn: Optional[torch.Tensor] = None,
                next_norm_gain: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if inference_params is None:
            return self.mixer_a.forward_whole_row(h, hn=hn, next_norm_gain=next_norm_gain)
        if hn is not None or next_norm_gain is not None:
            raise ValueError(f"DendriticNeuronLayer {self.layer_idx}: hn / next_norm_gain belong to the whole-row "
                             "regime")
        layers = getattr(inference_params, "layers", None)
        if layers is None or self.layer_idx not in layers:
            raise RuntimeError(f"DendriticNeuronLayer {self.layer_idx}: the inference cache holds no state for this "
                               "layer")
        return self.mixer_a(h, layers[self.layer_idx])

    def allocate_inference_cache(self, batch_size: int, max_seqlen: Optional[int] = None,
                                 dendritic_state_dtype: Optional[_StateDtype] = None,
                                 device: Optional[torch.device] = None) -> DendriticState:
        """The layer's zero state; ``dendritic_state_dtype`` None takes ``config.inference_dendritic_state_dtype``."""
        dt = self.inference_dendritic_state_dtype if dendritic_state_dtype is None else dendritic_state_dtype
        return self.mixer_a.allocate_inference_cache(batch_size, max_seqlen, dt, device)
