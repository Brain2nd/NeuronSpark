"""Pool attention of the hippocampal layer: the softmax-mixing retrieval pool over tagged synaptic events, with its
value residual, GLU tail, RoPE, PLIF write gate, GDN-family decay clock, induction bias and extra queries.

Whole-row math (one layer call, layer input h (T, B, D) fp32 after the lexical read, z = frozen_rms_norm(h) fp32;
H = 6 heads, dk = dv = 256; bf(.) = RNE to bf16, f(.) = widen to fp32):

    qkv  = bf(z @ qkv_proj^T) -> causal depthwise conv k = 4 (fp32 FMA chain from +0 in tap order on bf16 operands)
    q, k, v = split(qkv);   v = silu(v)                                       bf16
    lam  = pool_value_residual + anchor + repeat_256(tanh(f(bf(zc @ pool_vr_gate^T))))   (T, B, 1536) fp32,
           zc = (z_t - sum_{j<t} z_j / max(t, 1)) [t > 0]
    v    = bf(f(v) (1 + lam))                       hippocampal layer 0 (its post-SiLU v is the carry of the later
    layers)
         = bf(f(v) + lam f(carry))                  hippocampal layers 1 .. 5
    v[..., 1280:] = bf(f(v[..., 1280:]) + f(bf(bf(silu(f(bf(z A^T))) f(bf(z B^T))) @ U^T)))     GLU tail, last head
    q, k = rope(silu(q)), rope(silu(k))            interleaved pairs, every head on the head-width ladder (base 1e4)
    g    = gate_neuron(z)                           PLIF spikes (T, B, 1536) fp32
    a    = exp(-bf(exp(A_log)) softplus(f(bf(z @ W_decay^T))))       (T, B, 256) fp32, A_log = init + delta
    la   = mean_c log(max(a, 1e-6))                 (T, B) clock increment shared by the heads
    gamma_h = softplus(raw_h + log(e - 1)), raw = spectrum + delta;  gq_h = exp(log_h), log = spectrum + delta;
    wb_h = spectrum_h + pool_write_bias_h
    per head h (channels [256 h, 256 h + 256)):
        q_h = normalize(q[h]) gq_h;  k_h = normalize(k[h]);  gv_h = bf(bf(sigmoid(g[h] + wb_h - 2)) v[h])
        cum = cumsum_t(gamma_h la);   X_h = (m_2 beta_2h + m_3 beta_3h) + m_4 beta_4h      (B, T, T) fp32
        M_h = bf(where(j <= t, cum_t - cum_j, -inf) + X_h)                                   (B, 1, T, T)
        out_h = f(bf(softmax_j(16 q_h . k_h + M_h) @ gv_h))
    last head: + bf(bf(cat_m R_m @ W_out_in^T) @ W_out^T), R_m the same read with 4 extra queries
        qx_m = normalize(rope_5(silu(bf(bf(z A_x^T)[64 m : 64 m + 64] @ B_x[m]^T)))) exp(clamp(psi_m, -3, 3)) gq_5,
        psi = f(bf(z @ G_x^T))  (same keys, tagged values, clock and induction term as the head)
    attn = cat_h(out_h);  r = bf(((1 + w) rms(attn) * sigmoid(bf(h @ W_g^T))) @ W_o^T);  r = r - mean_c(r)

Regime differences kept explicitly in the code paths (each regime has its own numerics contract):

* match gain gq: bf16 in the whole row (q is bf16 when the gain is formed); fp32 for the six heads in prefill and
  decode (the gain multiplies the fp32 unit query); the extra queries form their gain in the bf16 query dtype in every
  regime;
* whole row: one ``ops.pool_read`` per layer reads every head and the extra queries (the bias and the induction
  matches formed per tile from the clocks and the id windows of ``induction_token_windows``; nothing T x T is
  materialised);
  prefill and decode write each call's events into per-head rings (unit keys, tagged values, absolute clocks; capacity
  W) and read the rings: prefill by ``ops.ring_write_prefill`` + ``ops.ring_recall_prefill`` (the induction term formed
  in-kernel from the matches), decode by the torch ring write and ``ops.ring_read_t1`` at the decode batch sizes
  ``RING_KERNEL_DECODE_SIZES`` and by the torch ring chain (+ ``ops.xq_form_t1`` for the extra queries) at the others;
* the pool clock: ``ops.cumsum0`` (whole row), ``ops.exact_cumsum0`` over the carried clock (prefill), one add
  (decode); the value-residual running mean: ``ops.cumsum0`` + ``ops.zc_center`` (whole row),
  ``ops.centred_running`` (prefill), torch over the carried sum (decode);
* short conv: ``ops.short_conv_train`` / ``ops.short_conv_silu_prefill`` (T >= 3, the SiLU in the store; a 2-token call takes the depthwise
  ``F.conv1d`` over the carried history) / ``ops.short_conv_step``;
* prefill only: the value-residual gate's centred input is written in bf16 (the operand dtype of its projection),
  ``lam``, the value residual and the GLU tail are one kernel, ``ops.vr_mix_prefill`` (``lam`` is not materialised),
  and the extra queries after their projections by ``ops.xq_queries_prefill`` (SiLU, RoPE,
  normalisation and gain of all M queries in one kernel, stored in bf16, the rounding the ring read applies); the
  whole row and decode keep the torch forms.

The forward runs inside a bf16 CUDA autocast region (``neuronspark.utils.precision.regime``): the expressions in the
code are evaluated under that autocast, which sets their dtypes (e.g. ``F.normalize`` of a bf16 tensor returns fp32,
the decode ring einsums are bf16 GEMMs).

Parameters (keys, flat on the layer): see :class:`PoolAttentionMixin`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuronspark.modules.linear import Linear, ShortConvWeight
from neuronspark.modules.norms import RMSNorm
from neuronspark.modules.plif import PLIFNeuron, centred_running_step, forked_rng, linear_init
from neuronspark.modules.rope import apply_rope_interleaved_ref, rope_head_band_tables
from neuronspark.utils.cache import HippocampalState

__all__ = [
    "PoolAttentionMixin",
    "pool_bias_gain_spectrum_init",
    "pool_dendritic_gain_spectrum_init",
    "pool_write_bias_spectrum_init",
    "value_residual_anchor",
    "RING_KERNEL_DECODE_SIZES",
    "POOL_BIAS_GAIN_RAW_SHIFT",
]

# ------------------------------------------------------------------------------------------------ fixed constants
# decay launch spectrum: half-lives log-uniform on [1, DECAY_TAU_MAX] tokens over the 256 decay channels
DECAY_TAU_MAX = 512.0
# floor of the decay inside the clock's log
DECAY_FLOOR = 1e-6
# eligibility-gain launch spectrum (head 0 .. last head), log-uniform
POOL_BIAS_GAIN_MAX, POOL_BIAS_GAIN_MIN = 1.0, 0.03
# match-gain factor launch spectrum (head 0 .. last head), log-uniform
POOL_DENDRITIC_GAIN_MAX, POOL_DENDRITIC_GAIN_MIN = 1.0, 0.37
# gamma = softplus(raw + log(e - 1)): raw 0 is gain 1 exactly
POOL_BIAS_GAIN_RAW_SHIFT = math.log(math.e - 1.0)
# clamp of the extra queries' log gain
XQ_PSI_CLAMP = 3.0
# forked launch streams
GLU_SEED_BASE = 974_000
XQ_SEED_BASE = 972_000
# decode batch sizes whose T = 1 ring read is the ring kernel (every other decode batch: the torch ring chain + the
# extra-query formation kernel): the decode route table
RING_KERNEL_DECODE_SIZES = frozenset((1, 32))


# ------------------------------------------------------------------------------------------------ launch spectra
def pool_bias_gain_spectrum_init(H: int, gain_max: float = POOL_BIAS_GAIN_MAX,
                                 gain_min: float = POOL_BIAS_GAIN_MIN) -> torch.Tensor:
    """(H,) fp32 raw launch values of the eligibility gains: gamma_h log-uniform from ``gain_max`` (head 0) to
    ``gain_min`` (last head), raw_h = log(expm1(gamma_h)) - log(e - 1), evaluated in float64 on the CPU and rounded
    once to fp32 (head 0 at gain 1.0 is raw 0.0 exactly)."""
    H = int(H)
    if H < 1 or not 0.0 < float(gain_min) < float(gain_max):
        raise ValueError(f"pool bias gain spectrum: H >= 1 and 0 < gain_min < gain_max, got {H}, {gain_min}, {gain_max}")
    log_gain = torch.linspace(math.log(float(gain_max)), math.log(float(gain_min)), H, dtype=torch.float64)
    gain = torch.exp(log_gain)
    shift = torch.log(torch.expm1(torch.ones((), dtype=torch.float64)))
    return (torch.log(torch.expm1(gain)) - shift).to(torch.float32)


def pool_dendritic_gain_spectrum_init(H: int, gain_max: float = POOL_DENDRITIC_GAIN_MAX,
                                      gain_min: float = POOL_DENDRITIC_GAIN_MIN) -> torch.Tensor:
    """(H,) fp32 log match-gain factors log f_h, f_h log-uniform from ``gain_max`` (head 0) to ``gain_min``, float64
    on the CPU rounded once to fp32."""
    H = int(H)
    if H < 1 or not 0.0 < float(gain_min) < float(gain_max):
        raise ValueError(f"pool dendritic gain spectrum: H >= 1 and 0 < gain_min < gain_max, got {H}, {gain_min}, "
                         f"{gain_max}")
    return torch.linspace(math.log(float(gain_max)), math.log(float(gain_min)), H, dtype=torch.float64).to(torch.float32)


def pool_write_bias_spectrum_init(H: int, span: float) -> torch.Tensor:
    """(H,) fp32 write-bias launch offsets linspace(0, span, H) (CPU)."""
    if int(H) < 2 or not float(span) > 0.0:
        raise ValueError(f"pool write-bias spectrum: H >= 2 and span > 0, got {H}, {span}")
    return torch.linspace(0.0, float(span), int(H), dtype=torch.float32)


def value_residual_anchor(pair: Sequence[float], d_value: int, heads: int, device=None) -> torch.Tensor:
    """(D_value,) fp32 anchor of a later hippocampal layer's value-residual mix: ``pair[0]`` on head 0's channels,
    ``pair[1]`` on the last head's, log-uniform in between when both are positive (linear otherwise)."""
    heads, d_value = int(heads), int(d_value)
    if heads < 2 or d_value % heads:
        raise ValueError(f"value-residual anchor: {heads} heads for D_value {d_value}")
    dvh = d_value // heads
    out = torch.empty((d_value,), dtype=torch.float32, device=device)
    a, b = float(pair[0]), float(pair[1])
    if heads == 2:
        out[:dvh].fill_(a)
        out[dvh:].fill_(b)
        return out
    for h in range(heads):
        if h == 0:
            v = a
        elif h == heads - 1:
            v = b
        elif a > 0.0 and b > 0.0:
            v = a * (b / a) ** (h / (heads - 1))
        else:
            v = a + (b - a) * (h / (heads - 1))
        out[h * dvh:(h + 1) * dvh].fill_(float(v))
    return out


# ------------------------------------------------------------------------------------------------ the mixin
class PoolAttentionMixin:
    """Mixin of :class:`neuronspark.modules.hippocampal_layer.HippocampalLayer`.

    Parameters (shape; launch): ``W_decay.weight`` (256, 1536; 0), ``decay_A_log_delta`` (256; 0),
    ``pool_bias_gain_delta`` / ``pool_dendritic_gain_delta`` / ``pool_write_bias`` (6; 0), ``pool_glu_a`` /
    ``pool_glu_b`` (128, 1536; Xavier, forked 974000), ``pool_glu_u`` (256, 128; 0), ``pool_value_residual`` (1536; 0),
    ``pool_vr_gate`` (6, 1536; 0), ``pool_xq_a`` (256, 1536), ``pool_xq_b.{0..3}`` (256, 64), ``pool_xq_out_in``
    (64, 1024) (Xavier, forked 972000, in this order), ``pool_xq_gain_proj`` (4, 1536; 0), ``pool_xq_out`` (256, 64; 0),
    ``qkv_proj.weight`` (4608, 1536; Xavier, drawn after the FFN), ``gate_neuron.*`` (PLIF 1536), ``qkv_conv.weight``
    (4608, 1, 4; identity: last tap 1), ``attn_out_norm.weight`` (1536; 0, zero-centred), ``attn_out_gate.weight``
    (1536, 1536; PyTorch default), ``attn_out_proj.weight`` (1536, 1536; N(0, out_std), drawn after the FFN),
    ``pool_ind_beta`` (18; 0, viewed (3 orders, 6 heads)).
    Persistent buffers: ``decay_A_log_init`` (256) = -linspace(0, ln 512, 256), ``pool_bias_gain_spectrum_raw`` (6),
    ``pool_dendritic_gain_spectrum_log`` (6), ``pool_write_bias_spectrum`` (6) = linspace(0, 6, 6),
    ``pool_value_residual_anchor`` (1536).  Non-persistent: ``rope_cos`` / ``rope_sin`` (8192, 768) fp32 and their bf16
    roundings ``rope_cos_bf16`` / ``rope_sin_bf16`` (gathered by the prefill: the rounding of the gathered rows).
    """

    # ------------------------------------------------------------------------------------------------ build
    def _build_pool_attention(self, config, layer_idx: int, hippocampal_index: int, device, dtype, init: str) -> None:
        D, Dk, Dv, H = int(config.D), int(config.D_key), int(config.D_value), int(config.hippocampal_heads)
        dk, dv = Dk // H, Dv // H
        f32 = torch.float32 if dtype is None else dtype
        launch = init == "launch"
        li = linear_init(init)

        def param(shape, fill: Optional[float] = None) -> nn.Parameter:
            if launch and fill is not None:
                return nn.Parameter(torch.full(shape, float(fill), device=device, dtype=f32))
            return nn.Parameter(torch.empty(shape, device=device, dtype=f32))

        self.D_key, self.D_value, self.hippocampal_heads = Dk, Dv, H
        self.hippocampal_head_dim, self.hippocampal_value_head_dim = dk, dv
        self.hippocampal_base_decay = float(config.hippocampal_base_decay)
        self.write_sparsity = float(config.hippocampal_write_sparsity)
        self.match_scale = float(config.hippocampal_dendritic_scale)
        self.induction_orders = tuple(int(n) for n in config.hippocampal_induction_bias_orders)
        self.xq_count = int(config.hippocampal_pool_extra_queries)
        self.xq_rank = int(config.hippocampal_pool_extra_query_rank)
        self.xq_out_rank = int(config.hippocampal_pool_extra_query_out_rank)
        self.glu_rank = int(config.hippocampal_pool_value_glu_rank)
        self.hippocampal_index = int(hippocampal_index)
        self.value_residual_first = self.hippocampal_index == 0
        # ---- decay (GDN family, vector width dk shared by the heads)
        self.W_decay = Linear(D, dk, device=device, dtype=dtype, init=li)
        if launch:
            with torch.no_grad():
                self.W_decay.weight.zero_()
        self.register_buffer("decay_A_log_init", -torch.linspace(0.0, math.log(DECAY_TAU_MAX), dk, dtype=torch.float32,
                                                                 device=device), persistent=True)
        self.decay_A_log_delta = param((dk,), 0.0)
        # ---- per-head pool spectra (buffers) + deltas
        self.register_buffer("pool_bias_gain_spectrum_raw",
                             pool_bias_gain_spectrum_init(H).to(device=device), persistent=True)
        self.pool_bias_gain_delta = param((H,), 0.0)
        self.register_buffer("pool_dendritic_gain_spectrum_log",
                             pool_dendritic_gain_spectrum_init(H).to(device=device), persistent=True)
        self.pool_dendritic_gain_delta = param((H,), 0.0)
        self.pool_write_bias = param((H,), 0.0)
        # ---- GLU tail of the last head
        g = self.glu_rank
        self.pool_glu_a = param((g, D))
        self.pool_glu_b = param((g, D))
        self.pool_glu_u = param((dv, g), 0.0)
        if launch:
            with forked_rng(device, GLU_SEED_BASE, layer_idx):
                nn.init.xavier_uniform_(self.pool_glu_a)
                nn.init.xavier_uniform_(self.pool_glu_b)
        # ---- value residual
        self.pool_value_residual = param((Dv,), 0.0)
        self.pool_vr_gate = param((H, D), 0.0)
        if self.value_residual_first:
            anchor = torch.zeros((Dv,), dtype=torch.float32, device=device)
        else:
            pairs = config.hippocampal_pool_value_residual_anchor
            if not 1 <= self.hippocampal_index <= len(pairs):
                raise ValueError(f"hippocampal layer rank {self.hippocampal_index}: no value-residual anchor pair "
                                 f"({len(pairs)} pairs for hippocampal layers 1 ..)")
            anchor = value_residual_anchor(pairs[self.hippocampal_index - 1], Dv, H, device)
        self.register_buffer("pool_value_residual_anchor", anchor, persistent=True)
        # ---- extra queries of the last head
        M, r, ro = self.xq_count, self.xq_rank, self.xq_out_rank
        self.pool_xq_a = param((M * r, D))
        self.pool_xq_b = nn.ParameterList([param((dk, r)) for _ in range(M)])
        self.pool_xq_gain_proj = param((M, D), 0.0)
        self.pool_xq_out_in = param((ro, M * dv))
        self.pool_xq_out = param((dv, ro), 0.0)
        if launch:
            with forked_rng(device, XQ_SEED_BASE, layer_idx):
                nn.init.xavier_uniform_(self.pool_xq_a)
                for b in self.pool_xq_b:
                    nn.init.xavier_uniform_(b)
                nn.init.xavier_uniform_(self.pool_xq_out_in)
        self.register_buffer("pool_write_bias_spectrum",
                             pool_write_bias_spectrum_init(H, config.hippocampal_pool_write_bias_spectrum_span).to(
                                 device=device), persistent=True)
        # ---- projections, the write-gate neuron, the qkv short conv, the output norm / gate / projection
        qkv = 2 * Dk + Dv
        self.qkv_proj = Linear(D, qkv, device=device, dtype=dtype, init=li)
        self.gate_neuron = PLIFNeuron(D, config.init_tau, centred=False, alpha=config.plif_surrogate_alpha,
                                      eta_rank=config.plif_eta_proj_rank, device=device, dtype=dtype, impl=self.impl,
                                      init=init)
        self.qkv_conv = ShortConvWeight(qkv, int(config.hippocampal_short_conv), device=device, dtype=dtype,
                                        init="empty")
        if launch:
            with torch.no_grad():
                # the depthwise Conv1d's default draw, then the identity launch (pass-through: last tap 1)
                nn.init.kaiming_uniform_(self.qkv_conv.weight, a=math.sqrt(5))
            self.qkv_conv.reset_parameters()
        self.attn_out_norm = RMSNorm(Dv, eps=float(config.rms_norm_eps), zero_centered=True, device=device,
                                     dtype=dtype, init="default" if launch else "empty")
        self.attn_out_gate = Linear(D, Dv, device=device, dtype=dtype, init=li)
        self.attn_out_proj = Linear(Dv, D, device=device, dtype=dtype, init=li)
        rope_cos, rope_sin = rope_head_band_tables(Dk, H, int(config.rope_max_seq_len), float(config.rope_base),
                                                   device=device)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.register_buffer("rope_cos_bf16", rope_cos.to(torch.bfloat16), persistent=False)
        self.register_buffer("rope_sin_bf16", rope_sin.to(torch.bfloat16), persistent=False)

    def _finish_pool_attention_init(self, out_std: float) -> None:
        """Launch draws taken after the FFN sub-block is built (global stream): qkv_proj Xavier-uniform, attn_out_proj
        N(0, out_std)."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.qkv_proj.weight)
            self.attn_out_proj.weight.normal_(std=float(out_std))

    def _build_induction_bias(self, config, device, dtype, init: str) -> None:
        n = len(self.induction_orders) * self.hippocampal_heads
        f32 = torch.float32 if dtype is None else dtype
        t = torch.zeros(n, device=device, dtype=f32) if init == "launch" else torch.empty(n, device=device, dtype=f32)
        self.pool_ind_beta = nn.Parameter(t)

    # ------------------------------------------------------------------------------------------------ constants
    @property
    def decay_A_log(self) -> torch.Tensor:
        """(256,) fp32 GDN rate exponent: launch spectrum + delta."""
        return self.decay_A_log_init + self.decay_A_log_delta

    def pool_bias_gains(self) -> torch.Tensor:
        """(H,) fp32 eligibility slopes gamma_h = softplus(spectrum + delta + log(e - 1))."""
        return F.softplus((self.pool_bias_gain_spectrum_raw + self.pool_bias_gain_delta).float()
                          + POOL_BIAS_GAIN_RAW_SHIFT)

    def pool_query_gain(self, dtype: torch.dtype) -> torch.Tensor:
        """(H,) match-gain factors exp(spectrum + delta) computed in fp32 and cast to ``dtype`` (the dtype of the
        query the gain multiplies)."""
        return torch.exp((self.pool_dendritic_gain_spectrum_log + self.pool_dendritic_gain_delta).float()).to(dtype)

    def pool_write_bias_total(self) -> torch.Tensor:
        """(H,) fp32 write-bias offsets: spectrum + delta."""
        return self.pool_write_bias_spectrum + self.pool_write_bias.float()

    def value_residual_lambda0(self) -> torch.Tensor:
        """(D_value,) fp32 static part of the value-residual mix: pool_value_residual + anchor."""
        return self.pool_value_residual.float() + self.pool_value_residual_anchor.float()

    def induction_beta(self) -> torch.Tensor:
        """(orders, H) view of pool_ind_beta (flat index n * H + h)."""
        return self.pool_ind_beta.view(len(self.induction_orders), self.hippocampal_heads)

    # ------------------------------------------------------------------------------------------------ shared pieces
    def _check_windows(self, ind, B: int, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = max(self.induction_orders)
        if (not isinstance(ind, (tuple, list)) or len(ind) != 2
                or any(not isinstance(w, torch.Tensor) or tuple(w.shape) != (B, T, n) or w.dtype != torch.int32
                       for w in ind)):
            got = None if not isinstance(ind, (tuple, list)) else [tuple(getattr(w, "shape", ())) for w in ind]
            raise RuntimeError(f"hippocampal layer: the whole-row induction bias needs the two int32 ({B}, {T}, {n}) id "
                               f"windows of induction_token_windows, got {got}")
        return ind[0], ind[1]

    def _check_ind(self, ind, shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, ...]:
        if (not isinstance(ind, (tuple, list)) or len(ind) != len(self.induction_orders)
                or any(not isinstance(m, torch.Tensor) or tuple(m.shape) != tuple(shape) or m.dtype != torch.bool
                       for m in ind)):
            raise RuntimeError(f"hippocampal layer: the induction bias needs one bool {tuple(shape)} match tensor per "
                               f"order "
                               f"{self.induction_orders}, got "
                               f"{None if not isinstance(ind, (tuple, list)) else [tuple(getattr(m, 'shape', ())) for m in ind]}")
        return tuple(ind)

    def _memory_decay(self, logit: torch.Tensor) -> torch.Tensor:
        """(T, B, 256, 1) fp32 decay a = exp(-bf16(exp(A_log)) softplus(f32(logit))) * hippocampal_base_decay from the
        bf16
        decay logit (T, B, 256)."""
        rate = torch.exp(self.decay_A_log).to(logit.dtype)
        a = torch.exp(-rate * F.softplus(logit.float()))
        a = a * self.hippocampal_base_decay
        return a.to(torch.float32).unsqueeze(-1)

    @staticmethod
    def _trace_increment(a: torch.Tensor, T: int, B: int, gain: torch.Tensor) -> torch.Tensor:
        """(T, B) fp32 gained clock increment gamma_h mean_c log(max(a, 1e-6)) of one head (whole row)."""
        _a = a.reshape(T, B, -1).float().clamp_min(DECAY_FLOOR)
        return _a.log().mean(dim=-1) * gain

    def _clock_increments(self, a: torch.Tensor, T: int, B: int) -> torch.Tensor:
        """(T, B, H) fp32 gained clock increments of every head (inference): the shared mean log decay times gamma_h."""
        H = self.hippocampal_heads
        _a = a.reshape(T, B, -1).float().clamp_min(DECAY_FLOOR)
        la = _a.log().mean(dim=-1, keepdim=True).expand(T, B, H)
        return la * self.pool_bias_gains().view(1, 1, H)

    def _tagged_values(self, g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """bf16 sigmoid(f32(g) - write_sparsity) times the bf16 values (g already carries the write bias)."""
        g_raw = g.float()
        if self.write_sparsity > 0.0:
            g_raw = g_raw - self.write_sparsity
        return torch.sigmoid(g_raw).to(v.dtype) * v

    def _value_residual_torch(self, v: torch.Tensor, lam: torch.Tensor, carry: Optional[torch.Tensor]):
        """(v', carry_out) of an inference call (torch expressions)."""
        if self.value_residual_first:
            return (v.float() * (1.0 + lam)).to(v.dtype), v
        self._check_carry(carry, v)
        return (v.float() + lam * carry.float()).to(v.dtype), carry

    @staticmethod
    def _check_carry(carry: Optional[torch.Tensor], v: torch.Tensor) -> None:
        if carry is None or tuple(carry.shape) != tuple(v.shape) or carry.dtype != v.dtype:
            raise RuntimeError("hippocampal layer: the value residual needs the first hippocampal layer's value stream "
                               f"of this call as the carry ({tuple(v.shape)} {v.dtype}), got "
                               f"{None if carry is None else (tuple(carry.shape), carry.dtype)}")

    def _glu_addend(self, z: torch.Tensor) -> torch.Tensor:
        """(T, B, dv) fp32 GLU addend of the last head's value channels (inference)."""
        lin = self._ops.linear
        gp = F.silu(lin(z, self.pool_glu_a).float()) * lin(z, self.pool_glu_b).float()
        return lin(gp.to(z.dtype), self.pool_glu_u).float()

    def _glu_tail_torch(self, v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """v with the GLU addend added to the last head's value channels (inference, torch)."""
        gv = self._glu_addend(z)
        dvh = self.hippocampal_value_head_dim
        return torch.cat([v[..., :self.D_value - dvh], (v[..., self.D_value - dvh:].float() + gv).to(v.dtype)], dim=-1)

    def _xq_band(self, t: torch.Tensor) -> torch.Tensor:
        """The last head's column band of a (..., D_key / 2) RoPE table."""
        half = self.hippocampal_head_dim // 2
        lo = (self.hippocampal_heads - 1) * half
        return t[..., lo:lo + half]

    def _xq_queries_torch(self, z: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, q_dtype: torch.dtype,
                          rope) -> torch.Tensor:
        """(B, M, T, dk) fp32 extra queries of a decode step (torch formation; ``rope`` the regime's interleaved
        rotation): unit rotated SiLU projections times exp(clamp(psi, -3, 3)) and the last head's match gain in the
        query dtype.  A prefill call forms the same queries with :meth:`_xq_queries_prefill`."""
        lin = self._ops.linear
        H, M, r = self.hippocampal_heads, self.xq_count, self.xq_rank
        head_gain = self.pool_query_gain(q_dtype)[H - 1]
        cos5, sin5 = self._xq_band(cos), self._xq_band(sin)
        u = lin(z, self.pool_xq_a)
        psi = lin(z, self.pool_xq_gain_proj).to(torch.float32).clamp(-XQ_PSI_CLAMP, XQ_PSI_CLAMP)
        queries = []
        for m in range(M):
            q_m = F.silu(lin(u[..., m * r:(m + 1) * r], self.pool_xq_b[m]))
            q_m = F.normalize(rope(q_m, cos5.to(q_m.dtype), sin5.to(q_m.dtype)), dim=-1)
            g = torch.exp(psi[..., m:m + 1]) * head_gain.to(torch.float32)
            queries.append(q_m * g.to(q_m.dtype))
        return torch.stack(queries, dim=0).permute(2, 0, 1, 3)

    def _xq_queries_prefill(self, z: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                            q_dtype: torch.dtype) -> torch.Tensor:
        """(B, M, T, dk) bf16 extra queries of a prefill call: the projections (bf16 GEMMs), then the formation of
        :meth:`_xq_queries_torch` (SiLU, interleaved RoPE, normalisation, gain) for all M queries in one kernel
        (``ops.xq_queries_prefill``), rounded once to bf16 -- the rounding the ring read applies to its queries."""
        o = self._ops
        lin = o.linear
        H, M, r = self.hippocampal_heads, self.xq_count, self.xq_rank
        head_gain = self.pool_query_gain(q_dtype)[H - 1]
        cos5, sin5 = self._xq_band(cos), self._xq_band(sin)
        u = lin(z, self.pool_xq_a)
        psi = lin(z, self.pool_xq_gain_proj).to(torch.float32).clamp(-XQ_PSI_CLAMP, XQ_PSI_CLAMP)
        qs = [lin(u[..., m * r:(m + 1) * r], self.pool_xq_b[m]) for m in range(M)]
        return o.xq_queries_prefill(qs, psi, head_gain, cos5, sin5)

    def _out_norm_gate_infer(self, attn: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Inference output norm x gate in one operator (``ops.rmsnorm_gate_rows``): the zero-centred rms norm of attn
        (fp32) times sigmoid(bf16(h @ attn_out_gate^T)) (bf16 sigmoid), rounded to bf16 for the output projection."""
        o = self._ops
        return o.rmsnorm_gate_rows(attn.to(h.dtype), self.attn_out_norm.eps, self.attn_out_norm.weight,
                                   o.linear(h, self.attn_out_gate.weight))

    def _out_proj_centred(self, attn: torch.Tensor) -> torch.Tensor:
        r = self._ops.linear(attn, self.attn_out_proj.weight)
        return r - r.mean(dim=-1, keepdim=True)

    # ------------------------------------------------------------------------------------------------ whole row
    def _attention_train(self, h: torch.Tensor, z: torch.Tensor, ind, carry: Optional[torch.Tensor]):
        """Whole-row attention sub-block: (r (T, B, D) bf16 centred residual, carry_out)."""
        o = self._ops
        lin = o.linear
        T, B, D = (int(s) for s in h.shape)
        H, dk, dv = self.hippocampal_heads, self.hippocampal_head_dim, self.hippocampal_value_head_dim
        qtok, ktok = self._check_windows(ind, B, T)
        # qkv projection (on the (T * B, D) view) and the causal short conv
        qkv = lin(z.reshape(T * B, D), self.qkv_proj.weight).view(T, B, -1)
        qkv = o.short_conv_train(qkv, self.qkv_conv.weight)
        q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
        v = F.silu(v)
        # value residual: lambda with the per-token gate over the strictly-past running mean of z
        lam = self.value_residual_lambda0()
        zf = z.float()
        zcum = torch.cat([torch.zeros_like(zf[:1]), o.cumsum0(zf)[:-1]], dim=0)
        g_vr = torch.tanh(lin(o.zc_center(zf, zcum), self.pool_vr_gate).to(torch.float32))
        lam = lam.view(1, 1, -1) + g_vr.repeat_interleave(dv, dim=-1)
        if self.value_residual_first:
            carry_out = v
            carry_in = None
        else:
            self._check_carry(carry, v)
            carry_out = carry_in = carry
        # GLU addend of the last head, applied with the value residual
        zc = z.contiguous()
        gp = F.silu(lin(zc, self.pool_glu_a).float()) * lin(zc, self.pool_glu_b).float()
        gv_add = lin(gp.to(z.dtype).contiguous(), self.pool_glu_u).float()
        v = o.value_residual(v, lam, carry_in, gv_add)
        # q / k SiLU + RoPE at positions 0 .. T - 1
        cos = self.rope_cos[:T].to(q.dtype)
        sin = self.rope_sin[:T].to(q.dtype)
        q = o.silu_rope(q, cos, sin)
        k = o.silu_rope(k, cos, sin)
        # write gate, decay
        gate = self.gate_neuron(z)
        a = self._memory_decay(lin(z.float().reshape(T * B, D), self.W_decay.weight).view(T, B, -1))
        # per-head pool constants and the head preparation
        gains = self.pool_bias_gains()
        qgain = self.pool_query_gain(q.dtype)
        wbias = self.pool_write_bias_total()
        qn, kn, gn = o.pool_heads(q, k, gate, qgain, wbias, H, dk)
        vs = o.value_split(v, H, dv)
        # one pool read for the layer: the H heads, then the last head's extra queries (its keys, tagged values,
        # clock and induction term)
        xq = self._extra_queries_train(z, cos, sin, qgain[H - 1])
        q_all = torch.cat([torch.stack(qn, dim=2), xq.to(qn[0].dtype)], dim=2)          # (T, B, H + M, dk)
        gv = torch.stack([self._tagged_values(gn[i], vs[i]) for i in range(H)], dim=2)  # (T, B, H, dv) bf16
        cum = torch.stack([o.cumsum0(self._trace_increment(a, T, B, gains[i])) for i in range(H)], dim=2)
        reads = o.pool_read(q_all, torch.stack(kn, dim=2), gv, cum, self.induction_beta(), qtok, ktok,
                            self.induction_orders, self.match_scale)                    # (T, B, H + M, dv) bf16
        heads = [reads[:, :, i].to(torch.float32) for i in range(H)]
        heads[-1] = heads[-1] + self._extra_query_out_train(reads[:, :, H:])
        attn = torch.cat(heads, dim=-1)
        # zero-centred output norm x sigmoid gate (the gate reads h), output projection, centring
        gate_logits = lin(h.reshape(T * B, D), self.attn_out_gate.weight).view(T, B, -1)
        attn = o.out_norm_gate(attn, self.attn_out_norm.weight, gate_logits, self.attn_out_norm.eps)
        return self._out_proj_centred(attn), carry_out

    def _extra_queries_train(self, z, cos, sin, head_gain) -> torch.Tensor:
        """(T, B, M, dk) the last head's extra queries (whole row): SiLU of the rank-r projections, RoPE on the head's
        band, normalised, times exp(clamp(psi)) and the head's match gain."""
        o = self._ops
        lin = o.linear
        M, r = self.xq_count, self.xq_rank
        cos5, sin5 = self._xq_band(cos), self._xq_band(sin)
        zc = z.contiguous()
        u = lin(zc, self.pool_xq_a)
        psi = lin(zc, self.pool_xq_gain_proj).to(torch.float32).clamp(-XQ_PSI_CLAMP, XQ_PSI_CLAMP)
        queries = []
        for m in range(M):
            q_m = F.silu(lin(u[..., m * r:(m + 1) * r].contiguous(), self.pool_xq_b[m]))
            q_m = F.normalize(o.rope_paired(q_m, cos5.to(q_m.dtype), sin5.to(q_m.dtype)), dim=-1)
            g = torch.exp(psi[..., m:m + 1]) * head_gain.to(torch.float32)
            queries.append(q_m * g.to(q_m.dtype))
        return torch.stack(queries, dim=2)

    def _extra_query_out_train(self, reads: torch.Tensor) -> torch.Tensor:
        """Whole-row addend (T, B, dv) of the last head from its extra queries' reads (T, B, M, dv) bf16."""
        lin = self._ops.linear
        T, B = int(reads.shape[0]), int(reads.shape[1])
        r = reads.reshape(T, B, self.xq_count * self.hippocampal_value_head_dim).to(torch.float32)
        return lin(lin(r, self.pool_xq_out_in).contiguous(), self.pool_xq_out)

    # ------------------------------------------------------------------------------------------------ prefill
    def _short_conv_two(self, qkv: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """2-token call: the depthwise conv over cat(carried history, x) (bf16 under autocast); history rolled to the
        last three inputs.  Returns (2, B, C) contiguous."""
        x = qkv.permute(1, 2, 0)                                                  # (B, C, T)
        full = torch.cat([hist.to(x.dtype), x], dim=2)                            # (B, C, 3 + T)
        out = F.conv1d(full, self.qkv_conv.weight, groups=int(x.shape[1]))
        hist.copy_(full[:, :, full.shape[2] - hist.shape[2]:])
        return out.permute(2, 0, 1).contiguous()

    def _attention_prefill(self, h: torch.Tensor, z: torch.Tensor, ind, carry, st: HippocampalState, pos0: torch.Tensor,
                           ptr0: torch.Tensor, W: int):
        """Prefill attention sub-block (T >= 2) over the carried rings: (r bf16 centred residual, carry_out)."""
        o = self._ops
        lin = o.linear
        T, B, D = (int(s) for s in h.shape)
        H, dk, dv = self.hippocampal_heads, self.hippocampal_head_dim, self.hippocampal_value_head_dim
        matches = self._check_ind(ind, (B, T, W))
        if T > W:
            raise RuntimeError(f"hippocampal layer: a call of {T} tokens exceeds the ring capacity {W}")
        qkv = lin(z, self.qkv_proj.weight)
        if T >= 3:
            qkv = o.short_conv_silu_prefill(qkv, st.conv_hist, self.qkv_conv.weight)      # the conv and its SiLU
        else:
            qkv = F.silu(self._short_conv_two(qkv, st.conv_hist))
        q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
        # value-residual mix and GLU tail in one kernel: the centred input in bf16 (its only consumer is the bf16 gate
        # projection, whose input cast is the same rounding); layer 0's unmixed values are the later layers' carry
        zc = o.centred_running(z, st.vr_sum, pos0, torch.bfloat16)
        if self.value_residual_first:
            carry_in, carry_out = None, v
        else:
            self._check_carry(carry, v)
            carry_in = carry_out = carry
        v = o.vr_mix_prefill(v, carry_in, lin(zc, self.pool_vr_gate), self.value_residual_lambda0(), self._glu_addend(z))
        positions = pos0.view(1, B) + torch.arange(T, device=h.device, dtype=torch.int64).view(T, 1)
        cos = self.rope_cos_bf16[positions].to(q.dtype)
        sin = self.rope_sin_bf16[positions].to(q.dtype)
        q = o.rope_interleaved(q, cos, sin)
        k = o.rope_interleaved(k, cos, sin)
        gate = self.gate_neuron(z, st.gate_neuron, pos0)
        a = self._memory_decay(lin(z.float(), self.W_decay.weight))
        # ring write of the call's events (unit keys, tagged values); the gained unit queries (fp32 gain)
        q_h = q.reshape(T, B, H, dk)
        k_h = k.reshape(T, B, H, dk)
        v_h = v.reshape(T, B, H, dv)
        g_h = gate.reshape(T, B, H, dk)
        qn = o.ring_write_prefill(k_h, q_h, g_h, v_h, self.pool_query_gain(torch.float32), self.pool_write_bias_total(),
                                  self.write_sparsity, st.pool_k, st.pool_gv, ptr0, T, B, H, W)
        # absolute clocks: the sequential cumulative sum from the carried clock
        la = self._clock_increments(a, T, B)
        run = torch.cat([st.pool_run.unsqueeze(0), la], dim=0)
        cum_abs = o.exact_cumsum0(run.reshape(T + 1, B * H)).view(T + 1, B, H)[1:]
        slots = (ptr0 + torch.arange(T, device=h.device, dtype=torch.int64)) % W
        st.pool_cum.index_copy_(2, slots, cum_abs.permute(1, 2, 0).contiguous())
        beta = self.induction_beta().to(torch.float32)
        out = o.ring_recall_prefill(qn.permute(1, 2, 0, 3), st.pool_k, st.pool_gv, st.pool_cum, cum_abs.permute(1, 2, 0),
                                    ptr0, (matches, beta), self.match_scale, T, B, H, W, dk, dv, h.dtype, False, 0)
        st.pool_run.copy_(cum_abs[-1])
        # extra queries of the last head over its ring
        xq = self._xq_queries_prefill(z, cos, sin, q.dtype)
        # the reads in bf16: their only consumer is the bf16 pool_xq_out_in projection
        reads = o.ring_recall_prefill(xq, st.pool_k, st.pool_gv, st.pool_cum, None, ptr0, (matches, beta),
                                      self.match_scale, T, B, self.xq_count, W, dk, dv, torch.bfloat16, True, H - 1)
        xq_add = lin(lin(reads, self.pool_xq_out_in), self.pool_xq_out)
        last = (H - 1) * dv
        attn = torch.cat([out[..., :last], out[..., last:] + xq_add], dim=-1)
        return self._out_proj_centred(self._out_norm_gate_infer(attn, h)), carry_out

    # ------------------------------------------------------------------------------------------------ decode
    def _attention_step(self, h: torch.Tensor, z: torch.Tensor, ind, carry, st: HippocampalState, pos0: torch.Tensor,
                        ptr0: torch.Tensor, W: int):
        """Single-token attention sub-block over the carried rings: (r bf16 centred residual, carry_out).  Every
        state write is in place (CUDA-graph replay safe)."""
        o = self._ops
        lin = o.linear
        T, B, D = (int(s) for s in h.shape)
        H, dk, dv, M = self.hippocampal_heads, self.hippocampal_head_dim, self.hippocampal_value_head_dim, self.xq_count
        matches = self._check_ind(ind, (B, 1, W))
        qkv = lin(z, self.qkv_proj.weight)
        qkv = o.short_conv_step(qkv[0], st.conv_hist, self.qkv_conv.weight).unsqueeze(0)
        q, k, v = qkv.split([self.D_key, self.D_key, self.D_value], dim=-1)
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        lam = self.value_residual_lambda0()
        zc = centred_running_step(z, st.vr_sum, pos0).to(z.dtype)
        g_vr = torch.tanh(lin(zc, self.pool_vr_gate).to(torch.float32))
        lam = lam.view(1, 1, -1) + g_vr.repeat_interleave(dv, dim=-1)
        v, carry_out = self._value_residual_torch(v, lam, carry)
        v = self._glu_tail_torch(v, z)
        positions = pos0.view(1, B)
        cos = self.rope_cos[positions].to(q.dtype)
        sin = self.rope_sin[positions].to(q.dtype)
        q = apply_rope_interleaved_ref(q, cos, sin)
        k = apply_rope_interleaved_ref(k, cos, sin)
        gate = self.gate_neuron(z, st.gate_neuron, pos0)
        a = self._memory_decay(lin(z.float(), self.W_decay.weight))
        ring_kernel = B in RING_KERNEL_DECODE_SIZES
        xq = self._xq_queries_torch(z, cos, sin, q.dtype, apply_rope_interleaved_ref) if ring_kernel else None
        # ring write (torch): unit keys, gained unit queries (fp32 gain), tagged values, absolute clocks
        k_hat = F.normalize(k.reshape(T, B, H, dk), dim=-1)
        q_h = F.normalize(q.reshape(T, B, H, dk), dim=-1)
        q_h = q_h * self.pool_query_gain(q_h.dtype).view(1, 1, H, 1)
        g_h = gate.reshape(T, B, H, dk).float() + self.pool_write_bias_total().view(1, 1, H, 1)
        gv = self._tagged_values(g_h, v.reshape(T, B, H, dv))
        cum_abs = st.pool_run.unsqueeze(0) + self._clock_increments(a, T, B).cumsum(dim=0)
        slots = (ptr0 + torch.arange(T, device=h.device, dtype=torch.int64)) % W
        st.pool_k.index_copy_(2, slots, k_hat.permute(1, 2, 0, 3).to(st.pool_k.dtype))
        st.pool_gv.index_copy_(2, slots, gv.permute(1, 2, 0, 3).to(st.pool_gv.dtype))
        st.pool_cum.index_copy_(2, slots, cum_abs.permute(1, 2, 0).contiguous())
        beta = self.induction_beta().to(torch.float32)
        if ring_kernel:
            # the ring kernel: every head's read and the extra queries' reads over the live slots (pos0 = tokens
            # before this step; ptr0 = this step's slot)
            red = o.ring_read_t1(q_h[0], xq[:, :, 0], cum_abs[0], st.pool_k, st.pool_gv, st.pool_cum, pos0, ptr0,
                                 matches, beta, self.match_scale)
            red = red.reshape(B, -1)
            base = red[:, :H * dv].reshape(1, B, H * dv).to(h.dtype)
            reads = red[:, H * dv:].reshape(B, M, 1, dv)
            r_x = reads.permute(2, 0, 1, 3).reshape(T, B, -1).to(h.dtype)
            xq_add = lin(lin(r_x, self.pool_xq_out_in), self.pool_xq_out)
        else:
            term = self._induction_ring_terms(matches, beta)                     # (B, H, 1, W) fp32
            base = self._ring_read_step_torch(q_h, cum_abs, st, term, h.dtype)
            xq_add = self._xq_read_step_torch(z, cos, sin, q.dtype, term[:, H - 1], st, ptr0, W, h.dtype)
        st.pool_run.copy_(cum_abs[-1])
        last = (H - 1) * dv
        attn = torch.cat([base[..., :last], base[..., last:] + xq_add], dim=-1)
        return self._out_proj_centred(self._out_norm_gate_infer(attn, h)), carry_out

    def _induction_ring_terms(self, matches, beta: torch.Tensor) -> torch.Tensor:
        """(B, H, T, W) fp32 induction terms of every head from the ring-layout matches: m_1 b_1, then + m_n b_n."""
        H = self.hippocampal_heads
        out = matches[0].unsqueeze(1).to(torch.float32) * beta[0].view(1, H, 1, 1)
        for n in range(1, len(matches)):
            out = out + matches[n].unsqueeze(1).to(torch.float32) * beta[n].view(1, H, 1, 1)
        return out

    @torch.no_grad()
    def _xq_weight_stack(self) -> torch.Tensor:
        """(M, dk, r) bf16 contiguous stack of ``pool_xq_b`` (the values the bf16 projection uses).  A documented
        cache: rebuilt whenever a weight's version, device or the stack's dtype changes, so build it in an eager call
        (e.g. the warm-up before a CUDA-graph capture) and it is reused by the captured steps."""
        key = tuple((int(w._version), w.device, w.data_ptr()) for w in self.pool_xq_b)
        cached = self.__dict__.get("_xq_wb_cache")
        if cached is None or cached[0] != key:
            wb = torch.stack([w.detach().to(torch.bfloat16) for w in self.pool_xq_b], dim=0).contiguous()
            self.__dict__["_xq_wb_cache"] = (key, wb)
            return wb
        return cached[1]

    def _ring_read_step_torch(self, q_h, cum_abs, st: HippocampalState, term, out_dtype) -> torch.Tensor:
        """(1, B, H dv) T = 1 ring read of every head over all W slots (empty slots: +inf clocks), torch chain."""
        T, B, H, _ = (int(s) for s in q_h.shape)
        dv = self.hippocampal_value_head_dim
        qh = q_h[0].to(st.pool_k.dtype).float()                                  # (B, H, dk)
        scores = torch.einsum("bhd,bhwd->bhw", qh, st.pool_k.float()) * self.match_scale
        bias = cum_abs[0].unsqueeze(-1) - st.pool_cum                            # (B, H, W)
        bias = bias + term[:, :, 0]
        bias = bias.to(st.pool_k.dtype).float()
        probs = torch.softmax(scores + bias, dim=-1)
        out = torch.einsum("bhw,bhwd->bhd", probs.to(st.pool_gv.dtype).float(), st.pool_gv.float())
        return out.reshape(1, B, H * dv).to(out_dtype)

    def _xq_read_step_torch(self, z, cos, sin, q_dtype, term5, st: HippocampalState, ptr0, W: int,
                            out_dtype) -> torch.Tensor:
        """(1, B, dv) T = 1 addend of the extra queries: ``ops.xq_form_t1`` forms the queries, the torch chain reads the
        last head's ring (+ the head's induction term) and the two output maps project."""
        o = self._ops
        lin = o.linear
        T, B = int(z.shape[0]), int(z.shape[1])
        H = self.hippocampal_heads
        u = lin(z, self.pool_xq_a)
        psi = lin(z, self.pool_xq_gain_proj)
        head_gain = self.pool_query_gain(q_dtype)[H - 1:H].to(torch.float32)
        wb = self._xq_weight_stack()
        cos5, sin5 = self._xq_band(cos), self._xq_band(sin)
        q = o.xq_form_t1(u[0], wb, cos5[0], sin5[0], psi[0], head_gain).unsqueeze(2)      # (B, M, 1, dk)
        i = H - 1
        qh = q.to(st.pool_k.dtype).float()
        k_ring, gv_ring, cum_ring = st.pool_k[:, i], st.pool_gv[:, i], st.pool_cum[:, i]
        slots = (ptr0 + torch.arange(T, device=z.device, dtype=torch.int64)) % W
        cum_t = cum_ring.index_select(1, slots)                                  # (B, 1)
        scores = torch.einsum("bmtd,bwd->bmtw", qh, k_ring.float()) * self.match_scale
        bias = cum_t.unsqueeze(-1) - cum_ring.unsqueeze(1)                       # (B, 1, W)
        bias = bias + term5
        bias = bias.to(st.pool_k.dtype).float()
        probs = torch.softmax(scores + bias.unsqueeze(1), dim=-1)
        out = torch.einsum("bmtw,bwd->bmtd", probs.to(st.pool_gv.dtype).float(), gv_ring.float())
        reads = out.permute(2, 0, 1, 3).reshape(T, B, -1).to(out_dtype)
        return lin(lin(reads, self.pool_xq_out_in), self.pool_xq_out)
