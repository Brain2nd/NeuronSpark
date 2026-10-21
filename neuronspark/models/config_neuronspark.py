"""Configuration of the NeuronSpark language model and its on-disk form (``config.json``).

``NeuronSparkConfig`` holds only the fields that change the computed function: a parameter shape, a numeric constant
of the forward math, of the initialisation or of the launch-time calibration.  The architecture itself (layer forms,
neuron model, mechanisms) is fixed in the code and is therefore not configurable.  The defaults are one configuration
of the architecture (the default configuration); a checkpoint directory carries its own ``config.json``.

Layout (ratio 3:1): layer ``i`` is a hippocampal layer iff ``(i + 1) % hippocampal_layer_interval == 0`` (in the default
24-layer configuration: layers 3, 7, 11, 15, 19, 23), every other layer is a dendritic neuron layer.

The single numerics option is ``inference_dendritic_state_dtype``: the storage dtype of the dendritic-layer state in the
inference cache, ``"bfloat16"`` (default: the state is rounded to bf16 once per decode step and once per prefill call)
or ``"float32"`` (the state is carried in fp32).  It changes no parameter and no training computation.

Configuration file format (``config.json`` of a checkpoint directory, see :mod:`neuronspark.utils.hf`): one JSON
object holding

* ``"model_type": "neuronspark"`` (:data:`MODEL_TYPE`, required in a file);
* ``"architectures": ["NeuronSparkLMHeadModel"]`` (:data:`ARCHITECTURES`; checked when present);
* one key per :class:`NeuronSparkConfig` field, named exactly as the field: integers and floats as JSON numbers,
  tuples as JSON arrays (``hippocampal_pool_value_residual_anchor`` as an array of ``[a, b]`` pairs), the dtype option
  as a string.

:meth:`NeuronSparkConfig.to_dict` / :meth:`~NeuronSparkConfig.to_json` write every field;
:meth:`~NeuronSparkConfig.from_dict` / :meth:`~NeuronSparkConfig.from_json` read the object back (a field that is
absent takes its default; any key that is not a field raises; the values are type-checked and validated).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple, Union

__all__ = [
    "NeuronSparkConfig",
    "MODEL_TYPE",
    "ARCHITECTURES",
    "DENDRITIC_STATE_DTYPES",
]

# "model_type" of the configuration file
MODEL_TYPE = "neuronspark"
# "architectures" of the configuration file
ARCHITECTURES = ("NeuronSparkLMHeadModel",)
# values of the inference_dendritic_state_dtype option
DENDRITIC_STATE_DTYPES = ("bfloat16", "float32")

_PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class NeuronSparkConfig:
    """Hyper-parameters of the NeuronSpark architecture (the defaults are the default configuration).

    Groups (consumer in parentheses):

    * vocabulary / trunk: ``vocab_size`` (embedding, tied readout, lexical order-1 rows, pad id ``V`` of the lexical
      and induction windows), ``D`` (model width), ``num_layers``, ``hippocampal_layer_interval``.
    * hippocampal layer, spiking FFN: ``D_ff``.
    * hippocampal layer, pool attention: ``D_key``, ``D_value``, ``hippocampal_heads`` (head width ``D_key /
      hippocampal_heads``), ``hippocampal_short_conv`` (qkv causal depthwise conv width), ``hippocampal_base_decay``
      (decay ``a = exp(-base * rate * softplus(logit))``), ``hippocampal_write_sparsity`` (write tag ``sigmoid(g + wb -
      2.0)``), ``hippocampal_pool_write_bias_spectrum_span`` (write-bias spectrum ``linspace(0, span, heads)``),
      ``hippocampal_pool_extra_queries`` / ``_extra_query_rank`` / ``_extra_query_out_rank`` (extra queries on the last
      head), ``hippocampal_pool_value_glu_rank`` (GLU tail of the last head), ``hippocampal_pool_value_residual_anchor``
      (anchor pairs of hippocampal layers 1..5; hippocampal layer 0 is the value source), ``hippocampal_read_copy_gain``
      (read gain ``1 + 4 pi c``), ``hippocampal_copy_novelty_gate_init`` (init of ``copy_novelty_b``),
      ``hippocampal_copy_novelty_gate_aux`` (weight of the copy-switch auxiliary loss),
      ``hippocampal_induction_bias_orders`` (n-gram orders of the induction matches).
    * product-key memory: ``hippocampal_pkm_heads``, ``_subkeys`` (slots = subkeys ** 2), ``_topk``, ``_query_dim``,
      ``_value_dim``, launch calibration ``_launch_ratio`` / ``_kmeans_iters``, ``hippocampal_pkm_copy_gate_strength``
      (``rho = 2 sigmoid(-strength * pi)``).
    * RoPE: ``rope_base``, ``rope_max_seq_len`` (table length; also the largest inference ring capacity).
    * PLIF neurons: ``init_tau``, ``plif_input2_init_tau`` (tau spectrum of ``input_neuron2``),
      ``plif_eta_proj_rank``, ``plif_surrogate_alpha`` (backward slope), ``plif_threshold_match_p`` (launch threshold
      match quantile).
    * lexical memory: ``lexical_memory_rows`` / ``_heads`` / ``_head_dims`` per n-gram order (order 1 first),
      ``lexical_memory_key_dim``, ``lexical_memory_init_std`` (table init std and fixed output scale ``1 / std``),
      ``lexical_memory_gate_bias`` (init of ``lex_beta``), ``lexical_memory_hash_seed``.
    * dendritic neuron layer: ``dendritic_expand``, ``dendritic_head_dim``, ``dendritic_state_size``,
      ``dendritic_chunk_size``, ``dendritic_theta0`` / ``dendritic_theta_floor`` (soma threshold ``theta = floor +
      softplus(rho)``, launched at ``theta0``), ``dendritic_surrogate_alpha``.
    * readout: ``readout_balance_widths`` (tiled balance projections, added width 16 first, then 32).
    * norms: ``dendritic_norm_eps`` (dendritic-layer pre-norm and gated norm), ``rms_norm_eps`` (every other RMSNorm).
    * inference: ``inference_dendritic_state_dtype`` (see the module docstring).
    """

    # ---- vocabulary / trunk
    vocab_size: int = 151669
    D: int = 1536
    num_layers: int = 24
    hippocampal_layer_interval: int = 4
    # ---- hippocampal layer: spiking FFN
    D_ff: int = 768
    # ---- hippocampal layer: pool attention
    D_key: int = 1536
    D_value: int = 1536
    hippocampal_heads: int = 6
    hippocampal_short_conv: int = 4
    hippocampal_base_decay: float = 1.0
    hippocampal_write_sparsity: float = 2.0
    hippocampal_pool_write_bias_spectrum_span: float = 6.0
    hippocampal_pool_extra_queries: int = 4
    hippocampal_pool_extra_query_rank: int = 64
    hippocampal_pool_extra_query_out_rank: int = 64
    hippocampal_pool_value_glu_rank: int = 128
    hippocampal_pool_value_residual_anchor: Tuple[Tuple[float, float], ...] = (
        (0.5, 0.5), (0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (0.5, 2.0))
    hippocampal_read_copy_gain: float = 4.0
    hippocampal_copy_novelty_gate_init: float = 4.0
    hippocampal_copy_novelty_gate_aux: float = 0.5
    hippocampal_induction_bias_orders: Tuple[int, ...] = (2, 3, 4)
    # ---- hippocampal layer: product-key memory
    hippocampal_pkm_heads: int = 4
    hippocampal_pkm_subkeys: int = 128
    hippocampal_pkm_topk: int = 16
    hippocampal_pkm_query_dim: int = 128
    hippocampal_pkm_value_dim: int = 768
    hippocampal_pkm_launch_ratio: float = 0.25
    hippocampal_pkm_kmeans_iters: int = 5
    hippocampal_pkm_copy_gate_strength: float = 4.0
    # ---- RoPE
    rope_base: float = 10000.0
    rope_max_seq_len: int = 8192
    # ---- PLIF neurons
    init_tau: float = 2.0
    plif_input2_init_tau: Tuple[float, float] = (2.0, 100.0)
    plif_eta_proj_rank: int = 16
    plif_surrogate_alpha: float = 8.0
    plif_threshold_match_p: float = 0.0
    # ---- lexical memory (model level)
    lexical_memory_rows: Tuple[int, ...] = (151669, 262139, 344951)
    lexical_memory_heads: Tuple[int, ...] = (1, 2, 2)
    lexical_memory_head_dims: Tuple[int, ...] = (96, 16, 16)
    lexical_memory_key_dim: int = 64
    lexical_memory_init_std: float = 0.02
    lexical_memory_gate_bias: float = -2.0
    lexical_memory_hash_seed: int = 12345
    # ---- dendritic neuron layer
    dendritic_expand: int = 6
    dendritic_head_dim: int = 64
    dendritic_state_size: int = 128
    dendritic_chunk_size: int = 256
    dendritic_theta0: float = 0.05
    dendritic_theta_floor: float = 0.02
    dendritic_surrogate_alpha: float = 8.0
    # ---- readout
    readout_balance_widths: Tuple[int, ...] = (16, 32)
    # ---- norms
    dendritic_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    # ---- inference (the one numerics option)
    inference_dendritic_state_dtype: str = "bfloat16"

    # ------------------------------------------------------------------------------------------------ validation
    def __post_init__(self) -> None:
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731  (frozen dataclass)
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name in _INT_FIELDS:
                set_(f.name, _as_int(f.name, v))
            elif f.name in _FLOAT_FIELDS:
                set_(f.name, _as_float(f.name, v))
            elif f.name in _INT_TUPLE_FIELDS:
                set_(f.name, tuple(_as_int(f.name, x) for x in _as_seq(f.name, v)))
            elif f.name in _FLOAT_TUPLE_FIELDS:
                set_(f.name, tuple(_as_float(f.name, x) for x in _as_seq(f.name, v)))
        set_("hippocampal_pool_value_residual_anchor",
             tuple(tuple(_as_float("hippocampal_pool_value_residual_anchor", x)
                         for x in _as_seq("hippocampal_pool_value_residual_anchor", pair))
                   for pair in _as_seq("hippocampal_pool_value_residual_anchor",
                                       self.hippocampal_pool_value_residual_anchor)))
        if not isinstance(self.inference_dendritic_state_dtype, str):
            raise TypeError("inference_dendritic_state_dtype must be a string, got "
                            f"{type(self.inference_dendritic_state_dtype).__name__}")
        self._validate()

    def _validate(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"NeuronSparkConfig: {msg}")

        for name in _INT_FIELDS:
            if name == "lexical_memory_hash_seed":
                need(getattr(self, name) >= 0, f"{name} must be non-negative, got {getattr(self, name)}")
                continue
            need(getattr(self, name) > 0, f"{name} must be positive, got {getattr(self, name)}")
        need(0.0 <= self.plif_threshold_match_p < 1.0,
             f"plif_threshold_match_p must lie in [0, 1), got {self.plif_threshold_match_p}")
        need(self.hippocampal_layer_interval <= self.num_layers,
             f"hippocampal_layer_interval {self.hippocampal_layer_interval} exceeds num_layers {self.num_layers}")
        need(self.D_key % self.hippocampal_heads == 0 and self.D_value % self.hippocampal_heads == 0,
             f"D_key {self.D_key} and D_value {self.D_value} must be multiples of hippocampal_heads "
             f"{self.hippocampal_heads}")
        need((self.D_key // self.hippocampal_heads) % 2 == 0,
             "the memory head width D_key / hippocampal_heads must be even (RoPE rotates pairs inside a head)")
        need((self.dendritic_expand * self.D) % self.dendritic_head_dim == 0,
             f"dendritic_expand * D = {self.dendritic_expand * self.D} must be a multiple of dendritic_head_dim "
             f"{self.dendritic_head_dim}")
        need(self.dendritic_chunk_size & (self.dendritic_chunk_size - 1) == 0,
             f"dendritic_chunk_size {self.dendritic_chunk_size} must be a power of 2")
        need(self.dendritic_theta0 > self.dendritic_theta_floor >= 0.0,
             f"dendritic_theta0 {self.dendritic_theta0} must exceed dendritic_theta_floor {self.dendritic_theta_floor} "
             ">= 0")
        need(self.dendritic_surrogate_alpha > 0.0 and self.plif_surrogate_alpha > 0.0,
             "surrogate slopes must be positive")
        need(self.dendritic_norm_eps > 0.0 and self.rms_norm_eps > 0.0, "norm eps must be positive")
        need(self.lexical_memory_init_std > 0.0, "lexical_memory_init_std must be positive")
        need(self.hippocampal_pkm_query_dim % 2 == 0, "hippocampal_pkm_query_dim must be even (two half-queries)")
        need(self.hippocampal_pkm_topk <= self.hippocampal_pkm_subkeys,
             "hippocampal_pkm_topk must not exceed hippocampal_pkm_subkeys")
        need(len(self.plif_input2_init_tau) == 2 and 1.0 < self.plif_input2_init_tau[0] <= self.plif_input2_init_tau[1],
             f"plif_input2_init_tau must be (tau_min, tau_max) with 1 < tau_min <= tau_max, got {self.plif_input2_init_tau}")
        need(self.init_tau > 1.0, f"init_tau must exceed 1, got {self.init_tau}")
        n_hippocampal = self.n_hippocampal_layers
        need(n_hippocampal >= 1, "the layout has no hippocampal layer")
        need(len(self.hippocampal_pool_value_residual_anchor) == n_hippocampal - 1,
             f"hippocampal_pool_value_residual_anchor needs {n_hippocampal - 1} pairs (hippocampal layers "
             f"1..{n_hippocampal - 1}), got {len(self.hippocampal_pool_value_residual_anchor)}")
        need(all(len(p) == 2 for p in self.hippocampal_pool_value_residual_anchor),
             "every hippocampal_pool_value_residual_anchor entry must be a pair")
        orders = self.hippocampal_induction_bias_orders
        need(len(orders) >= 1 and all(o >= 2 for o in orders) and list(orders) == sorted(set(orders)),
             f"hippocampal_induction_bias_orders must be strictly increasing orders >= 2, got {orders}")
        rows, heads, dims = self.lexical_memory_rows, self.lexical_memory_heads, self.lexical_memory_head_dims
        need(len(rows) == len(heads) == len(dims) >= 1,
             f"lexical_memory_rows / _heads / _head_dims must have one entry per order, got {rows} {heads} {dims}")
        need(rows[0] == self.vocab_size and heads[0] == 1,
             "lexical order 1 must be the exact-id table: rows[0] == vocab_size and heads[0] == 1")
        need((self.vocab_size + 1) ** len(rows) < (1 << 63),
             "the base-(V+1) key of the highest lexical order does not fit in int64")
        need(len(self.readout_balance_widths) >= 1, "readout_balance_widths must be non-empty")
        need(self.inference_dendritic_state_dtype in DENDRITIC_STATE_DTYPES,
             f"inference_dendritic_state_dtype must be one of {DENDRITIC_STATE_DTYPES}, got "
             f"{self.inference_dendritic_state_dtype!r}")

    # ------------------------------------------------------------------------------------------------ layout
    @property
    def layer_kinds(self) -> Tuple[str, ...]:
        """``"hippocampal"`` or ``"dendritic"`` per layer index."""
        return tuple("hippocampal" if (i + 1) % self.hippocampal_layer_interval == 0
                     else "dendritic" for i in range(self.num_layers))

    @property
    def hippocampal_layer_ids(self) -> Tuple[int, ...]:
        """Layer indices of the hippocampal layers: (3, 7, 11, 15, 19, 23)."""
        return tuple(i for i, k in enumerate(self.layer_kinds) if k == "hippocampal")

    @property
    def dendritic_layer_ids(self) -> Tuple[int, ...]:
        """Layer indices of the 18 dendritic neuron layers."""
        return tuple(i for i, k in enumerate(self.layer_kinds) if k == "dendritic")

    @property
    def n_hippocampal_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if (i + 1) % self.hippocampal_layer_interval == 0)

    @property
    def n_dendritic_layers(self) -> int:
        return self.num_layers - self.n_hippocampal_layers

    def is_hippocampal_layer(self, layer_idx: int) -> bool:
        return (int(layer_idx) + 1) % self.hippocampal_layer_interval == 0

    def hippocampal_index(self, layer_idx: int) -> int:
        """Rank 0..5 of a hippocampal layer among the hippocampal layers (0 = the value-residual source)."""
        ids = self.hippocampal_layer_ids
        if int(layer_idx) not in ids:
            raise ValueError(f"layer {layer_idx} is not a hippocampal layer (hippocampal layers: {ids})")
        return ids.index(int(layer_idx))

    # ------------------------------------------------------------------------------------------------ dendritic neuron
    # layer
    @property
    def dendritic_d_inner(self) -> int:
        """9216 = dendritic_expand * D."""
        return self.dendritic_expand * self.D

    @property
    def dendritic_nheads(self) -> int:
        """144 = dendritic_d_inner / dendritic_head_dim."""
        return self.dendritic_d_inner // self.dendritic_head_dim

    @property
    def dendritic_conv_width(self) -> int:
        """Causal depthwise conv width of the dendritic neuron population (fixed 4)."""
        return 4

    @property
    def dendritic_conv_dim(self) -> int:
        """9472 = dendritic_d_inner + 2 * dendritic_state_size (x, B, C channels; one state group)."""
        return self.dendritic_d_inner + 2 * self.dendritic_state_size

    @property
    def dendritic_in_proj_dim(self) -> int:
        """18832 = 2 * dendritic_d_inner + 2 * dendritic_state_size + dendritic_nheads (rows z | x | B | C | dt)."""
        return 2 * self.dendritic_d_inner + 2 * self.dendritic_state_size + self.dendritic_nheads

    @property
    def dendritic_residual_branches(self) -> int:
        """30 = 1 per dendritic neuron layer + 2 per hippocampal layer; sets the W_out init scale 1 / sqrt(30)."""
        return self.n_dendritic_layers + 2 * self.n_hippocampal_layers

    # ------------------------------------------------------------------------------------------------ hippocampal layer
    @property
    def hippocampal_head_dim(self) -> int:
        """256 = D_key / hippocampal_heads (also the vector-decay width)."""
        return self.D_key // self.hippocampal_heads

    @property
    def hippocampal_value_head_dim(self) -> int:
        """256 = D_value / hippocampal_heads."""
        return self.D_value // self.hippocampal_heads

    @property
    def hippocampal_qkv_dim(self) -> int:
        """4608 = 2 * D_key + D_value (qkv projection rows and short-conv channels)."""
        return 2 * self.D_key + self.D_value

    @property
    def hippocampal_dendritic_scale(self) -> float:
        """16.0 = sqrt(hippocampal_head_dim): the score scale of the unit-norm pool queries and keys."""
        return float(self.hippocampal_head_dim) ** 0.5

    @property
    def lexical_dim(self) -> int:
        """160 = sum over orders of heads * head_dim: the width of the model-level lexical vector."""
        return sum(h * d for h, d in zip(self.lexical_memory_heads, self.lexical_memory_head_dims))

    @property
    def lexical_order1_slice(self) -> Tuple[int, int]:
        """(0, 96): the order-1 channels of the lexical vector, the copy detector's operand."""
        return (0, int(self.lexical_memory_head_dims[0]))

    @property
    def pkm_slots(self) -> int:
        """16384 = hippocampal_pkm_subkeys ** 2."""
        return self.hippocampal_pkm_subkeys * self.hippocampal_pkm_subkeys

    @property
    def pkm_half_dim(self) -> int:
        """64 = hippocampal_pkm_query_dim / 2."""
        return self.hippocampal_pkm_query_dim // 2

    @property
    def out_std(self) -> float:
        """0.02 / sqrt(2 * num_layers) = 0.0028867513: init std of the output projections."""
        return 0.02 / math.sqrt(2 * max(self.num_layers, 1))

    # ------------------------------------------------------------------------------------------------ accounting
    def _plif_params(self, channels: int) -> int:
        # w, v_th, w_eta, b_eta (C each) + eta_V (rank, C) + eta_U (C, rank)
        return 4 * channels + 2 * self.plif_eta_proj_rank * channels

    @property
    def dendritic_layer_params(self) -> int:
        """43,149,488: pre_norm_gain, in_proj, conv_weight, conv_bias, dt_bias, A_log, D, gamma, W_out, rho."""
        D, di = self.D, self.dendritic_d_inner
        return (D + self.dendritic_in_proj_dim * D + self.dendritic_conv_dim * self.dendritic_conv_width
                + self.dendritic_conv_dim + 3 * self.dendritic_nheads + 2 * di + D * di)

    @property
    def hippocampal_layer_params(self) -> int:
        """37,770,926: the 70 parameter tensors of one hippocampal layer."""
        D, Dv, H = self.D, self.D_value, self.hippocampal_heads
        dk, dv, qkv = self.hippocampal_head_dim, self.hippocampal_value_head_dim, self.hippocampal_qkv_dim
        M = self.hippocampal_pool_extra_queries
        r, ro = self.hippocampal_pool_extra_query_rank, self.hippocampal_pool_extra_query_out_rank
        g, L, kd = self.hippocampal_pool_value_glu_rank, self.lexical_dim, self.lexical_memory_key_dim
        Hp, n = self.hippocampal_pkm_heads, self.hippocampal_pkm_subkeys
        dq, dvp = self.hippocampal_pkm_query_dim, self.hippocampal_pkm_value_dim
        ff = self.D_ff
        attention = (dk * D + dk                              # W_decay.weight, decay_A_log_delta
                     + qkv * D + qkv * self.hippocampal_short_conv  # qkv_proj, qkv_conv
                     + Dv + Dv * D + D * Dv                    # attn_out_norm, attn_out_gate, attn_out_proj
                     + self._plif_params(D)                    # gate_neuron
                     + 3 * H                                   # pool bias-gain / dendritic-gain deltas, write bias
                     + 2 * g * D + dv * g                      # pool_glu_a, pool_glu_b, pool_glu_u
                     + Dv + H * D                              # pool_value_residual, pool_vr_gate
                     + M * r * D + M * dk * r + M * D + ro * M * dv + dv * ro   # pool_xq_*
                     + len(self.hippocampal_induction_bias_orders) * H              # pool_ind_beta
                     + 2 * D + 1)                              # copy_novelty_w, copy_novelty_v, copy_novelty_b
        lexical = kd * D + kd * L + D * L + 2 * kd + 1        # lex_q, lex_k, lex_o, lex_q_gain, lex_k_gain, lex_beta
        ffn = (D + self._plif_params(D)                       # ffn_norm, input_neuron2
               + 2 * ff * D + D * ff + D * D                  # gate/up/down/skip projections
               + 2 * self._plif_params(ff)                    # gate/up neurons
               + D * D)                                       # ffn_out_proj
        pkm = (Hp * dq * D + dvp * D + D * dvp                # pkm_q, pkm_g, pkm_o
               + self.pkm_slots * dvp + Hp * 2 * n * (dq // 2)  # pkm_values, pkm_subkeys
               + dvp + 2 * Hp)                                # pkm_norm_gain, pkm_log_gain
        return attention + lexical + ffn + pkm

    @property
    def n_params_expected(self) -> int:
        """1,270,342,292 parameter elements (609 tensors)."""
        lexical_tables = sum(r * h * d for r, h, d in zip(self.lexical_memory_rows, self.lexical_memory_heads,
                                                          self.lexical_memory_head_dims))
        return (self.vocab_size * self.D + self.n_dendritic_layers * self.dendritic_layer_params
                + self.n_hippocampal_layers * self.hippocampal_layer_params + self.D + lexical_tables
                + sum(self.readout_balance_widths) * self.D)

    @property
    def n_hashed_lexical_tables(self) -> int:
        """4: the tables of the hashed orders (2 and 3), one per head."""
        return sum(self.lexical_memory_heads[1:])

    @property
    def n_buffers_expected(self) -> int:
        """10,908 persistent buffer elements (69 tensors): the lexical hash constants and, per hippocampal layer, the
        decay
        launch, the three pool spectra, the value-residual anchor and six PKM scalars."""
        per_hippocampal_layer = self.hippocampal_head_dim + 3 * self.hippocampal_heads + self.D_value + 6
        return 3 * self.n_hashed_lexical_tables + self.n_hippocampal_layers * per_hippocampal_layer

    @property
    def n_param_tensors_expected(self) -> int:
        """609 = 4 top level + lexical tables + 10 per dendritic neuron layer + 70 per hippocampal layer."""
        n_tables = sum(self.lexical_memory_heads)
        return (2 + len(self.readout_balance_widths) + n_tables + 10 * self.n_dendritic_layers
                + (66 + self.hippocampal_pool_extra_queries) * self.n_hippocampal_layers)

    @property
    def n_buffer_tensors_expected(self) -> int:
        """69 = 3 lexical hash buffers + 11 per hippocampal layer."""
        return 3 + 11 * self.n_hippocampal_layers

    @property
    def n_state_tensors(self) -> int:
        """678 = parameters + persistent buffers in the state dict."""
        return self.n_param_tensors_expected + self.n_buffer_tensors_expected

    # ------------------------------------------------------------------------------------------------ io
    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dictionary of the configuration file format (``model_type``, ``architectures``, every field)."""
        out: Dict[str, Any] = {"model_type": MODEL_TYPE, "architectures": list(ARCHITECTURES)}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == "hippocampal_pool_value_residual_anchor":
                v = [list(p) for p in v]
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out

    def to_json(self, path: _PathLike) -> None:
        """Write :meth:`to_dict` to ``path`` (UTF-8 JSON, indent 2, trailing newline)."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NeuronSparkConfig":
        """Read the configuration file format.  ``model_type`` / ``architectures`` (optional here) must hold their
        values when present; an absent field takes its default; any other key raises."""
        d = dict(d)
        mt = d.pop("model_type", MODEL_TYPE)
        if mt != MODEL_TYPE:
            raise ValueError(f"NeuronSparkConfig.from_dict: model_type {mt!r} is not {MODEL_TYPE!r}")
        arch = d.pop("architectures", list(ARCHITECTURES))
        if list(arch) != list(ARCHITECTURES):
            raise ValueError(f"NeuronSparkConfig.from_dict: architectures {arch!r} is not {list(ARCHITECTURES)!r}")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"NeuronSparkConfig.from_dict: unknown keys {unknown}")
        return cls(**d)

    @classmethod
    def from_json(cls, path: _PathLike) -> "NeuronSparkConfig":
        """Read a configuration file (:meth:`from_dict` of its JSON object)."""
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if not isinstance(d, dict):
            raise ValueError(f"{path}: the configuration is not a JSON object")
        return cls.from_dict(d)


# ---------------------------------------------------------------------------------------------------- field typing
_INT_FIELDS = frozenset((
    "vocab_size", "D", "num_layers", "hippocampal_layer_interval", "D_ff", "D_key", "D_value", "hippocampal_heads",
    "hippocampal_short_conv", "hippocampal_pool_extra_queries", "hippocampal_pool_extra_query_rank",
    "hippocampal_pool_extra_query_out_rank",
    "hippocampal_pool_value_glu_rank", "hippocampal_pkm_heads", "hippocampal_pkm_subkeys", "hippocampal_pkm_topk",
    "hippocampal_pkm_query_dim",
    "hippocampal_pkm_value_dim", "hippocampal_pkm_kmeans_iters", "rope_max_seq_len", "plif_eta_proj_rank",
    "lexical_memory_key_dim", "lexical_memory_hash_seed", "dendritic_expand", "dendritic_head_dim",
    "dendritic_state_size",
    "dendritic_chunk_size",
))
_FLOAT_FIELDS = frozenset((
    "hippocampal_base_decay", "hippocampal_write_sparsity", "hippocampal_pool_write_bias_spectrum_span",
    "hippocampal_read_copy_gain",
    "hippocampal_copy_novelty_gate_init", "hippocampal_copy_novelty_gate_aux", "hippocampal_pkm_launch_ratio",
    "hippocampal_pkm_copy_gate_strength", "rope_base", "init_tau", "plif_surrogate_alpha", "plif_threshold_match_p",
    "lexical_memory_init_std", "lexical_memory_gate_bias", "dendritic_theta0", "dendritic_theta_floor",
    "dendritic_surrogate_alpha",
    "dendritic_norm_eps", "rms_norm_eps",
))
_INT_TUPLE_FIELDS = frozenset((
    "hippocampal_induction_bias_orders", "lexical_memory_rows", "lexical_memory_heads", "lexical_memory_head_dims",
    "readout_balance_widths",
))
_FLOAT_TUPLE_FIELDS = frozenset(("plif_input2_init_tau",))


def _as_int(name: str, v: Any) -> int:
    if isinstance(v, bool):
        raise TypeError(f"NeuronSparkConfig.{name}: expected an integer, got a bool")
    if isinstance(v, float):
        if not v.is_integer():
            raise TypeError(f"NeuronSparkConfig.{name}: expected an integer, got {v!r}")
        return int(v)
    if not isinstance(v, int):
        raise TypeError(f"NeuronSparkConfig.{name}: expected an integer, got {type(v).__name__}")
    return int(v)


def _as_float(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"NeuronSparkConfig.{name}: expected a number, got {type(v).__name__}")
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"NeuronSparkConfig.{name}: must be finite, got {v!r}")
    return v


def _as_seq(name: str, v: Any):
    if isinstance(v, (str, bytes)) or not hasattr(v, "__iter__"):
        raise TypeError(f"NeuronSparkConfig.{name}: expected a sequence, got {type(v).__name__}")
    return list(v)
