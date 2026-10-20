"""Configuration of the NeuronSpark 1MIX language model and its on-disk form (``config.json``).

``NeuronSparkConfig`` holds only the fields that change the computed function: a parameter shape, a numeric constant
of the forward math, of the initialisation or of the launch-time calibration.  The architecture itself (layer forms,
neuron model, mechanisms) is fixed in the code and is therefore not configurable.  The defaults are the 1MIX 1.27 B
model; ``configs/onemix_1p3b.json`` is the same configuration on disk.

Layout (24 layers, ratio 3:1): layer ``i`` is a memory layer iff ``(i + 1) % memory_layer_interval == 0`` (layers 3, 7,
11, 15, 19, 23), every other layer is a selective-dendrite (SD) layer.

The single numerics option is ``inference_sd_state_dtype``: the storage dtype of the SD state in the inference cache,
``"bfloat16"`` (default: the state is rounded to bf16 once per decode step and once per prefill call) or ``"float32"``
(the state is carried in fp32).  It changes no parameter and no training computation.

Configuration file format (``config.json`` of a checkpoint directory, see :mod:`neuronspark.utils.hf`): one JSON
object holding

* ``"model_type": "neuronspark"`` (:data:`MODEL_TYPE`, required in a file);
* ``"architectures": ["NeuronSparkLMHeadModel"]`` (:data:`ARCHITECTURES`; checked when present);
* one key per :class:`NeuronSparkConfig` field, named exactly as the field: integers and floats as JSON numbers,
  tuples as JSON arrays (``memory_pool_value_residual_anchor`` as an array of ``[a, b]`` pairs), the dtype option as a
  string.

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
    "SD_STATE_DTYPES",
]

# "model_type" of the configuration file
MODEL_TYPE = "neuronspark"
# "architectures" of the configuration file
ARCHITECTURES = ("NeuronSparkLMHeadModel",)
# values of the inference_sd_state_dtype option
SD_STATE_DTYPES = ("bfloat16", "float32")

_PathLike = Union[str, "os.PathLike[str]"]


@dataclass(frozen=True)
class NeuronSparkConfig:
    """Hyper-parameters of the 1MIX architecture (defaults = the 1.27 B model).

    Groups (consumer in parentheses):

    * vocabulary / trunk: ``vocab_size`` (embedding, tied readout, lexical order-1 rows, pad id ``V`` of the lexical
      and induction windows), ``D`` (model width), ``num_layers``, ``memory_layer_interval``.
    * memory layer, spiking FFN: ``D_ff``.
    * memory layer, pool attention: ``D_key``, ``D_value``, ``memory_heads`` (head width ``D_key / memory_heads``),
      ``memory_short_conv`` (qkv causal depthwise conv width), ``memory_base_decay`` (decay
      ``a = exp(-base * rate * softplus(logit))``), ``memory_write_sparsity`` (write tag ``sigmoid(g + wb - 2.0)``),
      ``memory_pool_write_bias_spectrum_span`` (write-bias spectrum ``linspace(0, span, heads)``),
      ``memory_pool_extra_queries`` / ``_extra_query_rank`` / ``_extra_query_out_rank`` (extra queries on the last
      head), ``memory_pool_value_glu_rank`` (GLU tail of the last head), ``memory_pool_value_residual_anchor`` (anchor
      pairs of memory layers 1..5; memory layer 0 is the value source), ``memory_read_copy_gain`` (read gain
      ``1 + 4 pi c``), ``memory_copy_novelty_gate_init`` (init of ``copy_novelty_b``), ``memory_copy_novelty_gate_aux``
      (weight of the copy-switch auxiliary loss), ``memory_induction_bias_orders`` (n-gram orders of the induction
      match matrices).
    * product-key memory: ``memory_pkm_heads``, ``_subkeys`` (slots = subkeys ** 2), ``_topk``, ``_query_dim``,
      ``_value_dim``, launch calibration ``_launch_ratio`` / ``_kmeans_iters``, ``memory_pkm_copy_gate_strength``
      (``rho = 2 sigmoid(-strength * pi)``).
    * RoPE: ``rope_base``, ``rope_max_seq_len`` (table length; also the largest inference ring capacity).
    * PLIF neurons: ``init_tau``, ``plif_input2_init_tau`` (tau spectrum of ``input_neuron2``),
      ``plif_eta_proj_rank``, ``plif_surrogate_alpha`` (backward slope), ``plif_threshold_match_p`` (launch threshold
      match quantile).
    * lexical memory: ``lexical_memory_rows`` / ``_heads`` / ``_head_dims`` per n-gram order (order 1 first),
      ``lexical_memory_key_dim``, ``lexical_memory_init_std`` (table init std and fixed output scale ``1 / std``),
      ``lexical_memory_gate_bias`` (init of ``lex_beta``), ``lexical_memory_hash_seed``.
    * SD layer: ``sd_expand``, ``sd_head_dim``, ``sd_state_size``, ``sd_chunk_size``, ``sd_theta0`` /
      ``sd_theta_floor`` (soma threshold ``theta = floor + softplus(rho)``, launched at ``theta0``),
      ``sd_surrogate_alpha``.
    * readout: ``readout_balance_widths`` (tiled balance projections, added width 16 first, then 32).
    * norms: ``sd_norm_eps`` (SD pre-norm and gated norm), ``rms_norm_eps`` (every other RMSNorm).
    * inference: ``inference_sd_state_dtype`` (see the module docstring).
    """

    # ---- vocabulary / trunk
    vocab_size: int = 151669
    D: int = 1536
    num_layers: int = 24
    memory_layer_interval: int = 4
    # ---- memory layer: spiking FFN
    D_ff: int = 768
    # ---- memory layer: pool attention
    D_key: int = 1536
    D_value: int = 1536
    memory_heads: int = 6
    memory_short_conv: int = 4
    memory_base_decay: float = 1.0
    memory_write_sparsity: float = 2.0
    memory_pool_write_bias_spectrum_span: float = 6.0
    memory_pool_extra_queries: int = 4
    memory_pool_extra_query_rank: int = 64
    memory_pool_extra_query_out_rank: int = 64
    memory_pool_value_glu_rank: int = 128
    memory_pool_value_residual_anchor: Tuple[Tuple[float, float], ...] = (
        (0.5, 0.5), (0.5, 0.5), (0.5, 1.0), (0.5, 2.0), (0.5, 2.0))
    memory_read_copy_gain: float = 4.0
    memory_copy_novelty_gate_init: float = 4.0
    memory_copy_novelty_gate_aux: float = 0.5
    memory_induction_bias_orders: Tuple[int, ...] = (2, 3, 4)
    # ---- memory layer: product-key memory
    memory_pkm_heads: int = 4
    memory_pkm_subkeys: int = 128
    memory_pkm_topk: int = 16
    memory_pkm_query_dim: int = 128
    memory_pkm_value_dim: int = 768
    memory_pkm_launch_ratio: float = 0.25
    memory_pkm_kmeans_iters: int = 5
    memory_pkm_copy_gate_strength: float = 4.0
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
    # ---- SD layer
    sd_expand: int = 6
    sd_head_dim: int = 64
    sd_state_size: int = 128
    sd_chunk_size: int = 256
    sd_theta0: float = 0.05
    sd_theta_floor: float = 0.02
    sd_surrogate_alpha: float = 8.0
    # ---- readout
    readout_balance_widths: Tuple[int, ...] = (16, 32)
    # ---- norms
    sd_norm_eps: float = 1e-5
    rms_norm_eps: float = 1e-6
    # ---- inference (the one numerics option)
    inference_sd_state_dtype: str = "bfloat16"

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
        set_("memory_pool_value_residual_anchor",
             tuple(tuple(_as_float("memory_pool_value_residual_anchor", x)
                         for x in _as_seq("memory_pool_value_residual_anchor", pair))
                   for pair in _as_seq("memory_pool_value_residual_anchor", self.memory_pool_value_residual_anchor)))
        if not isinstance(self.inference_sd_state_dtype, str):
            raise TypeError("inference_sd_state_dtype must be a string, got "
                            f"{type(self.inference_sd_state_dtype).__name__}")
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
        need(self.memory_layer_interval <= self.num_layers,
             f"memory_layer_interval {self.memory_layer_interval} exceeds num_layers {self.num_layers}")
        need(self.D_key % self.memory_heads == 0 and self.D_value % self.memory_heads == 0,
             f"D_key {self.D_key} and D_value {self.D_value} must be multiples of memory_heads {self.memory_heads}")
        need((self.D_key // self.memory_heads) % 2 == 0,
             "the memory head width D_key / memory_heads must be even (RoPE rotates pairs inside a head)")
        need((self.sd_expand * self.D) % self.sd_head_dim == 0,
             f"sd_expand * D = {self.sd_expand * self.D} must be a multiple of sd_head_dim {self.sd_head_dim}")
        need(self.sd_chunk_size & (self.sd_chunk_size - 1) == 0, f"sd_chunk_size {self.sd_chunk_size} must be a power of 2")
        need(self.sd_theta0 > self.sd_theta_floor >= 0.0,
             f"sd_theta0 {self.sd_theta0} must exceed sd_theta_floor {self.sd_theta_floor} >= 0")
        need(self.sd_surrogate_alpha > 0.0 and self.plif_surrogate_alpha > 0.0, "surrogate slopes must be positive")
        need(self.sd_norm_eps > 0.0 and self.rms_norm_eps > 0.0, "norm eps must be positive")
        need(self.lexical_memory_init_std > 0.0, "lexical_memory_init_std must be positive")
        need(self.memory_pkm_query_dim % 2 == 0, "memory_pkm_query_dim must be even (two half-queries)")
        need(self.memory_pkm_topk <= self.memory_pkm_subkeys, "memory_pkm_topk must not exceed memory_pkm_subkeys")
        need(len(self.plif_input2_init_tau) == 2 and 1.0 < self.plif_input2_init_tau[0] <= self.plif_input2_init_tau[1],
             f"plif_input2_init_tau must be (tau_min, tau_max) with 1 < tau_min <= tau_max, got {self.plif_input2_init_tau}")
        need(self.init_tau > 1.0, f"init_tau must exceed 1, got {self.init_tau}")
        n_mem = self.n_memory_layers
        need(n_mem >= 1, "the layout has no memory layer")
        need(len(self.memory_pool_value_residual_anchor) == n_mem - 1,
             f"memory_pool_value_residual_anchor needs {n_mem - 1} pairs (memory layers 1..{n_mem - 1}), got "
             f"{len(self.memory_pool_value_residual_anchor)}")
        need(all(len(p) == 2 for p in self.memory_pool_value_residual_anchor),
             "every memory_pool_value_residual_anchor entry must be a pair")
        orders = self.memory_induction_bias_orders
        need(len(orders) >= 1 and all(o >= 2 for o in orders) and list(orders) == sorted(set(orders)),
             f"memory_induction_bias_orders must be strictly increasing orders >= 2, got {orders}")
        rows, heads, dims = self.lexical_memory_rows, self.lexical_memory_heads, self.lexical_memory_head_dims
        need(len(rows) == len(heads) == len(dims) >= 1,
             f"lexical_memory_rows / _heads / _head_dims must have one entry per order, got {rows} {heads} {dims}")
        need(rows[0] == self.vocab_size and heads[0] == 1,
             "lexical order 1 must be the exact-id table: rows[0] == vocab_size and heads[0] == 1")
        need((self.vocab_size + 1) ** len(rows) < (1 << 63),
             "the base-(V+1) key of the highest lexical order does not fit in int64")
        need(len(self.readout_balance_widths) >= 1, "readout_balance_widths must be non-empty")
        need(self.inference_sd_state_dtype in SD_STATE_DTYPES,
             f"inference_sd_state_dtype must be one of {SD_STATE_DTYPES}, got {self.inference_sd_state_dtype!r}")

    # ------------------------------------------------------------------------------------------------ layout
    @property
    def layer_kinds(self) -> Tuple[str, ...]:
        """``"memory"`` or ``"sd"`` per layer index."""
        return tuple("memory" if (i + 1) % self.memory_layer_interval == 0 else "sd" for i in range(self.num_layers))

    @property
    def memory_layer_ids(self) -> Tuple[int, ...]:
        """Layer indices of the memory layers: (3, 7, 11, 15, 19, 23)."""
        return tuple(i for i, k in enumerate(self.layer_kinds) if k == "memory")

    @property
    def sd_layer_ids(self) -> Tuple[int, ...]:
        """Layer indices of the 18 SD layers."""
        return tuple(i for i, k in enumerate(self.layer_kinds) if k == "sd")

    @property
    def n_memory_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if (i + 1) % self.memory_layer_interval == 0)

    @property
    def n_sd_layers(self) -> int:
        return self.num_layers - self.n_memory_layers

    def is_memory_layer(self, layer_idx: int) -> bool:
        return (int(layer_idx) + 1) % self.memory_layer_interval == 0

    def memory_index(self, layer_idx: int) -> int:
        """Rank 0..5 of a memory layer among the memory layers (0 = the value-residual source)."""
        ids = self.memory_layer_ids
        if int(layer_idx) not in ids:
            raise ValueError(f"layer {layer_idx} is not a memory layer (memory layers: {ids})")
        return ids.index(int(layer_idx))

    # ------------------------------------------------------------------------------------------------ SD layer
    @property
    def sd_d_inner(self) -> int:
        """9216 = sd_expand * D."""
        return self.sd_expand * self.D

    @property
    def sd_nheads(self) -> int:
        """144 = sd_d_inner / sd_head_dim."""
        return self.sd_d_inner // self.sd_head_dim

    @property
    def sd_conv_width(self) -> int:
        """Causal depthwise conv width of the SD mixer (fixed 4)."""
        return 4

    @property
    def sd_conv_dim(self) -> int:
        """9472 = sd_d_inner + 2 * sd_state_size (x, B, C channels; one state group)."""
        return self.sd_d_inner + 2 * self.sd_state_size

    @property
    def sd_in_proj_dim(self) -> int:
        """18832 = 2 * sd_d_inner + 2 * sd_state_size + sd_nheads (rows z | x | B | C | dt)."""
        return 2 * self.sd_d_inner + 2 * self.sd_state_size + self.sd_nheads

    @property
    def sd_residual_branches(self) -> int:
        """30 = 1 per SD layer + 2 per memory layer; sets the W_out init scale 1 / sqrt(30)."""
        return self.n_sd_layers + 2 * self.n_memory_layers

    # ------------------------------------------------------------------------------------------------ memory layer
    @property
    def memory_head_dim(self) -> int:
        """256 = D_key / memory_heads (also the vector-decay width)."""
        return self.D_key // self.memory_heads

    @property
    def memory_value_head_dim(self) -> int:
        """256 = D_value / memory_heads."""
        return self.D_value // self.memory_heads

    @property
    def memory_qkv_dim(self) -> int:
        """4608 = 2 * D_key + D_value (qkv projection rows and short-conv channels)."""
        return 2 * self.D_key + self.D_value

    @property
    def memory_dendritic_scale(self) -> float:
        """16.0 = sqrt(memory_head_dim): the score scale of the unit-norm pool queries and keys."""
        return float(self.memory_head_dim) ** 0.5

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
        """16384 = memory_pkm_subkeys ** 2."""
        return self.memory_pkm_subkeys * self.memory_pkm_subkeys

    @property
    def pkm_half_dim(self) -> int:
        """64 = memory_pkm_query_dim / 2."""
        return self.memory_pkm_query_dim // 2

    @property
    def out_std(self) -> float:
        """0.02 / sqrt(2 * num_layers) = 0.0028867513: init std of the output projections."""
        return 0.02 / math.sqrt(2 * max(self.num_layers, 1))

    # ------------------------------------------------------------------------------------------------ accounting
    def _plif_params(self, channels: int) -> int:
        # w, v_th, w_eta, b_eta (C each) + eta_V (rank, C) + eta_U (C, rank)
        return 4 * channels + 2 * self.plif_eta_proj_rank * channels

    @property
    def sd_layer_params(self) -> int:
        """43,149,488: pre_norm_gain, in_proj, conv_weight, conv_bias, dt_bias, A_log, D, gamma, W_out, rho."""
        D, di = self.D, self.sd_d_inner
        return (D + self.sd_in_proj_dim * D + self.sd_conv_dim * self.sd_conv_width + self.sd_conv_dim
                + 3 * self.sd_nheads + 2 * di + D * di)

    @property
    def memory_layer_params(self) -> int:
        """37,770,926: the 70 parameter tensors of one memory layer."""
        D, Dv, H = self.D, self.D_value, self.memory_heads
        dk, dv, qkv = self.memory_head_dim, self.memory_value_head_dim, self.memory_qkv_dim
        M, r, ro = self.memory_pool_extra_queries, self.memory_pool_extra_query_rank, self.memory_pool_extra_query_out_rank
        g, L, kd = self.memory_pool_value_glu_rank, self.lexical_dim, self.lexical_memory_key_dim
        Hp, n, dq, dvp = self.memory_pkm_heads, self.memory_pkm_subkeys, self.memory_pkm_query_dim, self.memory_pkm_value_dim
        ff = self.D_ff
        attention = (dk * D + dk                              # W_decay.weight, decay_A_log_delta
                     + qkv * D + qkv * self.memory_short_conv  # qkv_proj, qkv_conv
                     + Dv + Dv * D + D * Dv                    # attn_out_norm, attn_out_gate, attn_out_proj
                     + self._plif_params(D)                    # gate_neuron
                     + 3 * H                                   # pool bias-gain / dendritic-gain deltas, write bias
                     + 2 * g * D + dv * g                      # pool_glu_a, pool_glu_b, pool_glu_u
                     + Dv + H * D                              # pool_value_residual, pool_vr_gate
                     + M * r * D + M * dk * r + M * D + ro * M * dv + dv * ro   # pool_xq_*
                     + len(self.memory_induction_bias_orders) * H              # pool_ind_beta
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
        return (self.vocab_size * self.D + self.n_sd_layers * self.sd_layer_params
                + self.n_memory_layers * self.memory_layer_params + self.D + lexical_tables
                + sum(self.readout_balance_widths) * self.D)

    @property
    def n_hashed_lexical_tables(self) -> int:
        """4: the tables of the hashed orders (2 and 3), one per head."""
        return sum(self.lexical_memory_heads[1:])

    @property
    def n_buffers_expected(self) -> int:
        """10,908 persistent buffer elements (69 tensors): the lexical hash constants and, per memory layer, the decay
        launch, the three pool spectra, the value-residual anchor and six PKM scalars."""
        per_memory_layer = self.memory_head_dim + 3 * self.memory_heads + self.D_value + 6
        return 3 * self.n_hashed_lexical_tables + self.n_memory_layers * per_memory_layer

    @property
    def n_param_tensors_expected(self) -> int:
        """609 = 4 top level + lexical tables + 10 per SD layer + 70 per memory layer."""
        n_tables = sum(self.lexical_memory_heads)
        return (2 + len(self.readout_balance_widths) + n_tables + 10 * self.n_sd_layers
                + (66 + self.memory_pool_extra_queries) * self.n_memory_layers)

    @property
    def n_buffer_tensors_expected(self) -> int:
        """69 = 3 lexical hash buffers + 11 per memory layer."""
        return 3 + 11 * self.n_memory_layers

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
            if f.name == "memory_pool_value_residual_anchor":
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
    "vocab_size", "D", "num_layers", "memory_layer_interval", "D_ff", "D_key", "D_value", "memory_heads",
    "memory_short_conv", "memory_pool_extra_queries", "memory_pool_extra_query_rank", "memory_pool_extra_query_out_rank",
    "memory_pool_value_glu_rank", "memory_pkm_heads", "memory_pkm_subkeys", "memory_pkm_topk", "memory_pkm_query_dim",
    "memory_pkm_value_dim", "memory_pkm_kmeans_iters", "rope_max_seq_len", "plif_eta_proj_rank",
    "lexical_memory_key_dim", "lexical_memory_hash_seed", "sd_expand", "sd_head_dim", "sd_state_size", "sd_chunk_size",
))
_FLOAT_FIELDS = frozenset((
    "memory_base_decay", "memory_write_sparsity", "memory_pool_write_bias_spectrum_span", "memory_read_copy_gain",
    "memory_copy_novelty_gate_init", "memory_copy_novelty_gate_aux", "memory_pkm_launch_ratio",
    "memory_pkm_copy_gate_strength", "rope_base", "init_tau", "plif_surrogate_alpha", "plif_threshold_match_p",
    "lexical_memory_init_std", "lexical_memory_gate_bias", "sd_theta0", "sd_theta_floor", "sd_surrogate_alpha",
    "sd_norm_eps", "rms_norm_eps",
))
_INT_TUPLE_FIELDS = frozenset((
    "memory_induction_bias_orders", "lexical_memory_rows", "lexical_memory_heads", "lexical_memory_head_dims",
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
