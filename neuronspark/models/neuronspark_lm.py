"""The 1MIX language model: the backbone :class:`NeuronSparkModel` and the tied-readout LM head model
:class:`NeuronSparkLMHeadModel` (the counterparts of mamba_ssm's ``MixerModel`` / ``MambaLMHeadModel``).

Backbone (state-dict prefix ``backbone.``)::

    x    = embed_tokens(ids)                                  (T, B, D) fp32 residual stream
    lex  = lexical_memory(ids)                                (T, B, 160) fp32, model level, read by every memory layer
    ind  = induction match matrices (orders 2, 3, 4)          bool, model level
    copy = copy-detector statistics (maximum, lse)            (T, B) fp32 each, model level
    for i in 0 .. 23:   x = SDLayer_i(x)                      18 SD layers (i not in 3, 7, 11, 15, 19, 23)
                        x, carry = MemoryLayer_i(x, lex, ind, copy, carry)     6 memory layers (3:1 layout)
    h    = norm(x)                                            RMSNorm, eps 1e-6
    returned as (B, T, D) fp32

LM head: ``logits = bf16(h @ E^T)`` with the tied embedding ``E = backbone.embed_tokens.weight`` (151669 x 1536), then
the readout balance tiled across the vocabulary (width 16, then width 32, one bf16 rounding per width).  Training loss
(:meth:`NeuronSparkLMHeadModel.loss`): the mean token cross entropy of the balanced logits over the valid targets plus
the copy-switch auxiliary loss of the six memory layers' novelty predictors (weight ``memory_copy_novelty_gate_aux``).

Regimes (decided in :meth:`NeuronSparkModel.forward`, one code path each):

* ``inference_params`` None -> whole row (training, and teacher-forced scoring), under ``regime("train")`` (bf16
  autocast, TF32 on for fp32 matmuls).  T must be a multiple of ``sd_chunk_size`` (256) on the kernel path; the residual
  stream is the (T, B, D) view of the contiguous (B, T, D) embedding output; consecutive SD layers join one layer's
  residual addition with the next layer's pre-norm (``sd_resid_norm``) unless activation checkpointing is active.
  Works with or without autograd.
* ``inference_params`` given -> inference under ``regime("infer")`` (bf16 autocast, TF32 off, no reduced-precision bf16
  GEMM reductions) and ``torch.no_grad()``: T >= 2 is a prefill, T == 1 a decode step.  The stream is made contiguous
  T-major for T >= 2; the model-level session state (lexical window, induction keys ring, copy-operand ring) and every
  layer's carried state are read and updated in place; the final norm is ``rmsnorm_rows``; the device counters ``pos``
  / ``ring_ptr`` advance by T in place (CUDA-graph replay safe) and the host count ``seqlen_offset`` advances by T.

Activation checkpointing (``gradient_checkpointing = True``, whole row with autograd): every SD layer is recomputed in
backward (``use_reentrant=False, preserve_rng_state=False``), the memory layers stay resident, and the SD -> SD join is
not used (each checkpointed SD layer forms its own pre-norm).

Inference weights: :meth:`NeuronSparkLMHeadModel.prepare_inference` attaches a bf16 copy to every projection weight
(the tied embedding included); inference GEMMs read it (``neuronspark.ops.linear``).  A weight modified after the copy
was made raises at the next eager use; call ``prepare_inference()`` again.

Checkpoint: the state dict of :class:`NeuronSparkLMHeadModel` (678 tensors for the 1MIX configuration, every key
under ``backbone.``) is the ``model.safetensors`` of the checkpoint directory format defined in
:mod:`neuronspark.utils.hf`.  ``init="empty"`` builds without any random draw for a build followed by a checkpoint
load.

Launch (``init="launch"``): the embedding is drawn N(0, 0.02) on the global stream after ``nn.Embedding``'s own default
draw, then the layers in index order (each with its own draws, see the layer modules), the final norm (ones), the
readout balance (zeros) and the lexical memory (its private stream).  A fresh launch is calibrated with
:func:`neuronspark.utils.calibration.calibrate_fresh_model` before training; the parameter versions at the end of the
construction are recorded so that the calibration can refuse a model whose parameters were changed since.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from neuronspark import ops as _ops
from neuronspark.models.config_neuronspark import NeuronSparkConfig
from neuronspark.modules.copy_detector import allocate_copy_state, copy_stats_session, copy_stats_whole_row
from neuronspark.modules.induction import allocate_induction_state, induction_match_matrices, induction_session_match
from neuronspark.modules.lexical_memory import LexicalMemory
from neuronspark.modules.memory_layer import MemoryLayer
from neuronspark.modules.norms import RMSNorm
from neuronspark.modules.readout_balance import ReadoutBalance
from neuronspark.modules.sd_mixer import SDLayer
from neuronspark.ops.linear import cache_bf16_, drop_bf16_
from neuronspark.utils.cache import (
    DEFAULT_MAX_SEQLEN,
    MAX_DECODE_BATCH,
    RING_MULTIPLE,
    InferenceParams,
    resolve_sd_state_dtype,
)
from neuronspark.utils.generation import GenerationMixin
from neuronspark.utils.precision import check_fp32_parameters, regime

__all__ = [
    "NeuronSparkModel",
    "NeuronSparkLMHeadModel",
    "CausalLMOutput",
    "LossOutput",
    "MODEL_IMPLS",
    "MODEL_INITS",
    "EMBED_INIT_STD",
    "IGNORE_INDEX",
]

MODEL_IMPLS = ("kernel", "ref")
MODEL_INITS = ("launch", "empty")
# launch std of the tied embedding
EMBED_INIT_STD = 0.02
# target id that carries no loss
IGNORE_INDEX = -100
# 2-D parameters that are not projection weights (gathered or consumed in fp32 by their own operators): no bf16 copy
_NOT_PROJECTION_PREFIXES = ("lexical_memory.tables.",)
_NOT_PROJECTION_SUFFIXES = (".pkm_values", ".mixer_a.conv_weight")


class CausalLMOutput(NamedTuple):
    """``logits`` (B, R, V) bf16: the balanced readout of the last R positions (R = T, or ``num_last_tokens``)."""

    logits: torch.Tensor


class LossOutput(NamedTuple):
    """Training loss terms (fp32 scalars): ``loss = ce + aux``; ``n_valid`` = the number of targets != -100."""

    loss: torch.Tensor
    ce: torch.Tensor
    aux: torch.Tensor
    n_valid: torch.Tensor


def _check_choice(name: str, value: str, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}, got {value!r}")


def _as_token_ids(ids: torch.Tensor, what: str = "input_ids") -> torch.Tensor:
    if not isinstance(ids, torch.Tensor) or ids.dim() != 2:
        raise ValueError(f"{what} must be a (B, T) tensor of token ids, got "
                         f"{type(ids).__name__ if not isinstance(ids, torch.Tensor) else tuple(ids.shape)}")
    if ids.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"{what} must be int64 (or int32) token ids, got {ids.dtype}")
    if int(ids.shape[0]) < 1 or int(ids.shape[1]) < 1:
        raise ValueError(f"{what} is empty: {tuple(ids.shape)}")
    return ids if ids.dtype == torch.int64 else ids.to(torch.int64)


# ================================================================================================ backbone
class NeuronSparkModel(nn.Module):
    """``NeuronSparkModel(config, device=None, dtype=torch.float32, impl="kernel", init="launch")``.

    Attributes (state-dict keys relative to the backbone): ``embed_tokens`` (``nn.Embedding(V, D)``, the tied readout
    weight as well), ``layers`` (24 :class:`SDLayer` / :class:`MemoryLayer`), ``norm`` (``RMSNorm(D, 1e-6)``),
    ``balance_stack`` (:class:`ReadoutBalance`, widths (16, 32)), ``lexical_memory`` (:class:`LexicalMemory`).

    ``impl``: ``"kernel"`` (the fused operators) or ``"ref"`` (the pure-torch references, same signatures).
    ``dtype``: the parameters are fp32 masters; any other value raises.
    """

    def __init__(self, config: NeuronSparkConfig, device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = torch.float32, impl: str = "kernel", init: str = "launch") -> None:
        super().__init__()
        if not isinstance(config, NeuronSparkConfig):
            raise TypeError(f"NeuronSparkModel: expected a NeuronSparkConfig, got {type(config).__name__}")
        _check_choice("impl", impl, MODEL_IMPLS)
        _check_choice("init", init, MODEL_INITS)
        if dtype != torch.float32:
            raise TypeError(f"NeuronSparkModel: the parameters are fp32 masters (the operators cast their operands), "
                            f"got dtype {dtype}")
        self.config = config
        self.impl = impl
        self.gradient_checkpointing = False
        V, D = int(config.vocab_size), int(config.D)
        f32 = dict(device=device, dtype=torch.float32)
        launch = init == "launch"
        # registration order = state-dict order
        if launch:
            self.embed_tokens = nn.Embedding(V, D, **f32)
            with torch.no_grad():
                self.embed_tokens.weight.normal_(mean=0.0, std=EMBED_INIT_STD)
        else:
            self.embed_tokens = nn.Embedding(V, D, _weight=torch.empty((V, D), **f32))
        layers: List[nn.Module] = []
        for i in range(int(config.num_layers)):
            if config.is_memory_layer(i):
                layers.append(MemoryLayer(config, i, device=device, dtype=torch.float32, impl=impl, init=init))
            else:
                layers.append(SDLayer(config, i, device=device, dtype=torch.float32, impl=impl, init=init))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(D, eps=float(config.rms_norm_eps), device=device, dtype=torch.float32,
                            init="default" if launch else "empty")
        self.balance_stack = ReadoutBalance(D, tuple(config.readout_balance_widths), device=device,
                                            dtype=torch.float32, impl=impl, init=init)
        self.lexical_memory = LexicalMemory(config, device=device, init=init)
        if self.lexical_memory.order1_slice != tuple(config.lexical_order1_slice):
            raise RuntimeError(f"NeuronSparkModel: the lexical memory holds order 1 at {self.lexical_memory.order1_slice}"
                               f", the copy detector reads {tuple(config.lexical_order1_slice)}")
        check_fp32_parameters(self)
        # parameter versions at the end of a fresh launch (read by the launch calibration); None for an empty build
        self._launch_versions: Optional[Dict[str, int]] = self.parameter_versions() if launch else None

    # ------------------------------------------------------------------------------------------------ structure
    @property
    def memory_layers(self) -> List[MemoryLayer]:
        """The six memory layers in depth order."""
        return [m for m in self.layers if isinstance(m, MemoryLayer)]

    @property
    def sd_layers(self) -> List[SDLayer]:
        """The 18 SD layers in depth order."""
        return [m for m in self.layers if isinstance(m, SDLayer)]

    def parameter_versions(self) -> Dict[str, int]:
        """``{name: version counter}`` of every parameter (an in-place write bumps the counter)."""
        return {n: int(p._version) for n, p in self.named_parameters()}

    def extra_repr(self) -> str:
        c = self.config
        return (f"vocab_size={c.vocab_size}, D={c.D}, layers={c.num_layers} (SD {c.n_sd_layers}, memory "
                f"{c.n_memory_layers} at {c.memory_layer_ids}), impl={self.impl!r}")

    # ------------------------------------------------------------------------------------------------ forward
    def forward(self, input_ids: torch.Tensor, inference_params: Optional[InferenceParams] = None,
                aux_out: Optional[List[tuple]] = None, calib=None) -> torch.Tensor:
        """``input_ids`` (B, T) -> hidden states (B, T, D) fp32 (after the final norm).

        ``inference_params`` None: the whole-row regime; ``aux_out`` (a list) receives one ``(z, r, pi)`` per memory
        layer (the copy switch's operands, detached); ``calib`` is the launch-calibration capture object
        (:mod:`neuronspark.utils.calibration`).  ``inference_params`` given: prefill (T >= 2) or decode (T == 1) over
        the cache, under ``torch.no_grad()``.
        """
        ids = _as_token_ids(input_ids)
        if ids.device != self.embed_tokens.weight.device:
            raise RuntimeError(f"NeuronSparkModel: input_ids on {ids.device}, the model on "
                               f"{self.embed_tokens.weight.device}")
        if inference_params is None:
            return self._forward_whole_row(ids, aux_out, calib)
        if aux_out is not None or calib is not None:
            raise ValueError("NeuronSparkModel: aux_out / calib belong to the whole-row regime (no inference_params)")
        return self._forward_infer(ids, inference_params)

    def _forward_whole_row(self, ids: torch.Tensor, aux_out: Optional[List[tuple]], calib) -> torch.Tensor:
        cfg = self.config
        B, T = (int(s) for s in ids.shape)
        if self.impl == "kernel" and T % int(cfg.sd_chunk_size) != 0:
            raise RuntimeError(f"NeuronSparkModel: the whole-row regime runs full scan chunks: T = {T} is not a multiple "
                               f"of sd_chunk_size {cfg.sd_chunk_size} (pad the rows, or use an inference cache)")
        if aux_out is not None and not isinstance(aux_out, list):
            raise TypeError("NeuronSparkModel: aux_out must be a list")
        with regime("train"):
            x = self.embed_tokens(ids).transpose(0, 1)                       # (T, B, D) fp32, B-major storage
            lex = self.lexical_memory(ids)                                   # (T, B, 160) fp32
            ind = induction_match_matrices(ids, cfg.memory_induction_bias_orders, pad=int(cfg.vocab_size))
            copy = copy_stats_whole_row(lex, tuple(cfg.lexical_order1_slice))
            ckpt = bool(self.gradient_checkpointing) and torch.is_grad_enabled()
            carry: Optional[torch.Tensor] = None
            hn: Optional[torch.Tensor] = None
            n_layers = len(self.layers)
            for i, layer in enumerate(self.layers):
                if isinstance(layer, MemoryLayer):
                    if hn is not None:
                        raise RuntimeError(f"NeuronSparkModel: layer {i} is a memory layer, but the previous SD layer "
                                           "prepared an SD pre-norm for it")
                    # memory layers stay resident (never checkpointed)
                    x, carry = layer(x, lex=lex, ind=ind, copy=copy, carry=carry, aux_out=aux_out, calib=calib)
                    continue
                if ckpt:
                    x, _ = checkpoint(layer, x, None, use_reentrant=False, preserve_rng_state=False)
                    continue
                nxt = self.layers[i + 1] if i + 1 < n_layers else None
                gain = nxt.mixer_a.pre_norm_gain if isinstance(nxt, SDLayer) else None
                x, hn = layer(x, hn=hn, next_norm_gain=gain)
            if hn is not None:
                raise RuntimeError("NeuronSparkModel: the last SD layer prepared a pre-norm that no layer consumed")
            x = self.norm(x)
        return x.transpose(0, 1)                                             # (B, T, D)

    def _forward_infer(self, ids: torch.Tensor, ip: InferenceParams) -> torch.Tensor:
        cfg = self.config
        if not isinstance(ip, InferenceParams):
            raise TypeError(f"NeuronSparkModel: inference_params must be an InferenceParams, got {type(ip).__name__}")
        if torch.is_grad_enabled():
            raise RuntimeError("NeuronSparkModel: inference calls (an inference cache) run under torch.no_grad() (the "
                               "cache is updated in place)")
        B, T = (int(s) for s in ids.shape)
        if B != int(ip.max_batch_size):
            raise RuntimeError(f"NeuronSparkModel: {B} rows for a cache of {ip.max_batch_size} rows (every row of a cache "
                               "advances together; allocate a cache for this batch)")
        ip.check_room(T)
        W = int(ip.max_seqlen)
        D = int(cfg.D)
        o = _ops.namespace(self.impl)
        with regime("infer"):
            x = self.embed_tokens(ids).transpose(0, 1)                       # (T, B, D) fp32
            if T > 1:
                x = x.contiguous()                                           # T-major stream for the prefill
            lex = self.lexical_memory.session_vector(ids, ip.lexical)
            ind = induction_session_match(ids, ip.induction, ip.ring_ptr, W, cfg.memory_induction_bias_orders)
            copy = copy_stats_session(lex, ip.copy, ip.ring_ptr, ip.pos, W, tuple(cfg.lexical_order1_slice))
            carry: Optional[torch.Tensor] = None
            for layer in self.layers:
                if isinstance(layer, MemoryLayer):
                    x, carry = layer(x, lex=lex, ind=ind, copy=copy, carry=carry, inference_params=ip)
                else:
                    x, _ = layer(x, ip)
            x = o.rmsnorm_rows(x.reshape(T * B, D), float(self.norm.eps), weight=self.norm.weight).view(T, B, D)
            # the counters advance on the device (a replayed graph advances them too)
            ip.pos.add_(T)
            ip.ring_ptr.add_(T)
        ip.seqlen_offset += T
        return x.transpose(0, 1)                                             # (B, T, D)

    # ------------------------------------------------------------------------------------------------ cache
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int = DEFAULT_MAX_SEQLEN,
                                 sd_state_dtype: Optional[Union[str, torch.dtype]] = None,
                                 device: Optional[Union[str, torch.device]] = None) -> InferenceParams:
        """A zero (row-start) cache for ``batch_size`` rows with ring capacity ``max_seqlen`` (a multiple of 512, at
        most ``rope_max_seq_len``).  ``sd_state_dtype`` None takes ``config.inference_sd_state_dtype``."""
        cfg = self.config
        B, W = int(batch_size), int(max_seqlen)
        if not 1 <= B <= MAX_DECODE_BATCH:
            raise ValueError(f"allocate_inference_cache: batch_size must lie in [1, {MAX_DECODE_BATCH}], got {B}")
        if W <= 0 or W % RING_MULTIPLE != 0 or W > int(cfg.rope_max_seq_len):
            raise ValueError(f"allocate_inference_cache: max_seqlen (ring capacity) must be a positive multiple of "
                             f"{RING_MULTIPLE} and at most rope_max_seq_len {cfg.rope_max_seq_len}, got {W}")
        dt = resolve_sd_state_dtype(cfg.inference_sd_state_dtype if sd_state_dtype is None else sd_state_dtype)
        dev = self.embed_tokens.weight.device if device is None else torch.device(device)
        layers = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MemoryLayer):
                layers[i] = layer.allocate_inference_cache(B, W, device=dev)
            else:
                layers[i] = layer.allocate_inference_cache(B, W, sd_state_dtype=dt, device=dev)
        s0, s1 = (int(v) for v in cfg.lexical_order1_slice)
        pos, ring_ptr = InferenceParams.allocate_counters(B, dev)
        return InferenceParams(
            max_seqlen=W,
            max_batch_size=B,
            pos=pos,
            ring_ptr=ring_ptr,
            sd_state_dtype=dt,
            layers=layers,
            lexical=self.lexical_memory.allocate_inference_cache(B, device=dev),
            induction=allocate_induction_state(B, W, cfg.memory_induction_bias_orders, pad=int(cfg.vocab_size),
                                               device=dev),
            copy=allocate_copy_state(B, W, dim=s1 - s0, device=dev),
            rope_max_seq_len=int(cfg.rope_max_seq_len),
        )

    # ------------------------------------------------------------------------------------------------ bf16 weights
    def inference_linear_weights(self) -> List[Tuple[str, nn.Parameter]]:
        """``(name, parameter)`` of every projection weight (2-D, read by ``neuronspark.ops.linear``), the tied
        embedding included; the gathered tables (lexical tables, PKM values) and the SD conv taps are not included."""
        out = []
        for n, p in self.named_parameters():
            if p.dim() != 2:
                continue
            if n.startswith(_NOT_PROJECTION_PREFIXES) or n.endswith(_NOT_PROJECTION_SUFFIXES):
                continue
            out.append((n, p))
        return out


# ================================================================================================ LM head model
class NeuronSparkLMHeadModel(nn.Module, GenerationMixin):
    """``NeuronSparkLMHeadModel(config, device=None, dtype=torch.float32, impl="kernel", init="launch")``: the
    backbone (``backbone``) and the tied readout with the readout balance.

    * :meth:`forward` ``(input_ids, inference_params=None, num_last_tokens=0) -> CausalLMOutput(logits)`` (mamba's
      signature): the balanced logits (B, R, V) bf16 of the last R positions;
    * :meth:`loss` ``(input_ids, targets) -> LossOutput`` (training, whole row);
    * :meth:`allocate_inference_cache`, :meth:`prepare_inference` / :meth:`release_inference`;
    * :meth:`generate` (from :class:`neuronspark.utils.generation.GenerationMixin`), CUDA-graph decode by default;
    * :meth:`from_pretrained` / :meth:`save_pretrained` (:mod:`neuronspark.utils.hf`).
    """

    def __init__(self, config: NeuronSparkConfig, device: Optional[Union[str, torch.device]] = None,
                 dtype: torch.dtype = torch.float32, impl: str = "kernel", init: str = "launch") -> None:
        super().__init__()
        self.config = config
        self.impl = impl
        self.backbone = NeuronSparkModel(config, device=device, dtype=dtype, impl=impl, init=init)
        # the CUDA-graph decode cache of neuronspark.utils.generation (None until the first graph decode)
        self._decoding_cache = None

    @property
    def _o(self):
        return _ops.namespace(self.impl)

    @property
    def lm_head_weight(self) -> nn.Parameter:
        """The tied readout weight (the embedding table)."""
        return self.backbone.embed_tokens.weight

    @property
    def gradient_checkpointing(self) -> bool:
        return bool(self.backbone.gradient_checkpointing)

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, enabled: bool) -> None:
        self.backbone.gradient_checkpointing = bool(enabled)

    def extra_repr(self) -> str:
        return f"impl={self.impl!r}"

    # ------------------------------------------------------------------------------------------------ readout
    def _readout(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden (B, R, D) fp32 -> balanced logits (B, R, V) bf16 (the regime is the caller's)."""
        B, R, D = (int(s) for s in hidden.shape)
        rows = hidden.reshape(B * R, D)
        logits = self._o.linear(rows, self.backbone.embed_tokens.weight)       # (B * R, V) bf16
        logits = self.backbone.balance_stack.balance(logits, rows)
        return logits.view(B, R, int(logits.shape[-1]))

    def forward(self, input_ids: torch.Tensor, inference_params: Optional[InferenceParams] = None,
                num_last_tokens: int = 0) -> CausalLMOutput:
        """Balanced logits of the last ``num_last_tokens`` positions (0 = every position).  Whole row without
        ``inference_params`` (T a multiple of 256 on the kernel path; autograd allowed); prefill / decode over the
        cache with it (no autograd)."""
        n = int(num_last_tokens)
        if n < 0:
            raise ValueError(f"num_last_tokens must be >= 0, got {num_last_tokens}")
        with regime("train" if inference_params is None else "infer"):
            hidden = self.backbone(input_ids, inference_params=inference_params)
            if n > 0:
                hidden = hidden[:, -n:]
            logits = self._readout(hidden)
        return CausalLMOutput(logits=logits)

    def loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> LossOutput:
        """Training loss of a whole-row batch.

        ``input_ids`` (B, T) and ``targets`` (B, T): the pre-shifted next-token ids (``targets[:, t]`` is the token
        after ``input_ids[:, t]``), -100 ignored.  Under ``regime("train")``:

        * the backbone's hidden rows (B * T, D), the tied readout ``logits = linear(rows, E)`` (bf16; the fp32 weight
          gradient reaches E together with the embedding lookup's), the per-token cross entropy of the balanced logits
          by the fused readout-balance + cross-entropy operator (the balanced logits are never stored);
        * ``ce = sum(per * valid) / max(sum(valid), 1)``;
        * ``aux`` = the copy-switch auxiliary loss on the memory layers' detached operands of this call (weight
          ``memory_copy_novelty_gate_aux``; gradient to each layer's ``copy_novelty_w / _v / _b`` only), computed inside
          the training autocast region;
        * ``loss = ce + aux``.
        """
        ids = _as_token_ids(input_ids)
        tg = _as_token_ids(targets, "targets")
        if tuple(tg.shape) != tuple(ids.shape):
            raise ValueError(f"targets {tuple(tg.shape)} must have the shape of input_ids {tuple(ids.shape)}")
        o = self._o
        lam = float(self.config.memory_copy_novelty_gate_aux)
        with regime("train"):
            feats: List[tuple] = []
            hidden = self.backbone(ids, aux_out=feats)                           # (B, T, D) fp32
            B, T, D = (int(s) for s in hidden.shape)
            rows = hidden.reshape(B * T, D)
            logits = o.linear(rows, self.backbone.embed_tokens.weight)           # (B * T, V) bf16
            labels = tg.reshape(-1).contiguous()
            per = self.backbone.balance_stack.ce_per_token(logits, rows, labels)  # (B * T,) fp32
            valid = (labels != IGNORE_INDEX).to(per.dtype)
            n_valid = valid.sum()
            ce = (per * valid).sum() / n_valid.clamp(min=1.0)
            mem = self.backbone.memory_layers
            if len(feats) != len(mem):
                raise RuntimeError(f"NeuronSparkLMHeadModel.loss: {len(feats)} memory layers handed the copy switch's "
                                   f"operands, the model has {len(mem)}")
            if lam > 0.0:
                weights = [(l.copy_novelty_w, l.copy_novelty_v, l.copy_novelty_b) for l in mem]
                aux = o.copy_switch_aux(ids, tg.contiguous(), feats, weights, lam)
            else:
                aux = torch.zeros((), dtype=torch.float32, device=ce.device)
            loss = ce + aux
        return LossOutput(loss=loss, ce=ce, aux=aux, n_valid=n_valid)

    # ------------------------------------------------------------------------------------------------ inference
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int = DEFAULT_MAX_SEQLEN,
                                 sd_state_dtype: Optional[Union[str, torch.dtype]] = None,
                                 device: Optional[Union[str, torch.device]] = None) -> InferenceParams:
        """See :meth:`NeuronSparkModel.allocate_inference_cache`."""
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, sd_state_dtype=sd_state_dtype,
                                                      device=device)

    @torch.no_grad()
    def prepare_inference(self) -> "NeuronSparkLMHeadModel":
        """Attach (or refresh) the bf16 inference copy of every projection weight (``neuronspark.ops.linear``).
        Returns the model."""
        for _, p in self.backbone.inference_linear_weights():
            cache_bf16_(p)
        return self

    def release_inference(self) -> None:
        """Drop the bf16 weight copies and the CUDA-graph decode cache (whose graphs read those copies)."""
        self._decoding_cache = None
        for _, p in self.backbone.inference_linear_weights():
            drop_bf16_(p)

    # ------------------------------------------------------------------------------------------------ checkpoints
    @classmethod
    def from_pretrained(cls, path, device: Union[str, torch.device] = "cuda", allow_uncalibrated: bool = False,
                        impl: str = "kernel", inference_sd_state_dtype: Optional[str] = None
                        ) -> "NeuronSparkLMHeadModel":
        """Load a checkpoint directory (``config.json`` + ``model.safetensors``, the format of
        :mod:`neuronspark.utils.hf`); see :func:`neuronspark.utils.hf.load_pretrained`."""
        from neuronspark.utils.hf import load_pretrained

        return load_pretrained(path, device=device, allow_uncalibrated=allow_uncalibrated, impl=impl,
                               inference_sd_state_dtype=inference_sd_state_dtype, model_cls=cls)

    def save_pretrained(self, path, tokenizer_dir=None) -> dict:
        """Write the checkpoint directory (``config.json`` and ``model.safetensors`` with ``backbone.*`` keys); copy
        the tokenizer files of ``tokenizer_dir`` when given; returns the verification summary (see
        :func:`neuronspark.utils.hf.save_pretrained`)."""
        from neuronspark.utils.hf import save_pretrained

        return save_pretrained(self, path, tokenizer_dir=tokenizer_dir)
