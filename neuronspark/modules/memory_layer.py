"""Memory layer of 1MIX (6 instances, layers 3, 7, 11, 15, 19, 23; state-dict prefix ``layers.{i}.``).

One call, residual stream h (T, B, D) fp32:

     1. lexical read          h  <- h + alpha W_o m          (:mod:`neuronspark.modules.lexical_read`)
     2. z = frozen_rms_norm(h)                                fp32, gain 1, eps 1e-6
     3-12. pool attention     r  = centred attention residual (bf16) (:mod:`neuronspark.modules.pool_attention`)
    13. copy confidence       pi from the model-level copy statistics and the layer's sink tau
    14. novelty               c  = sigmoid(z . w + b + rms(r) . v)
    15. read gain             h' = h + bf16(f32(r) (1 + 4 pi c))  (:mod:`neuronspark.modules.copy_gate`)
    16. x_hat = ffn_norm(h')                                  fp32 RMSNorm with weight
    17. ffn = bf16(snn_ffn(input_neuron2(x_hat)) @ ffn_out_proj^T)
    18. y = PKM read(x_hat)                                    (:mod:`neuronspark.modules.pkm`)
    19. rho = 2 sigmoid(-4 pi)
    20. h_out = h' + f32(bf16(ffn + bf16(y_g - mean_c y_g))),  y_g = bf16(f32(y) rho)

``forward(h, *, lex, ind, copy, carry=None, inference_params=None, aux_out=None, calib=None) -> (h_out, carry)``:

* ``lex`` (T, B, 160) fp32: the model-level lexical vector of this call;
* ``ind``: one bool tensor per induction order, (B, T, T) whole-row match matrices or (B, T, W) ring matches of an
  inference call;
* ``copy``: the model-level copy-detector statistics ``(maximum, lse)``, each (T, B) fp32;
* ``carry``: the value stream (T, B, 1536) bf16 of memory layer 0 for this call (None into memory layer 0, which
  returns its own; the later layers return the carry they received);
* ``inference_params``: None -> whole-row regime (training, teacher-forced scoring; autograd allowed); an
  :class:`neuronspark.utils.cache.InferenceParams` -> prefill (T >= 2) or decode (T == 1) over the layer's carried
  :class:`MemoryState` (no autograd).  ``inference_params.pos`` / ``ring_ptr`` hold the counters at the call's start
  (the model advances them after the last layer);
* ``aux_out`` (whole row): a list that receives ``(z.detach(), r.detach(), pi)`` of this layer, the operands of the
  supervised copy switch;
* ``calib`` (whole row): the launch-calibration capture object (protocol in :mod:`neuronspark.modules.pkm`); while
  it is given, a layer whose PKM is not ready adds no PKM read and no read gain.

The forward must run inside a bf16 CUDA autocast region (``neuronspark.utils.precision.regime``), which selects the
regime's TF32 setting as well; it raises otherwise.

Construction: ``MemoryLayer(config, layer_idx, mem_index=None, device=None, dtype=None, impl="kernel",
init="launch")``.  ``mem_index`` (0 .. 5) is the rank among the memory layers (0 = the value-residual source);
``impl`` selects the operator namespace (``"kernel"`` | ``"ref"``); ``init="empty"`` allocates the parameters without
drawing (a build followed by ``load_state_dict``).  Keys: 70 parameters and 11 persistent buffers per layer,
registered flat on the layer by the four mixins, plus the submodules ``gate_neuron``, ``ffn_norm``,
``input_neuron2``, ``snn_ffn`` and ``ffn_out_proj``.  Loading a state dict that holds ``pkm_calibrated > 0.5`` marks the
PKM ready.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuronspark.modules.copy_gate import CopyGateMixin, NOVELTY_RMS_EPS
from neuronspark.modules.lexical_read import LexicalReadMixin
from neuronspark.modules.linear import Linear
from neuronspark.modules.norms import RMSNorm
from neuronspark.modules.pkm import PKMMixin
from neuronspark.modules.plif import PLIFNeuron, check_init, linear_init, rows_rmsnorm
from neuronspark.modules.pool_attention import PoolAttentionMixin
from neuronspark.modules.spiking_ffn import SpikingFFN
from neuronspark.ops import namespace
from neuronspark.utils.cache import InferenceParams, MemoryState

__all__ = ["MemoryLayer"]


def _require_bf16_autocast(t: torch.Tensor) -> None:
    if t.is_cuda and not (torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        raise RuntimeError("MemoryLayer: the forward runs inside a bf16 CUDA autocast region (enter "
                           "neuronspark.utils.precision.regime('train') or regime('infer'))")


class MemoryLayer(PoolAttentionMixin, LexicalReadMixin, CopyGateMixin, PKMMixin, nn.Module):
    """The memory layer (see the module docstring)."""

    def __init__(self, config, layer_idx: int, mem_index: Optional[int] = None,
                 device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None,
                 impl: str = "kernel", init: str = "launch") -> None:
        nn.Module.__init__(self)
        check_init(init)
        if not config.is_memory_layer(layer_idx):
            raise ValueError(f"layer {layer_idx} is not a memory layer (memory layers: {config.memory_layer_ids})")
        self.layer_idx = int(layer_idx)
        mem_index = config.memory_index(layer_idx) if mem_index is None else int(mem_index)
        if mem_index != config.memory_index(layer_idx):
            raise ValueError(f"layer {layer_idx} is memory layer {config.memory_index(layer_idx)}, not {mem_index}")
        self.impl = impl
        self._ops = namespace(impl)
        self.D = int(config.D)
        self.D_ff = int(config.D_ff)
        self.norm_eps = float(config.rms_norm_eps)
        launch = init == "launch"
        # attention side (W_decay .. attn_out_proj, gate_neuron, RoPE tables)
        self._build_pool_attention(config, self.layer_idx, mem_index, device, dtype, init)
        # FFN side
        self.ffn_norm = RMSNorm(self.D, eps=self.norm_eps, device=device, dtype=dtype,
                                init="default" if launch else "empty")
        self.input_neuron2 = PLIFNeuron(self.D, tuple(config.plif_input2_init_tau), centred=True,
                                        alpha=config.plif_surrogate_alpha, eta_rank=config.plif_eta_proj_rank,
                                        device=device, dtype=dtype, impl=impl, init=init)
        self.snn_ffn = SpikingFFN(self.D, self.D_ff, init_tau=config.init_tau, alpha=config.plif_surrogate_alpha,
                                  eta_rank=config.plif_eta_proj_rank, out_std=config.out_std, device=device,
                                  dtype=dtype, impl=impl, init=init)
        self.ffn_out_proj = Linear(self.D, self.D, device=device, dtype=dtype, init=linear_init(init))
        if launch:
            # the launch draws taken after the FFN sub-block is built (global stream, in this order)
            self._finish_pool_attention_init(config.out_std)
            with torch.no_grad():
                self.ffn_out_proj.weight.normal_(std=float(config.out_std))
        # forked streams: lexical read, then the copy arbitration / induction parameters, then the PKM
        self._build_lexical_read(config, self.layer_idx, device, dtype, init)
        self._build_copy_gate(config, device, dtype, init)
        self._build_induction_bias(config, device, dtype, init)
        self._build_pkm(config, self.layer_idx, device, dtype, init)

    # ------------------------------------------------------------------------------------------------ checkpoint
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        key = prefix + "pkm_calibrated"
        if key in state_dict:
            self._pkm_ready = bool(float(state_dict[key]) > 0.5)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}, mem_index={self.mem_index}, impl={self.impl!r}"

    # ------------------------------------------------------------------------------------------------ cache
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int,
                                 device: Optional[Union[str, torch.device]] = None) -> MemoryState:
        """Zero state of ``batch_size`` rows with rings of capacity ``max_seqlen`` (clocks +inf = empty slots)."""
        dev = self.pool_write_bias.device if device is None else torch.device(device)
        return MemoryState.allocate(int(batch_size), int(max_seqlen), qkv_dim=2 * self.D_key + self.D_value,
                                    conv_width=int(self.qkv_conv.width), heads=self.memory_heads,
                                    key_head_dim=self.memory_head_dim, value_head_dim=self.memory_value_head_dim,
                                    value_dim=self.D_value, d_model=self.D, d_ff=self.D_ff, device=dev)

    # ------------------------------------------------------------------------------------------------ forward
    def forward(self, h: torch.Tensor, *, lex: torch.Tensor, ind, copy: Tuple[torch.Tensor, torch.Tensor],
                carry: Optional[torch.Tensor] = None, inference_params: Optional[InferenceParams] = None,
                aux_out: Optional[List[tuple]] = None, calib=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if h.dim() != 3 or int(h.shape[-1]) != self.D or h.dtype != torch.float32:
            raise RuntimeError(f"MemoryLayer: h must be fp32 (T, B, {self.D}), got {tuple(h.shape)} {h.dtype}")
        _require_bf16_autocast(h)
        if inference_params is None:
            return self._forward_train(h, lex, ind, copy, carry, aux_out, calib)
        if torch.is_grad_enabled():
            raise RuntimeError("MemoryLayer: an inference call (inference_params given) runs under torch.no_grad()")
        if calib is not None or aux_out is not None:
            raise RuntimeError("MemoryLayer: calib / aux_out belong to the whole-row regime")
        ip = inference_params
        st = ip.layers.get(self.layer_idx)
        if not isinstance(st, MemoryState):
            raise RuntimeError(f"MemoryLayer {self.layer_idx}: the inference cache holds no MemoryState for this layer")
        B = int(h.shape[1])
        if B != int(ip.max_batch_size) or tuple(st.pool_cum.shape[:1]) != (B,):
            raise RuntimeError(f"MemoryLayer: batch {B} for a cache of {ip.max_batch_size} rows")
        if int(h.shape[0]) == 1:
            return self._forward_infer(h, lex, ind, copy, carry, st, ip, step=True)
        return self._forward_infer(h, lex, ind, copy, carry, st, ip, step=False)

    def _check_copy(self, copy, T: int, B: int) -> None:
        if (not isinstance(copy, (tuple, list)) or len(copy) != 2
                or any(not isinstance(c, torch.Tensor) or tuple(c.shape) != (T, B) or c.dtype != torch.float32
                       for c in copy)):
            raise RuntimeError(f"MemoryLayer: the copy-detector statistics must be (maximum, lse), fp32 ({T}, {B}) each")

    # ------------------------------------------------------------------------------------------------ whole row
    def _forward_train(self, h, lex, ind, copy, carry, aux_out, calib):
        o = self._ops
        T, B, _ = (int(s) for s in h.shape)
        self._check_copy(copy, T, B)
        mode = None if calib is None else calib.pkm_mode(self)
        h = self._lexical_read_train(h, lex)
        z = o.row_rmsnorm(h, None, self.norm_eps)
        if mode == "queries":
            calib.record(self, "copy_max", copy[0].detach().flatten().to(device="cpu", dtype=torch.float32))
        r, carry_out = self._attention_train(h, z, ind, carry)
        pi = None
        if calib is None or self._pkm_ready:
            pi = self.copy_confidence(copy)
            c = self.copy_novelty(z, r, lambda t: o.nov_norm(t, NOVELTY_RMS_EPS))
            if aux_out is not None:
                aux_out.append((z.detach(), r.detach(), pi.detach()))
            with torch.autocast(h.device.type, enabled=False):
                f = self.read_gain_factor(pi, c)
                h = o.read_gain_residual(h, r, f)
        else:
            h = h + r
        x_hat = o.row_rmsnorm(h, self.ffn_norm.weight, self.ffn_norm.eps)
        zs = self.input_neuron2(x_hat)
        ffn = o.linear(self.snn_ffn(zs), self.ffn_out_proj.weight)
        y = self._pkm_read_train(x_hat, r, calib, mode)
        if y is None:
            return h + ffn, carry_out
        if pi is None:
            raise RuntimeError("MemoryLayer: the PKM read ran without the copy confidence of this call")
        rho = self.pkm_copy_rho(pi)
        return o.pkm_tail(h, ffn, y, rho), carry_out

    # ------------------------------------------------------------------------------------------------ inference
    def _forward_infer(self, h, lex, ind, copy, carry, st: MemoryState, ip: InferenceParams, step: bool):
        o = self._ops
        T, B, _ = (int(s) for s in h.shape)
        self._check_copy(copy, T, B)
        pos0, ptr0, W = ip.pos, ip.ring_ptr, int(ip.max_seqlen)
        h = self._lexical_read_infer(h, lex)
        z = rows_rmsnorm(o, h, self.norm_eps, out_dtype=h.dtype)
        if step:
            r, carry_out = self._attention_step(h, z, ind, carry, st, pos0, ptr0, W)
        else:
            r, carry_out = self._attention_prefill(h, z, ind, carry, st, pos0, ptr0, W)
        pi = self.copy_confidence(copy)
        c = self.copy_novelty(z, r, lambda t: rows_rmsnorm(o, t, NOVELTY_RMS_EPS, out_dtype=torch.float32))
        with torch.autocast(h.device.type, enabled=False):
            f = self.read_gain_factor(pi, c)
            r = (r.float() * (1.0 + f).unsqueeze(-1)).to(r.dtype)
        h = h + r
        x_hat = rows_rmsnorm(o, h, self.ffn_norm.eps, weight=self.ffn_norm.weight, out_dtype=h.dtype)
        zs = self.input_neuron2(x_hat, st.input_neuron2, pos0)
        ffn = o.linear(self.snn_ffn(zs, st.ffn_gate, st.ffn_up, pos0), self.ffn_out_proj.weight)
        y = self._pkm_read_step(x_hat) if step else self._pkm_read_prefill(x_hat)
        with torch.autocast(h.device.type, enabled=False):
            rho = self.pkm_copy_rho(pi)
            y = (y.float() * rho.unsqueeze(-1)).to(y.dtype)
        ffn = ffn + (y - y.mean(dim=-1, keepdim=True))
        return h + ffn, carry_out
