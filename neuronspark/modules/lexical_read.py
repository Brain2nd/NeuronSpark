"""Lexical read-out of the hippocampal layer: the first step of every hippocampal layer, a gated read of the model-level
lexical n-gram vector m (:mod:`neuronspark.modules.lexical_memory`).

    hn    = h rsqrt(mean(h^2) + 1e-6)                                  fp32, no gain
    qf    = ZCN(f32(bf16(hn @ lex_q^T)), lex_q_gain)                   (T, B, K) fp32, K = 64
    kf    = ZCN(f32(bf16(m @ lex_k^T)), lex_k_gain)                    (T, B, K) fp32
    alpha = sigmoid( sum(qf * kf) / sqrt(K) + lex_beta )               (T, B, 1) fp32
    h'    = h + alpha * f32(bf16(m @ lex_o^T))                          fp32 residual

with ZCN(x, g) = x rsqrt(mean(x^2) + 1e-6) (1 + g) in fp32 (zero-centred gain, no mean subtraction).  m is the
(T, B, 160) fp32 lexical vector of this call (the whole row's, or the carried-window vector of an inference call: the
read is per token, so both regimes run the same arithmetic per position).

Ops per regime: whole row -- ``ops.row_rmsnorm_tee`` for hn (bf16) and the lex_q projection on hn's rows, then
``ops.lex_read_train`` for the 64-wide ZCN norms, alpha and the residual (one forward and one backward kernel,
:mod:`neuronspark.ops.triton.lex_read_train`); prefill (T >= 2) -- ``ops.rmsnorm_rows`` for hn, then
``ops.lex_read_prefill`` for the ZCN norms, alpha and the residual after the three projections (one kernel per call,
:mod:`neuronspark.ops.triton.lex_read`); decode (T == 1) -- ``ops.lex_read_step`` after the projections: the read
and the frozen norm of its result in one kernel.

Parameters (keys, flat on the layer): ``lex_q.weight`` (64, 1536), ``lex_k.weight`` (64, 160), ``lex_o.weight``
(1536, 160), ``lex_q_gain`` / ``lex_k_gain`` (64,) = 0, ``lex_beta`` (1,) = ``lexical_memory_gate_bias`` (-2.0).
Launch: the three projections are built and drawn on a forked stream seeded ``mechanism_init_seed(991_000,
layer_idx)`` (PyTorch's default draws, then lex_q / lex_k Xavier-uniform, lex_o N(0, out_std)).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from neuronspark.modules.linear import Linear
from neuronspark.modules.plif import forked_rng, linear_init, rows_rmsnorm

__all__ = ["LexicalReadMixin", "LEX_SEED_BASE", "LEX_EPS"]

# forked launch stream of the read's projections
LEX_SEED_BASE = 991_000
# eps of every norm of the read
LEX_EPS = 1e-6


class LexicalReadMixin:
    """Mixin of :class:`neuronspark.modules.hippocampal_layer.HippocampalLayer`: the lexical read, step 1 of the
    layer."""

    def _build_lexical_read(self, config, layer_idx: int, device, dtype, init: str) -> None:
        D = int(config.D)
        K = int(config.lexical_memory_key_dim)
        L = int(config.lexical_dim)
        f32 = torch.float32 if dtype is None else dtype
        self.lex_key_dim, self.lex_dim = K, L
        li = linear_init(init)
        if init == "launch":
            with forked_rng(device, LEX_SEED_BASE, layer_idx):
                self.lex_q = Linear(D, K, device=device, dtype=dtype, init=li)
                self.lex_k = Linear(L, K, device=device, dtype=dtype, init=li)
                self.lex_o = Linear(L, D, device=device, dtype=dtype, init=li)
                nn.init.xavier_uniform_(self.lex_q.weight)
                nn.init.xavier_uniform_(self.lex_k.weight)
                self.lex_o.weight.normal_(std=float(config.out_std))
            self.lex_q_gain = nn.Parameter(torch.zeros(K, device=device, dtype=f32))
            self.lex_k_gain = nn.Parameter(torch.zeros(K, device=device, dtype=f32))
            self.lex_beta = nn.Parameter(torch.full((1,), float(config.lexical_memory_gate_bias), device=device,
                                                    dtype=f32))
        else:
            self.lex_q = Linear(D, K, device=device, dtype=dtype, init=li)
            self.lex_k = Linear(L, K, device=device, dtype=dtype, init=li)
            self.lex_o = Linear(L, D, device=device, dtype=dtype, init=li)
            self.lex_q_gain = nn.Parameter(torch.empty(K, device=device, dtype=f32))
            self.lex_k_gain = nn.Parameter(torch.empty(K, device=device, dtype=f32))
            self.lex_beta = nn.Parameter(torch.empty((1,), device=device, dtype=f32))

    def _check_lex(self, h: torch.Tensor, lex: torch.Tensor) -> None:
        if (not isinstance(lex, torch.Tensor) or tuple(lex.shape[:2]) != tuple(h.shape[:2])
                or int(lex.shape[-1]) != self.lex_dim or lex.dtype not in (torch.float32, torch.bfloat16)):
            raise RuntimeError("lexical read: the hippocampal layer needs the model-level lexical vector of this call, "
                               f"fp32 (or its bf16 rounding) (T, B, {self.lex_dim}) for h {tuple(h.shape)}, got "
                               f"{None if not isinstance(lex, torch.Tensor) else (tuple(lex.shape), lex.dtype)}")

    def _lexical_read_train(self, h: torch.Tensor, lex: torch.Tensor) -> torch.Tensor:
        """Whole-row form: h (T, B, D) fp32, lex (T, B, 160) fp32 -> h' fp32."""
        self._check_lex(h, lex)
        o = self._ops
        # hn in bf16 (its only consumer is the lex_q projection); h is handed on by the norm so that the gradients of
        # the norm and of the residual are summed in the norm's backward
        T, B, D = (int(s) for s in h.shape)
        h, _, (hn,) = o.row_rmsnorm_tee(h, None, LEX_EPS, False, 1)
        q_raw = o.linear(hn.reshape(T * B, D), self.lex_q.weight).view(T, B, -1)
        m_in = lex.to(h.dtype)
        k_raw = o.linear(m_in, self.lex_k.weight)
        o_raw = o.linear(m_in, self.lex_o.weight)
        return o.lex_read_train(h, q_raw, k_raw, o_raw, self.lex_q_gain, self.lex_k_gain, self.lex_beta, LEX_EPS)

    def _lexical_read_infer(self, h: torch.Tensor, lex: torch.Tensor) -> torch.Tensor:
        """Prefill / decode form (any T >= 1): the same expressions per token; the norms, the gate and the residual
        after the projections run as ``ops.lex_read_prefill``."""
        self._check_lex(h, lex)
        o = self._ops
        hn = rows_rmsnorm(o, h, LEX_EPS, out_dtype=torch.bfloat16)          # its only consumer is the bf16 lex_q projection
        m_b = lex.to(torch.bfloat16)              # the projections' operand: the rounding of their input cast, once
        q_raw = o.linear(hn, self.lex_q.weight)
        k_raw = o.linear(m_b, self.lex_k.weight)
        o_raw = o.linear(m_b, self.lex_o.weight)
        return o.lex_read_prefill(h, q_raw, k_raw, o_raw, self.lex_q_gain, self.lex_k_gain, self.lex_beta, LEX_EPS)

    def _lexical_read_step(self, h: torch.Tensor, lex: torch.Tensor, norm_eps: float):
        """Decode form (T == 1): the read and the frozen norm of its result in ``ops.lex_read_step`` -> (h' (B, D)
        fp32, its bf16 rounding, z = rmsnorm(h') (B, D) fp32, its bf16 rounding)."""
        self._check_lex(h, lex)
        o = self._ops
        hn = rows_rmsnorm(o, h[0], LEX_EPS, out_dtype=torch.bfloat16)       # the operand of the lex_q projection
        m_in = lex[0].to(torch.bfloat16)                                    # the operand of the two projections of m
        return o.lex_read_step(h[0], o.linear(hn, self.lex_q.weight), o.linear(m_in, self.lex_k.weight),
                               o.linear(m_in, self.lex_o.weight), self.lex_q_gain, self.lex_k_gain, self.lex_beta,
                               LEX_EPS, norm_eps)
