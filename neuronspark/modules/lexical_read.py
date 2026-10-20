"""Lexical read-out of the memory layer: the first step of every memory layer, a gated read of the model-level
lexical n-gram vector m (:mod:`neuronspark.modules.lexical_memory`).

    hn    = h rsqrt(mean(h^2) + 1e-6)                                  fp32, no gain
    qf    = ZCN(f32(bf16(hn @ lex_q^T)), lex_q_gain)                   (T, B, K) fp32, K = 64
    kf    = ZCN(f32(bf16(m @ lex_k^T)), lex_k_gain)                    (T, B, K) fp32
    alpha = sigmoid( sum(qf * kf) / sqrt(K) + lex_beta )               (T, B, 1) fp32
    h'    = h + alpha * f32(bf16(m @ lex_o^T))                          fp32 residual

with ZCN(x, g) = x rsqrt(mean(x^2) + 1e-6) (1 + g) in fp32 (zero-centred gain, no mean subtraction).  m is the
(T, B, 160) fp32 lexical vector of this call (the whole row's, or the carried-window vector of an inference call: the
read is per token, so both regimes run the same arithmetic per position).

Ops per regime: whole row -- ``ops.row_rmsnorm`` for hn and the lex_q projection on the (T * B, D) view of hn, the
64-wide ZCN norms, alpha and the residual in torch; prefill (T >= 2) -- ``ops.rmsnorm_rows`` for hn, then
``ops.lex_read_prefill`` for the ZCN norms, alpha and the residual after the three projections (one kernel per call,
:mod:`neuronspark.ops.triton.lex_read`); decode (T == 1) -- ``ops.rmsnorm_rows`` for hn, the rest in torch.

Parameters (keys, flat on the layer): ``lex_q.weight`` (64, 1536), ``lex_k.weight`` (64, 160), ``lex_o.weight``
(1536, 160), ``lex_q_gain`` / ``lex_k_gain`` (64,) = 0, ``lex_beta`` (1,) = ``lexical_memory_gate_bias`` (-2.0).
Launch: the three projections are built and drawn on a forked stream seeded ``mechanism_init_seed(991_000,
layer_idx)`` (PyTorch's default draws, then lex_q / lex_k Xavier-uniform, lex_o N(0, out_std)).
"""
from __future__ import annotations

import math

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
    """Mixin of :class:`neuronspark.modules.memory_layer.MemoryLayer` (the lexical read, step 1 of the layer)."""

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
                or int(lex.shape[-1]) != self.lex_dim or lex.dtype != torch.float32):
            raise RuntimeError("lexical read: the memory layer needs the model-level lexical vector of this call, fp32 "
                               f"(T, B, {self.lex_dim}) for h {tuple(h.shape)}, got "
                               f"{None if not isinstance(lex, torch.Tensor) else (tuple(lex.shape), lex.dtype)}")

    def _lexical_gate(self, q_raw: torch.Tensor, k_raw: torch.Tensor) -> torch.Tensor:
        """(T, B, 1) fp32 alpha from the bf16 q / k projections (the ZCN norms, the scaled dot, the bias, sigmoid)."""
        qf = q_raw.float()
        qf = qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + LEX_EPS) * (1.0 + self.lex_q_gain.float())
        kf = k_raw.float()
        kf = kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + LEX_EPS) * (1.0 + self.lex_k_gain.float())
        return torch.sigmoid((qf * kf).sum(dim=-1, keepdim=True) / math.sqrt(self.lex_key_dim)
                             + self.lex_beta.float())

    def _lexical_read_train(self, h: torch.Tensor, lex: torch.Tensor) -> torch.Tensor:
        """Whole-row form: h (T, B, D) fp32, lex (T, B, 160) fp32 -> h' fp32."""
        self._check_lex(h, lex)
        o = self._ops
        T, B, D = (int(s) for s in h.shape)
        hn = o.row_rmsnorm(h, None, LEX_EPS)
        q_raw = o.linear(hn.reshape(T * B, D), self.lex_q.weight).view(T, B, -1)
        m_in = lex.to(h.dtype)
        alpha = self._lexical_gate(q_raw, o.linear(m_in, self.lex_k.weight))
        return h + (alpha * o.linear(m_in, self.lex_o.weight).float()).to(h.dtype)

    def _lexical_read_infer(self, h: torch.Tensor, lex: torch.Tensor) -> torch.Tensor:
        """Prefill / decode form (any T >= 1): the same expressions per token; a prefill call (T >= 2) runs the norms,
        the gate and the residual after the projections as ``ops.lex_read_prefill``."""
        self._check_lex(h, lex)
        o = self._ops
        hn = rows_rmsnorm(o, h, LEX_EPS, out_dtype=torch.bfloat16)          # its only consumer is the bf16 lex_q projection
        m_in = lex.to(h.dtype)
        if int(h.shape[0]) > 1:
            q_raw = o.linear(hn, self.lex_q.weight)
            k_raw = o.linear(m_in, self.lex_k.weight)
            o_raw = o.linear(m_in, self.lex_o.weight)
            return o.lex_read_prefill(h, q_raw, k_raw, o_raw, self.lex_q_gain, self.lex_k_gain, self.lex_beta, LEX_EPS)
        alpha = self._lexical_gate(o.linear(hn, self.lex_q.weight), o.linear(m_in, self.lex_k.weight))
        return h + (alpha * o.linear(m_in, self.lex_o.weight).float()).to(h.dtype)
