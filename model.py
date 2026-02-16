"""
SNNLanguageModel v7: SNN 隐状态空间语言模型（Parallel Scan 版本）

v6 → v7 变更：
  - K: 8 → 16 (16-bit 二进制编码精度)
  - 移除 W^(V)（见文档第 7.2 节）
  - 前向传播：parallel scan 并行处理全序列
  - 编码/解码：批量处理所有 token

架构：
  token → Embedding (trainable, D-dim)
  → encode_proj → sigmoid → K-bit 二进制编码 → K 帧 spike
  → L 个 SNNDecoderLayer（SNNBlock + SNNFFN，parallel scan）
  → K 帧 spike 二进制解码 → [0,1]^D
  → decode_proj → RMSNorm (trainable) → Embedding^T (tied) → logits

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.checkpoint import checkpoint

from atomic_ops import SNNDecoderLayer


@dataclass
class SNNModelOutput:
    """模型输出容器，对齐教程 CausalLMOutputWithPast 接口。"""
    last_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    """RMSNorm 归一化层。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SNNLanguageModel(nn.Module):
    """
    从零训练的 SNN 隐状态空间语言模型（v7: parallel scan）。

    Args:
        vocab_size: 词表大小（默认 6144，自训练 BPE）
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数（v7 默认 16）
        num_layers: SNN 解码层数
        D_ff: FFN 中间层维度
        v_th_min: 动态阈值下限
    """

    def __init__(
        self,
        vocab_size: int = 6144,
        D: int = 1024,
        N: int = 8,
        K: int = 16,
        num_layers: int = 20,
        D_ff: int = 3072,
        v_th_min: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff

        # ====== Embedding + Norm（全部可训练）======
        self.embed_tokens = nn.Embedding(vocab_size, D)
        self.norm = RMSNorm(D)

        # ====== 可训练编码/解码投影 ======
        self.encode_proj = nn.Linear(D, D)
        self.decode_proj = nn.Linear(D, D)

        # ====== SNN Decoder Layers ======
        self.layers = nn.ModuleList([
            SNNDecoderLayer(
                D=D, N=N, D_ff=D_ff, v_th_min=v_th_min,
                block_output_v_threshold=0.3 if i == 0 else 0.05,
                ffn_output_v_threshold=0.5 if i == 0 else 0.15,
                num_layers=num_layers,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])

        # ====== K-bit 二进制权重 ======
        self.register_buffer(
            'bit_weights',
            torch.tensor([2.0 ** (-(k + 1)) for k in range(K)]),
        )
        # ====== K-bit 编码缩放因子（并行编码用） ======
        self.register_buffer(
            'bit_scales',
            torch.tensor([2.0 ** (k + 1) for k in range(K)]),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化所有可训练权重（从零训练）。"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.encode_proj.weight)
        nn.init.zeros_(self.encode_proj.bias)
        nn.init.xavier_uniform_(self.decode_proj.weight)
        nn.init.zeros_(self.decode_proj.bias)

    def _encode_all_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        将所有 token 编码为 spike 帧序列。

        Args:
            token_ids: (batch, seq_len)

        Returns:
            spike_seq: (seq_len * K, batch, D) — 全部 T×K 帧的 spike
        """
        batch, seq_len = token_ids.shape

        # 1. Embedding: (batch, seq_len, D)
        emb = self.embed_tokens(token_ids)

        # 2. 编码: (batch, seq_len, D) → (batch, seq_len, D)
        h = torch.sigmoid(self.encode_proj(emb))

        # 3. K-bit 并行二进制编码 → (batch, seq_len, K, D)
        #    数学：h ∈ [0,1] 的二进制小数展开 bit_k = ⌊2^{k+1}·h⌋ mod 2
        #    STE: forward = bit_hard (二值 {0,1}), backward = ∂/∂h = 1（恒等）
        #    等价于逐位展开，但全部 K 位同时计算，无循环
        scaled = h.unsqueeze(2) * self.bit_scales.view(1, 1, self.K, 1)  # (batch, seq_len, K, D)
        bit_hard = torch.floor(scaled) % 2  # {0.0, 1.0}
        h_k = h.unsqueeze(2)  # (batch, seq_len, 1, D), broadcasts to (batch, seq_len, K, D)
        bits = h_k + (bit_hard - h_k).detach()  # STE: forward=bit_hard, backward=∂/∂h=1

        # 4. (batch, seq_len, K, D) → (seq_len*K, batch, D)
        spike_seq = bits.reshape(batch, seq_len * self.K, self.D)
        spike_seq = spike_seq.permute(1, 0, 2)  # (seq_len*K, batch, D)

        return spike_seq

    def _decode_all_tokens(self, spike_seq: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        将 spike 帧序列解码为所有 token 的实数表示。

        Args:
            spike_seq: (seq_len * K, batch, D)
            seq_len: token 序列长度

        Returns:
            decoded: (batch, seq_len, D)
        """
        batch = spike_seq.shape[1]

        # (seq_len*K, batch, D) → (batch, seq_len, K, D)
        spike_seq = spike_seq.permute(1, 0, 2)  # (batch, seq_len*K, D)
        spike_seq = spike_seq.reshape(batch, seq_len, self.K, self.D)

        # 二进制加权求和: sum(spike * 2^{-k})
        decoded = torch.einsum('bskd,k->bsd', spike_seq, self.bit_weights)

        return decoded  # (batch, seq_len, D)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        前向传播（v7: parallel scan 并行处理全序列）。

        Args:
            token_ids: (batch, seq_len) 的 token ID 序列
            target_ids: (batch, seq_len) 的目标 ID

        Returns:
            SNNModelOutput:
                若 target_ids 提供: .last_loss = mean cross-entropy loss
                若 target_ids is None: .logits = (batch, seq_len, vocab_size)
        """
        batch, seq_len = token_ids.shape

        # 重置所有 Layer 内部的神经元状态
        for layer_module in self.layers:
            functional.reset_net(layer_module)

        # ====== 1. 编码全部 token ======
        spike_seq = self._encode_all_tokens(token_ids)  # (seq_len*K, batch, D)

        # ====== 2. 逐层并行处理（梯度检查点：按层重计算，省显存） ======
        # 注意：checkpoint 反向时重跑 forward，必须先重置神经元状态回 0.0
        def _layer_forward(layer_mod, x):
            functional.reset_net(layer_mod)
            return layer_mod.forward_parallel(x)

        for layer_module in self.layers:
            spike_seq = checkpoint(
                _layer_forward, layer_module, spike_seq,
                use_reentrant=False,
            )

        # ====== 3. 解码全部 token ======
        decoded = self._decode_all_tokens(spike_seq, seq_len)  # (batch, seq_len, D)

        # ====== 4. 投影 → RMSNorm → tied LM Head ======
        h = self.decode_proj(decoded)  # (batch, seq_len, D)
        h = self.norm(h)               # (batch, seq_len, D)
        logits = F.linear(h, self.embed_tokens.weight)  # (batch, seq_len, vocab)

        if target_ids is not None:
            # 计算 loss
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = target_ids.reshape(-1)
            self.last_loss = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=0, reduction='none',
            )
            return SNNModelOutput(last_loss=self.last_loss)

        return SNNModelOutput(logits=logits)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        按功能分组的可训练参数，供 SPSA 使用。
        v7: 移除了 W_V 和 b_beta/b_alpha/b_th 中的 W^(V) 相关组。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.weight],
            'encode': list(self.encode_proj.parameters()),
            'decode': list(self.decode_proj.parameters()),
            # SNNBlock 参数（v7: 无 W_V）
            'W_in': [],
            'W_beta': [],
            'W_alpha': [],
            'W_th': [],
            'W_gate': [],
            'W_skip': [],
            'W_out': [],
            'b_beta': [],
            'b_alpha': [],
            'b_th': [],
            'block_output_neuron': [],
            # SNNFFN 参数
            'ffn_gate_proj': [],
            'ffn_up_proj': [],
            'ffn_down_proj': [],
            'ffn_skip_proj': [],
            'ffn_neurons': [],
        }

        for layer_module in self.layers:
            block = layer_module.snn_block
            ffn = layer_module.snn_ffn

            # SNNBlock 参数
            groups['W_in'].append(block.W_in.weight)
            groups['W_beta'].extend([block.W_beta_x.weight])
            groups['W_alpha'].extend([block.W_alpha_x.weight])
            groups['W_th'].extend([block.W_th_x.weight])
            groups['W_gate'].append(block.W_gate.weight)
            groups['W_skip'].append(block.W_skip.weight)
            groups['W_out'].append(block.W_out.weight)
            # v7: W_V 已移除
            groups['b_beta'].append(block.b_beta)
            groups['b_alpha'].append(block.b_alpha)
            groups['b_th'].append(block.b_th)
            groups['block_output_neuron'].append(block.output_neuron.w)

            # SNNFFN 参数
            groups['ffn_gate_proj'].append(ffn.gate_proj.weight)
            groups['ffn_up_proj'].append(ffn.up_proj.weight)
            groups['ffn_down_proj'].append(ffn.down_proj.weight)
            groups['ffn_skip_proj'].append(ffn.skip_proj.weight)
            groups['ffn_neurons'].extend([
                ffn.gate_neuron.w,
                ffn.up_neuron.w,
                ffn.output_neuron.w,
            ])

        return groups
