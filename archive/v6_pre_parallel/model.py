"""
SNNLanguageModel v6: SNN 隐状态空间语言模型 + FFN

架构：
  token → Embedding (trainable, D-dim)
  → encode_proj → sigmoid → K-bit 二进制编码 → K 帧 spike
  → L 个 SNNDecoderLayer（SNNBlock + SNNFFN，SPSA 训练）
  → K 帧 spike 二进制解码 → [0,1]^D
  → decode_proj → RMSNorm (trainable) → Embedding^T (tied) → logits

设计原则：
  - 小 vocab (6144) + 从零训练，所有参数可训练
  - SNNBlock 对标 Attention，SNNFFN 对标 MLP（SwiGLU → AND 门）
  - LM Head 与 Embedding 共享权重（tie_word_embeddings）
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

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
    从零训练的 SNN 隐状态空间语言模型。

    Args:
        vocab_size: 词表大小（默认 6144，自训练 BPE）
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数
        num_layers: SNN 解码层数（对齐 Qwen3 命名）
        D_ff: FFN 中间层维度（对齐 Qwen3 intermediate_size）
        v_th_min: 动态阈值下限
    """

    def __init__(
        self,
        vocab_size: int = 6144,
        D: int = 1024,
        N: int = 8,
        K: int = 8,
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
        # 阈值按深度递减：
        #   Layer 0: block_v_th=0.3, ffn_v_th=0.5
        #   Layer 1+: block_v_th=0.05, ffn_v_th=0.15
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

        self._init_weights()

    def _init_weights(self):
        """初始化所有可训练权重（从零训练）。"""
        # Embedding: normal_(0, 0.02)，参考 GPT-2 / happy-llm
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        # RMSNorm: 默认 ones 已在 __init__ 中设置
        # 编码/解码投影
        nn.init.xavier_uniform_(self.encode_proj.weight)
        nn.init.zeros_(self.encode_proj.bias)
        nn.init.xavier_uniform_(self.decode_proj.weight)
        nn.init.zeros_(self.decode_proj.bias)

    def _encode_token(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        将 embedding 编码为 K 帧二值 spike（MSB-first 二进制编码）。

        Args:
            x: token embedding, shape (batch, D)

        Returns:
            K 个 (batch, D) 的 {0,1} 张量列表
        """
        h = torch.sigmoid(self.encode_proj(x))  # (batch, D) → [0, 1]

        frames = []
        residual = h
        for k in range(self.K):
            bit = (residual >= 0.5).float()
            frames.append(bit)
            residual = (residual - bit * 0.5) * 2.0
        return frames

    def _decode_spikes(self, spike_frames: list[torch.Tensor]) -> torch.Tensor:
        """
        将 K 帧 spike 解码为 [0,1] 实数。

        Args:
            spike_frames: K 个 (batch, D) 的 {0,1} 张量列表

        Returns:
            decoded: shape (batch, D), 值域 [0, 1]
        """
        decoded = torch.zeros_like(spike_frames[0])
        for k, frame in enumerate(spike_frames):
            decoded = decoded + frame * self.bit_weights[k]
        return decoded

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        自回归前向传播（接口对齐教程 k_model.py）。

        Args:
            token_ids: (batch, seq_len) 的 token ID 序列（X = input_id[:-1]）
            target_ids: (batch, seq_len) 的目标 ID（Y = input_id[1:]）

        Returns:
            SNNModelOutput:
                若 target_ids 提供: .last_loss = (batch * seq_len,) per-token loss
                    （reduction='none', ignore_index=0，与教程一致）
                若 target_ids is None: .logits = (batch, seq_len, vocab_size)
        """
        batch, seq_len = token_ids.shape

        # 重置所有 Layer 内部的神经元状态
        for layer_module in self.layers:
            functional.reset_net(layer_module)

        if target_ids is not None:
            # 逐 token 计算 loss（避免 logits 全量驻留内存）
            all_losses = []
            for t in range(seq_len):
                logits_t = self._forward_one_token(token_ids[:, t])  # (batch, vocab)
                loss_t = F.cross_entropy(
                    logits_t, target_ids[:, t],
                    ignore_index=0, reduction='none',
                )  # (batch,)
                all_losses.append(loss_t)
                del logits_t
            # (seq_len, batch) → (batch, seq_len) → flatten → (batch * seq_len,)
            self.last_loss = torch.stack(all_losses, dim=1).reshape(-1)
            return SNNModelOutput(last_loss=self.last_loss)

        # 兼容模式：返回完整 logits（评估用）
        all_logits = []
        for t in range(seq_len):
            logits_t = self._forward_one_token(token_ids[:, t])
            all_logits.append(logits_t)
        return SNNModelOutput(logits=torch.stack(all_logits, dim=1))

    def _forward_one_token(self, token_emb_ids: torch.Tensor) -> torch.Tensor:
        """处理单个 token 的完整流程，返回 logits_t (batch, vocab_size)。"""
        # 1. Embedding
        emb = self.embed_tokens(token_emb_ids)  # (batch, D)

        # 2. 编码为 K 帧 spike
        spike_frames_in = self._encode_token(emb)

        # 3. SNN Decoder Layers 处理 K 帧
        spike_frames_out = []
        for k in range(self.K):
            spike = spike_frames_in[k]
            for layer_module in self.layers:
                spike = layer_module(spike)
            spike_frames_out.append(spike)

        # 4. 二进制解码 → [0,1]^D
        decoded = self._decode_spikes(spike_frames_out)

        # 5. 投影 → RMSNorm → tied LM Head
        h = self.decode_proj(decoded)       # (batch, D)
        h = self.norm(h)                    # (batch, D)
        logits_t = F.linear(h, self.embed_tokens.weight)  # (batch, vocab_size)
        return logits_t

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        按功能分组的可训练参数，供 IG-ZO 使用。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.weight],
            'encode': list(self.encode_proj.parameters()),
            'decode': list(self.decode_proj.parameters()),
            # SNNBlock 参数
            'W_in': [],
            'W_beta': [],
            'W_alpha': [],
            'W_th': [],
            'W_gate': [],
            'W_skip': [],
            'W_out': [],
            'W_V': [],
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
            groups['W_V'].extend([
                block.W_beta_V.weight,
                block.W_alpha_V.weight,
                block.W_th_V.weight,
            ])
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
