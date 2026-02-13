"""
SNNLanguageModel: 自回归 SNN 语言模型

架构：
  token → Embedding(vocab, D) → sigmoid → K-bit 二进制编码 → K 帧 spike
  → L 个 SNNBlock（每帧逐 Block 串行处理，step_mode='s'）
  → K 帧 spike 二进制解码 → [0,1]^D → Linear(D, vocab) → logits

时间步对齐：
  每个 token 对应 K 个 SNN 全局时间步，所有 Block 在每步同步处理。
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

from atomic_ops import SNNBlock


class SNNLanguageModel(nn.Module):
    """
    基于 SNN 隐状态空间的自回归语言模型。

    Args:
        vocab_size: 词表大小
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数
        num_blocks: SNN Block 层数
        v_th_min: 动态阈值下限
    """

    def __init__(
        self,
        vocab_size: int,
        D: int = 128,
        N: int = 8,
        K: int = 8,
        num_blocks: int = 2,
        v_th_min: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_blocks = num_blocks

        # ====== 编码层：token → 实数 → [0,1] ======
        self.embedding = nn.Embedding(vocab_size, D)
        # 投影到 [0,1] 范围，用于 K-bit 二进制编码
        self.encode_proj = nn.Linear(D, D)

        # ====== L 个 SNN Block ======
        self.blocks = nn.ModuleList([
            SNNBlock(D=D, N=N, v_th_min=v_th_min)
            for _ in range(num_blocks)
        ])

        # ====== 解码层：[0,1]^D → logits ======
        self.decode_head = nn.Linear(D, vocab_size)

        # 二进制权重：2^{-1}, 2^{-2}, ..., 2^{-K}（MSB-first）
        self.register_buffer(
            'bit_weights',
            torch.tensor([2.0 ** (-(k + 1)) for k in range(K)]),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.encode_proj.weight)
        nn.init.zeros_(self.encode_proj.bias)
        nn.init.xavier_uniform_(self.decode_head.weight)
        nn.init.zeros_(self.decode_head.bias)

    def _encode_token(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        将实数 token embedding 编码为 K 帧二值 spike。

        Args:
            x: token embedding, shape (batch, D), 实数

        Returns:
            spike_frames: K 个 (batch, D) 的 {0,1} 张量列表
        """
        # 投影到 [0, 1]
        h = torch.sigmoid(self.encode_proj(x))  # (batch, D)

        # K-bit 二进制编码（MSB-first，确定性）
        # h ∈ [0,1)，逐位提取：若 h >= 0.5 → bit=1, h -= 0.5; h *= 2; 重复
        frames = []
        residual = h
        for k in range(self.K):
            bit = (residual >= 0.5).float()  # {0, 1}
            frames.append(bit)
            residual = (residual - bit * 0.5) * 2.0
        return frames

    def _decode_spikes(self, spike_frames: list[torch.Tensor]) -> torch.Tensor:
        """
        将 K 帧 spike 二进制解码为 [0,1] 实数。

        Args:
            spike_frames: K 个 (batch, D) 的 {0,1} 张量列表

        Returns:
            decoded: shape (batch, D), 值域 [0, 1]
        """
        # ŷ_d = Σ_{k=1}^{K} spike_d[k] · 2^{-k}
        decoded = torch.zeros_like(spike_frames[0])
        for k, frame in enumerate(spike_frames):
            decoded = decoded + frame * self.bit_weights[k]
        return decoded

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        自回归前向传播。

        Args:
            token_ids: (batch, seq_len) 的 token ID 序列

        Returns:
            logits: (batch, seq_len, vocab_size) 的 logit 输出
        """
        batch, seq_len = token_ids.shape

        # 重置所有 Block 内部的神经元状态（V → 0）
        for block in self.blocks:
            functional.reset_net(block)

        all_logits = []

        for t in range(seq_len):
            # 1. Token → embedding
            emb = self.embedding(token_ids[:, t])  # (batch, D)

            # 2. 编码为 K 帧 spike
            spike_frames_in = self._encode_token(emb)

            # 3. 收集 K 帧输出 spike
            spike_frames_out = []

            for k in range(self.K):
                # 当前输入 spike 帧
                spike = spike_frames_in[k]  # (batch, D)

                # 逐 Block 串行处理（同一 SNN 时间步内）
                for block in self.blocks:
                    spike = block(spike)

                spike_frames_out.append(spike)

            # 4. 二进制解码 K 帧输出 → [0,1]^D
            decoded = self._decode_spikes(spike_frames_out)  # (batch, D)

            # 5. 输出头 → logits
            logits_t = self.decode_head(decoded)  # (batch, vocab_size)
            all_logits.append(logits_t)

        # (batch, seq_len, vocab_size)
        return torch.stack(all_logits, dim=1)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        返回按功能分组的参数，供 IG-ZO 使用。

        Returns:
            dict: 参数组名 → 参数列表
        """
        groups = {
            'embedding': [self.embedding.weight],
            'encode': list(self.encode_proj.parameters()),
            'decode': list(self.decode_head.parameters()),
            'W_in': [],
            'W_beta': [],
            'W_alpha': [],
            'W_th': [],
            'W_gate': [],
            'W_skip': [],
            'W_out': [],
            'W_V': [],      # voltage-gated N×N 矩阵
            'b_beta': [],
            'b_alpha': [],
            'b_th': [],
            'output_neuron': [],
        }

        for block in self.blocks:
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
            groups['output_neuron'].append(block.output_neuron.w)

        return groups
