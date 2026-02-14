"""
SNNLanguageModel v2: Qwen3 预训练语义层 + SNN 隐状态空间

架构：
  token → Qwen3 Embedding (frozen, 1024-dim)
  → encode_proj → sigmoid → K-bit 二进制编码 → K 帧 spike
  → L 个 SNNBlock（SPSA 训练，每帧逐 Block 串行处理）
  → K 帧 spike 二进制解码 → [0,1]^D
  → decode_proj → RMSNorm (frozen) → Embedding^T (frozen) → logits

设计原则：
  - 预训练层（Embedding + RMSNorm）提供成熟的语义空间，冻结不训练
  - SNN blocks 在语义空间中学习序列建模（唯一 SPSA 训练的部分）
  - LM Head 与 Embedding 共享权重（tie_word_embeddings）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from atomic_ops import SNNBlock


class RMSNorm(nn.Module):
    """RMSNorm（与 Qwen3 兼容）。"""

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
    基于 Qwen3 预训练语义层 + SNN 隐状态空间的语言模型。

    Args:
        vocab_size: 词表大小（Qwen3: 151936）
        D: 可见维度（Qwen3 hidden_size: 1024）
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数
        num_blocks: SNN Block 层数
        v_th_min: 动态阈值下限
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        D: int = 1024,
        N: int = 8,
        K: int = 8,
        num_blocks: int = 10,
        v_th_min: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_blocks = num_blocks

        # ====== 预训练冻结层 ======
        self.embed_tokens = nn.Embedding(vocab_size, D)
        self.norm = RMSNorm(D)

        # ====== 可训练编码/解码投影 ======
        self.encode_proj = nn.Linear(D, D)
        self.decode_proj = nn.Linear(D, D)

        # ====== SNN Blocks ======
        # 输出阈值按深度递减：
        #   Block 0: ~50% input, σ(I)≈0.42 → v_th=0.3 (~20% output)
        #   Block 1: input firing rate 急剧衰减 → v_th=0.05
        #   Block 2+: 信号稳定在 ~25-40% → v_th=0.05 维持传播
        output_v_thresholds = [0.3] + [0.05] * (num_blocks - 1)
        self.blocks = nn.ModuleList([
            SNNBlock(D=D, N=N, v_th_min=v_th_min,
                     output_v_threshold=output_v_thresholds[i])
            for i in range(num_blocks)
        ])

        # ====== K-bit 二进制权重 ======
        self.register_buffer(
            'bit_weights',
            torch.tensor([2.0 ** (-(k + 1)) for k in range(K)]),
        )

        self._init_trainable_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str = "Qwen/Qwen3-0.6B",
        **kwargs,
    ):
        """从预训练模型加载 Embedding + RMSNorm 并冻结。

        Args:
            pretrained_model_name: HuggingFace 模型名称
            **kwargs: 传递给 __init__ 的其他参数（N, K, num_blocks 等）
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"Loading pretrained weights from {pretrained_model_name}...")
        config = AutoConfig.from_pretrained(pretrained_model_name)
        pretrained = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            dtype=torch.float32,
        )

        # 提取权重
        embed_weight = pretrained.model.embed_tokens.weight.data.clone()
        norm_weight = pretrained.model.norm.weight.data.clone()
        vocab_size = config.vocab_size
        D = config.hidden_size

        # 释放预训练模型
        del pretrained
        torch.cuda.empty_cache()

        print(f"  vocab_size={vocab_size}, D={D}")

        # 构建模型
        model = cls(vocab_size=vocab_size, D=D, **kwargs)

        # 复制预训练权重
        model.embed_tokens.weight.data.copy_(embed_weight)
        model.norm.weight.data.copy_(norm_weight)

        # 冻结预训练层
        model.embed_tokens.weight.requires_grad = False
        model.norm.weight.requires_grad = False

        print("  Pretrained layers loaded and frozen.")
        return model

    def _init_trainable_weights(self):
        """初始化可训练的编码/解码投影层。"""
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
    ) -> torch.Tensor:
        """
        自回归前向传播。

        Args:
            token_ids: (batch, seq_len) 的 token ID 序列
            target_ids: (batch, seq_len) 的目标 ID。若提供，直接返回 scalar loss，
                        避免在内存中同时存在 (batch, seq_len, vocab_size) 张量。

        Returns:
            若 target_ids is None: logits (batch, seq_len, vocab_size)
            若 target_ids 提供:    scalar loss（交叉熵均值）
        """
        batch, seq_len = token_ids.shape

        # 重置所有 Block 内部的神经元状态
        for block in self.blocks:
            functional.reset_net(block)

        # 逐 token loss 累积模式：避免 logits 全量驻留
        if target_ids is not None:
            total_loss = 0.0
            for t in range(seq_len):
                logits_t = self._forward_one_token(token_ids[:, t])
                loss_t = F.cross_entropy(logits_t, target_ids[:, t])
                total_loss = total_loss + loss_t
                del logits_t, loss_t
            return total_loss / seq_len

        # 兼容模式：返回完整 logits（评估用）
        all_logits = []
        for t in range(seq_len):
            logits_t = self._forward_one_token(token_ids[:, t])
            all_logits.append(logits_t)
        return torch.stack(all_logits, dim=1)

    def _forward_one_token(self, token_emb_ids: torch.Tensor) -> torch.Tensor:
        """处理单个 token 的完整流程，返回 logits_t (batch, vocab_size)。"""
        # 1. 预训练 Embedding（frozen）
        emb = self.embed_tokens(token_emb_ids)  # (batch, D)

        # 2. 编码为 K 帧 spike
        spike_frames_in = self._encode_token(emb)

        # 3. SNN Blocks 处理 K 帧
        spike_frames_out = []
        for k in range(self.K):
            spike = spike_frames_in[k]
            for block in self.blocks:
                spike = block(spike)
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
        不包含冻结的预训练层。
        """
        groups = {
            'encode': list(self.encode_proj.parameters()),
            'decode': list(self.decode_proj.parameters()),
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
