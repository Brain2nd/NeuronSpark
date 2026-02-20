"""
SNNDecoderLayer: 单个 SNN 解码层（Pre-LN 连续残差流 + 动态 K 帧聚合）

  RMSNorm → PLIF → SNNBlock → 动态K聚合 → out_proj → 残差
  RMSNorm → PLIF → SNNFFN   → 动态K聚合 → out_proj → 残差

v8.0 动态 K（对标 Mamba Δ 选择性机制）：
  - 每个 token 的 K 帧 SNN 输出，学习自适应停止概率 p_halt
  - PonderNet 几何分布加权：λ_k = p_k · ∏_{j<k}(1-p_j)，归一化后加权聚合
  - 不同 token 有效步数不同：简单 token 早停（E[K]小），复杂 token 用满步数
  - ponder_cost 正则化：鼓励用更少步数完成简单 token 的处理

  数学推导：
    停止概率: p_k = σ(halt_proj(frame_k))         ∈ (0,1)
    生存概率: S_k = ∏_{j=1}^{k-1} (1 - p_j)       — 到第 k 步还没停
    权重:     λ_k = p_k · S_k                       — 恰好在第 k 步停止的概率
    归一化:   λ̂_k = λ_k / Σ_k λ_k                   — 确保权重和为 1
    聚合:     output = Σ_k λ̂_k · frame_k
    代价:     E[K] = Σ_k k · λ̂_k                    — 期望步数

v7.6 变更：K 帧层间聚合
  - SNN 子层输出 K 帧连续值（V_post 经投影），mean 聚合为 1 per token
  - 聚合后经 out_proj 投影，广播回 K 帧做残差
  - 使 β 的时间动力学通过 K 帧聚合梯度有效传播

对标 Qwen3DecoderLayer（Pre-LN 模式完全等价）:
  Qwen3:  RMSNorm → Attention → residual → RMSNorm → MLP → residual
  SNN:    RMSNorm → PLIF → SNNBlock → 动态K聚合 → out_proj → residual
        → RMSNorm → PLIF → SNNFFN   → 动态K聚合 → out_proj → residual
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, surrogate

from .plif_node import PLIFNode
from .rms_norm import RMSNorm
from .snn_block import SNNBlock
from .snn_ffn import SNNFFN
from .parallel_scan import plif_rowparam_forward


class SNNDecoderLayer(base.MemoryModule):
    """
    单个 SNN 解码层（连续残差流 + K 帧聚合版本）。

    层间传递连续值 h (TK, batch, D)，通过 PLIF 神经元转换为 spike，
    输入 SNN 子层处理后，K 帧聚合为 1 per token，经 out_proj 投影，
    广播回 K 帧做残差连接。

    K 帧聚合使 β 的时间动力学（控制 K 步内的膜电位演化）产生可微分的
    token 级效应，解决 β 梯度为纯噪声的问题。

    Args:
        D: 可见维度
        N: 状态扩展因子
        D_ff: FFN 中间层维度
        v_th_min: SNNBlock 动态阈值下限
        ffn_v_threshold: SNNFFN gate/up 神经元阈值
        K: 每 token 的 SNN 时间步数
        num_layers: 总层数（用于残差输出缩放 + SNNFFN down_proj 缩放）
        layer_idx: 当前层索引
    """

    def __init__(
        self,
        D: int,
        N: int,
        D_ff: int,
        v_th_min: float,
        ffn_v_threshold: float,
        K: int = 16,
        num_layers: int = 1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.D = D
        self.K = K

        self.snn_block = SNNBlock(
            D=D, N=N, v_th_min=v_th_min,
        )
        self.snn_ffn = SNNFFN(
            D=D, D_ff=D_ff,
            output_v_threshold=ffn_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
        )

        # Pre-LN 分支归一化: h → RMSNorm → PLIFNode
        self.block_norm = RMSNorm(D)
        self.ffn_norm = RMSNorm(D)

        # 输入神经元: RMSNorm(h) → V_post 膜电位激活（D 维可学习 β 和 V_th）
        self.input_neuron1 = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )
        self.input_neuron2 = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.5,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # 输出投影（突触）: spike (D) → 连续空间 (D)
        self.block_out_proj = nn.Linear(D, D, bias=False)
        self.ffn_out_proj = nn.Linear(D, D, bias=False)

        # ====== 动态 K: 停止投影（突触: SNN 输出 → 停止概率） ======
        # halt_proj: D → 1，每步每 token 产生一个停止 logit
        # PonderNet 几何分布加权，替代 uniform mean 聚合
        self.block_halt = nn.Linear(D, 1, bias=True)
        self.ffn_halt = nn.Linear(D, 1, bias=True)

        # 残差输出缩放初始化（GPT-2 style: σ = 0.02 / √(2·num_layers)）
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.block_out_proj.weight, std=std)
        nn.init.normal_(self.ffn_out_proj.weight, std=std)

        # halt 初始化: 小权重 + 负偏置 → p_halt ≈ 0.03 → 接近 uniform 聚合
        # σ(-3.5) ≈ 0.029, 几何分布归一化后 λ_1/λ_K ≈ 1.5, 接近均匀
        for halt in [self.block_halt, self.ffn_halt]:
            nn.init.xavier_uniform_(halt.weight)
            halt.weight.data.mul_(0.01)
            nn.init.constant_(halt.bias, -3.5)

    def _input_neuron_parallel(self, input_neuron, x):
        """
        输入 PLIF 神经元的 parallel scan 前向传播。

        完整 PLIF 动力学: V[t] = β·V[t-1] + (1-β)·x[t], spike = Θ(V-V_th), 软重置。
        输出膜电位 V_post 作为激活值（保留完整 SNN 动力学，但传递连续信号）。

        Args:
            input_neuron: PLIFNode 实例（D 维可学习 β 和 V_th）
            x: (TK, batch, D) — 连续值输入

        Returns:
            V_post: (TK, batch, D) — 膜电位（连续激活值）
        """
        TK, batch, D = x.shape

        beta = input_neuron.beta  # (D,)
        u = (1.0 - beta) * x  # (D,) broadcast → (TK, batch, D)

        v_init = input_neuron.v
        if isinstance(v_init, float):
            v_init = torch.zeros(batch, D, device=x.device, dtype=x.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = input_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=input_neuron.surrogate_function,
        )

        input_neuron.v = V_post[-1].detach()
        return V_post  # 膜电位作为激活值

    def _adaptive_aggregate(self, frames, halt_proj):
        """
        PonderNet 式自适应 K 帧聚合（动态 K 核心）。

        每步计算停止概率 p_k，用几何分布权重加权聚合，
        使不同 token 有不同的有效步数。

        数学:
          p_k = σ(halt_proj(frame_k))                 — 停止概率
          S_k = ∏_{j<k} (1-p_j)                       — 生存概率
          λ_k = p_k · S_k                             — 几何分布权重
          λ̂_k = λ_k / Σ λ_k                           — 归一化
          output = Σ λ̂_k · frame_k                    — 加权聚合
          E[K] = Σ k · λ̂_k                            — 期望步数（ponder cost）

        Args:
            frames: (seq_len, K, batch, D) — SNN 子层 K 帧输出
            halt_proj: nn.Linear(D, 1)    — 停止投影（突触）

        Returns:
            aggregated: (seq_len, batch, D) — 加权聚合结果
            ponder_cost: scalar             — 期望步数均值（正则化用）
        """
        seq_len, K, batch, D = frames.shape

        # ====== 1. 停止概率 ======
        halt_logits = halt_proj(frames).squeeze(-1)  # (seq_len, K, batch)
        p_halt = torch.sigmoid(halt_logits)           # (seq_len, K, batch) ∈ (0,1)
        # 数值安全: 防止 log(0)
        p_halt = p_halt.clamp(min=1e-7, max=1.0 - 1e-7)

        # ====== 2. 几何分布权重 ======
        # 生存概率 S_k = ∏_{j<k}(1-p_j), 用 log-space 累加
        log_1_minus_p = torch.log1p(-p_halt)          # (seq_len, K, batch)
        # exclusive cumsum: S_1=1, S_2=1-p_1, S_3=(1-p_1)(1-p_2), ...
        log_survive = torch.cumsum(log_1_minus_p, dim=1)
        log_survive = torch.cat([
            torch.zeros(seq_len, 1, batch, device=frames.device, dtype=frames.dtype),
            log_survive[:, :-1, :],
        ], dim=1)                                     # (seq_len, K, batch)
        survive = torch.exp(log_survive)              # (seq_len, K, batch)

        # λ_k = p_k · S_k
        halt_weights = p_halt * survive               # (seq_len, K, batch)

        # 归一化: Σ λ̂_k = 1
        halt_weights = halt_weights / (halt_weights.sum(dim=1, keepdim=True) + 1e-8)

        # ====== 3. 加权聚合 ======
        # (seq_len, K, batch, 1) × (seq_len, K, batch, D) → sum → (seq_len, batch, D)
        aggregated = (frames * halt_weights.unsqueeze(-1)).sum(dim=1)

        # ====== 4. Ponder cost: E[K] ======
        steps = torch.arange(1, K + 1, device=frames.device, dtype=frames.dtype)
        expected_k = (halt_weights * steps[None, :, None]).sum(dim=1)  # (seq_len, batch)
        ponder_cost = expected_k.mean()               # scalar

        return aggregated, ponder_cost

    def forward_parallel(self, h):
        """
        并行前向传播：连续残差流 + 动态 K 帧聚合。

        SNN 子层在 TK 维度处理（K 步时间动力学），输出后用 PonderNet
        自适应聚合 K 帧（不同 token 有效步数不同），经 out_proj 投影后
        广播回 TK 做残差。

        Args:
            h: (TK, batch, D) — 连续值输入

        Returns:
            h: (TK, batch, D) — 连续值输出
            ponder_cost: scalar — 两个子层的平均期望步数（正则化用）
        """
        TK, batch, D = h.shape
        K = self.K
        seq_len = TK // K

        # 子层 1: SNNBlock — RMSNorm → PLIFNode(V_post) → SNNBlock → 动态K聚合 → out_proj → 残差
        v_in = self._input_neuron_parallel(self.input_neuron1, self.block_norm(h))
        cont_block = self.snn_block.forward_parallel(v_in)  # (TK, batch, D), 连续值

        # 动态 K 帧聚合（PonderNet）: (TK, batch, D) → (seq_len, K, batch, D) → 加权 → (seq_len, batch, D)
        frames_block = cont_block.view(seq_len, K, batch, D)
        combined_block, pc_block = self._adaptive_aggregate(frames_block, self.block_halt)
        res_block = self.block_out_proj(combined_block)  # (seq_len, batch, D)
        res_block = res_block - res_block.mean(dim=-1, keepdim=True)  # 残差中心化

        # 广播回 TK：每 token 的残差复制 K 份
        h = h + res_block.repeat_interleave(K, dim=0)

        # 子层 2: SNNFFN — RMSNorm → PLIFNode(V_post) → SNNFFN → 动态K聚合 → out_proj → 残差
        v_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h))
        cont_ffn = self.snn_ffn.forward_parallel(v_in2)  # (TK, batch, D), 连续值

        frames_ffn = cont_ffn.view(seq_len, K, batch, D)
        combined_ffn, pc_ffn = self._adaptive_aggregate(frames_ffn, self.ffn_halt)
        res_ffn = self.ffn_out_proj(combined_ffn)
        res_ffn = res_ffn - res_ffn.mean(dim=-1, keepdim=True)

        h = h + res_ffn.repeat_interleave(K, dim=0)

        ponder_cost = (pc_block + pc_ffn) / 2.0  # 两个子层平均

        return h, ponder_cost

    def single_step_forward(self, h):
        """
        单步前向传播：连续残差流。

        注意：单步模式无法做动态 K 聚合（每步独立处理）。
        训练和推理均使用 forward_parallel（含动态 K 聚合）。
        此方法仅用于调试。

        Args:
            h: (batch, D) — 连续值输入

        Returns:
            h: (batch, D) — 连续值输出
            ponder_cost: scalar — 0.0（单步无 ponder cost）
        """
        # 子层 1: SNNBlock — RMSNorm → PLIFNode(V_post) → SNNBlock → out_proj → 残差
        _ = self.input_neuron1(self.block_norm(h))  # 触发 PLIF 动力学，更新 .v
        v_in = self.input_neuron1.v                  # V_post 膜电位作为激活值
        cont_block = self.snn_block.single_step_forward(v_in)
        res_block = self.block_out_proj(cont_block)
        h = h + res_block - res_block.mean(dim=-1, keepdim=True)

        # 子层 2: SNNFFN — RMSNorm → PLIFNode(V_post) → SNNFFN → out_proj → 残差
        _ = self.input_neuron2(self.ffn_norm(h))
        v_in2 = self.input_neuron2.v
        cont_ffn = self.snn_ffn.single_step_forward(v_in2)
        res_ffn = self.ffn_out_proj(cont_ffn)
        h = h + res_ffn - res_ffn.mean(dim=-1, keepdim=True)

        return h, torch.tensor(0.0, device=h.device)
