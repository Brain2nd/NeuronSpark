"""
SNNDecoderLayer: 单个 SNN 解码层（Pre-LN 连续残差流 + K 帧聚合）

  RMSNorm → PLIF → SNNBlock → K 帧 mean → out_proj → 残差
  RMSNorm → PLIF → SNNFFN   → K 帧 mean → out_proj → 残差

v7.6 变更：K 帧层间聚合
  - SNN 子层输出 K 帧连续值（V_post 经投影），mean 聚合为 1 per token
  - 聚合后经 out_proj 投影，广播回 K 帧做残差
  - 使 β 的时间动力学通过 K 帧聚合梯度有效传播

v7.5c 变更：引入 Pre-LN 式分支归一化
  - RMSNorm 仅归一化分支输入（送入 PLIFNode 之前），残差流 h 本身不被归一化
  - 解决 h 系统性漂负 + std 爆炸：out_proj 列和训练成负 → 每层残差偏负 → 20 层累加

对标 Qwen3DecoderLayer（Pre-LN 模式完全等价）:
  Qwen3:  RMSNorm → Attention → residual → RMSNorm → MLP → residual
  SNN:    RMSNorm → PLIF → SNNBlock → K聚合 → out_proj → residual
        → RMSNorm → PLIF → SNNFFN   → K聚合 → out_proj → residual
"""

import math

import torch
import torch.nn as nn
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

        # 残差输出缩放初始化（GPT-2 style: σ = 0.02 / √(2·num_layers)）
        std = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.block_out_proj.weight, std=std)
        nn.init.normal_(self.ffn_out_proj.weight, std=std)

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

    def forward_parallel(self, h):
        """
        并行前向传播：连续残差流 + K 帧聚合。

        SNN 子层在 TK 维度处理（K 步时间动力学），输出后聚合 K 帧为 1 per token，
        经 out_proj 投影后广播回 TK 做残差。这使 β 的时间效应通过聚合梯度传播。

        Args:
            h: (TK, batch, D) — 连续值输入

        Returns:
            h: (TK, batch, D) — 连续值输出
        """
        TK, batch, D = h.shape
        K = self.K
        seq_len = TK // K

        # 子层 1: SNNBlock — RMSNorm → PLIFNode(V_post) → SNNBlock → K聚合 → out_proj → 残差
        v_in = self._input_neuron_parallel(self.input_neuron1, self.block_norm(h))
        cont_block = self.snn_block.forward_parallel(v_in)  # (TK, batch, D), 连续值

        # K 帧聚合：(TK, batch, D) → (seq_len, K, batch, D) → mean → (seq_len, batch, D)
        combined_block = cont_block.view(seq_len, K, batch, D).mean(dim=1)
        res_block = self.block_out_proj(combined_block)  # (seq_len, batch, D)
        res_block = res_block - res_block.mean(dim=-1, keepdim=True)  # 残差中心化

        # 广播回 TK：每 token 的残差复制 K 份
        h = h + res_block.repeat_interleave(K, dim=0)

        # 子层 2: SNNFFN — RMSNorm → PLIFNode(V_post) → SNNFFN → K聚合 → out_proj → 残差
        v_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h))
        cont_ffn = self.snn_ffn.forward_parallel(v_in2)  # (TK, batch, D), 连续值

        combined_ffn = cont_ffn.view(seq_len, K, batch, D).mean(dim=1)
        res_ffn = self.ffn_out_proj(combined_ffn)
        res_ffn = res_ffn - res_ffn.mean(dim=-1, keepdim=True)

        h = h + res_ffn.repeat_interleave(K, dim=0)

        return h

    def single_step_forward(self, h):
        """
        单步前向传播：连续残差流。

        注意：单步模式无法做 K 帧聚合（每步独立处理）。
        训练和推理均使用 forward_parallel（含 K 帧聚合）。
        此方法仅用于调试。

        Args:
            h: (batch, D) — 连续值输入

        Returns:
            h: (batch, D) — 连续值输出
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

        return h
