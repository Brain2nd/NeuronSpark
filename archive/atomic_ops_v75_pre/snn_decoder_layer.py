"""
SNNDecoderLayer: 单个 SNN 解码层 = 输入PLIF → SNNBlock → out_proj → 残差
                                  → 输入PLIF → SNNFFN  → out_proj → 残差

v7.5 变更：引入连续残差流（Continuous Residual Stream）
  - 层间传递连续值 h，通过完整 PLIF 神经元转换为 spike 输入 SNN 子层
  - 输入PLIF神经元 + SNN子层 + 输出投影 + 残差连接（无层内归一化）
  - PLIFNode 直接接收原始 h — 保留幅度信息，V_th 作为 SNN 原生归一化机制
  - 所有 spike 生成都经过完整 PLIF 动力学: V[t]=β·V[t-1]+(1-β)·x[t], 阈值发放, 软重置
  - 所有普通 SNN 神经元均为 PLIFNode（D 维可学习 β 和 V_th，符合设计文档 5.5）
  - 解决 20 层梯度消失: ∂h_out/∂h_in = I + ∂(out_proj·snn(plif(h)))/∂h
  - 残差增长有界: spike ∈ {0,1}, out_proj 初始化小 → h 增长可控，V_th 自适应

对标 Qwen3DecoderLayer:
  Qwen3:  LayerNorm → Attention → residual → LayerNorm → MLP → residual
  SNN:    PLIF → SNNBlock → out_proj → residual
        → PLIF → SNNFFN   → out_proj → residual
"""

import math

import torch
import torch.nn as nn
from spikingjelly.activation_based import base, surrogate

from .plif_node import PLIFNode
from .snn_block import SNNBlock
from .snn_ffn import SNNFFN
from .parallel_scan import plif_rowparam_forward


class SNNDecoderLayer(base.MemoryModule):
    """
    单个 SNN 解码层（连续残差流版本）。

    层间传递连续值 h，通过完整 PLIF 神经元（膜电位动力学 + 阈值发放 + 软重置）
    转换为 spike，输入 SNN 子层处理后，经线性投影（突触）映射回连续空间做残差连接。

    Args:
        D: 可见维度
        N: 状态扩展因子
        D_ff: FFN 中间层维度
        v_th_min: SNNBlock 动态阈值下限
        block_output_v_threshold: SNNBlock 输出神经元阈值
        ffn_output_v_threshold: SNNFFN 输出神经元阈值
        num_layers: 总层数（用于残差输出缩放 + SNNFFN down_proj 缩放）
        layer_idx: 当前层索引
    """

    def __init__(
        self,
        D: int,
        N: int,
        D_ff: int,
        v_th_min: float,
        block_output_v_threshold: float,
        ffn_output_v_threshold: float,
        num_layers: int = 1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.D = D

        self.snn_block = SNNBlock(
            D=D, N=N, v_th_min=v_th_min,
            output_v_threshold=block_output_v_threshold,
        )
        self.snn_ffn = SNNFFN(
            D=D, D_ff=D_ff,
            output_v_threshold=ffn_output_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
        )

        # 输入神经元: continuous → spike（D 维可学习 β 和 V_th）
        # PLIFNode 直接接收原始 h，V_th 作为 SNN 原生归一化机制
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

        Args:
            input_neuron: PLIFNode 实例（D 维可学习 β 和 V_th）
            x: (TK, batch, D) — 原始连续值（未归一化，保留幅度信息）

        Returns:
            spike: (TK, batch, D) — 二值 spike {0, 1}
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
        return spike

    def forward_parallel(self, h):
        """
        并行前向传播：连续残差流。

        Args:
            h: (TK, batch, D) — 连续值输入

        Returns:
            h: (TK, batch, D) — 连续值输出
        """
        # 子层 1: SNNBlock（注意力等价）— PLIFNode 直接看原始 h
        spike_in = self._input_neuron_parallel(self.input_neuron1, h)
        spike_block = self.snn_block.forward_parallel(spike_in)
        h = h + self.block_out_proj(spike_block)

        # 子层 2: SNNFFN（前馈等价）— PLIFNode 直接看原始 h
        spike_in2 = self._input_neuron_parallel(self.input_neuron2, h)
        spike_ffn = self.snn_ffn.forward_parallel(spike_in2)
        h = h + self.ffn_out_proj(spike_ffn)

        return h

    def single_step_forward(self, h):
        """
        单步前向传播：连续残差流。

        Args:
            h: (batch, D) — 连续值输入

        Returns:
            h: (batch, D) — 连续值输出
        """
        # 子层 1: SNNBlock — PLIFNode 直接看原始 h
        spike_in = self.input_neuron1(h)
        spike_block = self.snn_block(spike_in)
        h = h + self.block_out_proj(spike_block)

        # 子层 2: SNNFFN — PLIFNode 直接看原始 h
        spike_in2 = self.input_neuron2(h)
        spike_ffn = self.snn_ffn(spike_in2)
        h = h + self.ffn_out_proj(spike_ffn)

        return h
