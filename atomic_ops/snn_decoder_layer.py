"""
SNNDecoderLayer: 单个 SNN 解码层（Pre-LN 连续残差流 + 位权脉冲编码）

  RMSNorm → PLIF → SNNBlock → spike × bit_weight → out_proj → 残差
  RMSNorm → PLIF → SNNFFN   → spike × bit_weight → out_proj → 残差

v7.6 变更：
  1. 位权脉冲编码（Bit-Weighted Spike Coding, MSB-first）
     - 每个 SNN 时间步 k 的 spike 乘以位权 w_k = 2^{K-1-k}（归一化后 Σw_k=K）
     - K 步 binary spike train 等价于 K-bit 二进制整数（65536 级精度）
  2. 可学习初始膜电位（v_init）
  3. 自适应 Surrogate Gradient

v7.5c 变更：引入 Pre-LN 式分支归一化
  - RMSNorm 仅归一化分支输入（送入 PLIFNode 之前），残差流 h 本身不被归一化
  - RMSNorm 控制分支输入 scale → PLIFNode V_th 恢复意义 → 打破漂移正反馈
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


# ====== Fused residual + mean-centering (torch.compile → single kernel) ======

@torch.compile(backend='inductor', fullgraph=True)
def _fused_residual_center(h: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """h + res - mean(res, dim=-1): 3 element-wise ops → 1 fused kernel."""
    return h + res - res.mean(dim=-1, keepdim=True)


class SNNDecoderLayer(base.MemoryModule):
    """
    单个 SNN 解码层（连续残差流 + 位权脉冲编码）。

    层间传递连续值 h，通过完整 PLIF 神经元（膜电位动力学 + 阈值发放 + 软重置）
    转换为 spike，输入 SNN 子层处理后，经位权加权 + 线性投影（突触）映射回连续空间做残差连接。

    位权脉冲编码（v7.6）：
      每个时间步 k 的 spike 乘以位权 w_k = 2^{K-1-k}（MSB-first，归一化 Σw_k=K）。
      K 步 binary spike → K-bit 二进制整数，信息容量从 ~11.5 bits 提升到 K bits。
      位权仅影响子层输出 → out_proj 的信号权重，层内 PLIF 递推不变。

    Args:
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数
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
        K: int,
        D_ff: int,
        v_th_min: float,
        block_output_v_threshold: float,
        ffn_output_v_threshold: float,
        num_layers: int = 1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.D = D
        self.K = K

        # ====== 位权脉冲编码（MSB-first，归一化 Σw_k = K） ======
        # w_k = 2^{K-1-k}: 步骤 0 = MSB（权重最大），步骤 K-1 = LSB（权重最小）
        # 归一化使总权重 = K，保持残差流信号量与均匀权重一致
        raw_weights = 2.0 ** torch.arange(K - 1, -1, -1, dtype=torch.float32)
        bit_weights = raw_weights * (K / raw_weights.sum())
        self.register_buffer('bit_weights', bit_weights)  # (K,)
        # 单步模式步数计数器（functional.reset_net 时自动重置为 0）
        self.register_memory('k_step', 0)

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

        # Pre-LN 分支归一化: h → RMSNorm → PLIFNode
        self.block_norm = RMSNorm(D)
        self.ffn_norm = RMSNorm(D)

        # 输入神经元: RMSNorm(h) → spike（D 维可学习 β 和 V_th）
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

    def _get_bit_weights_expanded(self, TK: int) -> torch.Tensor:
        """获取预计算的位权张量 (TK, 1, 1)，缓存避免重复 repeat。"""
        if not hasattr(self, '_bw_cache') or self._bw_cache.shape[0] != TK:
            self._bw_cache = self.bit_weights.repeat(TK // self.K).view(TK, 1, 1)
        return self._bw_cache

    def _apply_bit_weights(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """位权脉冲编码：给每个时间步的 spike 乘以 MSB-first 位权。

        位权使 K 步 binary spike train 等价于 K-bit 二进制整数：
          步骤 0 (MSB): spike × w_0 ≈ 8.0   ← 最高位（粗糙轮廓）
          步骤 K-1 (LSB): spike × w_{K-1} ≈ 0.0002  ← 最低位（精细化）

        层内 PLIF 递推不受影响（位权仅作用于子层输出 → out_proj 路径）。

        Args:
            spike_seq: (TK, batch, D) — 子层输出的 binary spike {0, 1}

        Returns:
            weighted: (TK, batch, D) — 位权加权后的 spike, 值域 {0, w_k}
        """
        bw = self._get_bit_weights_expanded(spike_seq.shape[0])
        return spike_seq * bw

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
            v_init = input_neuron.expand_v_init(batch, x.device, x.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = input_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=input_neuron.get_surrogate(u),
        )

        input_neuron.v = V_post[-1].detach()
        return spike

    def forward_parallel(self, h):
        """
        并行前向传播：连续残差流 + 位权脉冲编码。

        Args:
            h: (TK, batch, D) — 连续值输入

        Returns:
            h: (TK, batch, D) — 连续值输出
        """
        # 子层 1: SNNBlock — RMSNorm → PLIF → SNNBlock → spike×bit_weight → out_proj → 残差
        spike_in = self._input_neuron_parallel(self.input_neuron1, self.block_norm(h))
        spike_block = self.snn_block.forward_parallel(spike_in)
        spike_block = self._apply_bit_weights(spike_block)
        res_block = self.block_out_proj(spike_block)
        h = _fused_residual_center(h, res_block)

        # 子层 2: SNNFFN — RMSNorm → PLIF → SNNFFN → spike×bit_weight → out_proj → 残差
        spike_in2 = self._input_neuron_parallel(self.input_neuron2, self.ffn_norm(h))
        spike_ffn = self.snn_ffn.forward_parallel(spike_in2)
        spike_ffn = self._apply_bit_weights(spike_ffn)
        res_ffn = self.ffn_out_proj(spike_ffn)
        h = _fused_residual_center(h, res_ffn)

        return h

    def single_step_forward(self, h):
        """
        单步前向传播：连续残差流 + 位权脉冲编码。

        通过 k_step 计数器追踪 token 内步数，自动应用对应位权。

        Args:
            h: (batch, D) — 连续值输入

        Returns:
            h: (batch, D) — 连续值输出
        """
        # 当前步的位权
        bw = self.bit_weights[self.k_step]  # scalar

        # 子层 1: SNNBlock — RMSNorm → PLIFNode → SNNBlock → spike×bit_weight → out_proj → 残差
        spike_in = self.input_neuron1(self.block_norm(h))
        spike_block = self.snn_block.single_step_forward(spike_in)
        spike_block = spike_block * bw
        res_block = self.block_out_proj(spike_block)
        h = _fused_residual_center(h, res_block)

        # 子层 2: SNNFFN — RMSNorm → PLIFNode → SNNFFN → spike×bit_weight → out_proj → 残差
        spike_in2 = self.input_neuron2(self.ffn_norm(h))
        spike_ffn = self.snn_ffn.single_step_forward(spike_in2)
        spike_ffn = spike_ffn * bw
        res_ffn = self.ffn_out_proj(spike_ffn)
        h = _fused_residual_center(h, res_ffn)

        # 步数计数器递增（每 K 步自动回零）
        self.k_step = (self.k_step + 1) % self.K

        return h
