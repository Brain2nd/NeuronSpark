"""
SNNFFN: SNN 等价的 Feed-Forward Network

对标 Qwen3MLP 的 SwiGLU 结构：
  Qwen3 MLP:  down_proj( SiLU(gate_proj(x)) * up_proj(x) )
  SNN  FFN:   down_proj( gate_V_post * up_V_post ) + skip

膜电位门控（对标 SiLU gating）：
  gate/up 神经元完整 PLIF 动力学（积分+阈值+重置），
  输出膜电位 V_post 做连续乘法门控，替代 binary AND 门。

信号流：
  x → gate_proj → gate_neuron → V_post_gate
  x → up_proj   → up_neuron   → V_post_up
                   V_post_gate × V_post_up → gated
                   down_proj(gated) + skip_proj(x) → 连续输出
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, surrogate

from .plif_node import PLIFNode
from .parallel_scan import plif_rowparam_forward


class SNNFFN(base.MemoryModule):
    """
    SNN 等价的 Feed-Forward Network。

    Args:
        D: 可见维度（输入/输出 spike 维度）
        D_ff: 中间层维度（对标 Qwen3 intermediate_size）
        output_v_threshold: 输出神经元阈值
        num_layers: 总层数，用于 down_proj 缩放
        layer_idx: 当前层索引
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        D: int,
        D_ff: int,
        output_v_threshold: float = 0.3,
        num_layers: int = 1,
        layer_idx: int = 0,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.D = D
        self.D_ff = D_ff

        # ====== 三条投影路径（对标 SwiGLU: gate_proj, up_proj, down_proj） ======
        self.gate_proj = layer.Linear(D, D_ff, bias=False, step_mode='s')
        self.up_proj = layer.Linear(D, D_ff, bias=False, step_mode='s')
        self.down_proj = layer.Linear(D_ff, D, bias=False, step_mode='s')

        # ====== 残差路径 ======
        self.skip_proj = layer.Linear(D, D, bias=False, step_mode='s')

        # ====== 神经元（D 维或 D_ff 维可学习 β 和 V_th） ======
        # gate_neuron: 门控发放
        self.gate_neuron = PLIFNode(
            dim=D_ff,
            init_tau=2.0,
            v_threshold=output_v_threshold,
            surrogate_function=surrogate_function,
        )
        # up_neuron: 值发放
        self.up_neuron = PLIFNode(
            dim=D_ff,
            init_tau=2.0,
            v_threshold=output_v_threshold,
            surrogate_function=surrogate_function,
        )
        # ====== 参数初始化 ======
        self._initialize_parameters(num_layers)

    def _initialize_parameters(self, num_layers: int):
        """初始化投影权重。

        - gate_proj, up_proj, skip_proj: Kaiming uniform
        - down_proj: Kaiming uniform × 1/√(num_layers)，防深层梯度爆炸
        """
        for lin in [self.gate_proj, self.up_proj, self.skip_proj]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        self.down_proj.weight.data.mul_(1.0 / math.sqrt(num_layers))

    def forward_parallel(self, spike_in_seq: torch.Tensor) -> torch.Tensor:
        """
        并行前向传播：使用 parallel scan 处理全序列。

        v7.4 优化：
          - gate_proj + up_proj 合并为单次 matmul（2 launch → 1）
          - gate + up PLIF scan: row-param kernel（无需 expand+contiguous beta/v_th）
          - u_merged: 向量缩放替代 cat（1次 broadcast multiply 替代 2次 scale + 1次 cat）

        Args:
            spike_in_seq: (TK, batch, D) — 全部 T×K 帧的输入 spike

        Returns:
            continuous_out: (TK, batch, D) — 全部 T×K 帧的连续输出
        """
        TK, batch, D = spike_in_seq.shape
        D_ff = self.D_ff
        flat = spike_in_seq.reshape(TK * batch, D)

        # ====== Phase 1: 批量投影（gate+up 合并为 1 次 matmul） ======
        W_gate_up = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
        I_gate_up = F.linear(flat, W_gate_up).reshape(TK, batch, 2 * D_ff)
        I_skip = F.linear(flat, self.skip_proj.weight).reshape(TK, batch, D)

        # ====== Phase 2: Gate+Up 合并 PLIF scan（row-param kernel） ======
        beta_gate = self.gate_neuron.beta  # (D_ff,)
        beta_up = self.up_neuron.beta      # (D_ff,)
        surr = self.gate_neuron.surrogate_function

        # u_merged: 向量缩放（D_ff 维 β 直接 cat，无需 expand）
        scale_row = torch.cat([1.0 - beta_gate, 1.0 - beta_up])  # (2*D_ff,)
        u_merged = I_gate_up * scale_row  # (TK, batch, 2*D_ff), broadcast

        # beta_row / v_th_row: (batch, 2*D_ff) — D_ff 维可学习参数
        beta_row = torch.cat([beta_gate, beta_up])  # (2*D_ff,)
        beta_row = beta_row.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()

        v_th_row = torch.cat([self.gate_neuron.v_th, self.up_neuron.v_th])  # (2*D_ff,)
        v_th_row = v_th_row.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()

        # v_init_merged: (batch, 2*D_ff)
        v_init_gate = self.gate_neuron.v
        if isinstance(v_init_gate, float):
            v_init_gate = torch.zeros(batch, D_ff, device=flat.device, dtype=flat.dtype)
        v_init_up = self.up_neuron.v
        if isinstance(v_init_up, float):
            v_init_up = torch.zeros(batch, D_ff, device=flat.device, dtype=flat.dtype)
        v_init_merged = torch.cat([v_init_gate, v_init_up], dim=-1)

        # Row-param PLIF scan: beta/v_th 从寄存器读取，不占显存带宽
        spike_merged, V_post_merged = plif_rowparam_forward(
            beta_row, u_merged, v_th_row, v_init_merged,
            surrogate_function=surr,
        )

        # 用 V_post（膜电位）作为激活值，非 spike
        gate_v = V_post_merged[:, :, :D_ff]
        up_v = V_post_merged[:, :, D_ff:]
        self.gate_neuron.v = V_post_merged[-1, :, :D_ff].detach()
        self.up_neuron.v = V_post_merged[-1, :, D_ff:].detach()

        # ====== Phase 3: 连续门控（V_post × V_post，对标 SwiGLU）+ 降维 ======
        gated = gate_v * up_v  # (TK, batch, D_ff)
        gated_flat = gated.reshape(TK * batch, D_ff)
        I_out = F.linear(gated_flat, self.down_proj.weight).reshape(TK, batch, D) + I_skip

        # output_neuron 已移除：连续值由层级 K 帧聚合处理
        return I_out  # (TK, batch, D), 连续值

    def single_step_forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播。

        Args:
            spike_in: 二值脉冲输入, shape (batch, D), 值域 {0, 1}

        Returns:
            continuous_out: 连续输出, shape (batch, D)
        """
        # 门控路径 — V_post 膜电位激活
        _ = self.gate_neuron(self.gate_proj(spike_in))
        gate_v = self.gate_neuron.v  # V_post

        # 值路径 — V_post 膜电位激活
        _ = self.up_neuron(self.up_proj(spike_in))
        up_v = self.up_neuron.v  # V_post

        # 连续门控（对标 SwiGLU）
        gated = gate_v * up_v

        # 降维 + 残差
        I_out = self.down_proj(gated) + self.skip_proj(spike_in)  # R^D
        return I_out  # 连续值
