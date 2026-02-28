"""
SNNFFN: SNN 等价的 Feed-Forward Network（v10: +SEW 残差）

对标 Qwen3MLP 的 SwiGLU 结构：
  Qwen3 MLP:  down_proj( SiLU(gate_proj(x)) * up_proj(x) )
  SNN  FFN:   down_proj( gate_spike AND up_spike ) + skip → output_neuron → SEW

SiLU gating → spike 巧合检测（AND 门）：
  只有当 gate 和 up 通道同时发放时，信号才传递，实现非线性选择。

SEW 残差（v10）：
  spike_out = output_neuron_spike + spike_in
  提供 identity mapping 保证深度梯度流。

信号流：
  spike_in (p~20%) → gate_proj → gate_neuron → gate_spike (p~30%)
  spike_in (p~20%) → up_proj   → up_neuron   → up_spike (p~30%)
                      gate_spike AND up_spike → gated (p~9%)
                      down_proj(gated) + skip_proj(spike_in) → output_neuron → +spike_in → spike_out
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, surrogate

from .plif_node import PLIFNode
from .parallel_scan import plif_fixed_param_forward, plif_parallel_forward, plif_rowparam_forward_recompute
from .fp16_codec import binary_residual


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
        add_residual: bool = True,
    ):
        super().__init__()
        self.D = D
        self.D_ff = D_ff
        self.add_residual = add_residual

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
        # output_neuron: 输出发放
        self.output_neuron = PLIFNode(
            dim=D,
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
            spike_out_seq: (TK, batch, D) — 全部 T×K 帧的输出 spike
        """
        TK, batch, D = spike_in_seq.shape
        D_ff = self.D_ff
        flat = spike_in_seq.reshape(TK * batch, D)

        # ====== Phase 1: 批量投影（gate+up+skip 合并为 1 次 GEMM: 3 launch → 1） ======
        W_all = torch.cat([
            self.gate_proj.weight, self.up_proj.weight, self.skip_proj.weight,
        ], dim=0)  # (2*D_ff + D, D)
        proj_all = F.linear(flat, W_all)  # (TK*B, 2*D_ff + D)
        I_gate_up = proj_all[:, :2 * D_ff].reshape(TK, batch, 2 * D_ff)
        I_skip = proj_all[:, 2 * D_ff:].reshape(TK, batch, D)

        # ====== Phase 2: Gate+Up 合并 PLIF scan（row-param kernel） ======
        beta_gate = self.gate_neuron.beta  # (D_ff,)
        beta_up = self.up_neuron.beta      # (D_ff,)

        # u_merged: 向量缩放（D_ff 维 β 直接 cat，无需 expand）
        scale_row = torch.cat([1.0 - beta_gate, 1.0 - beta_up])  # (2*D_ff,)
        u_merged = I_gate_up * scale_row  # (TK, batch, 2*D_ff), broadcast

        # beta_row / v_th_row: (batch, 2*D_ff) — D_ff 维可学习参数
        beta_row = torch.cat([beta_gate, beta_up])  # (2*D_ff,)
        beta_row = beta_row.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()

        v_th_row = torch.cat([self.gate_neuron.v_th, self.up_neuron.v_th])  # (2*D_ff,)
        v_th_row = v_th_row.unsqueeze(0).expand(batch, 2 * D_ff).contiguous()

        # v_init_merged: (batch, 2*D_ff) — 可学习初始膜电位
        v_init_gate = self.gate_neuron.v
        if isinstance(v_init_gate, float):
            v_init_gate = self.gate_neuron.expand_v_init(batch, flat.device, flat.dtype)
        v_init_up = self.up_neuron.v
        if isinstance(v_init_up, float):
            v_init_up = self.up_neuron.expand_v_init(batch, flat.device, flat.dtype)
        v_init_merged = torch.cat([v_init_gate, v_init_up], dim=-1)

        # Row-param PLIF scan (recompute): beta/v_th 从寄存器读取，不占显存带宽
        # 自适应 surrogate: 根据 u_merged 的分布动态调整 alpha
        alpha = float(self.gate_neuron.get_surrogate(u_merged).alpha)
        spike_merged, V_last_merged = plif_rowparam_forward_recompute(
            beta_row, u_merged, v_th_row, v_init_merged, alpha,
        )

        gate_spike = spike_merged[:, :, :D_ff]
        up_spike = spike_merged[:, :, D_ff:]
        self.gate_neuron.v = V_last_merged[:, :D_ff].detach()
        self.up_neuron.v = V_last_merged[:, D_ff:].detach()

        # ====== Phase 3: AND 门 + 降维 ======
        gated = gate_spike * up_spike  # (TK, batch, D_ff)
        gated_flat = gated.reshape(TK * batch, D_ff)
        I_out = F.linear(gated_flat, self.down_proj.weight).reshape(TK, batch, D) + I_skip

        # ====== Phase 4: 输出神经元 parallel scan（D 维可学习 β 和 V_th） ======
        beta_out = self.output_neuron.beta  # (D,)
        v_init_out = self.output_neuron.v
        if isinstance(v_init_out, float):
            v_init_out = self.output_neuron.expand_v_init(batch, flat.device, flat.dtype)
        u_out = (1.0 - beta_out) * I_out  # (D,) broadcast → (TK, batch, D)

        beta_out_row = beta_out.unsqueeze(0).expand(batch, D).contiguous()
        v_th_out_row = self.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        alpha_out = float(self.output_neuron.get_surrogate(u_out).alpha)
        spike_out, V_last_out = plif_rowparam_forward_recompute(
            beta_out_row, u_out, v_th_out_row, v_init_out, alpha_out,
        )
        self.output_neuron.v = V_last_out.detach()

        # 二进制残差（替代 SEW 十进制加法）
        if self.add_residual:
            return binary_residual(spike_out, spike_in_seq)
        return spike_out  # MoE shared expert: 残差由 MoE 层统一处理

    def single_step_forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播。

        Args:
            spike_in: 二值脉冲输入, shape (batch, D), 值域 {0, 1}

        Returns:
            spike_out: 二值脉冲输出, shape (batch, D), 值域 {0, 1}
        """
        # 门控路径
        gate_spike = self.gate_neuron(self.gate_proj(spike_in))  # {0,1}^D_ff

        # 值路径
        up_spike = self.up_neuron(self.up_proj(spike_in))  # {0,1}^D_ff

        # AND 门：巧合检测
        gated = gate_spike * up_spike  # {0,1}^D_ff

        # 降维 + 残差 → 输出神经元 → SEW
        I_out = self.down_proj(gated) + self.skip_proj(spike_in)  # R^D
        return self.output_neuron(I_out) + spike_in  # SEW: {0,1}^D + {0,1}^D
