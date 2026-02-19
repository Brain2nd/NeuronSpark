"""
SNNFFN: SNN 等价的 Feed-Forward Network

对标 Qwen3MLP 的 SwiGLU 结构：
  Qwen3 MLP:  down_proj( SiLU(gate_proj(x)) * up_proj(x) )
  SNN  FFN:   down_proj( gate_spike AND up_spike ) + skip → output_neuron

SiLU gating → spike 巧合检测（AND 门）：
  只有当 gate 和 up 通道同时发放时，信号才传递，实现非线性选择。

信号流：
  spike_in (p~20%) → gate_proj → gate_neuron → gate_spike (p~30%)
  spike_in (p~20%) → up_proj   → up_neuron   → up_spike (p~30%)
                      gate_spike AND up_spike → gated (p~9%)
                      down_proj(gated) + skip_proj(spike_in) → output_neuron → spike_out
"""

import math
import torch
import torch.nn as nn
from spikingjelly.activation_based import base, layer, neuron, surrogate


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

        # ====== 神经元 ======
        # gate_neuron: 门控发放
        self.gate_neuron = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=output_v_threshold,
            v_reset=None,
            surrogate_function=surrogate_function,
            step_mode='s',
        )
        # up_neuron: 值发放
        self.up_neuron = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=output_v_threshold,
            v_reset=None,
            surrogate_function=surrogate_function,
            step_mode='s',
        )
        # output_neuron: 输出发放
        self.output_neuron = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=output_v_threshold,
            v_reset=None,
            surrogate_function=surrogate_function,
            step_mode='s',
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

        # 降维 + 残差 → 输出神经元
        I_out = self.down_proj(gated) + self.skip_proj(spike_in)  # R^D
        return self.output_neuron(I_out)  # {0,1}^D
