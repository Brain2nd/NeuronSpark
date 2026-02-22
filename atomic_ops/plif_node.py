"""
PLIFNode: D 维固定参数 PLIF 神经元（设计文档 5.5 "普通 SNN 神经元"）

与 SelectivePLIFNode 的区别：
  SelectivePLIF: β(t), α(t), V_th(t) 由输入每步动态计算（选择性记忆）
  PLIFNode:      β, V_th 为 D 维可学习参数，训练后固定（信号转换）

每个维度有独立的可学习参数：
  β_d = sigmoid(w_d): 时间常数（衰减率）
  V_th_d: 发放阈值
  v_init_d: 初始膜电位（v7.6 可学习，省暖机时间）

动力学（与 ParametricLIF 一致）：
  V[0] = v_init              (可学习初始状态)
  V[t] = β · V[t-1] + (1-β) · x[t]
  s[t] = Θ(V[t] - V_th)            (surrogate gradient)
  V[t] -= V_th · s[t]              (soft reset)
"""

import math

import torch
import torch.nn as nn
from spikingjelly.activation_based import base, surrogate


class PLIFNode(base.MemoryModule):
    """
    D 维固定参数 PLIF 神经元。

    Args:
        dim: 神经元数量（每个维度独立参数）
        init_tau: 初始时间常数 τ（β = 1 - 1/τ）
        v_threshold: 初始发放阈值
        surrogate_function: surrogate gradient 函数
        adaptive_surrogate: 是否启用自适应 surrogate alpha（v7.6）
        sigma_0: 参考输入标准差（输入 std=sigma_0 时 alpha 不变）
    """

    def __init__(
        self,
        dim: int,
        init_tau: float = 2.0,
        v_threshold: float = 0.5,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        adaptive_surrogate: bool = True,
        sigma_0: float = 1.0,
    ):
        super().__init__()
        # D 维可学习参数（随机初始化，每个维度独立）
        # w: 控制 β=sigmoid(w)，随机产生不同时间常数
        #    init_w ± 0.5 → β ∈ ~[sigmoid(w-0.5), sigmoid(w+0.5)]
        #    tau=2.0 时 w=0, β ∈ ~[0.38, 0.62]
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.empty(dim).normal_(init_w, 0.5))
        # v_th: 发放阈值，U[0.5x, 1.5x] 均匀分布产生维度间多样性
        self.v_th = nn.Parameter(torch.empty(dim).uniform_(
            v_threshold * 0.5, v_threshold * 1.5,
        ))
        self.surrogate_function = surrogate_function
        # 自适应 surrogate gradient（v7.6）
        # alpha_adaptive = alpha_0 * sigma_0 / sigma_V
        # 浅层 V 分布窄 → alpha 增大（更精确）；深层 V 分布宽 → alpha 减小（更覆盖）
        self.adaptive_surrogate = adaptive_surrogate
        self.sigma_0 = sigma_0
        # 可学习初始膜电位（v7.6: 省暖机时间，高 β 神经元收益最大）
        # 初始化为零 → 与旧版行为兼容，训练中学到最优 V[0]
        # ∂V[t]/∂v_init = β^t → 高 β（长记忆）神经元 v_init 梯度最强
        self.v_init = nn.Parameter(torch.zeros(dim))
        # 膜电位状态（functional.reset_net 时重置为 0.，首步展开为 v_init）
        self.register_memory('v', 0.)

    @property
    def beta(self):
        """D 维衰减率 β = sigmoid(w)，值域 (0, 1)。"""
        return torch.sigmoid(self.w)

    def get_surrogate(self, u: torch.Tensor = None):
        """返回 surrogate 函数，支持自适应 alpha。

        根据输入电流 u 的标准差动态调整 surrogate 宽度：
          alpha_adaptive = alpha_0 * sigma_0 / (sigma_V + eps)

        当 sigma_V < sigma_0 时 alpha 增大（surrogate 变窄，梯度更精确）；
        当 sigma_V > sigma_0 时 alpha 减小（surrogate 变宽，覆盖更多神经元）。

        Args:
            u: 输入电流张量（用于估计 sigma_V）。None 时返回固定 surrogate。

        Returns:
            surrogate 函数（SpikingJelly Sigmoid 对象）
        """
        if not self.adaptive_surrogate or u is None:
            return self.surrogate_function

        with torch.no_grad():
            sigma_V = u.std().clamp(min=1e-6)
            alpha_0 = float(self.surrogate_function.alpha)
            alpha_adaptive = alpha_0 * self.sigma_0 / sigma_V.item()
            alpha_adaptive = max(1.0, min(alpha_adaptive, 20.0))

        return surrogate.Sigmoid(alpha=alpha_adaptive)

    def expand_v_init(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """将 v_init (dim,) 扩展为 (batch, dim)，匹配运行 dtype。

        Args:
            batch: batch size
            device: 目标设备
            dtype: 运行精度（bfloat16/float16/float32）

        Returns:
            (batch, dim) 初始膜电位
        """
        return self.v_init.to(dtype=dtype).unsqueeze(0).expand(batch, -1)

    def forward(self, x):
        """
        单步前向传播。

        V[0] = v_init (可学习), V[t] = β·V[t-1] + (1-β)·x[t], spike = Θ(V-V_th), soft reset。

        Args:
            x: 输入电流, shape (batch, dim)

        Returns:
            spike: 二值脉冲, shape (batch, dim), 值域 {0, 1}
        """
        if isinstance(self.v, float):
            self.v = self.expand_v_init(x.shape[0], x.device, x.dtype)
        beta = self.beta
        self.v = beta * self.v + (1.0 - beta) * x
        spike = self.surrogate_function(self.v - self.v_th)
        self.v = self.v - spike * self.v_th  # soft reset
        return spike
