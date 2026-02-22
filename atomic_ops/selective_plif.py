"""
SelectivePLIFNode: 动态参数的 PLIF 神经元（v7.8: Resonate-and-Fire 振荡扩展）

与标准 ParametricLIFNode 的区别：
- β(t), α(t), V_th(t) 作为外部参数每步传入，不是内部 nn.Parameter
- v_init, w_init 为可学习参数（v7.6 可学习初始膜电位）
- 仅支持 step_mode='s'（单步模式）
- 仅支持 soft reset（v_reset=None）

一阶模式（ω=0，向后兼容）：
  V[t] = β(t) · V[t-1] + α(t) · I[t]
  s[t] = Θ(V[t] - V_th(t))
  V[t] -= V_th(t) · s[t]

二阶 Resonate-and-Fire 模式（ω>0）：
  V_pre[t] = β(t) · V_post[t-1] - ω(t) · W[t-1] + α(t) · I[t]
  W[t]     = ω(t) · V_post[t-1] + β(t) · W[t-1]
  s[t]     = Θ(V_pre[t] - V_th(t))
  V_post[t]= V_pre[t] - V_th(t) · s[t]   （soft reset，仅 V）

物理直觉：V=位移, W=速度, ω=弹簧刚度。
脉冲响应 V[t] ∝ r^t · cos(θt)，r=√(β²+ω²), θ=arctan(ω/β)。
不同神经元学到不同 ω → 自动形成频率分解（类似 Fourier basis）。
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate


class SelectivePLIFNode(neuron.BaseNode):
    """
    隐状态空间的核心神经元（支持 Resonate-and-Fire 振荡动力学）。

    接收外部动态计算的 β(t), α(t), V_th(t), ω(t)，执行：
      charge → fire → soft reset

    Args:
        dim: 神经元数量（D*N），用于 v_init/w_init 参数维度
        surrogate_function: surrogate gradient 函数，默认 Sigmoid(alpha=4.0)
        detach_reset: 是否在 reset 时 detach spike，默认 False
    """

    def __init__(
        self,
        dim: int,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        detach_reset: bool = False,
    ):
        # v_threshold=1.0 是占位值，实际使用外部传入的 v_th
        # v_reset=None 触发 soft reset 模式，register_memory('v', 0.)
        super().__init__(
            v_threshold=1.0,
            v_reset=None,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode='s',
            backend='torch',
            store_v_seq=False,
        )
        # 可学习初始膜电位
        self.v_init = nn.Parameter(torch.zeros(dim))
        # 可学习初始振荡状态（RF 二阶动力学）
        self.w_init = nn.Parameter(torch.zeros(dim))
        # W 状态（振荡"速度"分量）
        self.register_memory('w', 0.)

    def expand_v_init(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """将 v_init (dim,) 扩展为 (batch, dim)，匹配运行 dtype。"""
        return self.v_init.to(dtype=dtype).unsqueeze(0).expand(batch, -1)

    def expand_w_init(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """将 w_init (dim,) 扩展为 (batch, dim)，匹配运行 dtype。"""
        return self.w_init.to(dtype=dtype).unsqueeze(0).expand(batch, -1)

    def get_surrogate(self, u: torch.Tensor = None):
        """返回 surrogate 函数，支持自适应 alpha。"""
        if u is None:
            return self.surrogate_function

        with torch.no_grad():
            sigma_V = u.std().clamp(min=1e-6)
            alpha_0 = float(self.surrogate_function.alpha)
            alpha_adaptive = alpha_0 * 1.0 / sigma_V.item()  # sigma_0=1.0
            alpha_adaptive = max(1.0, min(alpha_adaptive, 20.0))

        return surrogate.Sigmoid(alpha=alpha_adaptive)

    def single_step_forward(
        self,
        x: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        v_th: torch.Tensor,
        omega: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        单步前向传播（支持一阶 PLIF 和二阶 RF）。

        Args:
            x:     输入电流 I[t],    shape (batch, D*N)
            beta:  衰减率 β(t),     shape (batch, D*N), 值域 (0, 1)
            alpha: 写入增益 α(t),   shape (batch, D*N), 值域 R+
            v_th:  动态阈值 V_th(t), shape (batch, D*N), 值域 R+
            omega: 振荡频率 ω(t),   shape (batch, D*N), 值域 R+ (None=一阶模式)

        Returns:
            spike: 二值脉冲 s[t],  shape (batch, D*N), 值域 {0, 1}
        """
        # Phase 0: 首步将 v/w 从 float 扩展为可学习初始状态
        if isinstance(self.v, float):
            self.v = self.expand_v_init(x.shape[0], x.device, x.dtype)
        else:
            self.v_float_to_tensor(x)

        if omega is not None:
            # 初始化 W 状态
            if isinstance(self.w, float):
                self.w = self.expand_w_init(x.shape[0], x.device, x.dtype)

            # Phase 1: RF Charge — 二阶耦合动力学
            v_pre = beta * self.v - omega * self.w + alpha * x
            w_new = omega * self.v + beta * self.w
            self.w = w_new
        else:
            # Phase 1: 标准 PLIF Charge
            v_pre = beta * self.v + alpha * x

        # Phase 2: Fire
        spike = self.surrogate_function(v_pre - v_th)

        # Phase 3: Soft Reset — 仅 V, W 不 reset
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = v_pre - spike_d * v_th

        return spike

    def extra_repr(self) -> str:
        return (
            f'v_reset={self.v_reset}, '
            f'detach_reset={self.detach_reset}, '
            f'step_mode={self.step_mode}, '
            f'surrogate={self.surrogate_function}'
        )
