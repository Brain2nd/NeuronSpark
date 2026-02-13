"""
SelectivePLIFNode: 动态参数的 PLIF 神经元

与标准 ParametricLIFNode 的区别：
- β(t), α(t), V_th(t) 作为外部参数每步传入，不是内部 nn.Parameter
- 本神经元无可训练参数，所有学习发生在 SNNBlock 的调制网络中
- 仅支持 step_mode='s'（单步模式）
- 仅支持 soft reset（v_reset=None）

状态方程：
  V[t] = β(t) · V[t-1] + α(t) · I[t]
  s[t] = Θ(V[t] - V_th(t))           (surrogate gradient)
  V[t] -= V_th(t) · s[t]              (soft reset)
"""

import torch
from spikingjelly.activation_based import neuron, surrogate


class SelectivePLIFNode(neuron.BaseNode):
    """
    隐状态空间的核心神经元。

    接收外部动态计算的 β(t), α(t), V_th(t)，执行：
      charge → fire → soft reset

    Args:
        surrogate_function: surrogate gradient 函数，默认 Sigmoid(alpha=4.0)
        detach_reset: 是否在 reset 时 detach spike，默认 False
    """

    def __init__(
        self,
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

    def single_step_forward(
        self,
        x: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        v_th: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步前向传播。

        Args:
            x:    输入电流 I[t],   shape (batch, D*N)
            beta: 衰减率 β(t),    shape (batch, D*N), 值域 (0, 1)
            alpha: 写入增益 α(t),  shape (batch, D*N), 值域 R+
            v_th: 动态阈值 V_th(t), shape (batch, D*N), 值域 R+

        Returns:
            spike: 二值脉冲 s[t],  shape (batch, D*N), 值域 {0, 1}
        """
        # Phase 0: 首步将 v 从 float 扩展为与输入同形的张量
        self.v_float_to_tensor(x)

        # Phase 1: Charge — 膜电位更新
        # V[t] = β(t) · V[t-1] + α(t) · I[t]
        self.v = beta * self.v + alpha * x

        # Phase 2: Fire — 使用动态 v_th（不是 self.v_threshold）
        # spike = Heaviside(V[t] - V_th(t))，反向用 surrogate gradient
        spike = self.surrogate_function(self.v - v_th)

        # Phase 3: Soft Reset — V[t] -= V_th(t) · s[t]
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.v = self.v - spike_d * v_th

        return spike

    def extra_repr(self) -> str:
        return (
            f'v_reset={self.v_reset}, '
            f'detach_reset={self.detach_reset}, '
            f'step_mode={self.step_mode}, '
            f'surrogate={self.surrogate_function}'
        )
