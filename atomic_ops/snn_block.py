"""
SNNBlock v7: 完整的 SNN 隐状态空间 Block（并行化版本）

v6 → v7 变更：
  - 移除 W^(V) (W_beta_V, W_alpha_V, W_th_V)，使 β/α/V_th 仅依赖 spike_in
  - 新增 forward_parallel: 使用 parallel scan 处理全序列
  - 保留 single_step_forward 用于调试

结构（每个 SNN 时间步）：
  spike_in {0,1}^D
    ├─→ W_in     → I[t] ∈ R^{D*N}
    ├─→ W_β^(x)  + b_β → σ      → β(t)
    ├─→ W_α^(x)  + b_α → softplus → α(t)
    ├─→ W_th^(x) + b_th → |·|+V_min → V_th(t)
    ├─→ W_gate   → sigmoid → gate ∈ (0,1)^D
    └─→ W_skip   → I_skip ∈ R^D

  SelectivePLIF(I, β, α, V_th) → s[t] ∈ {0,1}^{D*N}

  W_out · V_post[t] ⊙ gate + I_skip → 连续输出 ∈ R^D

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, surrogate

from .selective_plif import SelectivePLIFNode
from .parallel_scan import plif_parallel_forward


# ====== Fused modulation activations (torch.compile) ======
# Fuse sigmoid + softplus + abs + alpha*I into single kernel.
# 7-8 separate element-wise kernels → 1 fused kernel, ~4x speedup on DN-sized tensors.
# First call triggers JIT compilation (~seconds); cached for subsequent calls.

@torch.compile(backend='inductor', fullgraph=True)
def _fused_modulation(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th, v_th_min, I_all):
    beta = torch.sigmoid(raw_beta + b_beta)
    alpha = F.softplus(raw_alpha + b_alpha)
    v_th = v_th_min + torch.abs(raw_th + b_th)
    u = alpha * I_all
    return beta, u, v_th


class SNNBlock(base.MemoryModule):
    """
    单个 SNN Block（v7：并行化，无 W^(V)）。

    Args:
        D: 可见维度（Block 间通信的维度）
        N: 状态扩展因子（每个通道的隐神经元数）
        v_th_min: 动态阈值下限
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        D: int,
        N: int = 8,
        v_th_min: float = 0.1,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.D = D
        self.N = N
        self.v_th_min = v_th_min
        DN = D * N

        # ====== 六条并行输入投影（SNN 突触：spike 输入） ======
        self.W_in = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_beta_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_alpha_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_th_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_gate = layer.Linear(D, D, bias=False, step_mode='s')
        self.W_skip = layer.Linear(D, D, bias=False, step_mode='s')

        # ====== v7: W^(V) 已移除（见文档第 7.2 节） ======

        # ====== 调制偏置（结构化初始化） ======
        self.b_beta = nn.Parameter(torch.empty(DN))
        self.b_alpha = nn.Parameter(torch.empty(DN))
        self.b_th = nn.Parameter(torch.empty(DN))

        # ====== 输出投影：D*N → D（SNN 突触） ======
        self.W_out = layer.Linear(DN, D, bias=False, step_mode='s')

        # ====== 隐状态空间神经元（D*N 个，动态参数） ======
        self.hidden_neuron = SelectivePLIFNode(
            surrogate_function=surrogate_function,
            detach_reset=False,
        )

        # ====== 参数初始化 ======
        self._initialize_parameters()

    def _initialize_parameters(self):
        """功能引导初始化（v7：无 W^(V)）。详细推导见文档 5.8.10。"""
        D, N = self.D, self.N
        K_ref = 16  # v7: K=16

        # 目标 β 分布：多时间尺度 [0.80, 0.99]
        beta_values = torch.linspace(0.80, 0.99, N)

        # ====== 1. β 偏置：logit-spaced + 维度间随机扰动 ======
        b_beta_per_n = torch.log(beta_values / (1.0 - beta_values))
        # 以 per_n 值为均值，加 N(0, 0.1) 扰动打破 D 个通道的对称性
        self.b_beta.data.copy_(b_beta_per_n.repeat(D))
        self.b_beta.data.add_(torch.empty_like(self.b_beta).normal_(0, 0.1))

        # ====== 2. α 偏置：softplus(0.5413) ≈ 1.0 + 维度间随机扰动 ======
        # 以 0.5413 为均值，N(0, 0.1) 扰动 → α ∈ ~[0.7, 1.3]
        self.b_alpha.data.normal_(0.5413, 0.1)

        # ====== 3. W^(x) 权重 ======
        for lin in [self.W_in, self.W_gate, self.W_skip, self.W_out]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        for lin in [self.W_beta_x, self.W_alpha_x, self.W_th_x]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            lin.weight.data.mul_(0.1)

        # ====== 4. W_in 时间尺度缩放 ======
        scale_per_n = torch.sqrt(1.0 - beta_values ** 2)  # (N,)
        scale_DN = scale_per_n.repeat(D)  # (D*N,)
        with torch.no_grad():
            self.W_in.weight.mul_(scale_DN.unsqueeze(1))

        # ====== 5. b_th：σ_V 校准 ======
        # σ_V = sqrt(p/3) * sqrt(1 - β^{2K})
        # 其中 p 是输入 firing rate。旧版假设 p=0.5（σ_I=0.408），
        # 但实际 input_neuron firing rate 约 0.07~0.45，深层更低。
        # 用 p=0.15 保守估计，避免 v_th 过高导致死神经元。
        p_assumed = 0.15
        sigma_I_base = math.sqrt(p_assumed / 3.0)
        sigma_V_per_n = sigma_I_base * torch.sqrt(
            1.0 - beta_values ** (2 * K_ref)
        )
        target_p_fire = torch.linspace(0.25, 0.08, N)
        z_scores = math.sqrt(2.0) * torch.erfinv(
            2.0 * (1.0 - target_p_fire) - 1.0
        )
        target_V_th = sigma_V_per_n * z_scores
        b_th_per_n = torch.clamp(target_V_th - self.v_th_min, min=0.05)
        # 以 per_n 值为均值，加 N(0, 0.02) 扰动打破 D 个通道的对称性
        self.b_th.data.copy_(b_th_per_n.repeat(D))
        self.b_th.data.add_(torch.empty_like(self.b_th).normal_(0, 0.02))

        # ====== 6. W_out 发放率均衡缩放 ======
        out_scale_per_n = 1.0 / torch.sqrt(target_p_fire)
        out_scale_per_n = out_scale_per_n / out_scale_per_n.mean()
        out_scale_DN = out_scale_per_n.repeat(D)
        with torch.no_grad():
            self.W_out.weight.mul_(out_scale_DN.unsqueeze(0))

    def forward_parallel(self, spike_in_seq: torch.Tensor) -> torch.Tensor:
        """
        并行前向传播：使用 parallel scan 处理全序列。

        Args:
            spike_in_seq: (TK, batch, D) — 全部 T×K 帧的输入 spike

        Returns:
            continuous_out: (TK, batch, D) — 全部 T×K 帧的连续输出（V_post 经 W_out 投影）
        """
        TK, batch, D = spike_in_seq.shape
        DN = self.D * self.N

        # ====== Phase 1: 批量投影（全部 TK 帧同时计算）======
        flat = spike_in_seq.reshape(TK * batch, D)

        I_all = F.linear(flat, self.W_in.weight).reshape(TK, batch, DN)
        raw_beta = F.linear(flat, self.W_beta_x.weight).reshape(TK, batch, DN)
        raw_alpha = F.linear(flat, self.W_alpha_x.weight).reshape(TK, batch, DN)
        raw_th = F.linear(flat, self.W_th_x.weight).reshape(TK, batch, DN)
        gate_all = torch.sigmoid(
            F.linear(flat, self.W_gate.weight).reshape(TK, batch, D)
        )
        I_skip_all = F.linear(flat, self.W_skip.weight).reshape(TK, batch, D)

        # ====== Phase 1b: 融合激活（torch.compile → 单 kernel）======
        beta_all, u_hidden, v_th_all = _fused_modulation(
            raw_beta, self.b_beta, raw_alpha, self.b_alpha,
            raw_th, self.b_th, self.v_th_min, I_all,
        )

        # 获取隐神经元初始状态
        v_init_hidden = self.hidden_neuron.v
        if isinstance(v_init_hidden, float):
            v_init_hidden = torch.zeros(batch, DN, device=flat.device, dtype=flat.dtype)

        s_hidden, V_post_hidden, _ = plif_parallel_forward(
            beta_all, u_hidden, v_th_all, v_init_hidden, max_iter=3,
            surrogate_function=self.hidden_neuron.surrogate_function,
        )

        # 更新隐神经元状态（保存末步供下次调用）
        self.hidden_neuron.v = V_post_hidden[-1].detach()

        # ====== Phase 4: 输出投影（V_post → W_out: 连续梯度直通 β）======
        # 用 V_post（膜电压）代替 spike 作为 W_out 输入，消除 surrogate 梯度瓶颈：
        #   spike 路径: ∂spike/∂β = surrogate'(V-v_th) · V_prev ≈ 0（大部分时刻）
        #   V_post 路径: ∂V_post/∂β = V_prev（无 surrogate 阻断，每步都有梯度）
        v_flat = V_post_hidden.reshape(TK * batch, DN)
        I_out_all = F.linear(v_flat, self.W_out.weight).reshape(TK, batch, D)
        I_total_all = I_out_all * gate_all + I_skip_all  # (TK, batch, D)

        # output_neuron 已移除：连续值由层级 K 帧聚合处理
        return I_total_all  # (TK, batch, D), 连续值

    def single_step_forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播（v7：无 W^(V)，用于调试/兼容）。

        Args:
            spike_in: 二值脉冲输入, shape (batch, D), 值域 {0, 1}

        Returns:
            continuous_out: 连续输出, shape (batch, D)
        """
        V_prev = self.hidden_neuron.v
        if isinstance(V_prev, float):
            V_prev = torch.zeros(
                spike_in.shape[0], self.D * self.N,
                device=spike_in.device, dtype=spike_in.dtype,
            )

        I_t = self.W_in(spike_in)

        # v7: β 调制仅依赖 spike_in（无 W^(V)·V 项）
        beta = torch.sigmoid(self.W_beta_x(spike_in) + self.b_beta)
        alpha = F.softplus(self.W_alpha_x(spike_in) + self.b_alpha)
        v_th = self.v_th_min + torch.abs(self.W_th_x(spike_in) + self.b_th)

        gate = torch.sigmoid(self.W_gate(spike_in))
        I_skip = self.W_skip(spike_in)

        s_hidden = self.hidden_neuron(I_t, beta, alpha, v_th)

        # 用 V_post（膜电压）做输出投影，与 forward_parallel 一致
        V_post = self.hidden_neuron.v  # 发放+重置后的膜电位
        I_out = self.W_out(V_post)
        I_total = I_out * gate + I_skip

        return I_total  # 连续值
