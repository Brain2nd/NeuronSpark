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

  W_out · s[t] ⊙ gate + I_skip → 输出 PLIF → spike_out ∈ {0,1}^D

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, neuron, surrogate

from .selective_plif import SelectivePLIFNode
from .parallel_scan import plif_parallel_forward, plif_fixed_param_forward


class SNNBlock(base.MemoryModule):
    """
    单个 SNN Block（v7：并行化，无 W^(V)）。

    Args:
        D: 可见维度（Block 间通信的 spike 维度）
        N: 状态扩展因子（每个通道的隐神经元数）
        v_th_min: 动态阈值下限
        output_v_threshold: 输出神经元阈值（deeper blocks 需要更低值）
        surrogate_function: surrogate gradient 函数
    """

    def __init__(
        self,
        D: int,
        N: int = 8,
        v_th_min: float = 0.1,
        output_v_threshold: float = 0.3,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
    ):
        super().__init__()
        self.D = D
        self.N = N
        self.v_th_min = v_th_min
        self.output_v_threshold = output_v_threshold
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

        # ====== 输出神经元（D 个，固定参数 PLIF） ======
        self.output_neuron = neuron.ParametricLIFNode(
            init_tau=2.0,
            v_threshold=output_v_threshold,
            v_reset=None,  # soft reset
            surrogate_function=surrogate_function,
            step_mode='s',
        )

        # ====== 参数初始化 ======
        self._initialize_parameters()

    def _initialize_parameters(self):
        """功能引导初始化（v7：无 W^(V)）。详细推导见文档 5.8.10。"""
        D, N = self.D, self.N
        K_ref = 16  # v7: K=16

        # 目标 β 分布：多时间尺度 [0.80, 0.99]
        beta_values = torch.linspace(0.80, 0.99, N)

        # ====== 1. β 偏置：logit-spaced ======
        b_beta_per_n = torch.log(beta_values / (1.0 - beta_values))
        self.b_beta.data.copy_(b_beta_per_n.repeat(D))

        # ====== 2. α 偏置：softplus(0.5413) ≈ 1.0（单位写入增益） ======
        self.b_alpha.data.fill_(0.5413)

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
        sigma_I_base = math.sqrt(1.0 / 6.0)
        sigma_V_per_n = sigma_I_base * torch.sqrt(
            1.0 - beta_values ** (2 * K_ref)
        )
        target_p_fire = torch.linspace(0.25, 0.08, N)
        z_scores = math.sqrt(2.0) * torch.erfinv(
            2.0 * (1.0 - target_p_fire) - 1.0
        )
        target_V_th = sigma_V_per_n * z_scores
        b_th_per_n = torch.clamp(target_V_th - self.v_th_min, min=0.05)
        self.b_th.data.copy_(b_th_per_n.repeat(D))

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
            spike_out_seq: (TK, batch, D) — 全部 T×K 帧的输出 spike
        """
        TK, batch, D = spike_in_seq.shape
        DN = self.D * self.N

        # ====== Phase 1: 批量投影（全部 TK 帧同时计算）======
        flat = spike_in_seq.reshape(TK * batch, D)

        I_all = F.linear(flat, self.W_in.weight).reshape(TK, batch, DN)
        beta_all = torch.sigmoid(
            F.linear(flat, self.W_beta_x.weight).reshape(TK, batch, DN)
            + self.b_beta
        )
        alpha_all = F.softplus(
            F.linear(flat, self.W_alpha_x.weight).reshape(TK, batch, DN)
            + self.b_alpha
        )
        v_th_all = self.v_th_min + torch.abs(
            F.linear(flat, self.W_th_x.weight).reshape(TK, batch, DN)
            + self.b_th
        )
        gate_all = torch.sigmoid(
            F.linear(flat, self.W_gate.weight).reshape(TK, batch, D)
        )
        I_skip_all = F.linear(flat, self.W_skip.weight).reshape(TK, batch, D)

        # ====== Phase 2-3: 隐神经元 parallel scan + spike 迭代 ======
        u_hidden = alpha_all * I_all  # (TK, batch, DN)

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

        # ====== Phase 4: 输出投影 ======
        s_flat = s_hidden.reshape(TK * batch, DN)
        I_out_all = F.linear(s_flat, self.W_out.weight).reshape(TK, batch, D)
        I_total_all = I_out_all * gate_all + I_skip_all  # (TK, batch, D)

        # ====== Phase 5: 输出神经元 parallel scan ======
        # ParametricLIFNode: V[t] = beta_out * V[t-1] + (1-beta_out) * x[t]
        # 保持 beta_out 为 tensor（不调 .item()），梯度流向 output_neuron.w
        beta_out = torch.sigmoid(self.output_neuron.w)  # 0-dim tensor, requires_grad
        u_output = (1.0 - beta_out) * I_total_all  # (TK, batch, D)，标量 tensor 自动广播

        v_init_output = self.output_neuron.v
        if isinstance(v_init_output, float):
            v_init_output = torch.zeros(batch, D, device=flat.device, dtype=flat.dtype)

        spike_out, V_post_output = plif_fixed_param_forward(
            beta_out, u_output, self.output_neuron.v_threshold, v_init_output,
            max_iter=3,
            surrogate_function=self.output_neuron.surrogate_function,
        )

        # 更新输出神经元状态
        self.output_neuron.v = V_post_output[-1].detach()

        return spike_out  # (TK, batch, D)

    def single_step_forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播（v7：无 W^(V)，用于调试/兼容）。

        Args:
            spike_in: 二值脉冲输入, shape (batch, D), 值域 {0, 1}

        Returns:
            spike_out: 二值脉冲输出, shape (batch, D), 值域 {0, 1}
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

        I_out = self.W_out(s_hidden)
        I_total = I_out * gate + I_skip
        spike_out = self.output_neuron(I_total)

        return spike_out
