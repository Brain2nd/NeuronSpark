"""
SNNBlock v7.8: 完整的 SNN 隐状态空间 Block（v7.8: Resonate-and-Fire 振荡动力学）

v7 → v7.8 变更：
  - v7: 移除 W^(V), 新增 forward_parallel (parallel scan)
  - v7.8: 新增 W_omega 投影 → ω(t) 振荡频率, 二阶耦合 V/W 动力学
    · 七条并行输入投影: W_in, W_β, W_α, W_th, W_ω, W_gate, W_skip
    · 稳定性约束: √(β²+ω²) ≤ r_max < 1

结构（每个 SNN 时间步）：
  spike_in {0,1}^D
    ├─→ W_in     → I[t] ∈ R^{D*N}
    ├─→ W_β^(x)  + b_β → σ      → β(t)
    ├─→ W_α^(x)  + b_α → softplus → α(t)
    ├─→ W_th^(x) + b_th → |·|+V_min → V_th(t)
    ├─→ W_ω^(x)  + b_ω → softplus → ω(t)
    ├─→ W_gate   → sigmoid → gate ∈ (0,1)^D
    └─→ W_skip   → I_skip ∈ R^D

  稳定性约束: (β, ω) ← (β, ω) / max(1, √(β²+ω²)/r_max)

  RF-SelectivePLIF(I, β, α, V_th, ω) → s[t] ∈ {0,1}^{D*N}

  W_out · s[t] ⊙ gate + I_skip → 输出 PLIF → spike_out ∈ {0,1}^D

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节 + 第 10.6 节。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, neuron, surrogate

from .selective_plif import SelectivePLIFNode
from .plif_node import PLIFNode
from .parallel_scan import plif_rowparam_forward_sew, rf_plif_parallel_forward_recompute


@torch.compile(backend='inductor', fullgraph=True)
def _fused_gate_skip_scale(
    I_out: torch.Tensor, gate: torch.Tensor,
    I_skip: torch.Tensor, one_minus_beta: torch.Tensor,
) -> torch.Tensor:
    """融合 gate·out + skip → (1-β) 缩放 → u_output: 3 element-wise ops → 1 fused kernel。"""
    return (I_out * gate + I_skip) * one_minus_beta


# ====== Fused modulation activations (torch.compile) ======

@torch.compile(backend='inductor', fullgraph=True)
def _fused_modulation_rf(raw_beta, b_beta, raw_alpha, b_alpha, raw_th, b_th,
                          v_th_min, I_all, raw_omega, b_omega):
    """融合 sigmoid + softplus + abs + stability constraint → 单 kernel。

    稳定性约束: r = √(β²+ω²) ≤ r_max，保证振荡系统衰减。
    """
    beta = torch.sigmoid(raw_beta + b_beta)
    alpha = F.softplus(raw_alpha + b_alpha)
    v_th = v_th_min + torch.abs(raw_th + b_th)
    omega = F.softplus(raw_omega + b_omega)

    # 稳定性约束: √(β²+ω²) ≤ r_max=0.999
    r_sq = beta * beta + omega * omega
    r_max_sq = 0.999 * 0.999
    scale = torch.where(
        r_sq > r_max_sq,
        0.999 * torch.rsqrt(r_sq.clamp(min=1e-12)),
        torch.ones_like(r_sq),
    )
    beta = beta * scale
    omega = omega * scale

    u = alpha * I_all
    return beta, u, v_th, omega


class SNNBlock(base.MemoryModule):
    """
    单个 SNN Block（v7.8: Resonate-and-Fire 振荡动力学）。

    Args:
        D: 可见维度（Block 间通信的 spike 维度）
        N: 状态扩展因子（每个通道的隐神经元数）
        v_th_min: 动态阈值下限
        output_v_threshold: 输出神经元阈值
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

        # ====== 七条并行输入投影（v7.8: +W_omega） ======
        self.W_in = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_beta_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_alpha_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_th_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_omega_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_gate = layer.Linear(D, D, bias=False, step_mode='s')
        self.W_skip = layer.Linear(D, D, bias=False, step_mode='s')

        # ====== 调制偏置（结构化初始化） ======
        self.b_beta = nn.Parameter(torch.empty(DN))
        self.b_alpha = nn.Parameter(torch.empty(DN))
        self.b_th = nn.Parameter(torch.empty(DN))
        self.b_omega = nn.Parameter(torch.empty(DN))

        # ====== 输出投影：D*N → D ======
        self.W_out = layer.Linear(DN, D, bias=False, step_mode='s')

        # ====== 隐状态空间神经元（D*N 个，RF 动态参数） ======
        self.hidden_neuron = SelectivePLIFNode(
            dim=DN,
            surrogate_function=surrogate_function,
            detach_reset=False,
        )

        # ====== 输出神经元（D 个，固定参数 PLIF） ======
        self.output_neuron = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=output_v_threshold,
            surrogate_function=surrogate_function,
        )

        # ====== 参数初始化 ======
        self._initialize_parameters()

    def _initialize_parameters(self):
        """功能引导初始化（v7.8: 含 ω 初始化）。"""
        D, N = self.D, self.N
        K_ref = 16

        # 目标 β 分布：多时间尺度 [0.80, 0.99]
        beta_values = torch.linspace(0.80, 0.99, N)

        # ====== 1. β 偏置：logit-spaced + 维度间随机扰动 ======
        b_beta_per_n = torch.log(beta_values / (1.0 - beta_values))
        self.b_beta.data.copy_(b_beta_per_n.repeat(D))
        self.b_beta.data.add_(torch.empty_like(self.b_beta).normal_(0, 0.1))

        # ====== 2. α 偏置：softplus(0.5413) ≈ 1.0 + 维度间随机扰动 ======
        self.b_alpha.data.normal_(0.5413, 0.1)

        # ====== 3. ω 偏置：初始 ω 较小，softplus(-2.0) ≈ 0.127 ======
        # N 个神经元覆盖不同频率: 低频(-3.0)→高频(-0.5)
        # softplus(-3.0)≈0.049, softplus(-0.5)≈0.474 → θ ∈ [0.05, 0.44] rad/step
        b_omega_per_n = torch.linspace(-3.0, -0.5, N)
        self.b_omega.data.copy_(b_omega_per_n.repeat(D))
        self.b_omega.data.add_(torch.empty_like(self.b_omega).normal_(0, 0.1))

        # ====== 4. W^(x) 权重 ======
        for lin in [self.W_in, self.W_gate, self.W_skip, self.W_out]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        for lin in [self.W_beta_x, self.W_alpha_x, self.W_th_x, self.W_omega_x]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            lin.weight.data.mul_(0.1)

        # ====== 5. W_in 时间尺度缩放 ======
        scale_per_n = torch.sqrt(1.0 - beta_values ** 2)
        scale_DN = scale_per_n.repeat(D)
        with torch.no_grad():
            self.W_in.weight.mul_(scale_DN.unsqueeze(1))

        # ====== 6. b_th：σ_V 校准 ======
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
        self.b_th.data.copy_(b_th_per_n.repeat(D))
        self.b_th.data.add_(torch.empty_like(self.b_th).normal_(0, 0.02))

        # ====== 7. W_out 发放率均衡缩放 ======
        out_scale_per_n = 1.0 / torch.sqrt(target_p_fire)
        out_scale_per_n = out_scale_per_n / out_scale_per_n.mean()
        out_scale_DN = out_scale_per_n.repeat(D)
        with torch.no_grad():
            self.W_out.weight.mul_(out_scale_DN.unsqueeze(0))

    def forward_parallel(self, spike_in_seq: torch.Tensor) -> torch.Tensor:
        """
        并行前向传播：使用 RF parallel scan 处理全序列。

        Args:
            spike_in_seq: (TK, batch, D) — 全部 T×K 帧的输入 spike

        Returns:
            spike_out_seq: (TK, batch, D) — 全部 T×K 帧的输出 spike
        """
        TK, batch, D = spike_in_seq.shape
        DN = self.D * self.N

        # ====== Phase 1: 批量投影（合并 GEMM: 7 launch → 2 launch）======
        flat = spike_in_seq.reshape(TK * batch, D)

        # 5× D→DN 合并为 1 次 GEMM: (5*DN, D) @ (D, TK*B)^T
        W_dn5 = torch.cat([
            self.W_in.weight, self.W_beta_x.weight, self.W_alpha_x.weight,
            self.W_th_x.weight, self.W_omega_x.weight,
        ], dim=0)  # (5*DN, D)
        proj_dn5 = F.linear(flat, W_dn5)  # (TK*B, 5*DN)
        I_all, raw_beta, raw_alpha, raw_th, raw_omega = proj_dn5.split(DN, dim=-1)
        I_all = I_all.reshape(TK, batch, DN)
        raw_beta = raw_beta.reshape(TK, batch, DN)
        raw_alpha = raw_alpha.reshape(TK, batch, DN)
        raw_th = raw_th.reshape(TK, batch, DN)
        raw_omega = raw_omega.reshape(TK, batch, DN)

        # 2× D→D 合并为 1 次 GEMM: (2*D, D) @ (D, TK*B)^T
        W_d2 = torch.cat([self.W_gate.weight, self.W_skip.weight], dim=0)  # (2*D, D)
        proj_d2 = F.linear(flat, W_d2).reshape(TK, batch, 2 * D)  # (TK, B, 2*D)
        gate_all = torch.sigmoid(proj_d2[:, :, :D])
        I_skip_all = proj_d2[:, :, D:]

        # ====== Phase 1b: 融合激活 + 稳定性约束（torch.compile → 单 kernel）======
        beta_all, u_hidden, v_th_all, omega_all = _fused_modulation_rf(
            raw_beta, self.b_beta, raw_alpha, self.b_alpha,
            raw_th, self.b_th, self.v_th_min, I_all,
            raw_omega, self.b_omega,
        )

        # ====== Phase 2: RF PLIF parallel scan（V_post/W recompute，省 1GB/层）======
        v_init_hidden = self.hidden_neuron.v
        if isinstance(v_init_hidden, float):
            v_init_hidden = self.hidden_neuron.expand_v_init(batch, flat.device, flat.dtype)

        w_init_hidden = self.hidden_neuron.w
        if isinstance(w_init_hidden, float):
            w_init_hidden = self.hidden_neuron.expand_w_init(batch, flat.device, flat.dtype)

        s_hidden, v_last_hidden, w_last_hidden = rf_plif_parallel_forward_recompute(
            beta_all, omega_all, u_hidden, v_th_all,
            v_init_hidden, w_init_hidden,
            surrogate_function=self.hidden_neuron.get_surrogate(u_hidden),
        )

        # 更新隐神经元状态（直接用末步值）
        self.hidden_neuron.v = v_last_hidden.detach()
        self.hidden_neuron.w = w_last_hidden.detach()

        # ====== Phase 4: 输出投影 + 融合 gate·skip·scale (P2) ======
        s_flat = s_hidden.reshape(TK * batch, DN)
        I_out_all = F.linear(s_flat, self.W_out.weight).reshape(TK, batch, D)
        beta_out = self.output_neuron.beta
        # P2: fuse (I_out * gate + I_skip) * (1-beta) into single kernel
        u_output = _fused_gate_skip_scale(I_out_all, gate_all, I_skip_all, 1.0 - beta_out)

        # ====== Phase 5+6: 输出神经元 parallel scan + SEW 残差 (P1) ======
        v_init_output = self.output_neuron.v
        if isinstance(v_init_output, float):
            v_init_output = self.output_neuron.expand_v_init(batch, flat.device, flat.dtype)

        beta_out_row = beta_out.unsqueeze(0).expand(batch, D).contiguous()
        v_th_out_row = self.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        surr = self.output_neuron.get_surrogate(u_output)
        alpha = float(surr.alpha) if hasattr(surr, 'alpha') else 4.0

        # P1: fused PLIF + SEW (spike_sew = plif_spike + spike_in) in single Triton kernel
        spike_sew, V_post_output = plif_rowparam_forward_sew(
            beta_out_row, u_output, v_th_out_row, v_init_output,
            spike_in_seq, alpha,
        )

        self.output_neuron.v = V_post_output[-1].detach()

        return spike_sew

    def single_step_forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播（v7.8: RF 动力学）。

        Args:
            spike_in: 二值脉冲输入, shape (batch, D), 值域 {0, 1}

        Returns:
            spike_out: 二值脉冲输出, shape (batch, D), 值域 {0, 1}
        """
        V_prev = self.hidden_neuron.v
        if isinstance(V_prev, float):
            V_prev = self.hidden_neuron.expand_v_init(
                spike_in.shape[0], spike_in.device, spike_in.dtype,
            )

        I_t = self.W_in(spike_in)

        beta = torch.sigmoid(self.W_beta_x(spike_in) + self.b_beta)
        alpha = F.softplus(self.W_alpha_x(spike_in) + self.b_alpha)
        v_th = self.v_th_min + torch.abs(self.W_th_x(spike_in) + self.b_th)
        omega = F.softplus(self.W_omega_x(spike_in) + self.b_omega)

        # 稳定性约束
        r_sq = beta * beta + omega * omega
        r_max_sq = 0.999 * 0.999
        scale = torch.where(
            r_sq > r_max_sq,
            0.999 * torch.rsqrt(r_sq.clamp(min=1e-12)),
            torch.ones_like(r_sq),
        )
        beta = beta * scale
        omega = omega * scale

        gate = torch.sigmoid(self.W_gate(spike_in))
        I_skip = self.W_skip(spike_in)

        s_hidden = self.hidden_neuron(I_t, beta, alpha, v_th, omega)

        I_out = self.W_out(s_hidden)
        I_total = I_out * gate + I_skip
        spike_out = self.output_neuron(I_total)

        # SEW-ResNet ADD: spike-level identity mapping for gradient flow
        return spike_out + spike_in
