"""
SNNBlock: 完整的 SNN 隐状态空间 Block

结构（每个 SNN 时间步）：
  spike_in {0,1}^D
    ├─→ W_in     → I[t] ∈ R^{D*N}
    ├─→ W_β^(x)  ──┐+ W_β^(V)·V + b_β → σ      → β(t)
    ├─→ W_α^(x)  ──┐+ W_α^(V)·V + b_α → softplus → α(t)
    ├─→ W_th^(x) ──┐+ W_th^(V)·V + b_th → |·|+V_min → V_th(t)
    ├─→ W_gate   → sigmoid → gate ∈ (0,1)^D
    └─→ W_skip   → I_skip ∈ R^D

  SelectivePLIF(I, β, α, V_th) → s[t] ∈ {0,1}^{D*N}

  W_out · s[t] ⊙ gate + I_skip → 输出 PLIF → spike_out ∈ {0,1}^D
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, layer, neuron, surrogate

from .selective_plif import SelectivePLIFNode


class SNNBlock(base.MemoryModule):
    """
    单个 SNN Block。

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
        DN = D * N

        # ====== 六条并行输入投影（SNN 突触：spike 输入） ======
        self.W_in = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_beta_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_alpha_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_th_x = layer.Linear(D, DN, bias=False, step_mode='s')
        self.W_gate = layer.Linear(D, D, bias=False, step_mode='s')
        self.W_skip = layer.Linear(D, D, bias=False, step_mode='s')

        # ====== Voltage-gated 反馈矩阵：N×N，D 通道共享 ======
        # 标准 nn.Linear（NOT SNN 突触），操作连续值 V
        self.W_beta_V = nn.Linear(N, N, bias=False)
        self.W_alpha_V = nn.Linear(N, N, bias=False)
        self.W_th_V = nn.Linear(N, N, bias=False)

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
        # v_threshold 根据 block 深度校准：
        #   Block 0（~50% input）: v_threshold=0.3 → ~20% 输出
        #   Deeper blocks（input firing rate 衰减）: 需要更低 v_threshold
        # init_tau=2.0（β=0.5）：快速响应，适合 K-bit 二进制编码。
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
        """功能引导初始化：基于信号传播分析校准各参数，引导向预设职能演化。

        设计原则：
        1. 多时间尺度 β 分布（HiPPO 启发）
        2. W_in 时间尺度缩放：均衡不同 β 的稳态 σ_V
        3. b_th σ_V 校准：基于目标发放率设置阈值
        4. W_out 发放率均衡：放大低发放率神经元的贡献
        5. W^(V) 结构化初始化：对角自反馈
        详细推导见文档 5.8.10。
        """
        D, N = self.D, self.N
        K_ref = 8  # 信号传播分析的参考时间步数

        # 目标 β 分布：多时间尺度 [0.80, 0.99]
        beta_values = torch.linspace(0.80, 0.99, N)

        # ====== 1. β 偏置：logit-spaced ======
        b_beta_per_n = torch.log(beta_values / (1.0 - beta_values))
        self.b_beta.data.copy_(b_beta_per_n.repeat(D))

        # ====== 2. α 偏置：softplus(0.5413) ≈ 1.0（单位写入增益） ======
        self.b_alpha.data.fill_(0.5413)

        # ====== 3. W^(x) 权重 ======
        # 信号路径：Kaiming uniform（需要合理电流幅度）
        for lin in [self.W_in, self.W_gate, self.W_skip, self.W_out]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        # 调制路径：Kaiming × 0.1（偏置主导，训练学习输入依赖性）
        # V_th = V_min + |b_th + W_th·spike|, 若 W_th 太大则 |·| 的噪声淹没 b_th
        for lin in [self.W_beta_x, self.W_alpha_x, self.W_th_x]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            lin.weight.data.mul_(0.1)

        # ====== 4. W_in 时间尺度缩放 ======
        # 稳态 σ²_V ∝ σ²_I / (1-β²)，乘 √(1-β²) 使不同时间尺度 σ_V 接近
        # 短程（β=0.80）: scale=0.60, 长程（β=0.99）: scale=0.14
        scale_per_n = torch.sqrt(1.0 - beta_values ** 2)  # (N,)
        scale_DN = scale_per_n.repeat(D)  # (D*N,)
        with torch.no_grad():
            self.W_in.weight.mul_(scale_DN.unsqueeze(1))  # 逐行缩放

        # ====== 5. b_th：σ_V 校准 ======
        # 缩放后: σ²_V,n = σ²_{I,base} · (1 - β_n^{2K})
        # 其中 σ²_{I,base} = 1/6（Kaiming fan_in=D, spike p=0.5）
        sigma_I_base = math.sqrt(1.0 / 6.0)  # ≈ 0.408
        sigma_V_per_n = sigma_I_base * torch.sqrt(
            1.0 - beta_values ** (2 * K_ref)
        )
        # 目标发放率：短程 ~25% → 长程 ~8% 线性过渡
        target_p_fire = torch.linspace(0.25, 0.08, N)
        # V_th = σ_V · Φ^{-1}(1 - p_fire)
        z_scores = math.sqrt(2.0) * torch.erfinv(
            2.0 * (1.0 - target_p_fire) - 1.0
        )
        target_V_th = sigma_V_per_n * z_scores
        b_th_per_n = torch.clamp(target_V_th - self.v_th_min, min=0.05)
        self.b_th.data.copy_(b_th_per_n.repeat(D))

        # ====== 6. W_out 发放率均衡缩放 ======
        # 低发放率神经元贡献少 → 放大其 W_out 列权重
        # 缩放因子 1/√(p_fire_n)，归一化到均值 1
        out_scale_per_n = 1.0 / torch.sqrt(target_p_fire)
        out_scale_per_n = out_scale_per_n / out_scale_per_n.mean()
        out_scale_DN = out_scale_per_n.repeat(D)  # (D*N,)
        with torch.no_grad():
            self.W_out.weight.mul_(out_scale_DN.unsqueeze(0))  # 逐列缩放

        # ====== 7. W^(V) 结构化初始化 ======
        # 对角 +ε：自身 V 反馈到自身调制参数（自监测）
        # 非对角为零：训练初期各时间尺度独立演化
        eps = 0.05
        for W_V in [self.W_beta_V, self.W_alpha_V, self.W_th_V]:
            nn.init.zeros_(W_V.weight)
            W_V.weight.data.fill_diagonal_(eps)

    def _voltage_gated(self, V: torch.Tensor, W_V: nn.Linear) -> torch.Tensor:
        """
        对 V ∈ (batch, D*N) 应用共享 N×N 矩阵。
        reshape → matmul → reshape，等价于 D 个通道各自做 N×N 变换。
        """
        batch = V.shape[0]
        V_reshaped = V.view(batch * self.D, self.N)  # (batch*D, N)
        V_proj = W_V(V_reshaped)                      # (batch*D, N)
        return V_proj.view(batch, self.D * self.N)    # (batch, D*N)

    def single_step_forward(self, spike_in: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播。

        Args:
            spike_in: 二值脉冲输入, shape (batch, D), 值域 {0, 1}

        Returns:
            spike_out: 二值脉冲输出, shape (batch, D), 值域 {0, 1}
        """
        # 获取 V[t-1]（隐神经元当前膜电位）
        V_prev = self.hidden_neuron.v
        if isinstance(V_prev, float):
            # 首步：v 还是 float 0.，构造全零张量
            V_prev = torch.zeros(
                spike_in.shape[0], self.D * self.N,
                device=spike_in.device, dtype=spike_in.dtype,
            )

        # ====== 阶段一：六条并行输入路径 ======

        # 输入电流
        I_t = self.W_in(spike_in)  # (batch, D*N)

        # β 调制：W_β^(x)·spike + W_β^(V)·V + b_β → sigmoid
        beta = torch.sigmoid(
            self.W_beta_x(spike_in)
            + self._voltage_gated(V_prev, self.W_beta_V)
            + self.b_beta
        )

        # α 调制：W_α^(x)·spike + W_α^(V)·V + b_α → softplus
        alpha = F.softplus(
            self.W_alpha_x(spike_in)
            + self._voltage_gated(V_prev, self.W_alpha_V)
            + self.b_alpha
        )

        # V_th 调制：V_min + |W_th^(x)·spike + W_th^(V)·V + b_th|
        v_th = self.v_th_min + torch.abs(
            self.W_th_x(spike_in)
            + self._voltage_gated(V_prev, self.W_th_V)
            + self.b_th
        )

        # 门控：sigmoid(W_gate · spike)
        gate = torch.sigmoid(self.W_gate(spike_in))  # (batch, D)

        # 残差
        I_skip = self.W_skip(spike_in)  # (batch, D)

        # ====== 阶段二：隐状态空间 ======
        s_hidden = self.hidden_neuron(I_t, beta, alpha, v_th)  # (batch, D*N)

        # ====== 阶段三：门控 + 残差 + 输出 ======
        I_out = self.W_out(s_hidden)           # (batch, D)
        I_total = I_out * gate + I_skip        # (batch, D)
        spike_out = self.output_neuron(I_total)  # (batch, D)

        return spike_out
