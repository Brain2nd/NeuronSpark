"""
SNNAdamW: SNN 专用优化器 — AdamW + 神经动力学增强（完整 10 阶段）。

step() 内部 10 个阶段：

  ── 梯度预处理 ──
  1. Natural Gradient 补偿（b_beta/b_alpha/b_omega 梯度除以激活导数）
  2. 梯度裁剪（全参数 clip_grad_norm_）
  ── 参数更新 ──
  3. AdamW 标准步（super().step()）
  ── 软正则化（梯度级微调） ──
  4. 双侧 Lyapunov 惩罚（Gradient Flossing: 推 β 远离 0 和 1 两端）
  5. β 多样性维护（同一 D 通道内 N 个 β 互斥）
  6. ω 多样性维护（同一 D 通道内 N 个 ω 互斥）
  7. α 多样性维护（同一 D 通道内 N 个 α 互斥）
  ── 硬约束（安全网） ──
  8. β 范围钳制（sigmoid(b_beta) ∈ [β_min, β_max]，防癫痫+防死寂）
  9. V_th 上界钳制（|b_th| ≤ b_th_max，防死寂）
  10. RF 稳定性投影（√(β²+ω²) ≤ r_max，防癫痫，最终权威）

设计原则：
  - 软正则化 (Phase 4-7) 提供持续的分布引导力
  - 硬约束 (Phase 8-10) 作为安全网，防止极端情况
  - Phase 10 最后执行，RF 稳定性是不可违反的绝对约束

文献基础：
  - Gradient Flossing (Krein+, NeurIPS'23): Lyapunov 指数正则化
  - Curse of Memory (Zucchet+, NeurIPS'24): β→1 参数敏感度 O(1/(1-β²)^{3/2})
  - LRU (Orvieto+, ICML'23): 离散线性递归稳定性约束
  - Rhythm-SNN (Nature Comm.'25): 异构振荡频率多样性

参数组标记：
  每个 param_group 可包含 'dynamics' 和 'N' 键：
    'dynamics': 'b_beta' | 'b_alpha' | 'b_omega' | 'b_th' | None
    'N': int  (状态扩展因子，b_beta/b_alpha/b_omega reshape (D*N,) → (D,N))
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW


def _softplus_inv(x):
    """softplus 反函数: log(exp(x) - 1)，数值稳定版。

    identity: softplus(softplus_inv(x)) == x
    公式: log(exp(x) - 1) = x + log(1 - exp(-x))
    """
    return x + torch.log1p(-torch.exp(-x))


class SNNAdamW(AdamW):
    """SNN 专用优化器：AdamW + 10 阶段神经动力学增强。

    Args:
        params: 参数组（每组可含 'dynamics' 和 'N' 键标记神经动力学参数）
        lr: 基础学习率
        betas: Adam 动量系数
        eps: Adam epsilon
        weight_decay: 默认 weight decay
        grad_clip: 梯度裁剪阈值（0 = 不裁剪）
        max_comp: Natural Gradient 补偿因子上限
        lyapunov_strength: 双侧 Lyapunov 正则化强度
        beta_diversity_strength: β 多样性排斥强度
        omega_diversity_strength: ω 多样性排斥强度
        alpha_diversity_strength: α 多样性排斥强度
        beta_range: (β_min, β_max) 硬钳制范围
        b_th_max: b_th 绝对值上限（V_th 上界 = v_th_min + b_th_max）
        rf_r_max: RF 谱半径上限 √(β²+ω²) ≤ r_max
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        max_comp: float = 100.0,
        lyapunov_strength: float = 1e-4,
        beta_diversity_strength: float = 1e-4,
        omega_diversity_strength: float = 1e-4,
        alpha_diversity_strength: float = 1e-4,
        beta_range: tuple = (0.3, 0.999),
        b_th_max: float = 2.0,
        rf_r_max: float = 0.999,
    ):
        self.grad_clip = grad_clip
        self.max_comp = max_comp
        self.lyapunov_strength = lyapunov_strength
        self.beta_diversity_strength = beta_diversity_strength
        self.omega_diversity_strength = omega_diversity_strength
        self.alpha_diversity_strength = alpha_diversity_strength
        self.beta_range = beta_range
        self.b_th_max = b_th_max
        self.rf_r_max = rf_r_max
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @torch.no_grad()
    def step(self, closure=None):
        # ================================================================
        # Phase 1: Natural Gradient 补偿
        # ----------------------------------------------------------------
        # 消除 sigmoid/softplus 激活函数对 b_beta/b_alpha/b_omega 的梯度衰减。
        # β = sigmoid(b_beta), sigmoid'(z) = β(1-β)
        #   → 当 β→0.99 时 sigmoid' = 0.0099，梯度衰减 100×
        # 补偿: grad /= activation'(b)，等价于在 β/α/ω 空间做梯度下降。
        # ================================================================
        for group in self.param_groups:
            dynamics = group.get('dynamics')
            if dynamics is None:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                if dynamics == 'b_beta':
                    beta = torch.sigmoid(p.data)
                    sigmoid_deriv = (beta * (1.0 - beta)).clamp(min=1.0 / self.max_comp)
                    p.grad.div_(sigmoid_deriv)
                elif dynamics == 'b_alpha':
                    softplus_deriv = torch.sigmoid(p.data).clamp(min=0.1)
                    p.grad.div_(softplus_deriv)
                elif dynamics == 'b_omega':
                    softplus_deriv = torch.sigmoid(p.data).clamp(min=0.1)
                    p.grad.div_(softplus_deriv)

        # ================================================================
        # Phase 2: 梯度裁剪
        # ================================================================
        if self.grad_clip > 0:
            all_params = [p for g in self.param_groups for p in g['params']
                          if p.grad is not None]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)

        # ================================================================
        # Phase 3: AdamW 标准步
        # ================================================================
        loss = super().step(closure)

        # ================================================================
        # Phase 4: 双侧 Lyapunov 惩罚 (Gradient Flossing)
        # ----------------------------------------------------------------
        # 原始 (单侧): penalty = (log β)²
        #   β→0: penalty→+∞ ✓  |  β→1: penalty→0 ✗ (死寂方向无惩罚)
        #
        # 修正 (双侧): penalty = (log β)² + (log(1-β))²
        #   β→0: (log β)²→+∞ ✓  |  β→1: (log(1-β))²→+∞ ✓
        #   β=0.5: penalty 最小 (中央平衡点)
        #
        # d/db [(log β)² + (log(1-β))²]
        #   = 2·log(β)·(1-β) − 2·β·log(1-β)
        #
        # 证明:
        #   d/db [log(σ(b))]² = 2·log(β)·σ'(b)/β = 2·log(β)·(1-β)
        #   d/db [log(σ(-b))]² = 2·log(1-β)·(-σ'(b))/(1-β) = -2·β·log(1-β)
        # ================================================================
        if self.lyapunov_strength > 0:
            for group in self.param_groups:
                if group.get('dynamics') != 'b_beta':
                    continue
                group_lr = group['lr']
                for p in group['params']:
                    beta = torch.sigmoid(p.data)
                    log_beta = torch.log(beta.clamp(min=1e-7))
                    log_1m_beta = torch.log((1.0 - beta).clamp(min=1e-7))
                    penalty_grad = (2.0 * log_beta * (1.0 - beta)
                                    - 2.0 * beta * log_1m_beta)
                    p.data.sub_(penalty_grad,
                                alpha=self.lyapunov_strength * group_lr)

        # ================================================================
        # Phase 5: β 多样性维护
        # ----------------------------------------------------------------
        # 同一 D 通道内 N 个神经元的 β 应覆盖多个时间尺度。
        # 排斥力 = -(β - mean_β)，推离通道均值。
        # 链式法则转回 logit 空间: Δb = repulsion · β·(1-β)
        # ================================================================
        if self.beta_diversity_strength > 0:
            for group in self.param_groups:
                if group.get('dynamics') != 'b_beta':
                    continue
                N = group.get('N')
                if N is None or N <= 1:
                    continue
                group_lr = group['lr']
                for p in group['params']:
                    beta = torch.sigmoid(p.data)
                    D = beta.numel() // N
                    beta_2d = beta.reshape(D, N)
                    mean_beta = beta_2d.mean(dim=1, keepdim=True)
                    repulsion = -(beta_2d - mean_beta)
                    d_b = repulsion * beta_2d * (1.0 - beta_2d)
                    p.data.add_(d_b.reshape(-1),
                                alpha=self.beta_diversity_strength * group_lr)

        # ================================================================
        # Phase 6: ω 多样性维护
        # ----------------------------------------------------------------
        # 同一 D 通道内 N 个 ω 应覆盖不同振荡频率 (Rhythm-SNN)。
        # softplus'(b) = sigmoid(b)，链式法则转回 b_omega 空间。
        # ================================================================
        if self.omega_diversity_strength > 0:
            for group in self.param_groups:
                if group.get('dynamics') != 'b_omega':
                    continue
                N = group.get('N')
                if N is None or N <= 1:
                    continue
                group_lr = group['lr']
                for p in group['params']:
                    omega = F.softplus(p.data)
                    D = omega.numel() // N
                    omega_2d = omega.reshape(D, N)
                    mean_omega = omega_2d.mean(dim=1, keepdim=True)
                    repulsion = -(omega_2d - mean_omega)
                    sigmoid_b = torch.sigmoid(p.data).reshape(D, N)
                    d_b = repulsion * sigmoid_b
                    p.data.add_(d_b.reshape(-1),
                                alpha=self.omega_diversity_strength * group_lr)

        # ================================================================
        # Phase 7: α 多样性维护
        # ----------------------------------------------------------------
        # 同一 D 通道内 N 个 α (写入增益) 应有差异化响应幅度。
        # α = softplus(b_alpha), softplus'(b) = sigmoid(b)
        # ================================================================
        if self.alpha_diversity_strength > 0:
            for group in self.param_groups:
                if group.get('dynamics') != 'b_alpha':
                    continue
                N = group.get('N')
                if N is None or N <= 1:
                    continue
                group_lr = group['lr']
                for p in group['params']:
                    alpha = F.softplus(p.data)
                    D = alpha.numel() // N
                    alpha_2d = alpha.reshape(D, N)
                    mean_alpha = alpha_2d.mean(dim=1, keepdim=True)
                    repulsion = -(alpha_2d - mean_alpha)
                    sigmoid_b = torch.sigmoid(p.data).reshape(D, N)
                    d_b = repulsion * sigmoid_b
                    p.data.add_(d_b.reshape(-1),
                                alpha=self.alpha_diversity_strength * group_lr)

        # ================================================================
        # Phase 8: β 范围硬钳制（防癫痫 + 防死寂）
        # ----------------------------------------------------------------
        # β < β_min (默认 0.3): 记忆 < 2 步，癫痫风险
        # β > β_max (默认 0.999): K=16 步几乎不充电 → 死寂
        #   β=0.999 → V[16] = (1-0.999^16)·x ≈ 0.016·x → 事实上的死神经元
        #
        # 钳制 b_beta ∈ [logit(β_min), logit(β_max)]
        # ================================================================
        beta_min, beta_max = self.beta_range
        b_beta_lo = torch.logit(torch.tensor(beta_min))  # ≈ -0.847
        b_beta_hi = torch.logit(torch.tensor(beta_max))  # ≈ 6.907
        for group in self.param_groups:
            if group.get('dynamics') != 'b_beta':
                continue
            for p in group['params']:
                p.data.clamp_(b_beta_lo.item(), b_beta_hi.item())

        # ================================================================
        # Phase 9: V_th 上界钳制（防死寂）
        # ----------------------------------------------------------------
        # V_th = v_th_min + |raw_th + b_th|
        # 如果 |b_th| 增长过大，即使 raw_th=0 也 V_th >> σ_V → 永远不发放
        # 钳制 |b_th| ≤ b_th_max (默认 2.0)
        # ================================================================
        if self.b_th_max > 0:
            for group in self.param_groups:
                if group.get('dynamics') != 'b_th':
                    continue
                for p in group['params']:
                    p.data.clamp_(-self.b_th_max, self.b_th_max)

        # ================================================================
        # Phase 10: RF 稳定性投影（最终权威，不可违反）
        # ----------------------------------------------------------------
        # Resonate-and-Fire 二阶系统谱半径: r = √(β² + ω²)
        # 必须 r ≤ r_max (默认 0.999)，否则振荡发散 → 癫痫
        #
        # 投影: 若 r > r_max，等比缩放 (β, ω) ← (β, ω) · r_max/r
        # 然后映射回参数空间: b_beta ← logit(β'), b_omega ← softplus⁻¹(ω')
        #
        # 必须最后执行：Phase 4-5 可能推高 β, Phase 8 可能钳制 β 下界，
        # 都可能改变 r。只有 Phase 10 保证最终 r ≤ r_max。
        # ================================================================
        rf_r_max_sq = self.rf_r_max ** 2

        # 收集 b_beta 和 b_omega 参数（model.get_param_groups 保证同序）
        b_beta_params = []
        b_omega_params = []
        for group in self.param_groups:
            dyn = group.get('dynamics')
            if dyn == 'b_beta':
                b_beta_params.extend(group['params'])
            elif dyn == 'b_omega':
                b_omega_params.extend(group['params'])

        if len(b_beta_params) == len(b_omega_params) and b_beta_params:
            for p_beta, p_omega in zip(b_beta_params, b_omega_params):
                beta = torch.sigmoid(p_beta.data)
                omega = F.softplus(p_omega.data)
                r_sq = beta * beta + omega * omega
                violation = r_sq > rf_r_max_sq

                if not violation.any():
                    continue

                scale = self.rf_r_max * torch.rsqrt(r_sq[violation])
                beta_proj = (beta[violation] * scale).clamp(1e-7, 1.0 - 1e-7)
                omega_proj = (omega[violation] * scale).clamp(min=1e-7)

                # 映射回参数空间
                p_beta.data[violation] = torch.log(beta_proj / (1.0 - beta_proj))
                p_omega.data[violation] = _softplus_inv(omega_proj)

        return loss
