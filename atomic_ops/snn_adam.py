"""
SNNAdam: SNN 专用优化器 — Adam + 神经动力学增强（无 weight decay，对齐 HappyLLM 预训练）。

与 SNNAdamW 的唯一区别：继承 Adam（无解耦 weight decay），对齐 HappyLLM 预训练原始设计。
SFT 阶段使用 SNNAdamW（带 weight decay）。

step() 内部 11 个阶段（与 SNNAdamW 完全一致）：

  ── 梯度预处理 ──
  1. Natural Gradient 补偿（b_beta/b_alpha/b_omega 梯度除以激活导数）
  2. 梯度裁剪（全参数 clip_grad_norm_，foreach 加速）
  ── 参数更新 ──
  3. Adam 标准步 + Magma 掩码（super().step() + 动量对齐随机掩码，可选）
  ── 软正则化（梯度级微调） ──
  4. 双侧 Lyapunov 惩罚（Gradient Flossing: 推 β 远离 0 和 1 两端）
  5. β 多样性维护（同一 D 通道内 N 个 β 互斥）
  6. ω 多样性维护（同一 D 通道内 N 个 ω 互斥）
  7. α 多样性维护（同一 D 通道内 N 个 α 互斥）
  8. V_th 多样性维护（同一 D 通道内 N 个 V_th 互斥 — 神经元趋同惩罚）
  ── 硬约束（安全网） ──
  9. β 范围钳制（sigmoid(b_beta) ∈ [β_min, β_max]，防癫痫+防死寂）
  10. V_th 上界钳制（|b_th| ≤ b_th_max，防死寂）
  11. RF 稳定性投影（√(β²+ω²) ≤ r_max，防癫痫，最终权威）

合并参数组: train_ddp.py 传入的参数组按 lr_mult 合并为 ~2 组
  · (lr_mult=1.0): 权重矩阵 + norm + router
  · (lr_mult=10.0): 神经元 + dynamics 参数

文献基础：
  - Gradient Flossing (Krein+, NeurIPS'23): Lyapunov 指数正则化
  - Curse of Memory (Zucchet+, NeurIPS'24): β→1 参数敏感度 O(1/(1-β²)^{3/2})
  - LRU (Orvieto+, ICML'23): 离散线性递归稳定性约束
  - Rhythm-SNN (Nature Comm.'25): 异构振荡频率多样性
  - Magma (Google, arXiv 2602.15322, Feb '26): 动量对齐梯度掩码
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam

from .snn_adamw import _softplus_inv


class SNNAdam(Adam):
    """SNN 专用优化器：Adam + 10 阶段神经动力学增强（无 weight decay）。

    对齐 HappyLLM 预训练设计：预训练阶段不使用 weight decay。
    内部合并参数组 + foreach 加速，与 SNNAdamW 相同的 10 阶段增强。

    Args:
        params: 参数组（每组可含 'dynamics' 和 'N' 键标记神经动力学参数）
        lr: 基础学习率
        betas: Adam 动量系数
        eps: Adam epsilon
        grad_clip: 梯度裁剪阈值（0 = 不裁剪）
        max_comp: Natural Gradient 补偿因子上限
        lyapunov_strength: 双侧 Lyapunov 正则化强度
        beta_diversity_strength: β 多样性排斥强度
        omega_diversity_strength: ω 多样性排斥强度
        alpha_diversity_strength: α 多样性排斥强度
        vth_diversity_strength: V_th 多样性排斥强度（神经元趋同惩罚）
        beta_range: (β_min, β_max) 硬钳制范围
        b_th_max: b_th 绝对值上限（V_th 上界 = v_th_min + b_th_max）
        rf_r_max: RF 谱半径上限 √(β²+ω²) ≤ r_max
        magma: 启用 Magma 动量对齐梯度掩码（默认关闭）
        magma_tau: Magma 温度 τ（cosine similarity 缩放因子）
        magma_p: Magma 掩码丢弃率（0.5 = 50% block 不更新）
        magma_ema: Magma 对齐分数 EMA 系数
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        grad_clip: float = 1.0,
        max_comp: float = 100.0,
        lyapunov_strength: float = 1e-4,
        beta_diversity_strength: float = 1e-4,
        omega_diversity_strength: float = 1e-4,
        alpha_diversity_strength: float = 1e-4,
        vth_diversity_strength: float = 1e-4,
        beta_range: tuple = (0.3, 0.999),
        b_th_max: float = 2.0,
        rf_r_max: float = 0.999,
        magma: bool = False,
        magma_tau: float = 2.0,
        magma_p: float = 0.5,
        magma_ema: float = 0.9,
    ):
        self.grad_clip = grad_clip
        self.max_comp = max_comp
        self.lyapunov_strength = lyapunov_strength
        self.beta_diversity_strength = beta_diversity_strength
        self.omega_diversity_strength = omega_diversity_strength
        self.alpha_diversity_strength = alpha_diversity_strength
        self.vth_diversity_strength = vth_diversity_strength
        self.beta_range = beta_range
        self.b_th_max = b_th_max
        self.rf_r_max = rf_r_max
        self.magma = magma
        self.magma_tau = magma_tau
        self.magma_p = magma_p
        self.magma_ema = magma_ema

        # ---- 提取 dynamics 元数据 ----
        self._dyn_params = {'b_beta': [], 'b_alpha': [], 'b_omega': [], 'b_th': []}
        self._dyn_N = {}  # id(p) → N (状态扩展因子)

        # ---- 合并参数组（按 lr_mult 去重，无 weight_decay 维度） ----
        merged = {}  # lr_mult → group dict
        dynamics_lr_mult = 1.0

        for group in params:
            dynamics = group.get('dynamics')
            N_val = group.get('N')
            lr_mult = group.get('lr_mult', 1.0)

            for p in group['params']:
                if dynamics in self._dyn_params:
                    self._dyn_params[dynamics].append(p)
                    if N_val is not None:
                        self._dyn_N[id(p)] = N_val
                    dynamics_lr_mult = lr_mult

            if lr_mult not in merged:
                merged[lr_mult] = {
                    'params': [],
                    'lr': lr * lr_mult,
                    'lr_mult': lr_mult,
                }
            merged[lr_mult]['params'].extend(group['params'])

        self._dynamics_lr_mult = dynamics_lr_mult

        # 预计算硬钳制边界
        self._b_beta_lo = torch.logit(torch.tensor(beta_range[0])).item()
        self._b_beta_hi = torch.logit(torch.tensor(beta_range[1])).item()

        super().__init__(
            list(merged.values()), lr=lr, betas=betas, eps=eps,
            foreach=True,
        )

    def _get_dynamics_lr(self):
        """获取当前 dynamics 参数组的学习率。"""
        for g in self.param_groups:
            if abs(g.get('lr_mult', 1.0) - self._dynamics_lr_mult) < 0.01:
                return g['lr']
        return self.defaults['lr'] * self._dynamics_lr_mult

    @torch.no_grad()
    def step(self, closure=None):
        # ================================================================
        # Phase 1: Natural Gradient 补偿
        # ================================================================
        for p in self._dyn_params['b_beta']:
            if p.grad is None:
                continue
            beta = torch.sigmoid(p.data)
            p.grad.div_((beta * (1.0 - beta)).clamp(min=1.0 / self.max_comp))

        for p in self._dyn_params['b_alpha']:
            if p.grad is None:
                continue
            p.grad.div_(torch.sigmoid(p.data).clamp(min=0.1))

        for p in self._dyn_params['b_omega']:
            if p.grad is None:
                continue
            p.grad.div_(torch.sigmoid(p.data).clamp(min=0.1))

        # ================================================================
        # Phase 2: 梯度裁剪
        # ================================================================
        if self.grad_clip > 0:
            all_params = [p for g in self.param_groups for p in g['params']
                          if p.grad is not None]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip,
                                               foreach=True)

        # ================================================================
        # Phase 3: Adam 标准步 + Magma 掩码
        # ----------------------------------------------------------------
        # Magma (arXiv 2602.15322): per-block 随机掩码 + 动量对齐缩放
        #   1. cos(momentum, gradient) → sigmoid(·/τ) → EMA 平滑 → 对齐分数 s
        #   2. Bernoulli 掩码 (p=0.5): 50% block 不更新
        #   3. 保留块: Δθ 缩放 s; 丢弃块: 恢复 θ_old
        # 动量 (exp_avg/exp_avg_sq) 始终 dense 更新，不受掩码影响。
        # ================================================================
        _magma_info = {}
        if self.magma:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'exp_avg' not in state:
                        continue  # 第一步无动量，跳过
                    m_flat = state['exp_avg'].reshape(-1)
                    g_flat = p.grad.reshape(-1)
                    cos_sim = (torch.dot(m_flat, g_flat)
                               / (m_flat.norm() * g_flat.norm() + 1e-8))
                    s_tilde = torch.sigmoid(cos_sim / self.magma_tau)
                    if 'magma_s' not in state:
                        state['magma_s'] = s_tilde.clone()
                    else:
                        state['magma_s'].mul_(self.magma_ema).add_(
                            s_tilde, alpha=1.0 - self.magma_ema)
                    keep = (torch.rand(1, device=p.device).item() >= self.magma_p)
                    _magma_info[id(p)] = (
                        p.data.clone(), state['magma_s'].item(), keep)

        loss = super().step(closure)

        if self.magma and _magma_info:
            for group in self.param_groups:
                for p in group['params']:
                    info = _magma_info.get(id(p))
                    if info is None:
                        continue
                    theta_old, s, keep = info
                    if keep:
                        # 保留块: θ = θ_old + s·Δθ
                        p.data.sub_(theta_old).mul_(s).add_(theta_old)
                    else:
                        # 丢弃块: 恢复原始参数
                        p.data.copy_(theta_old)

        dyn_lr = self._get_dynamics_lr()

        # ================================================================
        # Phase 4: 双侧 Lyapunov 惩罚 (Gradient Flossing)
        # ================================================================
        if self.lyapunov_strength > 0:
            coeff = self.lyapunov_strength * dyn_lr
            for p in self._dyn_params['b_beta']:
                beta = torch.sigmoid(p.data)
                log_beta = torch.log(beta.clamp(min=1e-7))
                log_1m_beta = torch.log((1.0 - beta).clamp(min=1e-7))
                penalty_grad = (2.0 * log_beta * (1.0 - beta)
                                - 2.0 * beta * log_1m_beta)
                p.data.sub_(penalty_grad, alpha=coeff)

        # ================================================================
        # Phase 5: β 多样性维护
        # ================================================================
        if self.beta_diversity_strength > 0:
            coeff = self.beta_diversity_strength * dyn_lr
            for p in self._dyn_params['b_beta']:
                N = self._dyn_N.get(id(p))
                if N is None or N <= 1:
                    continue
                beta = torch.sigmoid(p.data)
                D = beta.numel() // N
                beta_2d = beta.reshape(D, N)
                mean_beta = beta_2d.mean(dim=1, keepdim=True)
                repulsion = (beta_2d - mean_beta)
                d_b = repulsion * beta_2d * (1.0 - beta_2d)
                p.data.add_(d_b.reshape(-1), alpha=coeff)

        # ================================================================
        # Phase 6: ω 多样性维护
        # ================================================================
        if self.omega_diversity_strength > 0:
            coeff = self.omega_diversity_strength * dyn_lr
            for p in self._dyn_params['b_omega']:
                N = self._dyn_N.get(id(p))
                if N is None or N <= 1:
                    continue
                omega = F.softplus(p.data)
                D = omega.numel() // N
                omega_2d = omega.reshape(D, N)
                mean_omega = omega_2d.mean(dim=1, keepdim=True)
                repulsion = (omega_2d - mean_omega)
                sigmoid_b = torch.sigmoid(p.data).reshape(D, N)
                d_b = repulsion * sigmoid_b
                p.data.add_(d_b.reshape(-1), alpha=coeff)

        # ================================================================
        # Phase 7: α 多样性维护
        # ================================================================
        if self.alpha_diversity_strength > 0:
            coeff = self.alpha_diversity_strength * dyn_lr
            for p in self._dyn_params['b_alpha']:
                N = self._dyn_N.get(id(p))
                if N is None or N <= 1:
                    continue
                alpha_val = F.softplus(p.data)
                D = alpha_val.numel() // N
                alpha_2d = alpha_val.reshape(D, N)
                mean_alpha = alpha_2d.mean(dim=1, keepdim=True)
                repulsion = (alpha_2d - mean_alpha)
                sigmoid_b = torch.sigmoid(p.data).reshape(D, N)
                d_b = repulsion * sigmoid_b
                p.data.add_(d_b.reshape(-1), alpha=coeff)

        # ================================================================
        # Phase 8: V_th 多样性维护（神经元趋同惩罚）
        # ----------------------------------------------------------------
        # V_th = v_th_min + |raw_th + b_th|, ∂V_th/∂b_th = sign(·)
        # 在 b_th 空间做 centroid repulsion，等效推斥 V_th。
        # 与 Phase 5-7 组合 → 4D (β,ω,α,V_th) pairwise 趋同惩罚。
        # ================================================================
        if self.vth_diversity_strength > 0:
            coeff = self.vth_diversity_strength * dyn_lr
            for p in self._dyn_params['b_th']:
                N = self._dyn_N.get(id(p))
                if N is None or N <= 1:
                    continue
                D = p.data.numel() // N
                bth_2d = p.data.reshape(D, N)
                mean_bth = bth_2d.mean(dim=1, keepdim=True)
                repulsion = (bth_2d - mean_bth)
                p.data.add_(repulsion.reshape(-1), alpha=coeff)

        # ================================================================
        # Phase 9: β 范围硬钳制（防癫痫 + 防死寂）  [原 Phase 8]
        # ================================================================
        for p in self._dyn_params['b_beta']:
            p.data.clamp_(self._b_beta_lo, self._b_beta_hi)

        # ================================================================
        # Phase 10: V_th 上界钳制（防死寂）
        # ================================================================
        if self.b_th_max > 0:
            for p in self._dyn_params['b_th']:
                p.data.clamp_(-self.b_th_max, self.b_th_max)

        # ================================================================
        # Phase 11: RF 稳定性投影（最终权威，不可违反）
        # ================================================================
        b_beta_list = self._dyn_params['b_beta']
        b_omega_list = self._dyn_params['b_omega']

        if len(b_beta_list) == len(b_omega_list) and b_beta_list:
            rf_r_max_sq = self.rf_r_max ** 2
            for p_beta, p_omega in zip(b_beta_list, b_omega_list):
                beta = torch.sigmoid(p_beta.data)
                omega = F.softplus(p_omega.data)
                r_sq = beta * beta + omega * omega
                violation = r_sq > rf_r_max_sq

                if not violation.any():
                    continue

                scale = self.rf_r_max * torch.rsqrt(r_sq[violation])
                beta_proj = (beta[violation] * scale).clamp(1e-7, 1.0 - 1e-7)
                omega_proj = (omega[violation] * scale).clamp(min=1e-7)

                p_beta.data[violation] = torch.log(beta_proj / (1.0 - beta_proj))
                p_omega.data[violation] = _softplus_inv(omega_proj)

        return loss
