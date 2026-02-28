"""
MoE-SNNFFN: SNN 专用 Mixture-of-Experts 前馈网络（v3: 堆叠参数 + 直接 Alpha）

DeepSeek-V3 模式:
- 1 个共享 expert（always active，处理所有 token）
- E 个路由 expert（top-k 选择，输出加权）
- Auxiliary-loss-free 负载均衡（expert_bias sign rule）

训练时所有 expert 运行（SNN 时序状态连续），输出按 routing 权重加权。
推理时仅运行 shared + top-k expert（真正稀疏加速）。

v3 优化（vs v2 batched）:
- 堆叠 nn.Parameter: 消除 Python 循环重建批量张量（省 ~0.85ms）
  · expert_W_gus / expert_W_down: 直接 (E, out, in) 存储，bmm 无需 stack/cat
  · expert_gu_w / expert_gu_v_th / ...: (E, dim) 存储，单次 sigmoid 替代 E 次循环
- 直接 Alpha 浮点接口: plif_rowparam_forward_alpha 跳过 surrogate 对象创建
- 采样版自适应 alpha: 从前 4096 元素采样 std 替代全张量 u.std()
- torch.bincount 替代 expert count 循环
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import base, surrogate

from .snn_ffn import SNNFFN
from .parallel_scan import plif_rowparam_forward_alpha, plif_rowparam_forward_recompute
from .fp16_codec import fp16_decode, fp16_encode, binary_encode_ste


class MoESNNFFN(base.MemoryModule):
    """MoE-SNNFFN: 1 shared SNNFFN + E routed experts（v3 堆叠参数）。

    路由 expert 参数以堆叠 nn.Parameter 存储（非 nn.ModuleList），
    消除 forward 中的 Python 循环瓶颈。

    Args:
        D: 可见维度（输入/输出 spike 维度）
        D_ff_shared: 共享 expert 中间层维度（默认 D）
        D_ff_expert: 路由 expert 中间层维度（默认 D//2）
        num_experts: 路由 expert 数量
        top_k: 每 token 选中的 expert 数
        K: 每 token 的 SNN 时间步数
        output_v_threshold: 输出神经元阈值
        num_layers: 总层数，用于 down_proj 缩放
        layer_idx: 当前层索引
    """

    def __init__(
        self,
        D: int,
        D_ff_shared: int,
        D_ff_expert: int,
        num_experts: int,
        top_k: int,
        K: int,
        output_v_threshold: float = 0.15,
        num_layers: int = 1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.D = D
        self.D_ff = D_ff_expert
        self.num_experts = num_experts
        self.top_k = top_k
        self.K = K

        # 共享 expert（always active，残差由 MoE 层统一处理）
        self.shared_expert = SNNFFN(
            D=D, D_ff=D_ff_shared,
            output_v_threshold=output_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
            add_residual=False,
        )

        # ====== 堆叠路由 expert 参数（替代 nn.ModuleList） ======
        E = num_experts
        D_ff = D_ff_expert

        # 投影权重：(E, out_dim, in_dim)，bmm 直接使用
        self.expert_W_gus = nn.Parameter(torch.empty(E, 2 * D_ff + D, D))    # gate+up+skip
        self.expert_W_down = nn.Parameter(torch.empty(E, D, D_ff))            # down_proj

        # 神经元参数：(E, dim)
        # gate+up 神经元: 2*D_ff 维
        self.expert_gu_w = nn.Parameter(torch.empty(E, 2 * D_ff))             # sigmoid → beta
        self.expert_gu_v_th = nn.Parameter(torch.empty(E, 2 * D_ff))          # 发放阈值
        self.expert_gu_v_init = nn.Parameter(torch.zeros(E, 2 * D_ff))        # 初始膜电位

        # output 神经元: D 维
        self.expert_out_w = nn.Parameter(torch.empty(E, D))                    # sigmoid → beta
        self.expert_out_v_th = nn.Parameter(torch.empty(E, D))                 # 发放阈值
        self.expert_out_v_init = nn.Parameter(torch.zeros(E, D))               # 初始膜电位

        # 神经元膜电位状态（functional.reset_net 自动重置为 0.）
        self.register_memory('expert_gu_v', 0.)
        self.register_memory('expert_out_v', 0.)

        # Router: 基于 spike firing rate 做路由（v10: 不再依赖连续 h）
        self.router = nn.Linear(D, num_experts, bias=False)

        # DeepSeek-V3 expert_bias（非梯度参数，sign rule 更新）
        self.register_buffer('expert_bias', torch.zeros(num_experts))

        # surrogate 参数
        self.surrogate_alpha_0 = 4.0
        self.sigma_0 = 1.0

        # 初始化
        self._initialize_expert_parameters(num_layers, output_v_threshold)

    def _initialize_expert_parameters(self, num_layers: int, v_threshold: float):
        """初始化堆叠 expert 参数，匹配 SNNFFN 的初始化模式。

        投影权重:
        - gate/up/skip: Kaiming uniform per expert slice
        - down: Kaiming uniform × 1/√(num_layers)

        神经元参数:
        - w: normal_(init_w, 0.5)  where init_w = -log(tau-1), tau=2.0
        - v_th: uniform_(v_threshold*0.5, v_threshold*1.5)
        - v_init: zeros (already initialized)
        """
        E = self.num_experts
        D = self.D
        D_ff = self.D_ff

        # ---- 投影权重 ----
        for e in range(E):
            # gate_proj: (D_ff, D) — Kaiming uniform
            nn.init.kaiming_uniform_(self.expert_W_gus.data[e, :D_ff, :], a=math.sqrt(5))
            # up_proj: (D_ff, D) — Kaiming uniform
            nn.init.kaiming_uniform_(self.expert_W_gus.data[e, D_ff:2 * D_ff, :], a=math.sqrt(5))
            # skip_proj: (D, D) — Kaiming uniform
            nn.init.kaiming_uniform_(self.expert_W_gus.data[e, 2 * D_ff:, :], a=math.sqrt(5))
            # down_proj: (D, D_ff) — Kaiming uniform × 1/√(num_layers)
            nn.init.kaiming_uniform_(self.expert_W_down.data[e], a=math.sqrt(5))
            self.expert_W_down.data[e].mul_(1.0 / math.sqrt(num_layers))

        # ---- 神经元参数（匹配 PLIFNode.__init__）----
        init_w = -math.log(2.0 - 1.0)  # tau=2.0 → init_w=0
        # gate+up 神经元
        self.expert_gu_w.data.normal_(init_w, 0.5)
        self.expert_gu_v_th.data.uniform_(v_threshold * 0.5, v_threshold * 1.5)
        # output 神经元
        self.expert_out_w.data.normal_(init_w, 0.5)
        self.expert_out_v_th.data.uniform_(v_threshold * 0.5, v_threshold * 1.5)

    def _compute_adaptive_alpha(self, u, sample_size=4096):
        """从前 sample_size 个元素采样 std（替代全张量 u.std()）。

        Returns:
            alpha: float — 自适应 surrogate alpha 值
        """
        with torch.no_grad():
            if u.numel() <= sample_size:
                sigma = u.std().clamp(min=1e-6).item()
            else:
                sigma = u.reshape(-1)[:sample_size].std().clamp(min=1e-6).item()
            return max(1.0, min(self.surrogate_alpha_0 * self.sigma_0 / sigma, 20.0))

    def _batched_expert_forward(self, spike_in_seq):
        """批量执行所有路由 expert（Per-Token Independent Scan）。

        优化: Per-Token Reshape — TK=8192 拆分为 T×K=512×16 独立短序列。
        Routed expert 不应有跨 token 膜电位状态（routing 是 per-token 决策），
        拆分后 PLIF scan 从 K=8192 变为 K=16，grid blocks 从 ~64 增至 ~16384。

        Args:
            spike_in_seq: (TK, B, D) — spike 输入

        Returns:
            expert_outs: (E, TK, B, D) — 所有 expert 的 spike 输出
        """
        E = self.num_experts
        TK, B, D = spike_in_seq.shape
        D_ff = self.D_ff
        K_steps = self.K
        T = TK // K_steps
        TB = T * B
        flat = spike_in_seq.reshape(TK * B, D)

        # ====== Phase 1: 批量 GEMM — gate+up+skip 投影（无 stack/cat） ======
        proj = torch.bmm(
            flat.unsqueeze(0).expand(E, -1, -1),
            self.expert_W_gus.transpose(1, 2),
        )
        I_gate_up = proj[:, :, :2 * D_ff].reshape(E, TK, B, 2 * D_ff)
        I_skip = proj[:, :, 2 * D_ff:].reshape(E, TK, B, D)

        # ====== Phase 2: Per-Token PLIF scan — gate+up 神经元 ======
        beta_gu = torch.sigmoid(self.expert_gu_w)           # (E, 2*D_ff)
        scale_gu = 1.0 - beta_gu                            # (E, 2*D_ff)
        u_merged = I_gate_up * scale_gu.reshape(E, 1, 1, 2 * D_ff)

        # Per-Token Reshape: (E, TK, B, dim) → (K, E*TB, dim)
        # rows 按 E-major 排列: [e0_t0b0, e0_t0b1, ..., e0_t511b3, e1_t0b0, ...]
        u_flat = (u_merged
            .reshape(E, T, K_steps, B, 2 * D_ff)
            .permute(2, 0, 1, 3, 4)
            .reshape(K_steps, E * TB, 2 * D_ff)
            .contiguous())

        # beta/v_th/v_init: (E, dim) → (E*TB, dim)
        # E-major: each expert's param repeated TB times, contiguous per expert
        beta_row = beta_gu.unsqueeze(1).expand(E, TB, -1).reshape(E * TB, 2 * D_ff).contiguous()
        v_th_row = self.expert_gu_v_th.unsqueeze(1).expand(E, TB, -1).reshape(E * TB, 2 * D_ff).contiguous()
        v_init = self.expert_gu_v_init.to(dtype=flat.dtype).unsqueeze(1).expand(E, TB, -1).reshape(E * TB, 2 * D_ff).contiguous()

        alpha_gu = self._compute_adaptive_alpha(u_flat)
        spike_flat, _ = plif_rowparam_forward_recompute(
            beta_row, u_flat, v_th_row, v_init, alpha_gu,
        )  # (K, E*TB, 2*D_ff) — AND 门直接在此布局操作

        # ====== Phase 3: AND 门（在 kernel 输出布局上直接操作，无 reshape 副本）======
        gate_spike = spike_flat[:, :, :D_ff]     # (K, E*TB, D_ff)
        up_spike = spike_flat[:, :, D_ff:]       # (K, E*TB, D_ff)
        gated = gate_spike * up_spike             # (K, E*TB, D_ff)

        # ====== Phase 4: Reshape 回 + 批量 down_proj GEMM ======
        # (K, E*TB, D_ff) → (K, E, T, B, D_ff) → (E, T, K, B, D_ff) → (E, TK*B, D_ff)
        gated_flat = (gated
            .reshape(K_steps, E, T, B, D_ff)
            .permute(1, 2, 0, 3, 4)
            .reshape(E, TK * B, D_ff)
            .contiguous())
        I_out = torch.bmm(gated_flat, self.expert_W_down.transpose(1, 2))
        I_out = I_out.reshape(E, TK, B, D) + I_skip

        # ====== Phase 5: Per-Token output PLIF scan ======
        beta_out = torch.sigmoid(self.expert_out_w)
        scale_out = 1.0 - beta_out
        u_out = I_out * scale_out.reshape(E, 1, 1, D)

        u_out_flat = (u_out
            .reshape(E, T, K_steps, B, D)
            .permute(2, 0, 1, 3, 4)
            .reshape(K_steps, E * TB, D)
            .contiguous())

        beta_out_row = beta_out.unsqueeze(1).expand(E, TB, -1).reshape(E * TB, D).contiguous()
        v_th_out_row = self.expert_out_v_th.unsqueeze(1).expand(E, TB, -1).reshape(E * TB, D).contiguous()
        v_init_out = self.expert_out_v_init.to(dtype=flat.dtype).unsqueeze(1).expand(E, TB, -1).reshape(E * TB, D).contiguous()

        alpha_out = self._compute_adaptive_alpha(u_out_flat)
        spike_out_flat, _ = plif_rowparam_forward_recompute(
            beta_out_row, u_out_flat, v_th_out_row, v_init_out, alpha_out,
        )  # (K, E*TB, D)

        # Reshape 回: (K, E*TB, D) → (E, TK, B, D)
        expert_outs = (spike_out_flat
            .reshape(K_steps, E, T, B, D)
            .permute(1, 2, 0, 3, 4)
            .reshape(E, TK, B, D)
            .contiguous())
        return expert_outs

    def _single_expert_forward(self, e_idx, spike_in):
        """单个 expert 的单步前向（推理用）。

        用参数索引 self.expert_W_gus[e_idx] 等，保持简单。

        Args:
            e_idx: expert 索引
            spike_in: (B, D) — spike 输入

        Returns:
            spike_out: (B, D)
        """
        B, D = spike_in.shape
        D_ff = self.D_ff

        # gate+up+skip 投影
        proj = F.linear(spike_in, self.expert_W_gus[e_idx])  # (B, 2*D_ff+D)
        I_gate = proj[:, :D_ff]
        I_up = proj[:, D_ff:2 * D_ff]
        I_skip = proj[:, 2 * D_ff:]

        # gate+up 神经元（单步）
        beta_gu = torch.sigmoid(self.expert_gu_w[e_idx])  # (2*D_ff,)
        beta_gate = beta_gu[:D_ff]
        beta_up = beta_gu[D_ff:]
        v_th_gate = self.expert_gu_v_th[e_idx, :D_ff]
        v_th_up = self.expert_gu_v_th[e_idx, D_ff:]

        # gate 神经元单步
        if isinstance(self.expert_gu_v, float):
            v_gate = self.expert_gu_v_init[e_idx, :D_ff].to(dtype=spike_in.dtype).unsqueeze(0).expand(B, D_ff)
            v_up = self.expert_gu_v_init[e_idx, D_ff:].to(dtype=spike_in.dtype).unsqueeze(0).expand(B, D_ff)
        else:
            # expert_gu_v is (E*B, 2*D_ff) in parallel mode, but in single step we need per-expert state
            # In single step mode, states are managed differently
            v_gate = self.expert_gu_v_init[e_idx, :D_ff].to(dtype=spike_in.dtype).unsqueeze(0).expand(B, D_ff)
            v_up = self.expert_gu_v_init[e_idx, D_ff:].to(dtype=spike_in.dtype).unsqueeze(0).expand(B, D_ff)

        u_gate = (1.0 - beta_gate) * I_gate
        v_gate = beta_gate * v_gate + u_gate
        surr = surrogate.Sigmoid(alpha=self.surrogate_alpha_0)
        gate_spike = surr(v_gate - v_th_gate)
        v_gate = v_gate - gate_spike * v_th_gate

        u_up = (1.0 - beta_up) * I_up
        v_up = beta_up * v_up + u_up
        up_spike = surr(v_up - v_th_up)
        v_up = v_up - up_spike * v_th_up

        # AND 门 + down_proj + skip
        gated = gate_spike * up_spike
        I_out = F.linear(gated, self.expert_W_down[e_idx]) + I_skip

        # output 神经元单步
        beta_o = torch.sigmoid(self.expert_out_w[e_idx])  # (D,)
        v_th_o = self.expert_out_v_th[e_idx]

        if isinstance(self.expert_out_v, float):
            v_out = self.expert_out_v_init[e_idx].to(dtype=spike_in.dtype).unsqueeze(0).expand(B, D)
        else:
            v_out = self.expert_out_v_init[e_idx].to(dtype=spike_in.dtype).unsqueeze(0).expand(B, D)

        u_out = (1.0 - beta_o) * I_out
        v_out = beta_o * v_out + u_out
        spike_out = surr(v_out - v_th_o)
        v_out = v_out - spike_out * v_th_o

        return spike_out

    def forward_parallel(self, spike_in_seq):
        """
        并行前向传播（v10: router 使用 spike firing rate）。

        Args:
            spike_in_seq: (TK, B, D) — spike 输入

        Returns:
            spike_out: (TK, B, D)
            router_info: dict — routing 统计（用于 expert_bias 更新）
        """
        TK, B, D = spike_in_seq.shape
        K = self.K
        T = TK // K
        E = self.num_experts

        # ---- Router: spike firing rate 平均 K 步 → per-token routing ----
        spike_rate = spike_in_seq.reshape(T, K, B, D).mean(dim=1)  # (T, B, D), ∈ [0,1]
        router_logits = self.router(spike_rate)  # (T, B, E)

        # Top-k 选择（DeepSeek-V3: 加 bias 影响选择，softmax 用原始 logits）
        biased_logits = router_logits + self.expert_bias
        _, top_k_idx = biased_logits.topk(self.top_k, dim=-1)  # (T, B, k)

        # Softmax 归一化（对选中 expert 的原始 logits）
        top_k_original = router_logits.gather(-1, top_k_idx)
        routing_weights = F.softmax(top_k_original, dim=-1)  # (T, B, k)

        # 统计 expert 选中次数（bincount 替代循环）
        counts = torch.bincount(top_k_idx.reshape(-1), minlength=E).float()

        # ---- 高效路由权重构建（scatter_add_ 替代 E 次 mask 循环）----
        full_weights = torch.zeros(T, B, E, device=spike_in_seq.device, dtype=spike_in_seq.dtype)
        full_weights.scatter_add_(-1, top_k_idx, routing_weights)
        # (T, B, E) → (TK, B, E)
        weights_tk = full_weights.unsqueeze(1).expand(T, K, B, E).reshape(TK, B, E)

        # ---- Shared expert: 始终处理所有 token ----
        shared_out = self.shared_expert.forward_parallel(spike_in_seq)  # (TK, B, D)

        # ---- Routed experts: decode-combine-encode（保证 IEEE 754 位结构完整） ----
        if self.training:
            # 训练: 批量执行所有 expert（无循环）
            expert_spikes = self._batched_expert_forward(spike_in_seq)  # (E, TK, B, D) binary

            # Decode 到 per-token 连续空间
            shared_val = fp16_decode(shared_out, T, K)  # (B, T, D)
            expert_vals = torch.stack(
                [fp16_decode(expert_spikes[e], T, K) for e in range(E)], dim=0
            )  # (E, B, T, D)

            # Per-token 连续空间加权合并
            # full_weights: (T, B, E) → (E, B, T, 1)
            w = full_weights.permute(2, 1, 0).unsqueeze(-1)  # (E, B, T, 1)
            combined = shared_val + (expert_vals * w).sum(dim=0)  # (B, T, D)

            # 加残差: decode(spike_in) + combined
            residual_val = fp16_decode(spike_in_seq, T, K)  # (B, T, D)
            total = combined + residual_val  # (B, T, D)

            # 重编码为二进制（STE backward）
            out = binary_encode_ste(total)  # (TK, B, D) binary {0,1}
        else:
            # 推理: 仅运行选中的 expert（真正稀疏加速）
            out_val = fp16_decode(shared_out, T, K)  # (B, T, D)
            active_experts = top_k_idx.unique()
            for e_idx_val in active_experts:
                e_idx = e_idx_val.item()
                expert_spike = self._inference_expert_parallel(e_idx, spike_in_seq)
                expert_val = fp16_decode(expert_spike, T, K)  # (B, T, D)
                w_e = full_weights[:, :, e_idx].permute(1, 0).unsqueeze(-1)  # (B, T, 1)
                out_val = out_val + expert_val * w_e
            # 加残差
            residual_val = fp16_decode(spike_in_seq, T, K)
            total = out_val + residual_val
            out = fp16_encode(total, 16)  # 推理不需 STE，直接 encode

        return out, {
            'expert_counts': counts.detach(),
            'top_k_idx': top_k_idx.detach(),
        }

    def _inference_expert_parallel(self, e_idx, spike_in_seq):
        """推理时单个 expert 的 parallel forward（用参数索引）。

        Args:
            e_idx: expert 索引
            spike_in_seq: (TK, B, D)

        Returns:
            spike_out: (TK, B, D)
        """
        TK, B, D = spike_in_seq.shape
        D_ff = self.D_ff
        flat = spike_in_seq.reshape(TK * B, D)

        # gate+up+skip 投影
        proj = F.linear(flat, self.expert_W_gus[e_idx])  # (TK*B, 2*D_ff+D)
        I_gate_up = proj[:, :2 * D_ff].reshape(TK, B, 2 * D_ff)
        I_skip = proj[:, 2 * D_ff:].reshape(TK, B, D)

        # gate+up PLIF
        beta_gu = torch.sigmoid(self.expert_gu_w[e_idx])  # (2*D_ff,)
        scale_gu = 1.0 - beta_gu
        u_merged = I_gate_up * scale_gu

        beta_row = beta_gu.unsqueeze(0).expand(B, 2 * D_ff).contiguous()
        v_th_row = self.expert_gu_v_th[e_idx].unsqueeze(0).expand(B, 2 * D_ff).contiguous()
        v_init_gu = self.expert_gu_v_init[e_idx].to(dtype=flat.dtype).unsqueeze(0).expand(B, 2 * D_ff)

        alpha_gu = self._compute_adaptive_alpha(u_merged)
        spike_merged, V_post_merged = plif_rowparam_forward_alpha(
            beta_row, u_merged, v_th_row, v_init_gu, alpha_gu,
        )

        gate_spike = spike_merged[:, :, :D_ff]
        up_spike = spike_merged[:, :, D_ff:]
        gated = gate_spike * up_spike

        # down_proj + skip
        gated_flat = gated.reshape(TK * B, D_ff)
        I_out = F.linear(gated_flat, self.expert_W_down[e_idx]).reshape(TK, B, D) + I_skip

        # output PLIF
        beta_out = torch.sigmoid(self.expert_out_w[e_idx])
        u_out = (1.0 - beta_out) * I_out
        beta_out_row = beta_out.unsqueeze(0).expand(B, D).contiguous()
        v_th_out_row = self.expert_out_v_th[e_idx].unsqueeze(0).expand(B, D).contiguous()
        v_init_out = self.expert_out_v_init[e_idx].to(dtype=flat.dtype).unsqueeze(0).expand(B, D)

        alpha_out = self._compute_adaptive_alpha(u_out)
        spike_out, _ = plif_rowparam_forward_alpha(
            beta_out_row, u_out, v_th_out_row, v_init_out, alpha_out,
        )
        return spike_out

    def single_step_forward(self, spike_in):
        """
        单步前向传播（推理/生成用，v10: router 使用 spike）。

        Args:
            spike_in: (B, D) — spike 输入

        Returns:
            spike_out: (B, D)
        """
        # Router: 单步直接用 spike 做路由（binary {0,1} 表示当前活跃特征）
        router_logits = self.router(spike_in)  # (B, E)
        biased_logits = router_logits + self.expert_bias
        _, top_k_idx = biased_logits.topk(self.top_k, dim=-1)  # (B, k)
        top_k_original = router_logits.gather(-1, top_k_idx)
        routing_weights = F.softmax(top_k_original, dim=-1)  # (B, k)

        # Shared expert
        out = self.shared_expert.single_step_forward(spike_in)

        # 仅运行被选中的 expert
        for e_idx in range(self.num_experts):
            mask = (top_k_idx == e_idx)  # (B, k) bool
            if not mask.any():
                continue
            weight = (routing_weights * mask.float()).sum(dim=-1, keepdim=True)  # (B, 1)
            expert_out = self._single_expert_forward(e_idx, spike_in)
            out = out + expert_out * weight

        return out

    def update_expert_bias(self, expert_counts, update_rate=0.001):
        """DeepSeek-V3 式 auxiliary-loss-free 负载均衡。

        通过 sign rule 更新 expert_bias，使选中次数低于平均的 expert
        获得更高的 bias（更容易被选中），反之亦然。

        Args:
            expert_counts: (E,) — 各 expert 被选中的次数
            update_rate: bias 更新步长
        """
        target = expert_counts.sum() / self.num_experts
        self.expert_bias += update_rate * torch.sign(target - expert_counts)
