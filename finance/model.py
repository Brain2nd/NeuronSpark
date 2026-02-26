"""
SNNFinanceModel: SNN 隐状态空间金融交易模型（连续残差流 + FP16 边界编解码）

架构（三段式边界）：
  model.encode(features)     → spike_seq       # 输入边界：input_proj → FP16 位提取 → 16帧 spike
  model.snn_forward(spike)   → h_out           # 纯 SNN 核心（连续残差流，12 层）
  model.decode(h_out, seq)   → decisions, reconstruction
                                                # 输出边界：PLIFNode → spike → FP16 位重建 → heads

输出（多维连续交易决策，每资产 4 维）：
  decisions[..., 0] = position     ∈ [-1, +1]         方向+仓位（tanh）
  decisions[..., 1] = leverage     ∈ [1, max_leverage] 杠杆倍率（sigmoid 缩放）
  decisions[..., 2] = stop_loss    ∈ [sl_min, sl_max]  止损距离%（sigmoid 缩放）
  decisions[..., 3] = take_profit  ∈ [tp_min, tp_max]  止盈距离%（sigmoid 缩放）

梯度流:
  features → input_proj ──[detach]──→ spike_seq → SNN layers → h_out
    → output_neuron(surrogate) → spike → fp16_reconstruct → decode_proj → norm
    → h @ input_proj.weight    # tied reconstruction: input_proj 梯度路径
    → decision_head            # 交易决策输出

三阶段训练（统一 I/O: features → decisions）:
  Stage 1 (BC):  Huber loss, targets from rule-based strategy auto-generated
  Stage 2 (SFT): Huber loss, targets from backtested optimal decisions
  Stage 3 (RL):  PPO/A2C, reward = PnL（model 直接输出 continuous action）
"""

from dataclasses import dataclass
from typing import Optional

import os
import sys

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate
from torch.utils.checkpoint import checkpoint

from atomic_ops import SNNDecoderLayer
from atomic_ops.plif_node import PLIFNode
from atomic_ops.rms_norm import RMSNorm
from atomic_ops.parallel_scan import plif_rowparam_forward
from atomic_ops.fp16_codec import fp16_encode, fp16_decode
from atomic_ops.lateral_inhibition import LateralInhibition


# 决策向量维度索引
POS = 0   # position:    方向+仓位
LEV = 1   # leverage:    杠杆倍率
SL  = 2   # stop_loss:   止损距离
TP  = 3   # take_profit: 止盈距离
N_DECISION_DIMS = 4


@dataclass
class FinanceModelOutput:
    """模型输出容器。"""
    last_loss: Optional[torch.Tensor] = None
    decisions: Optional[torch.Tensor] = None        # (batch, seq_len, n_assets, 4)
    reconstruction: Optional[torch.Tensor] = None


class SNNFinanceModel(nn.Module):
    """
    SNN 金融交易模型（连续多维决策输出）。

    复用 NeuronSpark LM 的完整 SNN 核心（Resonate-and-Fire 动力学 + 位权脉冲编码），
    替换输入/输出边界：
      - embed_tokens → input_proj (Linear: n_features → D)
      - tied LM head → decision_head (Linear: D → n_assets×4) + tied reconstruction

    Args:
        n_features: 输入特征维度（默认 494）
        n_assets: 交易资产数（默认 1，单品种；3 = 三个交易对）
        D, N, K, num_layers, D_ff, v_th_min: SNN 核心参数
        max_leverage: 杠杆上限（默认 10）
        sl_range: 止损距离范围 (min%, max%)（默认 (0.005, 0.10) = 0.5%~10%）
        tp_range: 止盈距离范围 (min%, max%)（默认 (0.01, 0.20) = 1%~20%）
        recon_weight: 重建辅助损失权重
    """

    def __init__(
        self,
        n_features: int = 987,
        n_assets: int = 1,
        D: int = 256,
        N: int = 8,
        K: int = 16,
        num_layers: int = 12,
        D_ff: int = 768,
        v_th_min: float = 0.1,
        max_leverage: float = 10.0,
        sl_range: tuple[float, float] = (0.005, 0.10),
        tp_range: tuple[float, float] = (0.01, 0.20),
        recon_weight: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_assets = n_assets
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff
        self.recon_weight = recon_weight

        # 决策输出范围参数
        self.max_leverage = max_leverage
        self.sl_min, self.sl_max = sl_range
        self.tp_min, self.tp_max = tp_range

        # ====== Input Projection（替代 LM 的 embed_tokens）======
        self.input_proj = nn.Linear(n_features, D, bias=False)
        self.norm = LateralInhibition(D)

        # ====== 解码投影 ======
        self.decode_proj = nn.Linear(D, D)

        # ====== Decision Head（连续多维交易决策）======
        # 每资产 4 维: [position, leverage, stop_loss, take_profit]
        self.decision_head = nn.Linear(D, n_assets * N_DECISION_DIMS)

        # ====== 输出 RMSNorm + 输出神经元 ======
        self.output_norm = RMSNorm(D)
        self.output_neuron = PLIFNode(
            dim=D,
            init_tau=2.0,
            v_threshold=0.3,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
        )

        # ====== SNN Decoder Layers ======
        self.layers = nn.ModuleList([
            SNNDecoderLayer(
                D=D, N=N, K=K, D_ff=D_ff, v_th_min=v_th_min,
                block_output_v_threshold=0.05,
                ffn_output_v_threshold=0.15,
                num_layers=num_layers,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        """初始化所有可训练权重。"""
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.decode_proj.weight)
        nn.init.zeros_(self.decode_proj.bias)
        # decision_head 初始化: 小值让初始输出接近中间值
        # position tanh(0)=0（空仓），leverage sigmoid(0)=0.5（中等杠杆），
        # sl/tp sigmoid(0)=0.5（中间止损/止盈）
        nn.init.normal_(self.decision_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decision_head.bias)

    # ================================================================
    # 决策激活
    # ================================================================

    def _activate_decisions(self, raw: torch.Tensor) -> torch.Tensor:
        """将线性头的原始输出激活到各维度的物理范围。

        Args:
            raw: (batch, seq_len, n_assets, 4) 原始输出

        Returns:
            decisions: (batch, seq_len, n_assets, 4) 激活后
              [0] position    ∈ [-1, +1]
              [1] leverage    ∈ [1, max_leverage]
              [2] stop_loss   ∈ [sl_min, sl_max]
              [3] take_profit ∈ [tp_min, tp_max]
        """
        position    = torch.tanh(raw[..., POS])
        leverage    = 1.0 + (self.max_leverage - 1.0) * torch.sigmoid(raw[..., LEV])
        stop_loss   = self.sl_min + (self.sl_max - self.sl_min) * torch.sigmoid(raw[..., SL])
        take_profit = self.tp_min + (self.tp_max - self.tp_min) * torch.sigmoid(raw[..., TP])
        return torch.stack([position, leverage, stop_loss, take_profit], dim=-1)

    def _normalize_decisions(self, decisions: torch.Tensor) -> torch.Tensor:
        """将决策向量归一化到 [0, 1]，消除量纲差异以计算 loss。

        Args:
            decisions: (batch, seq_len, n_assets, 4) 激活后的决策

        Returns:
            normalized: (batch, seq_len, n_assets, 4) 各维度 ∈ [0, 1]
        """
        pos_norm = (decisions[..., POS] + 1.0) / 2.0                                  # [-1,1] → [0,1]
        lev_norm = (decisions[..., LEV] - 1.0) / (self.max_leverage - 1.0)             # [1,max] → [0,1]
        sl_norm  = (decisions[..., SL] - self.sl_min) / (self.sl_max - self.sl_min)    # [min,max] → [0,1]
        tp_norm  = (decisions[..., TP] - self.tp_min) / (self.tp_max - self.tp_min)    # [min,max] → [0,1]
        return torch.stack([pos_norm, lev_norm, sl_norm, tp_norm], dim=-1)

    # ================================================================
    # 三段式前向
    # ================================================================

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """输入边界：features → spike_seq。

        Args:
            features: (batch, seq_len, n_features) 金融特征

        Returns:
            spike_seq: (seq_len*K, batch, D), detached binary {0,1}
        """
        projected = self.input_proj(features)
        return fp16_encode(projected, K=self.K)

    def snn_forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """SNN 核心：spike_seq → h_out。"""
        h = spike_seq

        def _layer_forward(layer_mod, x):
            functional.reset_net(layer_mod)
            return layer_mod.forward_parallel(x)

        for layer_module in self.layers:
            h = checkpoint(
                _layer_forward, layer_module, h,
                use_reentrant=False,
            )
        return h

    def _output_neuron_parallel(self, h: torch.Tensor) -> torch.Tensor:
        """输出 PLIF 神经元的 parallel scan 前向：连续 h → binary spike。"""
        TK, batch, D = h.shape

        beta = self.output_neuron.beta
        u = (1.0 - beta) * h

        v_init = self.output_neuron.v
        if isinstance(v_init, float):
            v_init = self.output_neuron.expand_v_init(batch, h.device, h.dtype)

        beta_row = beta.unsqueeze(0).expand(batch, D).contiguous()
        v_th_row = self.output_neuron.v_th.unsqueeze(0).expand(batch, D).contiguous()

        spike, V_post = plif_rowparam_forward(
            beta_row, u, v_th_row, v_init,
            surrogate_function=self.output_neuron.get_surrogate(u),
        )

        self.output_neuron.v = V_post[-1].detach()
        return spike

    def decode(self, h_out: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """输出边界：连续 h → 决策 + 重建。

        Returns:
            decisions: (batch, seq_len, n_assets, 4) 激活后的交易决策
            reconstruction: (batch, seq_len, n_features) 输入重建
        """
        h_out = self.output_norm(h_out)
        spikes = self._output_neuron_parallel(h_out)
        decoded = fp16_decode(spikes, seq_len, K=self.K)
        h = self.decode_proj(decoded)
        h = self.norm(h)

        # Decision head: (batch, seq_len, n_assets * 4) → reshape → activate
        raw = self.decision_head(h)
        batch = h.shape[0]
        raw = raw.view(batch, seq_len, self.n_assets, N_DECISION_DIMS)
        decisions = self._activate_decisions(raw)

        # Tied reconstruction head: 梯度路径到 input_proj.weight
        reconstruction = h @ self.input_proj.weight

        return decisions, reconstruction

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
    ) -> FinanceModelOutput:
        """
        前向传播。

        Args:
            features: (batch, seq_len, n_features) 金融特征序列
            targets: (batch, seq_len, n_assets, 4) 目标决策向量
                     [position, leverage, stop_loss, take_profit]
            loss_mask: (batch, seq_len) 有效位置掩码

        Returns:
            FinanceModelOutput
        """
        batch, seq_len, _ = features.shape

        for layer_module in self.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.output_neuron)

        spike_seq = self.encode(features)
        h_out = self.snn_forward(spike_seq)
        decisions, reconstruction = self.decode(h_out, seq_len)

        if targets is not None:
            # 归一化到 [0,1] 消除量纲差异后计算 loss
            pred_norm = self._normalize_decisions(decisions)
            target_norm = self._normalize_decisions(targets)

            # 各维度独立 MSE，再按维度加权
            # position 最重要 (方向决定盈亏)，leverage 次之，sl/tp 辅助
            dim_weights = torch.tensor(
                [2.0, 1.5, 1.0, 1.0], device=pred_norm.device, dtype=pred_norm.dtype
            )  # (4,)
            sq_err = (pred_norm - target_norm).pow(2)  # (B, S, A, 4)
            weighted_sq_err = sq_err * dim_weights     # broadcast (4,)

            if loss_mask is not None:
                mask = loss_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
                decision_loss = (weighted_sq_err * mask).sum() / mask.sum().clamp(min=1) / (self.n_assets * dim_weights.sum())
            else:
                decision_loss = weighted_sq_err.mean() * (4.0 / dim_weights.sum())

            recon_loss = F.mse_loss(reconstruction, features)
            total_loss = decision_loss + self.recon_weight * recon_loss

            return FinanceModelOutput(
                last_loss=total_loss,
                decisions=decisions,
                reconstruction=reconstruction,
            )

        return FinanceModelOutput(decisions=decisions, reconstruction=reconstruction)

    # ================================================================
    # 梯度补偿 + 参数分组
    # ================================================================

    def compensate_modulation_gradients(self, max_comp: float = 100.0):
        """Natural Gradient 补偿（与 LM 版完全相同）。"""
        for layer_module in self.layers:
            block = layer_module.snn_block

            if block.b_beta.grad is not None:
                with torch.no_grad():
                    beta = torch.sigmoid(block.b_beta.data)
                    sigmoid_deriv = (beta * (1.0 - beta)).clamp(min=1.0 / max_comp)
                    block.b_beta.grad.div_(sigmoid_deriv)

            if block.b_alpha.grad is not None:
                with torch.no_grad():
                    softplus_deriv = torch.sigmoid(block.b_alpha.data).clamp(min=0.1)
                    block.b_alpha.grad.div_(softplus_deriv)

            if block.b_omega.grad is not None:
                with torch.no_grad():
                    softplus_deriv = torch.sigmoid(block.b_omega.data).clamp(min=0.1)
                    block.b_omega.grad.div_(softplus_deriv)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """按功能分组的可训练参数。"""
        groups = {
            'input_proj': [self.input_proj.weight],
            'decision_head': list(self.decision_head.parameters()),
            'norm': [self.norm.gain],
            'decode': list(self.decode_proj.parameters()),
            'output_neuron': [self.output_neuron.w, self.output_neuron.v_th,
                              self.output_neuron.v_init],
            'rms_norms': [self.output_norm.weight],
            'residual_projs': [],
            'input_neurons': [],
            'W_in': [], 'W_beta': [], 'W_alpha': [], 'W_th': [], 'W_omega': [],
            'W_gate': [], 'W_skip': [], 'W_out': [],
            'b_beta': [], 'b_alpha': [], 'b_th': [], 'b_omega': [],
            'block_output_neuron': [],
            'ffn_gate_proj': [], 'ffn_up_proj': [], 'ffn_down_proj': [],
            'ffn_skip_proj': [], 'ffn_neurons': [],
        }

        for layer_module in self.layers:
            block = layer_module.snn_block

            groups['residual_projs'].extend([
                layer_module.block_out_proj.weight,
                layer_module.ffn_out_proj.weight,
            ])
            groups['input_neurons'].extend([
                layer_module.input_neuron1.w,
                layer_module.input_neuron1.v_th,
                layer_module.input_neuron1.v_init,
                layer_module.input_neuron2.w,
                layer_module.input_neuron2.v_th,
                layer_module.input_neuron2.v_init,
            ])
            groups['rms_norms'].extend([
                layer_module.block_norm.weight,
                layer_module.ffn_norm.weight,
            ])

            groups['W_in'].append(block.W_in.weight)
            groups['W_beta'].extend([block.W_beta_x.weight])
            groups['W_alpha'].extend([block.W_alpha_x.weight])
            groups['W_th'].extend([block.W_th_x.weight])
            groups['W_omega'].extend([block.W_omega_x.weight])
            groups['W_gate'].append(block.W_gate.weight)
            groups['W_skip'].append(block.W_skip.weight)
            groups['W_out'].append(block.W_out.weight)
            groups['b_beta'].append(block.b_beta)
            groups['b_alpha'].append(block.b_alpha)
            groups['b_th'].append(block.b_th)
            groups['b_omega'].append(block.b_omega)
            groups['block_output_neuron'].extend([
                block.output_neuron.w,
                block.output_neuron.v_th,
                block.output_neuron.v_init,
                block.hidden_neuron.v_init,
                block.hidden_neuron.w_init,
            ])

            ffn = layer_module.snn_ffn
            groups['ffn_gate_proj'].append(ffn.gate_proj.weight)
            groups['ffn_up_proj'].append(ffn.up_proj.weight)
            groups['ffn_down_proj'].append(ffn.down_proj.weight)
            groups['ffn_skip_proj'].append(ffn.skip_proj.weight)
            groups['ffn_neurons'].extend([
                ffn.gate_neuron.w, ffn.gate_neuron.v_th, ffn.gate_neuron.v_init,
                ffn.up_neuron.w, ffn.up_neuron.v_th, ffn.up_neuron.v_init,
                ffn.output_neuron.w, ffn.output_neuron.v_th, ffn.output_neuron.v_init,
            ])

        return groups

    def get_layer_indices(self) -> dict[int, int]:
        """Return {id(param): layer_idx} for all layer-indexed parameters."""
        mapping = {}
        for i, layer_module in enumerate(self.layers):
            for p in layer_module.parameters():
                mapping[id(p)] = i
        return mapping
