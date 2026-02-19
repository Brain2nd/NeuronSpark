"""
SNNLanguageModel v7.5b: SNN 隐状态空间语言模型（连续残差流 + FP16 边界编码）

v7.5 → v7.5b 变更：
  - 编码层改为 FP16 二进制编码（IEEE 754 位模式，固定预处理，无可训练参数）
  - 解码层改为时间步均值池化（SNN rate coding 自然解码）
  - 删除 encode_proj, bit_weights, bit_scales（共减少 ~590K 参数）
  - 编码/解码作为模型边界操作，不内嵌在模型中

架构（三段式边界）：
  model.encode(token_ids)    → spike_seq       # 输入边界：embed + fp16 位提取
  model.snn_forward(spike)   → h_out           # 纯 SNN 核心（连续残差流）
  model.decode(h_out, seq)   → logits          # 输出边界：均值池化 + proj + norm + tied LM head

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.checkpoint import checkpoint

from atomic_ops import SNNDecoderLayer
from atomic_ops.fp16_codec import fp16_encode, fp16_decode
from atomic_ops.lateral_inhibition import LateralInhibition


@dataclass
class SNNModelOutput:
    """模型输出容器，对齐教程 CausalLMOutputWithPast 接口。"""
    last_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class SNNLanguageModel(nn.Module):
    """
    从零训练的 SNN 隐状态空间语言模型（v7: parallel scan）。

    Args:
        vocab_size: 词表大小（默认 6144，自训练 BPE）
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数（v7 默认 16）
        num_layers: SNN 解码层数
        D_ff: FFN 中间层维度
        v_th_min: 动态阈值下限
    """

    def __init__(
        self,
        vocab_size: int = 6144,
        D: int = 1024,
        N: int = 8,
        K: int = 16,
        num_layers: int = 20,
        D_ff: int = 3072,
        v_th_min: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff

        # ====== Embedding + Norm（全部可训练）======
        self.embed_tokens = nn.Embedding(vocab_size, D)
        self.norm = LateralInhibition(D)

        # ====== 解码投影（编码已改为 FP16 边界操作，无可训练参数）======
        self.decode_proj = nn.Linear(D, D)

        # ====== SNN Decoder Layers ======
        self.layers = nn.ModuleList([
            SNNDecoderLayer(
                D=D, N=N, D_ff=D_ff, v_th_min=v_th_min,
                block_output_v_threshold=0.05,
                ffn_output_v_threshold=0.15,
                num_layers=num_layers,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        """初始化所有可训练权重（从零训练）。"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.decode_proj.weight)
        nn.init.zeros_(self.decode_proj.bias)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入边界：token_ids → spike_seq。

        Embedding lookup + FP16 二进制编码，作为完整的输入边界操作。
        输出 detached，编码不参与梯度计算。

        Returns: (seq_len*K, batch, D), detached binary {0,1}
        """
        emb = self.embed_tokens(token_ids)       # (batch, seq_len, D)
        return fp16_encode(emb, K=self.K)         # (seq_len*K, batch, D), detached

    def snn_forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """SNN 核心：spike_seq → h_out。

        纯 SNN 层计算，带梯度检查点。

        Returns: (seq_len*K, batch, D), 连续值
        """
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

    def decode(self, h_out: torch.Tensor, seq_len: int) -> torch.Tensor:
        """输出边界：最后一层输出 → logits。

        fp16 均值池化 + 投影 + 归一化 + tied LM head，
        作为完整的输出边界操作。

        Returns: (batch, seq_len, vocab_size)
        """
        decoded = fp16_decode(h_out, seq_len, K=self.K)  # (batch, seq_len, D)
        h = self.decode_proj(decoded)                     # (batch, seq_len, D)
        h = self.norm(h)                                  # (batch, seq_len, D)
        return F.linear(h, self.embed_tokens.weight)      # (batch, seq_len, vocab)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        前向传播（v7.5b: 三段式边界架构）。

        encode(token_ids) → spike_seq           # 输入边界（embed + fp16 位提取）
        snn_forward(spike_seq) → h_out          # 纯 SNN 核心
        decode(h_out, seq_len) → logits         # 输出边界（decode + proj + norm + tied logits）

        梯度流:
          emb ──[detach]──→ spike_seq → SNN layers → h_out → mean_pool → decode_proj → norm → logits
          embed_tokens.weight 通过 tied LM head 获得梯度
          SNN 参数通过 surrogate gradient 从 loss 反传获得梯度
        """
        batch, seq_len = token_ids.shape

        # 重置所有 Layer 内部的神经元状态
        for layer_module in self.layers:
            functional.reset_net(layer_module)

        # 三段式
        spike_seq = self.encode(token_ids)        # 输入边界
        h_out = self.snn_forward(spike_seq)       # SNN 核心
        logits = self.decode(h_out, seq_len)      # 输出边界

        if target_ids is not None:
            logits_flat = logits.reshape(-1, self.vocab_size)
            targets_flat = target_ids.reshape(-1)
            self.last_loss = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=0, reduction='none',
            )
            return SNNModelOutput(last_loss=self.last_loss)

        return SNNModelOutput(logits=logits)

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        按功能分组的可训练参数。
        v7.5: 新增 residual_projs 和 input_neurons 组。
        v7.5b: 移除层内 LateralInhibition，仅保留输出 LI。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.gain],
            'decode': list(self.decode_proj.parameters()),
            # 残差流组件（v7.5 新增）
            'residual_projs': [],
            'input_neurons': [],
            # SNNBlock 参数（v7: 无 W_V）
            'W_in': [],
            'W_beta': [],
            'W_alpha': [],
            'W_th': [],
            'W_gate': [],
            'W_skip': [],
            'W_out': [],
            'b_beta': [],
            'b_alpha': [],
            'b_th': [],
            'block_output_neuron': [],
            # SNNFFN 参数
            'ffn_gate_proj': [],
            'ffn_up_proj': [],
            'ffn_down_proj': [],
            'ffn_skip_proj': [],
            'ffn_neurons': [],
        }

        for layer_module in self.layers:
            block = layer_module.snn_block
            ffn = layer_module.snn_ffn

            # 残差流组件（v7.5: 残差投影 + 输入神经元，无层内归一化）
            groups['residual_projs'].extend([
                layer_module.block_out_proj.weight,
                layer_module.ffn_out_proj.weight,
            ])
            groups['input_neurons'].extend([
                layer_module.input_neuron1.w,
                layer_module.input_neuron1.v_th,
                layer_module.input_neuron2.w,
                layer_module.input_neuron2.v_th,
            ])

            # SNNBlock 参数
            groups['W_in'].append(block.W_in.weight)
            groups['W_beta'].extend([block.W_beta_x.weight])
            groups['W_alpha'].extend([block.W_alpha_x.weight])
            groups['W_th'].extend([block.W_th_x.weight])
            groups['W_gate'].append(block.W_gate.weight)
            groups['W_skip'].append(block.W_skip.weight)
            groups['W_out'].append(block.W_out.weight)
            # v7: W_V 已移除
            groups['b_beta'].append(block.b_beta)
            groups['b_alpha'].append(block.b_alpha)
            groups['b_th'].append(block.b_th)
            groups['block_output_neuron'].extend([
                block.output_neuron.w,
                block.output_neuron.v_th,
            ])

            # SNNFFN 参数
            groups['ffn_gate_proj'].append(ffn.gate_proj.weight)
            groups['ffn_up_proj'].append(ffn.up_proj.weight)
            groups['ffn_down_proj'].append(ffn.down_proj.weight)
            groups['ffn_skip_proj'].append(ffn.skip_proj.weight)
            groups['ffn_neurons'].extend([
                ffn.gate_neuron.w, ffn.gate_neuron.v_th,
                ffn.up_neuron.w, ffn.up_neuron.v_th,
                ffn.output_neuron.w, ffn.output_neuron.v_th,
            ])

        return groups
