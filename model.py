"""
SNNLanguageModel v10: 纯 spike-to-spike SNN 语言模型

架构（三段式边界）：
  model.encode(token_ids)    → spike_seq       # 输入边界：embed → FP16 位提取 → 16帧 spike
  model.snn_forward(spike)   → spike_out       # 纯 SNN 核心（spike-to-spike，20 层 binary_residual）
  model.decode(spike_out)    → logits          # 输出边界：spike → FP16 位重建 → proj → norm → tied head

v10 变更（移除连续残差流）：
  - 删除: output_norm, output_neuron, compensate_modulation_gradients
  - SNN 核心层间直接传递 binary spike，不再有连续残差旁路
  - 深度梯度流由 binary_residual (decode → add → encode, STE identity) 保证
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
    纯 spike-to-spike SNN 语言模型（v10: 无连续残差流）。

    Args:
        vocab_size: 词表大小（默认 6144，自训练 BPE）
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数（默认 16）
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
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
        D_ff_shared: int = None,
        D_ff_expert: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.D = D
        self.N = N
        self.K = K
        self.num_layers = num_layers
        self.D_ff = D_ff
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.moe_top_k = top_k
        self.D_ff_shared = D_ff_shared
        self.D_ff_expert = D_ff_expert

        # ====== Embedding + Norm（全部可训练）======
        self.embed_tokens = nn.Embedding(vocab_size, D)
        self.norm = LateralInhibition(D)

        # ====== 解码投影 ======
        self.decode_proj = nn.Linear(D, D)

        # ====== SNN Decoder Layers ======
        self.layers = nn.ModuleList([
            SNNDecoderLayer(
                D=D, N=N, K=K, D_ff=D_ff, v_th_min=v_th_min,
                block_output_v_threshold=0.05,
                ffn_output_v_threshold=0.15,
                num_layers=num_layers,
                layer_idx=i,
                use_moe=use_moe,
                num_experts=num_experts,
                top_k=top_k,
                D_ff_shared=D_ff_shared,
                D_ff_expert=D_ff_expert,
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
        """SNN 核心：spike_seq → spike_out（纯 spike-to-spike）。

        每层 spike → spike，binary_residual (STE identity) 保证深度梯度流。

        Returns: (seq_len*K, batch, D), spike
        """
        spike = spike_seq

        def _layer_forward(layer_mod, x):
            functional.reset_net(layer_mod)
            return layer_mod(x)  # 走 __call__ → forward()，触发 FSDP allgather

        for layer_module in self.layers:
            spike = checkpoint(
                _layer_forward, layer_module, spike,
                use_reentrant=False,
            )
        return spike

    def decode(self, spike_out: torch.Tensor, seq_len: int) -> torch.Tensor:
        """输出边界：spike → FP16 位重建 → logits。

        梯度流: loss → logits → norm → decode_proj → fp16_reconstruct
                → surrogate_grad(SNN layers)

        Returns: (batch, seq_len, vocab_size)
        """
        decoded = fp16_decode(spike_out, seq_len, K=self.K)  # (batch, seq_len, D)
        h = self.decode_proj(decoded)                         # (batch, seq_len, D)
        h = self.norm(h)                                      # (batch, seq_len, D)
        return F.linear(h, self.embed_tokens.weight)          # (batch, seq_len, vocab)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成（SNN 神经元状态跨 token 连续维护）。

        1. Prefill: forward_parallel 并行处理 prompt，建立所有神经元 V 状态
        2. Autoregressive: 逐 token 生成，每 token 用 forward_parallel 处理 K=16 帧
           复用 Triton parallel scan kernel，神经元 V 状态跨 token 连续传递

        Args:
            prompt_ids: (batch, prompt_len) token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（<=0 = greedy）
            top_k: top-k 采样（None/0 = 不限制）
            eos_token_id: 遇到此 token 停止生成

        Returns:
            (batch, prompt_len + generated_len) 完整序列
        """
        batch, prompt_len = prompt_ids.shape

        # 重置所有神经元（新序列的初始条件 V=0）
        for layer_module in self.layers:
            functional.reset_net(layer_module)

        # ====== Prefill: parallel 处理整个 prompt ======
        spike_seq = self.encode(prompt_ids)  # (prompt_len*K, batch, D)
        spike = spike_seq
        for layer_module in self.layers:
            spike = layer_module(spike)

        # Output path: spike → fp16_decode → proj → norm → logits
        decoded = fp16_decode(spike, prompt_len, K=self.K)
        h_dec = self.decode_proj(decoded)
        h_dec = self.norm(h_dec)
        logits = F.linear(h_dec, self.embed_tokens.weight)

        # 采样第一个新 token
        next_token = self._sample(logits[:, -1, :], temperature, top_k)
        generated = [next_token]

        # ====== Autoregressive: 逐 token，forward_parallel 处理 K 帧 ======
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 编码单 token → K 帧 spike
            emb = self.embed_tokens(next_token)       # (batch, 1, D)
            spike_frames = fp16_encode(emb, K=self.K)  # (K, batch, D)

            # K 帧通过 SNN（走 forward() 触发 FSDP allgather）
            spike = spike_frames
            for layer_module in self.layers:
                spike = layer_module(spike)

            # Output path: spike → fp16_decode → proj → logits
            decoded = fp16_decode(spike, 1, K=self.K)  # (batch, 1, D)
            h_dec = self.decode_proj(decoded)
            h_dec = self.norm(h_dec)
            logits = F.linear(h_dec, self.embed_tokens.weight)

            next_token = self._sample(logits[:, -1, :], temperature, top_k)
            generated.append(next_token)

        return torch.cat([prompt_ids, torch.cat(generated, dim=1)], dim=1)

    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """从 logits 采样（temperature + top-k）。

        Returns: (batch, 1)
        """
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor = None,
    ) -> SNNModelOutput:
        """
        前向传播（v10: 纯 spike-to-spike 架构）。

        encode → spike_seq           # 输入边界（embed → FP16 位提取）
        snn_forward → spike_out      # 纯 SNN 核心（spike-to-spike，binary_residual 梯度流）
        decode → logits              # 输出边界（spike → FP16 位重建 → proj → logits）
        """
        batch, seq_len = token_ids.shape

        # 重置所有神经元状态
        for layer_module in self.layers:
            functional.reset_net(layer_module)

        # 三段式
        spike_seq = self.encode(token_ids)        # 输入边界
        spike_out = self.snn_forward(spike_seq)   # SNN 核心
        logits = self.decode(spike_out, seq_len)  # 输出边界

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
        按功能分组的可训练参数（v10: 无连续残差流组件）。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.gain],
            'decode': list(self.decode_proj.parameters()),
            # SNNBlock 参数（v7.8: +W_omega/b_omega）
            'W_in': [],
            'W_beta': [],
            'W_alpha': [],
            'W_th': [],
            'W_omega': [],
            'W_gate': [],
            'W_skip': [],
            'W_out': [],
            'b_beta': [],
            'b_alpha': [],
            'b_th': [],
            'b_omega': [],
            'block_output_neuron': [],
            # SNNFFN 参数
            'ffn_gate_proj': [],
            'ffn_up_proj': [],
            'ffn_down_proj': [],
            'ffn_skip_proj': [],
            'ffn_neurons': [],
            # MoE 路由 expert 参数
            'ffn_expert_projs': [],
            'ffn_expert_neurons': [],
            'ffn_router': [],
        }

        for layer_module in self.layers:
            block = layer_module.snn_block

            # SNNBlock 参数
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

            # SNNFFN 参数（MoE 模式: shared expert 沿用现有 key，routed expert 用新 key）
            if layer_module.use_moe:
                moe = layer_module.snn_ffn
                shared = moe.shared_expert
                groups['ffn_gate_proj'].append(shared.gate_proj.weight)
                groups['ffn_up_proj'].append(shared.up_proj.weight)
                groups['ffn_down_proj'].append(shared.down_proj.weight)
                groups['ffn_skip_proj'].append(shared.skip_proj.weight)
                groups['ffn_neurons'].extend([
                    shared.gate_neuron.w, shared.gate_neuron.v_th, shared.gate_neuron.v_init,
                    shared.up_neuron.w, shared.up_neuron.v_th, shared.up_neuron.v_init,
                    shared.output_neuron.w, shared.output_neuron.v_th, shared.output_neuron.v_init,
                ])
                groups['ffn_expert_projs'].extend([
                    moe.expert_W_gus, moe.expert_W_down,
                ])
                groups['ffn_expert_neurons'].extend([
                    moe.expert_gu_w, moe.expert_gu_v_th, moe.expert_gu_v_init,
                    moe.expert_out_w, moe.expert_out_v_th, moe.expert_out_v_init,
                ])
                groups['ffn_router'].append(moe.router.weight)
            else:
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
