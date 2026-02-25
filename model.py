"""
SNNLanguageModel v7.5b: SNN 隐状态空间语言模型（连续残差流 + FP16 边界编解码）

架构（三段式边界）：
  model.encode(token_ids)    → spike_seq       # 输入边界：embed → FP16 位提取 → 16帧 spike
  model.snn_forward(spike)   → h_out           # 纯 SNN 核心（连续残差流，20 层）
  model.decode(h_out, seq)   → logits          # 输出边界：PLIFNode → spike → FP16 位重建 → proj → norm → tied head

编解码对称性：
  encode: 连续值 → IEEE 754 float16 位提取 → 16 帧 binary {0,1}（detach，固定预处理）
  decode: 16 帧 binary {0,1} ← 输出神经元 ← 连续 h → IEEE 754 位重建 → 连续值（可微分）

数学原理见 SNN_SELECTIVE_STATE_SPACE.md 第 7 节。
"""

from dataclasses import dataclass
from typing import Optional

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

    def _output_neuron_parallel(self, h: torch.Tensor) -> torch.Tensor:
        """输出 PLIF 神经元的 parallel scan 前向：连续 h → binary spike。

        Args:
            h: (TK, batch, D) 连续值（SNN 最后一层输出）

        Returns:
            spike: (TK, batch, D) 二值 {0, 1}
        """
        TK, batch, D = h.shape

        beta = self.output_neuron.beta  # (D,)
        u = (1.0 - beta) * h  # PLIF: u = (1-β) · x

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

    def decode(self, h_out: torch.Tensor, seq_len: int) -> torch.Tensor:
        """输出边界：连续 h → 输出神经元 → spike → FP16 位重建 → logits。

        梯度流: loss → logits → norm → decode_proj → fp16_reconstruct
                → surrogate_grad(output_neuron) → h_out → SNN layers

        Returns: (batch, seq_len, vocab_size)
        """
        h_out = self.output_norm(h_out)                    # RMSNorm: 控制 scale
        spikes = self._output_neuron_parallel(h_out)   # (TK, batch, D), binary {0,1}
        decoded = fp16_decode(spikes, seq_len, K=self.K)  # IEEE 754 位重建 → (batch, seq_len, D)
        h = self.decode_proj(decoded)                      # (batch, seq_len, D)
        h = self.norm(h)                                   # (batch, seq_len, D)
        return F.linear(h, self.embed_tokens.weight)       # (batch, seq_len, vocab)

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
        functional.reset_net(self.output_neuron)

        # ====== Prefill: parallel 处理整个 prompt ======
        spike_seq = self.encode(prompt_ids)  # (prompt_len*K, batch, D)
        h = spike_seq
        for layer_module in self.layers:
            h = layer_module.forward_parallel(h)
        # 此时所有层的所有神经元 .v 状态 = prompt 末尾状态

        # Output path: norm → output_neuron → fp16_decode → proj → norm → logits
        h_norm = self.output_norm(h)
        output_spikes = self._output_neuron_parallel(h_norm)
        decoded = fp16_decode(output_spikes, prompt_len, K=self.K)
        h_dec = self.decode_proj(decoded)
        h_dec = self.norm(h_dec)
        logits = F.linear(h_dec, self.embed_tokens.weight)

        # 采样第一个新 token
        next_token = self._sample(logits[:, -1, :], temperature, top_k)
        generated = [next_token]

        # ====== Autoregressive: 逐 token，forward_parallel 处理 K 帧 ======
        # 等价性证明：layer-by-layer forward_parallel ≡ frame-by-frame single_step
        #   因为层间无反馈（Layer i 的输出不影响 Layer j<i 的状态），
        #   每层内的时间递推 V[t]=β·V[t-1]+(1-β)·x[t] 与处理顺序无关。
        # 优势：复用 Triton parallel scan kernel，避免 K×L 次 Python 循环。
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # 编码单 token → K 帧 spike
            emb = self.embed_tokens(next_token)       # (batch, 1, D)
            spike_frames = fp16_encode(emb, K=self.K)  # (K, batch, D)

            # K 帧通过 SNN（forward_parallel 复用 Triton parallel scan）
            # 不 reset — 神经元 .v 从上一 token 末尾状态继续
            h = spike_frames
            for layer_module in self.layers:
                h = layer_module.forward_parallel(h)

            # Output path: norm → output_neuron → fp16_decode → proj → logits
            h_norm = self.output_norm(h)
            output_spikes = self._output_neuron_parallel(h_norm)
            decoded = fp16_decode(output_spikes, 1, K=self.K)  # (batch, 1, D)
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
        前向传播（v7.5b: 三段式边界架构）。

        encode → spike_seq           # 输入边界（embed → FP16 位提取）
        snn_forward → h_out          # 纯 SNN 核心（连续残差流）
        decode → logits              # 输出边界（PLIFNode → spike → FP16 位重建 → proj → logits）

        梯度流:
          emb ──[detach]──→ spike_seq → SNN layers → h_out
            → output_neuron(surrogate) → spike → fp16_reconstruct → decode_proj → norm → logits
          embed_tokens.weight 通过 tied LM head 获得梯度
          SNN 参数通过 surrogate gradient 从 loss 反传获得梯度
        """
        batch, seq_len = token_ids.shape

        # 重置所有神经元状态
        for layer_module in self.layers:
            functional.reset_net(layer_module)
        functional.reset_net(self.output_neuron)

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

    def compensate_modulation_gradients(self, max_comp: float = 100.0):
        """
        Natural Gradient 补偿：消除 sigmoid/softplus 激活函数对 b_beta/b_alpha 的梯度衰减。

        问题：β = sigmoid(W·x + b_beta)，sigmoid 在高 β 区（β=0.99, sigmoid'=0.01）
        梯度衰减 100x，导致长期记忆神经元（高 β）几乎无法训练。

        补偿：∂L/∂b_compensated = ∂L/∂b / activation'(b)
        等价于在 β/α 空间做梯度下降，而非在 logit/pre-softplus 空间。

        Args:
            max_comp: 补偿因子上限（防止极端值导致不稳定）
        """
        for layer_module in self.layers:
            block = layer_module.snn_block

            # b_beta: sigmoid 饱和补偿
            # sigmoid'(z) = sigmoid(z) · (1 - sigmoid(z)) = β · (1-β)
            if block.b_beta.grad is not None:
                with torch.no_grad():
                    beta = torch.sigmoid(block.b_beta.data)
                    sigmoid_deriv = (beta * (1.0 - beta)).clamp(min=1.0 / max_comp)
                    block.b_beta.grad.div_(sigmoid_deriv)

            # b_alpha: softplus 补偿（较温和，softplus'(z) = sigmoid(z)）
            if block.b_alpha.grad is not None:
                with torch.no_grad():
                    softplus_deriv = torch.sigmoid(block.b_alpha.data).clamp(min=0.1)
                    block.b_alpha.grad.div_(softplus_deriv)

            # b_omega: softplus 补偿（同 b_alpha）
            if block.b_omega.grad is not None:
                with torch.no_grad():
                    softplus_deriv = torch.sigmoid(block.b_omega.data).clamp(min=0.1)
                    block.b_omega.grad.div_(softplus_deriv)

            # b_th: |·| 导数为 ±1，无衰减，不需要补偿

    def get_param_groups(self) -> dict[str, list[nn.Parameter]]:
        """
        按功能分组的可训练参数。
        v7.5b: output_neuron 归入 neuron 组。
        """
        groups = {
            'embedding': [self.embed_tokens.weight],
            'norm': [self.norm.gain],
            'decode': list(self.decode_proj.parameters()),
            # 输出神经元
            'output_neuron': [self.output_neuron.w, self.output_neuron.v_th,
                              self.output_neuron.v_init],
            # RMSNorm（Pre-LN 分支归一化）
            'rms_norms': [self.output_norm.weight],
            # 残差流组件
            'residual_projs': [],
            'input_neurons': [],
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

            # 残差流组件
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
                # 共享 expert → 现有参数组（同 learning rate / weight decay）
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
                # 路由 expert → 堆叠参数（v3: 无 ModuleList）
                groups['ffn_expert_projs'].extend([
                    moe.expert_W_gus, moe.expert_W_down,
                ])
                groups['ffn_expert_neurons'].extend([
                    moe.expert_gu_w, moe.expert_gu_v_th, moe.expert_gu_v_init,
                    moe.expert_out_w, moe.expert_out_v_th, moe.expert_out_v_init,
                ])
                # Router
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
