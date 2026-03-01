"""
SNNDecoderLayer: 纯 spike-to-spike 解码层

  spike_in → SNNBlock → SNNFFN → spike_out

v10 变更（移除连续残差流）：
  - 删除: RMSNorm, 输入 PLIF 神经元, 位权编码, out_proj, 连续残差
  - 层间直接传递 binary spike {0, 1}
  - 深度梯度流由 binary_residual (decode → add → encode, STE) 保证
  - 每层时序窗口 ~1/(1-beta) 步，20 层堆叠 → ~120 tokens 有效上下文
"""

import torch
from spikingjelly.activation_based import base

from .snn_block import SNNBlock
from .snn_ffn import SNNFFN


class SNNDecoderLayer(base.MemoryModule):
    """
    纯 spike-to-spike SNN 解码层。

    spike_in → SNNBlock → SNNFFN → spike_out
    梯度通过 binary_residual (STE identity) 穿越所有层，无连续残差旁路。

    Args:
        D: 可见维度
        N: 状态扩展因子
        K: 每 token 的 SNN 时间步数
        D_ff: FFN 中间层维度
        v_th_min: SNNBlock 动态阈值下限
        block_output_v_threshold: SNNBlock 输出神经元阈值
        ffn_output_v_threshold: SNNFFN 输出神经元阈值
        num_layers: 总层数（用于 SNNFFN down_proj 缩放）
        layer_idx: 当前层索引
        use_moe: 是否启用 MoE-SNNFFN
        num_experts: 路由 expert 数量
        top_k: 每 token 选中的 expert 数
        D_ff_shared: 共享 expert 中间层维度（默认 D）
        D_ff_expert: 路由 expert 中间层维度（默认 D//2）
    """

    def __init__(
        self,
        D: int,
        N: int,
        K: int,
        D_ff: int,
        v_th_min: float,
        block_output_v_threshold: float,
        ffn_output_v_threshold: float,
        num_layers: int = 1,
        layer_idx: int = 0,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 2,
        D_ff_shared: int = None,
        D_ff_expert: int = None,
    ):
        super().__init__()
        self.D = D
        self.K = K

        self.snn_block = SNNBlock(
            D=D, N=N, v_th_min=v_th_min,
            output_v_threshold=block_output_v_threshold,
        )

        self.use_moe = use_moe
        if use_moe:
            from .moe_snn_ffn import MoESNNFFN
            D_ff_shared = D_ff_shared or D
            D_ff_expert = D_ff_expert or D // 2
            self.snn_ffn = MoESNNFFN(
                D=D, D_ff_shared=D_ff_shared, D_ff_expert=D_ff_expert,
                num_experts=num_experts, top_k=top_k, K=K,
                output_v_threshold=ffn_output_v_threshold,
                num_layers=num_layers,
                layer_idx=layer_idx,
            )
        else:
            self.snn_ffn = SNNFFN(
                D=D, D_ff=D_ff,
                output_v_threshold=ffn_output_v_threshold,
                num_layers=num_layers,
                layer_idx=layer_idx,
            )

    def forward(self, spike_in):
        """
        前向传播（FSDP 入口）：代理到 forward_parallel。

        FSDP 只拦截 __call__ → forward() 路径来 allgather 参数。
        必须通过此方法而非直接调 forward_parallel()。

        Args:
            spike_in: (TK, batch, D) — binary spike {0, 1}

        Returns:
            spike_out: (TK, batch, D) — binary spike {0, 1}
        """
        return self.forward_parallel(spike_in)

    def forward_parallel(self, spike_in):
        """
        并行前向传播：纯 spike-to-spike。

        注意：多卡 FSDP 时必须通过 forward() 调用，不要直接调此方法。

        Args:
            spike_in: (TK, batch, D) — binary spike {0, 1}

        Returns:
            spike_out: (TK, batch, D) — binary spike {0, 1}
        """
        # 子层 1: SNNBlock（内含 binary_residual）
        spike_block = self.snn_block.forward_parallel(spike_in)

        # 子层 2: SNNFFN（内含 binary_residual）
        if self.use_moe:
            spike_ffn, router_info = self.snn_ffn.forward_parallel(spike_block)
            self._last_router_info = router_info
        else:
            spike_ffn = self.snn_ffn.forward_parallel(spike_block)

        return spike_ffn

    def single_step_forward(self, spike_in):
        """
        单步前向传播：纯 spike-to-spike。

        Args:
            spike_in: (batch, D) — binary spike {0, 1}

        Returns:
            spike_out: (batch, D) — binary spike {0, 1}
        """
        spike_block = self.snn_block.single_step_forward(spike_in)

        if self.use_moe:
            spike_ffn = self.snn_ffn.single_step_forward(spike_block)
        else:
            spike_ffn = self.snn_ffn.single_step_forward(spike_block)

        return spike_ffn
