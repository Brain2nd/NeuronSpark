"""
SNNDecoderLayer: 单个 SNN 解码层 = SNNBlock（注意力） + SNNFFN（前馈）

对标 Qwen3DecoderLayer:
  Qwen3: RMSNorm → Attention → residual → RMSNorm → MLP → residual
  SNN:   SNNBlock(spike) → SNNFFN(spike)

注：SNN 已在内部处理残差（skip connection），不需要额外的 add-norm。
"""

from spikingjelly.activation_based import base

from .snn_block import SNNBlock
from .snn_ffn import SNNFFN


class SNNDecoderLayer(base.MemoryModule):
    """
    单个 SNN 解码层。

    Args:
        D: 可见维度
        N: 状态扩展因子
        D_ff: FFN 中间层维度
        v_th_min: SNNBlock 动态阈值下限
        block_output_v_threshold: SNNBlock 输出神经元阈值
        ffn_output_v_threshold: SNNFFN 输出神经元阈值
        num_layers: 总层数（传给 SNNFFN 做 down_proj 缩放）
        layer_idx: 当前层索引
    """

    def __init__(
        self,
        D: int,
        N: int,
        D_ff: int,
        v_th_min: float,
        block_output_v_threshold: float,
        ffn_output_v_threshold: float,
        num_layers: int = 1,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.snn_block = SNNBlock(
            D=D, N=N, v_th_min=v_th_min,
            output_v_threshold=block_output_v_threshold,
        )
        self.snn_ffn = SNNFFN(
            D=D, D_ff=D_ff,
            output_v_threshold=ffn_output_v_threshold,
            num_layers=num_layers,
            layer_idx=layer_idx,
        )

    def single_step_forward(self, spike_in):
        """
        spike_in {0,1}^D → SNNBlock → SNNFFN → spike_out {0,1}^D
        """
        spike_mid = self.snn_block(spike_in)  # 注意力
        spike_out = self.snn_ffn(spike_mid)   # 前馈
        return spike_out
