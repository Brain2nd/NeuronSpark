"""
RMSNorm: 残差流分支归一化（Pre-LN 模式）。

位置: h → RMSNorm → PLIFNode → SNN子层 → out_proj → 残差
作用: 控制送入 PLIFNode 的输入 scale，防止残差流漂移/爆炸。
      仅归一化分支输入，残差流本身不被归一化。

对标 Qwen3/LLaMA 的 Pre-LN RMSNorm。
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    x_norm = x / RMS(x) * weight
    RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: 归一化维度
        eps: 数值稳定性
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)
