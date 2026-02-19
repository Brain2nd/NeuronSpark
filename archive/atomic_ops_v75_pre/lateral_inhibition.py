"""
LateralInhibition: 侧抑制归一化（Divisive Normalization）

神经科学基础：
  Carandini & Heeger (2012) "Normalization as a canonical neural computation"
  侧抑制是大脑中最基本的计算原语之一：兴奋性神经元的活动通过抑制性中间神经元池
  反馈调节，实现增益控制（gain control）。

SNN 机制：
  1. 兴奋性群体活动:  activity_i = h_i²
  2. 抑制性中间神经元池: pool = mean(activity) = mean(h²)
  3. 分裂抑制 (shunting inhibition): h_norm = h / sqrt(pool + ε)
  4. 增益调制 (gain modulation): output = gain · h_norm

替换 RMSNorm：数学操作等价，但在 SNN 框架中有明确的神经科学对应——
  RMSNorm 是 divisive normalization 的特例。

Triton fused kernel:
  - 前向: {mean(h²), rsqrt, element-wise mul} → 1 kernel launch
  - 反向: {recompute norm, grad_gain, grad_h} → 1 kernel launch
  - 每行 (D dim) 一个 block，行间并行
"""

import os

import torch
import torch.nn as nn
from spikingjelly.activation_based import base


# ============================================================
# Triton fused kernels
# ============================================================

_SYSTEM_PTXAS = '/usr/local/cuda-13.0/bin/ptxas'
if os.path.exists(_SYSTEM_PTXAS) and 'TRITON_PTXAS_PATH' not in os.environ:
    os.environ['TRITON_PTXAS_PATH'] = _SYSTEM_PTXAS

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


if _HAS_TRITON:

    @triton.jit
    def _li_fwd_kernel(
        X_ptr, GAIN_ptr, OUT_ptr,
        stride_row,
        D: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Forward: out = x * rsqrt(mean(x²) + eps) * gain

        Grid: (num_rows,). Each program processes one row of D elements.
        Computation in float32; storage in input dtype.
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        off = row * stride_row + cols

        # Load in float32
        x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
        gain = tl.load(GAIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Inhibitory pool: population activity
        variance = tl.sum(x * x, axis=0) / D
        rrms = 1.0 / tl.sqrt(variance + eps)

        # Divisive inhibition + gain modulation
        out = x * rrms * gain

        tl.store(OUT_ptr + off, out, mask=mask)

    @triton.jit
    def _li_bwd_kernel(
        DOUT_ptr, X_ptr, GAIN_ptr,
        DX_ptr, DGAIN_ptr,
        stride_row,
        D: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Backward: grad_x, grad_gain (per-row, reduced externally).

        Grid: (num_rows,).
        d_x = rrms * (d_out * gain - x_hat * mean(d_out * gain * x_hat))
        d_gain_row = d_out * x_hat  (sum across rows done outside kernel)
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        off = row * stride_row + cols

        dout = tl.load(DOUT_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + off, mask=mask, other=0.0).to(tl.float32)
        gain = tl.load(GAIN_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Recompute forward (avoid saving intermediate tensors)
        variance = tl.sum(x * x, axis=0) / D
        rrms = 1.0 / tl.sqrt(variance + eps)
        x_hat = x * rrms

        # grad_gain (per-row contribution)
        dgain = dout * x_hat
        tl.store(DGAIN_ptr + off, dgain, mask=mask)

        # grad_x: rrms * (dout*gain - x_hat * mean(dout*gain*x_hat))
        dout_gain = dout * gain
        dot = tl.sum(dout_gain * x_hat, axis=0) / D
        dx = (dout_gain - x_hat * dot) * rrms

        tl.store(DX_ptr + off, dx, mask=mask)


class _LateralInhibitionTriton(torch.autograd.Function):
    """Triton-accelerated lateral inhibition (divisive normalization)."""

    @staticmethod
    def forward(ctx, x, gain, eps):
        orig_shape = x.shape
        D = x.shape[-1]
        x_2d = x.reshape(-1, D).contiguous()
        N = x_2d.shape[0]

        out = torch.empty_like(x_2d)

        BLOCK_D = triton.next_power_of_2(D)
        _li_fwd_kernel[(N,)](
            x_2d, gain, out,
            x_2d.stride(0),
            D=D, eps=eps, BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_2d, gain)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.N = N
        ctx.D = D

        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, gain = ctx.saved_tensors
        D = ctx.D
        N = ctx.N

        grad_2d = grad_output.reshape(N, D).contiguous()

        dx = torch.empty_like(x_2d)
        dgain_rows = torch.empty_like(x_2d)

        BLOCK_D = triton.next_power_of_2(D)
        _li_bwd_kernel[(N,)](
            grad_2d, x_2d, gain,
            dx, dgain_rows,
            x_2d.stride(0),
            D=D, eps=ctx.eps, BLOCK_D=BLOCK_D,
        )

        # Reduce per-row dgain across all rows
        dgain = dgain_rows.sum(dim=0)

        return dx.reshape(ctx.orig_shape), dgain, None


# ============================================================
# Public module
# ============================================================

class LateralInhibition(base.MemoryModule):
    """
    侧抑制归一化层（Divisive Normalization）。

    通过抑制性中间神经元池实现增益控制。

    数学：
      pool = mean(h², dim=-1)            # 抑制性池：群体活动水平
      h_norm = h / sqrt(pool + ε)        # 分裂抑制 (shunting inhibition)
      output = gain · h_norm             # 增益调制 (gain modulation)

    等价于 RMSNorm，但在 SNN 框架中对应 divisive normalization
    (Carandini & Heeger, 2012)，是神经科学中最基本的计算原语之一。

    CUDA: Triton fused kernel（前向+反向各 1 次 launch）
    CPU:  PyTorch fallback

    Args:
        dim: 特征维度（D）
        eps: 数值稳定性
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if _HAS_TRITON and h.is_cuda:
            return _LateralInhibitionTriton.apply(h, self.gain, self.eps)
        # PyTorch fallback
        variance = h.pow(2).mean(-1, keepdim=True)
        h_norm = h * torch.rsqrt(variance + self.eps)
        return self.gain * h_norm

    def extra_repr(self):
        return f'dim={self.dim}, eps={self.eps}'
