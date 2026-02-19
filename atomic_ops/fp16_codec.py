"""
FP16 二进制编码/解码 — 模型边界操作（无可训练参数）。

IEEE 754 float16 位布局（K=16 时间步）:
  时间步:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  位:     sign  E4   E3   E2   E1   E0   M9   M8   M7   M6   M5   M4   M3   M2   M1   M0
  含义:   符号  ←── 指数(bias=15) ──→  ←────────── 尾数(隐含 1.xxx) ──────────→

编码: embedding → IEEE 754 float16 位提取 → 16 帧二值 spike（detach，固定预处理）
解码: 16 帧二值 spike → IEEE 754 位重建 → 连续值（可微分，梯度通过 surrogate grad 传播）
"""

import torch
from torch import Tensor


def fp16_encode(emb: Tensor, K: int = 16) -> Tensor:
    """FP16 二进制编码（模型边界操作，固定预处理）。

    将连续 embedding 转为 IEEE 754 float16 位模式，作为 SNN 的 spike 输入。

    Args:
        emb: (batch, seq_len, D) 连续 embedding
        K: 时间步数（必须为 16，对应 float16 的 16 位）

    Returns:
        spike_seq: (seq_len*K, batch, D) 二值 {0, 1}, detached
    """
    batch, seq_len, D = emb.shape

    # 转为 float16 获取 IEEE 754 位模式
    # clamp 防止 overflow 产生 Inf（float16 最大值 65504）
    emb_fp16 = emb.float().clamp(-65504.0, 65504.0).half()
    bits_int = emb_fp16.view(torch.int16)  # (batch, seq_len, D)

    # 提取 16 位（MSB first: sign, exponent, mantissa）
    shifts = torch.arange(15, -1, -1, device=emb.device)  # [15, 14, ..., 0]
    # bits_int: (batch, seq_len, D) → unsqueeze → (batch, seq_len, 1, D)
    # shifts: (K,) → view → (1, 1, K, 1)
    bits = ((bits_int.unsqueeze(2) >> shifts.view(1, 1, K, 1)) & 1)  # (batch, seq_len, K, D)

    # 转为计算 dtype 并 detach（编码不参与梯度）
    bits = bits.to(emb.dtype).detach()

    # reshape → (seq_len*K, batch, D)
    return bits.reshape(batch, seq_len * K, D).permute(1, 0, 2).contiguous()


def fp16_decode(spikes: Tensor, seq_len: int, K: int = 16) -> Tensor:
    """FP16 精确位解码：从 16 个二值 spike 重建 float16 值。

    fp16_encode 的精确逆操作。全程可微分——梯度通过 IEEE 754 重建公式
    传到每个 spike 输出，再经 surrogate gradient 传入 SNN。

    IEEE 754 float16 重建:
      Normal   (exp > 0):  (-1)^sign * 2^(exp - 15) * (1 + mant_frac)
      Subnormal (exp = 0): (-1)^sign * 2^(-14)      * mant_frac
      其中 mant_frac = Σ mant_bit_i * 2^{-(i+1)}, i=0..9

    Args:
        spikes: (seq_len*K, batch, D) 二值 {0, 1}（输出神经元的 spike）
        seq_len: token 序列长度
        K: 时间步数（= 16）

    Returns:
        decoded: (batch, seq_len, D) 连续值
    """
    batch, D = spikes.shape[1], spikes.shape[2]

    # (seq_len*K, batch, D) → (batch, seq_len, K, D)
    s = spikes.permute(1, 0, 2).reshape(batch, seq_len, K, D)

    # ---- Sign: bit 0 ----
    sign = 1.0 - 2.0 * s[:, :, 0, :]  # +1 or -1

    # ---- Exponent: bits 1-5, 加权求和 → 整数 0~31 ----
    exp_weights = torch.tensor(
        [16.0, 8.0, 4.0, 2.0, 1.0],
        device=spikes.device, dtype=spikes.dtype,
    )
    exp_val = (s[:, :, 1:6, :] * exp_weights.view(1, 1, 5, 1)).sum(dim=2)

    # ---- Mantissa fraction: bits 6-15, 加权求和 → [0, 1) ----
    mant_weights = torch.tensor(
        [2.0 ** (-i) for i in range(1, 11)],
        device=spikes.device, dtype=spikes.dtype,
    )
    mant_frac = (s[:, :, 6:, :] * mant_weights.view(1, 1, 10, 1)).sum(dim=2)

    # ---- IEEE 754 重建 ----
    # Normal:    (-1)^s * 2^(exp-15) * (1 + mant_frac)
    # Subnormal: (-1)^s * 2^(-14)   * mant_frac
    is_normal = (exp_val > 0)

    normal_val = sign * torch.exp2(exp_val - 15.0) * (1.0 + mant_frac)
    subnormal_val = sign * (2.0 ** -14) * mant_frac

    return torch.where(is_normal, normal_val, subnormal_val)
