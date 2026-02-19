"""
FP16 二进制编码/解码 — 模型边界操作（无可训练参数）。

IEEE 754 float16 位布局（K=16 时间步）:
  时间步:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  位:     sign  E4   E3   E2   E1   E0   M9   M8   M7   M6   M5   M4   M3   M2   M1   M0
  含义:   符号  ←── 指数(bias=15) ──→  ←────────── 尾数(隐含 1.xxx) ──────────→

编码是固定数据预处理（类比高斯位置编码），不参与梯度计算。
解码使用均值池化（SNN rate coding 的自然解码方式）。
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


def fp16_decode(h: Tensor, seq_len: int, K: int = 16) -> Tensor:
    """时间步均值池化（模型边界操作）。

    连续残差流经过多层 SNN 后，h 已非二值，FP16 位重建无意义。
    均值池化 = SNN rate coding 的自然解码。

    Args:
        h: (seq_len*K, batch, D) 连续值（SNN 层输出）
        seq_len: token 序列长度
        K: 每 token 的时间步数

    Returns:
        decoded: (batch, seq_len, D)
    """
    batch, D = h.shape[1], h.shape[2]
    # (seq_len*K, batch, D) → (batch, seq_len*K, D) → (batch, seq_len, K, D)
    h = h.permute(1, 0, 2).reshape(batch, seq_len, K, D)
    return h.mean(dim=2)
