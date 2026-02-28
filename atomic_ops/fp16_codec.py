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


# ====== 二进制残差连接（替代 SEW 十进制加法） ======

def _binary_residual_naive(spike_a: Tensor, spike_b: Tensor) -> Tensor:
    """未融合 baseline（仅用于 benchmark 对比）: fp16_decode → add → fp16_encode。

    ~20 个独立 kernel launch，8.6 ms/call。
    """
    TK = spike_a.shape[0]
    T = TK // 16
    val_a = fp16_decode(spike_a, T, 16)
    val_b = fp16_decode(spike_b, T, 16)
    return fp16_encode(val_a + val_b, 16)


def _binary_residual_fwd(spike_a: Tensor, spike_b: Tensor) -> Tensor:
    """内联 decode+add+encode 全流程，供 torch.compile 融合。

    inductor 将 ~20 个独立 kernel（sign/exp/mant 加权求和、exp2、where、
    clamp、bit shift...）融合为 2-3 个 fused kernel。

    Benchmark (B=3, T=512, D=1024):
      naive (unfused):  8.6 ms → compiled: 1.4 ms (6.3x 加速)
    """
    TK, B, D = spike_a.shape
    T = TK // 16
    K = 16
    dtype = spike_a.dtype
    device = spike_a.device

    sa = spike_a.permute(1, 0, 2).reshape(B, T, K, D)
    sb = spike_b.permute(1, 0, 2).reshape(B, T, K, D)

    # 共享权重常量（torch.compile 捕获为 constant）
    exp_w = torch.tensor([16., 8., 4., 2., 1.], device=device, dtype=dtype)
    mant_w = torch.tensor(
        [0.5, 0.25, 0.125, 0.0625, 0.03125,
         0.015625, 0.0078125, 0.00390625,
         0.001953125, 0.0009765625],
        device=device, dtype=dtype,
    )

    # Decode spike_a (inline fp16_decode)
    sign_a = 1.0 - 2.0 * sa[:, :, 0, :]
    exp_a = (sa[:, :, 1:6, :] * exp_w.view(1, 1, 5, 1)).sum(dim=2)
    mant_a = (sa[:, :, 6:, :] * mant_w.view(1, 1, 10, 1)).sum(dim=2)
    normal_a = sign_a * torch.exp2(exp_a - 15.0) * (1.0 + mant_a)
    subnormal_a = sign_a * (2.0 ** -14) * mant_a
    val_a = torch.where(exp_a > 0, normal_a, subnormal_a)

    # Decode spike_b (inline fp16_decode)
    sign_b = 1.0 - 2.0 * sb[:, :, 0, :]
    exp_b = (sb[:, :, 1:6, :] * exp_w.view(1, 1, 5, 1)).sum(dim=2)
    mant_b = (sb[:, :, 6:, :] * mant_w.view(1, 1, 10, 1)).sum(dim=2)
    normal_b = sign_b * torch.exp2(exp_b - 15.0) * (1.0 + mant_b)
    subnormal_b = sign_b * (2.0 ** -14) * mant_b
    val_b = torch.where(exp_b > 0, normal_b, subnormal_b)

    # Add + Encode (inline fp16_encode)
    val_sum = val_a + val_b
    sum_fp16 = val_sum.float().clamp(-65504.0, 65504.0).half()
    sum_int = sum_fp16.view(torch.int16)
    shifts = torch.arange(15, -1, -1, device=device)
    bits = ((sum_int.unsqueeze(2) >> shifts.view(1, 1, K, 1)) & 1).to(dtype)
    return bits.reshape(B, T * K, D).permute(1, 0, 2).contiguous()


# torch.compile 融合: ~20 kernel → 2-3 fused kernel, 6.3x 加速
_binary_residual_compiled = torch.compile(
    _binary_residual_fwd, backend='inductor', fullgraph=True,
)


class _BinaryResidual(torch.autograd.Function):
    """二进制加法 autograd wrapper（torch.compile 融合版）。

    Forward: 精确的 IEEE 754 二进制加法，inductor 融合 decode+add+encode。
    Backward: STE identity gradient — 两条路径都传完整梯度。
    """

    @staticmethod
    def forward(ctx, spike_a, spike_b):
        return _binary_residual_compiled(spike_a, spike_b)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


def binary_residual(spike_a: Tensor, spike_b: Tensor) -> Tensor:
    """二进制残差加法（替代 SEW 的十进制 spike_a + spike_b）。

    将两个二值 spike 序列在 IEEE 754 float16 语义下正确相加：
    decode → continuous add → re-encode，inductor 融合为 2-3 个 kernel。
    输出严格 {0,1}，6.3x 快于未融合版本。

    K=16 hardcoded（IEEE 754 float16 = 16 bits）。

    Args:
        spike_a: (TK, B, D) 二值 spike 序列
        spike_b: (TK, B, D) 二值 spike 序列

    Returns:
        (TK, B, D) 二值 spike 序列，decode 后等于 decode(a) + decode(b)
    """
    return _BinaryResidual.apply(spike_a, spike_b)


class _BinaryEncodeSTE(torch.autograd.Function):
    """连续值 → 二进制编码，STE backward 传梯度。

    用于 MoE combine 后重编码：forward 精确 encode，backward 用 mean-over-K
    近似将 (TK, B, D) 梯度聚合回 (B, T, D)。
    不 save_for_backward（只用 shape），省显存。
    """

    @staticmethod
    def forward(ctx, continuous_btd):
        ctx.B, ctx.T, ctx.D = continuous_btd.shape
        return fp16_encode(continuous_btd, 16)  # (TK, B, D) binary

    @staticmethod
    def backward(ctx, grad_output):
        B, T, D = ctx.B, ctx.T, ctx.D
        # grad_output: (T*16, B, D) → (B, T, 16, D) → mean over K → (B, T, D)
        grad_btd = grad_output.permute(1, 0, 2).reshape(B, T, 16, D).mean(dim=2)
        return grad_btd


def binary_encode_ste(continuous_btd: Tensor) -> Tensor:
    """连续值 → 二进制编码（STE backward）。

    MoE combine 后重编码用：forward 精确，backward 将 K=16 帧梯度
    平均聚合回 per-token 梯度。

    Args:
        continuous_btd: (B, T, D) 连续值

    Returns:
        (T*16, B, D) 二值 spike 序列
    """
    return _BinaryEncodeSTE.apply(continuous_btd)
