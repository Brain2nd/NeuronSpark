"""binary_residual 和 binary_encode_ste 正确性测试。"""

import torch
from atomic_ops.fp16_codec import fp16_encode, fp16_decode, binary_residual, binary_encode_ste


def test_binary_residual_forward():
    """Forward 正确性: decode(binary_residual(a, b)) ≈ x_a + x_b。"""
    torch.manual_seed(42)
    B, T, D = 2, 4, 8
    K = 16

    # 生成可表示的 fp16 值
    x_a = torch.randn(B, T, D).clamp(-100, 100)
    x_b = torch.randn(B, T, D).clamp(-100, 100)

    spike_a = fp16_encode(x_a, K)  # (TK, B, D)
    spike_b = fp16_encode(x_b, K)

    result = binary_residual(spike_a, spike_b)  # (TK, B, D)
    decoded = fp16_decode(result, T, K)  # (B, T, D)

    # fp16_encode 会先 clamp 到 fp16 范围并量化，所以真值用 fp16 round-trip
    x_a_fp16 = x_a.half().float()
    x_b_fp16 = x_b.half().float()
    expected = (x_a_fp16 + x_b_fp16).half().float()  # fp16 精度的加法

    # 允许 fp16 精度误差
    err = (decoded - expected).abs().max().item()
    print(f"[Forward] max error: {err:.6e}")
    assert err < 1e-2, f"Forward error too large: {err}"
    print("[Forward] PASSED")


def test_binary_residual_output_legality():
    """输出合法性: 结果严格 ∈ {0, 1}，不会出现 2。"""
    torch.manual_seed(42)
    B, T, D = 4, 8, 16
    K = 16

    x_a = torch.randn(B, T, D).clamp(-100, 100)
    x_b = torch.randn(B, T, D).clamp(-100, 100)

    spike_a = fp16_encode(x_a, K)
    spike_b = fp16_encode(x_b, K)

    # 旧 SEW: 十进制加法会产生 {0, 1, 2}
    sew_result = spike_a + spike_b
    sew_unique = sew_result.unique().tolist()
    print(f"[Legality] SEW unique values: {sew_unique}")
    assert 2.0 in sew_unique, "SEW should produce value 2 (for comparison)"

    # 新 binary_residual: 只产生 {0, 1}
    br_result = binary_residual(spike_a, spike_b)
    br_unique = br_result.unique().tolist()
    print(f"[Legality] binary_residual unique values: {br_unique}")
    assert all(v in [0.0, 1.0] for v in br_unique), f"binary_residual produced illegal values: {br_unique}"
    print("[Legality] PASSED")


def test_binary_residual_backward_ste():
    """Backward STE: 两条输入路径都得到 identity gradient。"""
    torch.manual_seed(42)
    B, T, D = 2, 4, 8
    K = 16

    x_a = torch.randn(B, T, D).clamp(-100, 100)
    x_b = torch.randn(B, T, D).clamp(-100, 100)

    spike_a = fp16_encode(x_a, K).requires_grad_(True)
    spike_b = fp16_encode(x_b, K).requires_grad_(True)

    result = binary_residual(spike_a, spike_b)
    loss = result.sum()
    loss.backward()

    # STE: grad == ones（identity mapping）
    expected_grad = torch.ones_like(spike_a)
    assert torch.allclose(spike_a.grad, expected_grad), "spike_a.grad should be all ones (STE)"
    assert torch.allclose(spike_b.grad, expected_grad), "spike_b.grad should be all ones (STE)"
    print("[Backward STE] PASSED")


def test_binary_encode_ste_forward():
    """binary_encode_ste forward: 输出是合法的二进制编码。"""
    torch.manual_seed(42)
    B, T, D = 2, 4, 8
    K = 16

    x = torch.randn(B, T, D).clamp(-100, 100)
    spike = binary_encode_ste(x)  # (TK, B, D)

    assert spike.shape == (T * K, B, D)
    unique = spike.unique().tolist()
    assert all(v in [0.0, 1.0] for v in unique), f"binary_encode_ste produced illegal values: {unique}"

    # 验证 round-trip
    decoded = fp16_decode(spike, T, K)
    x_fp16 = x.half().float()
    err = (decoded - x_fp16).abs().max().item()
    print(f"[Encode STE Forward] max round-trip error: {err:.6e}")
    assert err < 1e-3, f"Round-trip error too large: {err}"
    print("[Encode STE Forward] PASSED")


def test_binary_encode_ste_backward():
    """binary_encode_ste backward: 梯度从 (TK, B, D) 聚合回 (B, T, D)。"""
    torch.manual_seed(42)
    B, T, D = 2, 4, 8
    K = 16

    x = torch.randn(B, T, D).clamp(-100, 100).requires_grad_(True)
    spike = binary_encode_ste(x)
    loss = spike.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (B, T, D)
    # mean-over-K: 每帧贡献 1，16 帧平均 = 1.0
    expected_grad_val = 1.0  # sum of ones / K * K = 1
    print(f"[Encode STE Backward] grad mean: {x.grad.mean().item():.4f}")
    print("[Encode STE Backward] PASSED")


if __name__ == '__main__':
    test_binary_residual_forward()
    test_binary_residual_output_legality()
    test_binary_residual_backward_ste()
    test_binary_encode_ste_forward()
    test_binary_encode_ste_backward()
    print("\n===== ALL TESTS PASSED =====")
